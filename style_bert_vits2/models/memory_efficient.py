"""
メモリ効率化のためのバケツ化とメモリプール管理モジュール
"""

import threading
import time
from typing import Dict, Optional, Tuple

import torch

from style_bert_vits2.logging import logger


# バケツサイズの定義（2のべき乗系列で拡張）
# 短い系列から長い系列まで効率的にカバー
BUCKET_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048]

# BERT専用バケツサイズ（トークン長最適化）
# DeBERTa-v2対応、断片化削減とFLOPs増加のバランス
BERT_BUCKET_SIZES = [32, 64, 96, 128, 192, 256, 384, 512]

# グローバルメモリプール（dtype別に管理）
# key: (dtype, device) -> dict[bucket_size -> (tensor, last_used_time)]
_memory_pools: dict[tuple[torch.dtype, str], dict[int, tuple[torch.Tensor, float]]] = {}

# プール用途別管理（Mid-lived vs Long-lived分離）
# key: (pool_type, dtype, device) -> dict[bucket_size -> (tensor, last_used_time)]
_typed_memory_pools: dict[
    tuple[str, torch.dtype, str], dict[int, tuple[torch.Tensor, float]]
] = {}

# メモリプール操作のスレッドセーフティ確保
_pool_lock = threading.Lock()

# メモリプールの設定
POOL_MAX_AGE_SECONDS = 900  # 15分間未使用のテンソルは解放
POOL_MAX_MEMORY_GB = 2.0  # プール全体の最大メモリ使用量
POOL_CHECK_INTERVAL = 60  # プールチェック間隔（秒）

# 最後のプールチェック時刻
_last_pool_check = time.time()


def get_bucket_size(actual_length: int, max_overhead_ratio: float = 1.5) -> int:
    """
    実際の長さに対して最適なバケツサイズを返す

    Args:
        actual_length: 実際の系列長
        max_overhead_ratio: 許容する最大オーバーヘッド比率（デフォルト1.5倍）

    Returns:
        最適なバケツサイズ
    """
    # 最大オーバーヘッド比率を超えない最小のバケツサイズを選択
    for bucket_size in BUCKET_SIZES:
        if bucket_size >= actual_length:
            if bucket_size <= actual_length * max_overhead_ratio:
                return bucket_size
            # オーバーヘッドが大きすぎる場合は実長を返す
            return actual_length

    # バケツサイズが見つからない場合は実長を返す
    return actual_length


def get_pool_key(dtype: torch.dtype, device: str) -> tuple[torch.dtype, str]:
    """プールのキーを生成（dtype別管理）"""
    return (dtype, device)


def get_typed_pool_key(
    pool_type: str, dtype: torch.dtype, device: str
) -> tuple[str, torch.dtype, str]:
    """用途別プールのキーを生成"""
    return (pool_type, dtype, device)


def get_bert_bucket_size(actual_length: int, max_overhead_ratio: float = 1.3) -> int:
    """
    BERTトークン長に対して最適なバケツサイズを返す

    Args:
        actual_length: 実際のトークン長
        max_overhead_ratio: 許容する最大オーバーヘッド比率（デフォルト1.3倍）

    Returns:
        最適なバケツサイズ
    """
    # BERT専用のバケツサイズから選択
    for bucket_size in BERT_BUCKET_SIZES:
        if bucket_size >= actual_length:
            if bucket_size <= actual_length * max_overhead_ratio:
                return bucket_size
            # オーバーヘッドが大きすぎる場合は実長を返す
            return actual_length

    # バケツサイズが見つからない場合は実長を返す（長文対応）
    return actual_length


def _check_and_cleanup_pools():
    """古いテンソルを解放し、メモリ上限をチェック"""
    global _last_pool_check

    current_time = time.time()
    if current_time - _last_pool_check < POOL_CHECK_INTERVAL:
        return

    _last_pool_check = current_time
    total_memory_bytes = 0
    keys_to_remove = []

    # 古いテンソルの検出とメモリ使用量計算
    for pool_key, pool in _memory_pools.items():
        bucket_sizes_to_remove = []

        for bucket_size, (tensor, last_used) in pool.items():
            if current_time - last_used > POOL_MAX_AGE_SECONDS:
                bucket_sizes_to_remove.append(bucket_size)
            else:
                total_memory_bytes += tensor.element_size() * tensor.numel()

        # 古いテンソルを削除
        for bucket_size in bucket_sizes_to_remove:
            del pool[bucket_size]
            logger.debug(f"Released old tensor: {pool_key}, size={bucket_size}")

        if not pool:
            keys_to_remove.append(pool_key)

    # 空のプールを削除
    for key in keys_to_remove:
        del _memory_pools[key]

    # メモリ上限チェック
    total_memory_gb = total_memory_bytes / (1024**3)
    if total_memory_gb > POOL_MAX_MEMORY_GB:
        logger.warning(
            f"Memory pool exceeds limit: {total_memory_gb:.2f}GB > {POOL_MAX_MEMORY_GB}GB. "
            "Clearing oldest tensors..."
        )
        _evict_oldest_tensors(total_memory_gb - POOL_MAX_MEMORY_GB)


def _evict_oldest_tensors(memory_to_free_gb: float):
    """最も古いテンソルから順に解放"""
    # すべてのテンソルを最終使用時刻でソート
    all_tensors = []
    for pool_key, pool in _memory_pools.items():
        for bucket_size, (tensor, last_used) in pool.items():
            memory_gb = tensor.element_size() * tensor.numel() / (1024**3)
            all_tensors.append((last_used, pool_key, bucket_size, memory_gb))

    all_tensors.sort()  # 最も古いものから

    freed_memory = 0
    for last_used, pool_key, bucket_size, memory_gb in all_tensors:
        if freed_memory >= memory_to_free_gb:
            break

        if pool_key in _memory_pools and bucket_size in _memory_pools[pool_key]:
            del _memory_pools[pool_key][bucket_size]
            freed_memory += memory_gb
            logger.debug(
                f"Evicted tensor: {pool_key}, size={bucket_size}, freed={memory_gb:.3f}GB"
            )


def allocate_bucketed_tensor(
    shape: tuple[int, ...],
    actual_length: int,
    length_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    use_pool: bool = True,
) -> tuple[torch.Tensor, int]:
    """
    バケツ化されたテンソルを割り当てる

    Args:
        shape: テンソルの形状
        actual_length: 実際の系列長
        length_dim: 長さ次元のインデックス
        dtype: データ型
        device: デバイス
        use_pool: メモリプールを使用するか

    Returns:
        (allocated_tensor, bucket_size): 割り当てられたテンソルとバケツサイズ
    """
    bucket_size = get_bucket_size(actual_length)

    # バケツ化された形状を作成
    bucketed_shape = list(shape)
    bucketed_shape[length_dim] = bucket_size

    if use_pool and bucket_size in BUCKET_SIZES:
        # 定期的なクリーンアップ
        _check_and_cleanup_pools()

        pool_key = get_pool_key(dtype, str(device))
        if pool_key not in _memory_pools:
            _memory_pools[pool_key] = {}

        pool = _memory_pools[pool_key]

        # プールから取得または新規作成
        if bucket_size in pool:
            # 既存のテンソルを再利用
            tensor, _ = pool[bucket_size]
            if tensor.shape == tuple(bucketed_shape):
                # 最終使用時刻を更新
                pool[bucket_size] = (tensor, time.time())
                return tensor, bucket_size

        # 新規作成してプールに追加（初期化なし）
        tensor = torch.empty(bucketed_shape, dtype=dtype, device=device)
        pool[bucket_size] = (tensor, time.time())
        return tensor, bucket_size

    # プールを使わない場合は通常の割り当て（初期化なし）
    return torch.empty(bucketed_shape, dtype=dtype, device=device), bucket_size


def copy_to_bucketed_tensor(
    source: torch.Tensor,
    length_dim: int = -1,
    model_name: str = "default",  # 互換性のため残すが使用しない
    use_pool: bool = True,
) -> tuple[torch.Tensor, int]:
    """
    既存のテンソルをバケツ化されたテンソルにコピー

    Args:
        source: ソーステンソル
        length_dim: 長さ次元のインデックス
        model_name: (非推奨) モデル名
        use_pool: メモリプールを使用するか

    Returns:
        (bucketed_tensor, actual_length): バケツ化されたテンソルと実際の長さ
    """
    actual_length = source.shape[length_dim]

    # バケツ化されたテンソルを割り当て
    bucketed_tensor, bucket_size = allocate_bucketed_tensor(
        source.shape,
        actual_length,
        length_dim,
        source.dtype,
        source.device,
        use_pool,
    )

    # パディング部分を含めて全体をゼロで初期化
    # CUDAカーネルが未初期化メモリにアクセスしてエラーになることを防ぐ
    bucketed_tensor.zero_()

    # データをコピー（実際の長さ分のみ）
    if length_dim == -1 or length_dim == len(source.shape) - 1:
        bucketed_tensor[..., :actual_length] = source
    elif length_dim == 0:
        bucketed_tensor[:actual_length] = source
    elif length_dim == 1:
        bucketed_tensor[:, :actual_length] = source
    elif length_dim == 2:
        bucketed_tensor[:, :, :actual_length] = source
    else:
        raise ValueError(f"Unsupported length_dim: {length_dim}")

    return bucketed_tensor, actual_length


def clear_memory_pools(model_name: str | None = None):
    """
    メモリプールをクリア

    Args:
        model_name: (非推奨) 特定のモデルのプールのみクリアする場合は指定
    """
    global _memory_pools

    # model_name引数は無視（後方互換性のため残す）
    _memory_pools.clear()
    logger.info("All memory pools cleared")

    # CUDAキャッシュもクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_pool_stats() -> dict[str, dict[str, any]]:
    """
    メモリプールの統計情報を取得

    Returns:
        プールごとの統計情報
    """
    stats = {}
    total_memory_bytes = 0

    for pool_key, pool in _memory_pools.items():
        dtype, device = pool_key
        pool_memory_bytes = 0
        bucket_info = {}

        for bucket_size, (tensor, last_used) in pool.items():
            memory_bytes = tensor.element_size() * tensor.numel()
            pool_memory_bytes += memory_bytes
            age_seconds = time.time() - last_used

            bucket_info[str(bucket_size)] = {
                "memory_mb": memory_bytes / (1024**2),
                "age_seconds": int(age_seconds),
            }

        total_memory_bytes += pool_memory_bytes

        stats[f"{dtype}_{device}"] = {
            "num_buckets": len(pool),
            "memory_mb": pool_memory_bytes / (1024**2),
            "buckets": bucket_info,
        }

    stats["_total"] = {
        "memory_mb": total_memory_bytes / (1024**2),
        "memory_gb": total_memory_bytes / (1024**3),
        "limit_gb": POOL_MAX_MEMORY_GB,
    }

    return stats


def bucket_bert_inputs(
    inputs: dict[str, torch.Tensor],
    pool_type: str = "bert_mid_lived",
    max_overhead_ratio: float = 1.3,
    use_pool: bool = True,
) -> tuple[dict[str, torch.Tensor], int]:
    """
    BERT入力テンソルをバケツ化する

    Args:
        inputs: BERT入力辞書（input_ids, attention_mask, token_type_ids等）
        pool_type: プール種別（Mid-lived管理用）
        max_overhead_ratio: 許容する最大オーバーヘッド比率
        use_pool: メモリプールを使用するか

    Returns:
        (bucketed_inputs, actual_length): バケツ化された入力辞書と実際の長さ
    """
    # input_idsから実際のトークン長を取得
    input_ids = inputs["input_ids"]
    actual_length = input_ids.shape[1]  # [batch_size, seq_len]
    bucket_size = get_bert_bucket_size(actual_length, max_overhead_ratio)

    # 実長と同じ場合はバケツ化不要
    if bucket_size == actual_length:
        return inputs, actual_length

    bucketed_inputs = {}

    with _pool_lock:  # スレッドセーフティ確保
        for key, tensor in inputs.items():
            if key in ["input_ids", "attention_mask", "token_type_ids"]:
                # バケツ化対象のテンソル
                bucketed_tensor, _ = _allocate_typed_bucketed_tensor(
                    tensor.shape,
                    actual_length,
                    length_dim=1,  # seq_len次元
                    dtype=tensor.dtype,
                    device=tensor.device,
                    pool_type=pool_type,
                    use_pool=use_pool,
                )

                # パディング部分のみゼロ初期化（効率化）
                if bucket_size > actual_length:
                    bucketed_tensor[:, actual_length:] = 0

                # 実データをコピー
                bucketed_tensor[:, :actual_length] = tensor
                bucketed_inputs[key] = bucketed_tensor
            else:
                # バケツ化不要のテンソル（そのまま）
                bucketed_inputs[key] = tensor

    return bucketed_inputs, actual_length


def _allocate_typed_bucketed_tensor(
    shape: tuple[int, ...],
    actual_length: int,
    length_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    pool_type: str,
    use_pool: bool = True,
) -> tuple[torch.Tensor, int]:
    """
    用途別プールでバケツ化されたテンソルを割り当てる（内部関数）
    """
    if pool_type == "bert_mid_lived":
        bucket_size = get_bert_bucket_size(actual_length)
    else:
        bucket_size = get_bucket_size(actual_length)

    # バケツ化された形状を作成
    bucketed_shape = list(shape)
    bucketed_shape[length_dim] = bucket_size

    if use_pool and (
        (pool_type == "bert_mid_lived" and bucket_size in BERT_BUCKET_SIZES)
        or (pool_type != "bert_mid_lived" and bucket_size in BUCKET_SIZES)
    ):
        pool_key = get_typed_pool_key(pool_type, dtype, str(device))
        if pool_key not in _typed_memory_pools:
            _typed_memory_pools[pool_key] = {}

        pool = _typed_memory_pools[pool_key]

        # プールから取得または新規作成
        if bucket_size in pool:
            tensor, _ = pool[bucket_size]
            if tensor.shape == tuple(bucketed_shape):
                # 最終使用時刻を更新
                pool[bucket_size] = (tensor, time.time())
                return tensor, bucket_size

        # 新規作成してプールに追加（BERT用は完全初期化）
        if pool_type == "bert_mid_lived":
            tensor = torch.zeros(bucketed_shape, dtype=dtype, device=device)
        else:
            tensor = torch.empty(bucketed_shape, dtype=dtype, device=device)
        pool[bucket_size] = (tensor, time.time())
        return tensor, bucket_size

    # プールを使わない場合（BERT用は完全初期化）
    if pool_type == "bert_mid_lived":
        return torch.zeros(bucketed_shape, dtype=dtype, device=device), bucket_size
    else:
        return torch.empty(bucketed_shape, dtype=dtype, device=device), bucket_size


def clear_typed_memory_pools(pool_type: str | None = None):
    """
    用途別メモリプールをクリア（Mid-lived専用クリア対応）

    Args:
        pool_type: クリアするプール種別。Noneの場合は全プールをクリア
    """
    global _typed_memory_pools

    with _pool_lock:
        if pool_type is None:
            _typed_memory_pools.clear()
            logger.info("All typed memory pools cleared")
        else:
            keys_to_remove = []
            for key in _typed_memory_pools.keys():
                if key[0] == pool_type:  # pool_typeが一致
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del _typed_memory_pools[key]

            logger.info(f"Typed memory pools '{pool_type}' cleared")

    # CUDAキャッシュもクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_typed_memory_pool_stats() -> dict[str, dict[str, any]]:
    """
    用途別メモリプールの統計情報を取得

    Returns:
        プール種別ごとの統計情報
    """
    stats = {}
    total_memory_bytes = 0

    with _pool_lock:
        for pool_key, pool in _typed_memory_pools.items():
            pool_type, dtype, device = pool_key
            pool_memory_bytes = 0
            bucket_info = {}

            for bucket_size, (tensor, last_used) in pool.items():
                memory_bytes = tensor.element_size() * tensor.numel()
                pool_memory_bytes += memory_bytes
                age_seconds = time.time() - last_used

                bucket_info[str(bucket_size)] = {
                    "memory_mb": memory_bytes / (1024**2),
                    "age_seconds": int(age_seconds),
                }

            total_memory_bytes += pool_memory_bytes

            pool_name = f"{pool_type}_{dtype}_{device}"
            stats[pool_name] = {
                "num_buckets": len(pool),
                "memory_mb": pool_memory_bytes / (1024**2),
                "buckets": bucket_info,
            }

    stats["_total"] = {
        "memory_mb": total_memory_bytes / (1024**2),
        "memory_gb": total_memory_bytes / (1024**3),
        "limit_gb": POOL_MAX_MEMORY_GB,
    }

    return stats
