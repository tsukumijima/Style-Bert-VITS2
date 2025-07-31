"""
テンソルパディング・メモリプール管理モジュール

音声合成における可変長テンソルを固定サイズにパディングすることで、
PyTorchのメモリアロケータの効率化とCUDA断片化の削減を実現する。

環境変数 PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:True"
の使用を前提として設計されている。
"""

import os
import threading
import time
from typing import Any

import torch

from style_bert_vits2.logging import logger


# PYTORCH_CUDA_ALLOC_CONF環境変数の確認
_CUDA_ALLOC_CONF = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "")
if "cudaMallocAsync" not in _CUDA_ALLOC_CONF:
    logger.warning(
        "PYTORCH_CUDA_ALLOC_CONF environment variable does not include 'cudaMallocAsync'. "
        "For optimal memory efficiency, set: "
        "PYTORCH_CUDA_ALLOC_CONF='backend:cudaMallocAsync,expandable_segments:True'"
    )

# パディングサイズの定義（最適化された固定サイズ）
# VITS2用：音素・トークン系列長に対応
SEQUENCE_PADDING_SIZES = [8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]

# BERT用：より細かい粒度でトークン長をカバー
BERT_TOKEN_PADDING_SIZES = [
    32,
    48,
    64,
    80,
    96,
    128,
    160,
    192,
    256,
    320,
    384,
    512,
    768,
    1024,
]

# グローバルメモリプール
# key: (pool_type, dtype, device_str, shape_signature) -> dict[padded_size -> (tensor, last_used_time)]
_memory_pools: dict[
    tuple[str, torch.dtype, str, str], dict[int, tuple[torch.Tensor, float]]
] = {}

# プール管理用のロック
_pool_lock = threading.Lock()

# プール設定
POOL_MAX_AGE_SECONDS = 600  # 10分間未使用のテンソルは解放
POOL_MAX_MEMORY_GB = 1.5  # プール全体の最大メモリ使用量
POOL_CLEANUP_INTERVAL = 30  # プールクリーンアップ間隔（秒）

# 最後のクリーンアップ時刻
_last_cleanup_time = time.time()


def get_optimal_padding_size(
    actual_length: int,
    padding_sizes: list[int],
    max_overhead_ratio: float = 1.3,
) -> int:
    """
    実際の長さに対して最適なパディングサイズを決定する。

    Args:
        actual_length: 実際のテンソル長
        padding_sizes: 利用可能なパディングサイズのリスト
        max_overhead_ratio: 許容する最大オーバーヘッド比率

    Returns:
        最適なパディングサイズ
    """
    for padding_size in padding_sizes:
        if padding_size >= actual_length:
            overhead_ratio = padding_size / actual_length
            if overhead_ratio <= max_overhead_ratio:
                return padding_size
            # オーバーヘッドが大きすぎる場合は実長を返す（パディングなし）
            return actual_length

    # 最大パディングサイズを超える場合は実長を返す
    return actual_length


def _generate_shape_signature(shape: tuple[int, ...], length_dim: int) -> str:
    """
    テンソル形状のシグネチャを生成する（長さ次元を除く）。

    Args:
        shape: テンソル形状
        length_dim: 長さ次元のインデックス

    Returns:
        形状シグネチャ文字列
    """
    signature_parts = []
    for i, dim_size in enumerate(shape):
        if i == length_dim:
            signature_parts.append("L")  # 長さ次元を表す
        else:
            signature_parts.append(str(dim_size))
    return "x".join(signature_parts)


def _cleanup_memory_pools() -> None:
    """
    古いテンソルを解放し、メモリ上限をチェックする。
    """
    global _last_cleanup_time

    current_time = time.time()
    if current_time - _last_cleanup_time < POOL_CLEANUP_INTERVAL:
        return

    _last_cleanup_time = current_time
    total_memory_bytes = 0
    pools_to_remove = []

    # 古いテンソルの検出とメモリ使用量計算
    for pool_key, pool in _memory_pools.items():
        sizes_to_remove = []

        for padded_size, (tensor, last_used) in pool.items():
            age = current_time - last_used
            if age > POOL_MAX_AGE_SECONDS:
                sizes_to_remove.append(padded_size)
                logger.debug(
                    f"Removing aged tensor: {pool_key}, size={padded_size}, age={age:.1f}s"
                )
            else:
                total_memory_bytes += tensor.element_size() * tensor.numel()

        # 古いテンソルを削除
        for size in sizes_to_remove:
            del pool[size]

        if not pool:
            pools_to_remove.append(pool_key)

    # 空のプールを削除
    for pool_key in pools_to_remove:
        del _memory_pools[pool_key]

    # メモリ上限チェック
    total_memory_gb = total_memory_bytes / (1024**3)
    if total_memory_gb > POOL_MAX_MEMORY_GB:
        logger.warning(
            f"Memory pool exceeds limit: {total_memory_gb:.2f}GB > {POOL_MAX_MEMORY_GB}GB. "
            "Evicting oldest tensors..."
        )
        _evict_oldest_tensors(total_memory_gb - POOL_MAX_MEMORY_GB)


def _evict_oldest_tensors(memory_to_free_gb: float) -> None:
    """
    最も古いテンソルから順に解放してメモリを確保する。

    Args:
        memory_to_free_gb: 解放すべきメモリ量（GB）
    """
    # 全テンソルを最終使用時刻でソート
    all_tensors = []
    for pool_key, pool in _memory_pools.items():
        for padded_size, (tensor, last_used) in pool.items():
            memory_gb = tensor.element_size() * tensor.numel() / (1024**3)
            all_tensors.append((last_used, pool_key, padded_size, memory_gb))

    all_tensors.sort()  # 最も古いものから

    freed_memory = 0.0
    for last_used, pool_key, padded_size, memory_gb in all_tensors:
        if freed_memory >= memory_to_free_gb:
            break

        if pool_key in _memory_pools and padded_size in _memory_pools[pool_key]:
            del _memory_pools[pool_key][padded_size]
            freed_memory += memory_gb
            logger.debug(
                f"Evicted tensor: {pool_key}, size={padded_size}, freed={memory_gb:.3f}GB"
            )


def _get_pooled_tensor(
    shape: tuple[int, ...],
    padded_length: int,
    length_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    pool_type: str,
    use_pool: bool,
    zero_init: bool = False,
) -> torch.Tensor:
    """
    プールからテンソルを取得または新規作成する。

    Args:
        shape: 元のテンソル形状
        padded_length: パディング後の長さ
        length_dim: 長さ次元のインデックス
        dtype: データ型
        device: デバイス
        pool_type: プール種別
        use_pool: プールを使用するか
        zero_init: ゼロ初期化するか

    Returns:
        パディングされたテンソル
    """
    # パディング後の形状を計算
    padded_shape = list(shape)
    padded_shape[length_dim] = padded_length
    padded_shape_tuple = tuple(padded_shape)

    if not use_pool:
        # プール使用なし：直接作成
        if zero_init:
            return torch.zeros(padded_shape_tuple, dtype=dtype, device=device)
        else:
            return torch.empty(padded_shape_tuple, dtype=dtype, device=device)

    # プール使用：プールから取得または作成
    shape_signature = _generate_shape_signature(shape, length_dim)
    pool_key = (pool_type, dtype, str(device), shape_signature)

    with _pool_lock:
        # 定期クリーンアップ
        _cleanup_memory_pools()

        if pool_key not in _memory_pools:
            _memory_pools[pool_key] = {}

        pool = _memory_pools[pool_key]

        # 既存のテンソルをチェック
        if padded_length in pool:
            tensor, _ = pool[padded_length]
            if tensor.shape == padded_shape_tuple:
                # 使用時刻を更新して返す
                pool[padded_length] = (tensor, time.time())
                logger.debug(f"Reused pooled tensor: {pool_key}, size={padded_length}")
                return tensor
            else:
                # 形状が一致しない場合は削除
                del pool[padded_length]
                logger.debug(
                    f"Removed mismatched tensor: {pool_key}, size={padded_length}"
                )

        # 新規作成してプールに追加
        if zero_init:
            tensor = torch.zeros(padded_shape_tuple, dtype=dtype, device=device)
        else:
            tensor = torch.empty(padded_shape_tuple, dtype=dtype, device=device)

        pool[padded_length] = (tensor, time.time())
        logger.debug(f"Created new pooled tensor: {pool_key}, size={padded_length}")

        return tensor


def pad_sequence_tensor(
    source: torch.Tensor,
    length_dim: int = -1,
    pool_type: str = "sequence",
    max_overhead_ratio: float = 1.3,
    use_pool: bool = True,
) -> tuple[torch.Tensor, int]:
    """
    音素・トークン系列テンソルをパディングする（VITS2用）。

    Args:
        source: ソーステンソル
        length_dim: 長さ次元のインデックス
        pool_type: プール種別
        max_overhead_ratio: 許容する最大オーバーヘッド比率
        use_pool: メモリプールを使用するか

    Returns:
        (padded_tensor, actual_length): パディングされたテンソルと実際の長さ
    """
    if length_dim < 0:
        length_dim = len(source.shape) + length_dim

    actual_length = source.shape[length_dim]
    padded_length = get_optimal_padding_size(
        actual_length, SEQUENCE_PADDING_SIZES, max_overhead_ratio
    )

    # パディングが不要な場合はそのまま返す
    if padded_length == actual_length:
        return source, actual_length

    # パディングされたテンソルを取得
    padded_tensor = _get_pooled_tensor(
        source.shape,
        padded_length,
        length_dim,
        source.dtype,
        source.device,
        pool_type,
        use_pool,
        zero_init=True,  # VITS2ではパディング部分をゼロ初期化
    )

    # データをコピー
    if length_dim == 0:
        padded_tensor[:actual_length] = source
    elif length_dim == 1:
        padded_tensor[:, :actual_length] = source
    elif length_dim == 2:
        padded_tensor[:, :, :actual_length] = source
    elif length_dim == 3:
        padded_tensor[:, :, :, :actual_length] = source
    else:
        # 汎用的なスライシング（パフォーマンスは劣る）
        slices = [slice(None)] * len(source.shape)
        slices[length_dim] = slice(actual_length)
        padded_tensor[tuple(slices)] = source

    return padded_tensor, actual_length


def pad_bert_inputs(
    inputs: dict[str, torch.Tensor],
    pool_type: str = "bert",
    max_overhead_ratio: float = 1.25,
    use_pool: bool = True,
) -> tuple[dict[str, torch.Tensor], int]:
    """
    BERT入力辞書をパディングする。

    Args:
        inputs: BERT入力辞書（input_ids, attention_mask等）
        pool_type: プール種別
        max_overhead_ratio: 許容する最大オーバーヘッド比率
        use_pool: メモリプールを使用するか

    Returns:
        (padded_inputs, actual_length): パディングされた入力辞書と実際の長さ
    """
    # input_idsから実際のトークン長を取得
    if "input_ids" not in inputs:
        raise ValueError("input_ids is required in BERT inputs")

    input_ids = inputs["input_ids"]
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D tensor, got {input_ids.dim()}D")

    actual_length = input_ids.shape[1]  # [batch_size, seq_len]
    padded_length = get_optimal_padding_size(
        actual_length, BERT_TOKEN_PADDING_SIZES, max_overhead_ratio
    )

    # パディングが不要な場合はそのまま返す
    if padded_length == actual_length:
        return inputs, actual_length

    padded_inputs = {}

    for key, tensor in inputs.items():
        if key in ["input_ids", "attention_mask", "token_type_ids"]:
            # パディング対象のテンソル
            padded_tensor = _get_pooled_tensor(
                tensor.shape,
                padded_length,
                1,  # seq_len次元
                tensor.dtype,
                tensor.device,
                f"{pool_type}_{key}",
                use_pool,
                zero_init=True,  # BERTではパディング部分をゼロ初期化
            )

            # データをコピー
            padded_tensor[:, :actual_length] = tensor
            padded_inputs[key] = padded_tensor
        else:
            # パディング不要のテンソル（そのまま）
            padded_inputs[key] = tensor

    return padded_inputs, actual_length


def clear_memory_pools(pool_type: str | None = None) -> None:
    """
    メモリプールをクリアする。

    Args:
        pool_type: クリアするプール種別。Noneの場合は全プールをクリア。
    """
    global _memory_pools

    with _pool_lock:
        if pool_type is None:
            cleared_count = len(_memory_pools)
            _memory_pools.clear()
            logger.info(f"Cleared all memory pools ({cleared_count} pools)")
        else:
            keys_to_remove = []
            for key in _memory_pools.keys():
                if key[0] == pool_type:  # pool_typeが一致
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del _memory_pools[key]

            logger.info(
                f"Cleared memory pools of type '{pool_type}' ({len(keys_to_remove)} pools)"
            )

    # CUDAキャッシュもクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_pool_stats() -> dict[str, Any]:
    """
    メモリプールの統計情報を取得。

    Returns:
        プール統計情報
    """
    stats: dict[str, Any] = {}
    total_memory_bytes = 0
    total_tensors = 0

    with _pool_lock:
        for pool_key, pool in _memory_pools.items():
            pool_type, dtype, device, shape_sig = pool_key
            pool_memory_bytes = 0
            tensor_info = {}

            for padded_size, (tensor, last_used) in pool.items():
                memory_bytes = tensor.element_size() * tensor.numel()
                pool_memory_bytes += memory_bytes
                age_seconds = time.time() - last_used

                tensor_info[str(padded_size)] = {
                    "memory_mb": memory_bytes / (1024**2),
                    "age_seconds": int(age_seconds),
                    "shape": list(tensor.shape),
                }

            total_memory_bytes += pool_memory_bytes
            total_tensors += len(pool)

            pool_name = f"{pool_type}_{dtype}_{device}_{shape_sig}"
            stats[pool_name] = {
                "num_tensors": len(pool),
                "memory_mb": pool_memory_bytes / (1024**2),
                "tensors": tensor_info,
            }

    stats["_summary"] = {
        "total_pools": len(_memory_pools),
        "total_tensors": total_tensors,
        "total_memory_mb": total_memory_bytes / (1024**2),
        "total_memory_gb": total_memory_bytes / (1024**3),
        "memory_limit_gb": POOL_MAX_MEMORY_GB,
        "usage_ratio": (total_memory_bytes / (1024**3)) / POOL_MAX_MEMORY_GB,
    }

    return stats


def set_memory_pool_config(
    max_memory_gb: float | None = None,
    max_age_seconds: int | None = None,
    cleanup_interval: int | None = None,
) -> None:
    """
    メモリプールの設定を変更する。

    Args:
        max_memory_gb: プール全体の最大メモリ使用量（GB）
        max_age_seconds: テンソルの最大保持時間（秒）
        cleanup_interval: クリーンアップ間隔（秒）
    """
    global POOL_MAX_MEMORY_GB, POOL_MAX_AGE_SECONDS, POOL_CLEANUP_INTERVAL

    if max_memory_gb is not None:
        POOL_MAX_MEMORY_GB = max_memory_gb
        logger.info(f"Memory pool max memory set to {max_memory_gb}GB")

    if max_age_seconds is not None:
        POOL_MAX_AGE_SECONDS = max_age_seconds
        logger.info(f"Memory pool max age set to {max_age_seconds}s")

    if cleanup_interval is not None:
        POOL_CLEANUP_INTERVAL = cleanup_interval
        logger.info(f"Memory pool cleanup interval set to {cleanup_interval}s")
