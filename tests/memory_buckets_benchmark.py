#!/usr/bin/env python3
"""
Usage: PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:True" .venv/bin/python -m tests.memory_buckets_benchmark [--device cuda] [--model koharune-ami] [--iterations 30]

メモリ効率化テンソルパディングのベンチマークスクリプト

本番環境同様にPYTORCH_CUDA_ALLOC_CONFを設定して実行してください。
"""

import os


# PYTORCH_CUDA_ALLOC_CONFの確認と警告
cuda_alloc_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "")
if "cudaMallocAsync" not in cuda_alloc_conf:
    print("WARNING: PYTORCH_CUDA_ALLOC_CONF is not set for optimal memory management!")
    print(
        'Please run with: PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:True"'
    )
    print(
        "This benchmark may not accurately reflect production memory behavior without proper configuration."
    )
    print()
else:
    print(f"Using PYTORCH_CUDA_ALLOC_CONF: {cuda_alloc_conf}")
    print()

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from style_bert_vits2.constants import (
    BASE_DIR,
    DEFAULT_LENGTH,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models.infer import infer
from style_bert_vits2.models.tensor_padding import (
    clear_memory_pools,
    get_memory_pool_stats,
)
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder

from .utils import save_benchmark_audio, set_random_seeds


# テストテキスト（様々な長さ）
TEST_TEXTS = [
    "こんにちは。",  # 短い
    "今日はとてもいい天気ですね。",  # 中程度
    "昨日は雨が降っていましたが、今日は晴れてとても気持ちがいいです。",  # やや長い
    "人工知能の発展により、音声合成技術も大きく進歩しました。より自然で表現豊かな音声を生成できるようになっています。",  # 長い
    "春の訪れとともに、桜の花が咲き始めました。淡いピンク色の花びらが風に舞い、街全体が華やかな雰囲気に包まれています。多くの人々が花見を楽しみ、笑顔があふれる季節となりました。",  # とても長い
]


class MemoryTracker:
    """GPUメモリ使用量を追跡するクラス"""

    def __init__(self, device: str):
        self.device = device
        self.snapshots: list[dict[str, float]] = []

    def snapshot(self, label: str):
        """現在のメモリ使用量をスナップショット"""
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            fragmentation = reserved - allocated
        else:
            allocated = reserved = fragmentation = 0.0

        self.snapshots.append(
            {
                "label": label,  # type: ignore
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "fragmentation_gb": fragmentation,
            }
        )

    def print_report(self):
        """メモリ使用状況のレポートを出力"""
        print("\n=== Memory Usage Report ===")
        print(
            f"{'Label':<30} {'Allocated (GB)':>15} {'Reserved (GB)':>15} {'Fragmentation (GB)':>20}"
        )
        print("-" * 82)

        for snap in self.snapshots:
            print(
                f"{snap['label']:<30} {snap['allocated_gb']:>15.3f} "
                f"{snap['reserved_gb']:>15.3f} {snap['fragmentation_gb']:>20.3f}"
            )


def measure_inference_performance(
    model: TTSModel,
    text: str,
    device: str,
    use_padding: bool,
    use_fp16: bool,
    save_audio: bool = False,
) -> tuple[float, float, NDArray[np.float32]]:
    """
    推論のパフォーマンスを測定する

    Args:
        save_audio: 音声ファイルを保存するかどうか

    Returns:
        tuple: (推論時間, ピークメモリ使用量MB, 音声データ)
    """
    # メモリプール統計をクリア（新しい測定のため）
    if use_padding:
        clear_memory_pools()

    # テキスト情報をログ出力
    logger.info(
        f"Processing text (length: {len(text)}): '{text[:50]}{'...' if len(text) > 50 else ''}'"
    )

    # メモリ統計をリセット
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    net_g = model.net_g
    assert net_g is not None

    style_vec = model.get_style_vector(0, 1.0)  # デフォルトスタイル

    # 推論を実行
    with torch.inference_mode():
        audio_data = infer(
            text=text,
            style_vec=style_vec,
            sdp_ratio=DEFAULT_SDP_RATIO,
            noise_scale=DEFAULT_NOISE,
            noise_scale_w=DEFAULT_NOISEW,
            length_scale=DEFAULT_LENGTH,
            sid=0,
            language=Languages.JP,
            hps=model.hyper_parameters,
            net_g=net_g,
            device=device,
            use_fp16=use_fp16,
            clear_cuda_cache=False,  # ベンチマーク中はキャッシュクリアを無効化
            enable_tensor_padding=use_padding,
        )

    end_time = time.perf_counter()
    inference_time = end_time - start_time

    # パディング統計情報を出力
    if use_padding:
        pool_stats = get_memory_pool_stats()
        logger.info(f"  Padding enabled - Pool stats: {pool_stats}")

    # ピークメモリ使用量を取得
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    else:
        peak_memory = 0.0

    # 音声ファイル保存（オプション）
    if save_audio:
        output_type = "with_padding" if use_padding else "without_padding"
        save_benchmark_audio(
            audio_data,
            model.hyper_parameters.data.sampling_rate,
            text,
            "tensor_padding_benchmark",
            output_type,
        )

    return inference_time, peak_memory, audio_data


def run_benchmark(
    device: str = "cuda",
    model_name: str = "koharune-ami",
    num_iterations: int = 30,
    use_fp16: bool = True,
    fix_seed: bool = False,
) -> None:
    """ベンチマークを実行する"""

    # ランダムシード固定
    if fix_seed:
        set_random_seeds()

    print("=" * 80)
    print("Style-Bert-VITS2 メモリ効率化テンソルパディングベンチマーク")
    print("=" * 80)
    print(f"デバイス: {device}")
    print(f"モデル: {model_name}")
    print(f"反復回数: {num_iterations}")
    print(f"FP16: {use_fp16}")
    print(f"ランダムシード固定: {'有効' if fix_seed else '無効'}")
    print("=" * 80)

    # BERT メモリ使用量を測定
    bert_memory_usage = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        memory_before_bert = torch.cuda.memory_allocated() / (1024 * 1024)
        logger.info("Loading BERT model for memory measurement...")
        bert_models.load_model(Languages.JP, device_map=device, use_fp16=use_fp16)
        memory_after_bert = torch.cuda.memory_allocated() / (1024 * 1024)
        bert_memory_usage = memory_after_bert - memory_before_bert

        print(f"BERT メモリ使用量: {bert_memory_usage:.2f}MB")
        print("=" * 80)

    # モデルホルダーを初期化
    model_holder = TTSModelHolder(
        BASE_DIR / "model_assets",
        device,
        onnx_providers=[],
        ignore_onnx=True,
        use_fp16=use_fp16,
    )

    if len(model_holder.models_info) == 0:
        print("エラー: 音声合成モデルが見つかりませんでした。")
        return

    # 指定されたモデルを検索
    model_info = None
    for info in model_holder.models_info:
        if info.name == model_name:
            model_info = info
            break

    if model_info is None:
        print(f'エラー: モデル "{model_name}" が見つかりませんでした。')
        print("利用可能なモデル:")
        for info in model_holder.models_info:
            print(f"  - {info.name}")
        return

    # Safetensors 形式のモデルファイルを検索
    model_files = [
        f
        for f in model_info.files
        if f.endswith(".safetensors") and not f.startswith(".")
    ]
    if len(model_files) == 0:
        print(
            f'エラー: モデル "{model_name}" の .safetensors ファイルが見つかりませんでした。'
        )
        return

    model_file = model_files[0]
    print(f"使用するモデルファイル: {model_file}")
    print()

    # モデルをロード
    model = model_holder.get_model(model_name, model_file)
    model.load()

    # パディングなしでベンチマーク
    print("\n" + "=" * 60)
    print("パディングなしでベンチマーク実行中...")
    print("=" * 60)

    tracker_without = MemoryTracker(device)
    tracker_without.snapshot("Initial")

    # ウォームアップ
    for i in range(2):
        _, _, _ = measure_inference_performance(
            model, TEST_TEXTS[0], device, use_padding=False, use_fp16=use_fp16
        )

    torch.cuda.empty_cache()
    gc.collect()
    tracker_without.snapshot("After warmup")

    # 本測定
    inference_times_without = []
    for i in range(num_iterations):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        # 最後の数回のみ音声を保存（主観評価用）
        save_audio = i >= max(0, num_iterations - len(TEST_TEXTS))
        inference_time, peak_memory, _ = measure_inference_performance(
            model,
            text,
            device,
            use_padding=False,
            use_fp16=use_fp16,
            save_audio=save_audio,
        )
        inference_times_without.append(inference_time)

        if (i + 1) % 10 == 0:
            tracker_without.snapshot(f"After {i + 1} iterations")
            print(f"  反復 {i + 1}/{num_iterations} 完了")

    tracker_without.snapshot("Final (before cleanup)")

    # パディングありでベンチマーク
    print("\n" + "=" * 60)
    print("パディングありでベンチマーク実行中...")
    print("=" * 60)

    tracker_with = MemoryTracker(device)
    tracker_with.snapshot("Initial")

    # ウォームアップ
    for i in range(2):
        _, _, _ = measure_inference_performance(
            model, TEST_TEXTS[0], device, use_padding=True, use_fp16=use_fp16
        )

    torch.cuda.empty_cache()
    gc.collect()
    tracker_with.snapshot("After warmup")

    # 本測定
    inference_times_with = []
    for i in range(num_iterations):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        # 最後の数回のみ音声を保存（主観評価用）
        save_audio = i >= max(0, num_iterations - len(TEST_TEXTS))
        inference_time, peak_memory, _ = measure_inference_performance(
            model,
            text,
            device,
            use_padding=True,
            use_fp16=use_fp16,
            save_audio=save_audio,
        )
        inference_times_with.append(inference_time)

        if (i + 1) % 10 == 0:
            tracker_with.snapshot(f"After {i + 1} iterations")
            print(f"  反復 {i + 1}/{num_iterations} 完了")

    tracker_with.snapshot("Final (before cleanup)")

    # クリーンアップ
    clear_memory_pools()
    torch.cuda.empty_cache()
    gc.collect()
    tracker_with.snapshot("After cleanup")

    # モデルをアンロード
    model.unload()

    # 結果比較
    print("\n" + "=" * 60)
    print("ベンチマーク結果比較")
    print("=" * 60)

    # 推論時間比較
    avg_without = np.mean(inference_times_without)
    avg_with = np.mean(inference_times_with)
    overhead = (avg_with / avg_without - 1) * 100

    print("\n推論時間:")
    print(
        f"  パディングなし: {avg_without:.3f} s (標準偏差: {np.std(inference_times_without):.3f})"
    )
    print(
        f"  パディングあり: {avg_with:.3f} s (標準偏差: {np.std(inference_times_with):.3f})"
    )
    print(f"  オーバーヘッド: {overhead:+.1f}%")

    # メモリ断片化比較
    final_without = next(
        s for s in tracker_without.snapshots if s["label"] == "Final (before cleanup)"
    )
    final_with = next(
        s for s in tracker_with.snapshots if s["label"] == "Final (before cleanup)"
    )

    print("\nメモリ断片化:")
    print(f"  パディングなし: {final_without['fragmentation_gb']:.3f} GB")
    print(f"  パディングあり: {final_with['fragmentation_gb']:.3f} GB")
    print(
        f"  削減量: {final_without['fragmentation_gb'] - final_with['fragmentation_gb']:.3f} GB"
    )

    # メモリプール統計
    pool_stats = get_memory_pool_stats()
    if pool_stats:
        print("\nメモリプール統計:")
        for pool_name, stats in pool_stats.items():
            print(f"  {pool_name}: {stats}")

    # 詳細メモリレポート
    print("\n=== パディングなし - メモリ使用状況 ===")
    tracker_without.print_report()

    print("\n=== パディングあり - メモリ使用状況 ===")
    tracker_with.print_report()

    # 判定
    print("\n=== 総合評価 ===")
    if overhead < 5:
        print("✓ オーバーヘッドは許容範囲内です (< 5%)")
    elif overhead < 15:
        print("⚠ 軽微なオーバーヘッドがあります (5-15%)")
    else:
        print("✗ 大きなオーバーヘッドがあります (> 15%)")

    fragmentation_reduction = (
        final_without["fragmentation_gb"] - final_with["fragmentation_gb"]
    )
    if fragmentation_reduction > 0.1:
        print("✓ メモリ断片化が大幅に改善されました")
    elif fragmentation_reduction > 0.05:
        print("✓ メモリ断片化が改善されました")
    else:
        print("⚠ メモリ断片化の改善は限定的です")

    # 音声ファイル保存の通知
    audio_dir = Path("tests/wavs/tensor_padding_benchmark")
    if audio_dir.exists() and any(audio_dir.iterdir()):
        print(f"\n🎧 音声ファイルが保存されました: {audio_dir}")
        print("   - *_without_padding.wav: パディングなしの音声")
        print("   - *_with_padding.wav: パディングありの音声")
        print("   主観評価で音質の比較をしてください。")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient tensor padding benchmark"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="koharune-ami",
        help="Model name to use (default: koharune-ami)",
    )
    parser.add_argument(
        "--iterations", type=int, default=30, help="Number of iterations (default: 30)"
    )
    parser.add_argument(
        "--fp16",
        dest="use_fp16",
        action="store_true",
        help="Use FP16 inference (default)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="use_fp16",
        action="store_false",
        help="Disable FP16 inference",
    )
    parser.add_argument(
        "--fix-seed",
        action="store_true",
        help="Fix random seed for reproducibility",
    )
    parser.set_defaults(use_fp16=True)

    args = parser.parse_args()

    try:
        run_benchmark(
            device=args.device,
            model_name=args.model,
            num_iterations=args.iterations,
            use_fp16=args.use_fp16,
            fix_seed=args.fix_seed,
        )
    except KeyboardInterrupt:
        print("\nベンチマークが中断されました。")
    except Exception as ex:
        logger.exception(
            f"テンソルパディングベンチマーク実行中にエラーが発生しました: {ex}"
        )


if __name__ == "__main__":
    main()
