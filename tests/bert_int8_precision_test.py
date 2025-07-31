#!/usr/bin/env python3
"""
BERT 8bit 量子化精度評価テストスクリプト

このスクリプトは実際のユースケースに近い形で BERT モデルの 8bit 量子化による影響を評価する。
各精度設定で一度だけモデルをロードし、複数のテキストで推論を実行してウォーミングアップ後の性能を測定する。

使用方法:
    .venv/bin/python -m tests.bert_int8_precision_test [--device cuda] [--warmup-runs 3] [--test-runs 5]
"""

import argparse
import time
from typing import NotRequired, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.distance import cosine

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models, clean_text_with_given_phone_tone
from style_bert_vits2.nlp.japanese.bert_feature import extract_bert_feature

from .utils import set_random_seeds


class BertConfig(TypedDict):
    """BERT設定の型定義。"""

    name: str
    use_fp16: bool
    use_int8: bool
    llm_int8_threshold: NotRequired[float]
    llm_int8_skip_modules: NotRequired[list[str] | None]


# 測定用サンプルテキスト
BENCHMARK_TEXTS = [
    "こんにちは、今日はいい天気ですね。",
    "人工知能の発展により、私たちの生活は大きく変わりつつあります。",
    "日本の四季は美しく、春の桜、夏の花火、秋の紅葉、冬の雪景色と、それぞれに魅力があります。",
    "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。",
    "最近の機械学習技術の進歩は目覚ましく、特に大規模言語モデルの登場により自然言語処理分野が大きく変化している。",
    "東京スカイツリーは高さ634メートルで、世界で最も高い自立式電波塔として知られています。",
]


def measure_bert_inference_time(
    texts: list[str],
    device: str,
    warmup_runs: int = 3,
    test_runs: int = 5,
) -> tuple[list[NDArray[np.float32]], float, float]:
    """
    BERT 推論時間を測定する（モデルロード済み前提）。

    Returns:
        tuple: (特徴量リスト, 平均推論時間, 標準偏差)
    """
    features = []
    inference_times = []

    # ウォーミングアップ実行
    for i in range(warmup_runs):
        for text in texts:
            norm_text, _, _, word2ph, sep_text, _, _ = clean_text_with_given_phone_tone(
                text,
                Languages.JP,
                use_jp_extra=False,
                raise_yomi_error=False,
            )

            with torch.inference_mode():
                _ = extract_bert_feature(
                    text=norm_text,
                    word2ph=word2ph,
                    device=device,
                    sep_text=sep_text,
                )
        print(f"    ウォーミングアップ {i + 1}/{warmup_runs} 完了")

    # 本測定
    for run in range(test_runs):
        run_features = []
        run_start_time = time.perf_counter()

        for text in texts:
            norm_text, _, _, word2ph, sep_text, _, _ = clean_text_with_given_phone_tone(
                text,
                Languages.JP,
                use_jp_extra=False,
                raise_yomi_error=False,
            )

            with torch.inference_mode():
                bert_feature = extract_bert_feature(
                    text=norm_text,
                    word2ph=word2ph,
                    device=device,
                    sep_text=sep_text,
                )

                # FP32 に統一して numpy に変換（すべての精度で統一する）
                bert_feature_np = bert_feature.float().cpu().numpy()

                run_features.append(bert_feature_np)

        run_end_time = time.perf_counter()
        run_time = run_end_time - run_start_time
        inference_times.append(run_time)

        if run == 0:  # 最初の実行の特徴量を保存
            features = run_features

        print(f"    測定実行 {run + 1}/{test_runs}: {run_time:.3f}秒")

    avg_time = float(np.mean(inference_times))
    std_time = float(np.std(inference_times))

    return features, avg_time, std_time


def compute_cosine_similarity_batch(
    features1: list[NDArray[np.float32]], features2: list[NDArray[np.float32]]
) -> list[float]:
    """複数の特徴量ペアのコサイン類似度を計算する。"""
    similarities = []
    for f1, f2 in zip(features1, features2):
        vec1_flat = f1.flatten()
        vec2_flat = f2.flatten()
        cos_distance = cosine(vec1_flat, vec2_flat)
        cos_similarity = 1 - cos_distance
        similarities.append(cos_similarity)
    return similarities


def test_configuration(
    name: str,
    use_fp16: bool = False,
    use_int8: bool = False,
    llm_int8_threshold: float = 6.0,
    llm_int8_skip_modules: list[str] | None = None,
    device: str = "cuda",
    warmup_runs: int = 3,
    test_runs: int = 5,
) -> tuple[list[NDArray[np.float32]], float, float, float]:
    """
    指定された設定でテストを実行する。

    Returns:
        tuple: (特徴量リスト, 平均推論時間, 標準偏差, ピークメモリ使用量)
    """
    print(f"\n{'=' * 60}")
    print(f"設定: {name}")
    print(f"{'=' * 60}")

    # 既存のモデルをアンロード
    if bert_models.is_model_loaded(Languages.JP):
        bert_models.unload_model(Languages.JP)
        if device == "cuda":
            torch.cuda.empty_cache()

    # メモリ統計をリセット
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    print("モデルロード中...")
    load_start_time = time.perf_counter()

    # モデルをロード
    bert_models.load_model(
        Languages.JP,
        device_map=device,
        use_fp16=use_fp16,
        use_int8=use_int8,
        llm_int8_threshold=llm_int8_threshold,
        llm_int8_skip_modules=llm_int8_skip_modules,
    )

    load_time = time.perf_counter() - load_start_time
    print(f"モデルロード完了: {load_time:.3f}秒")

    # 推論時間を測定
    print("推論性能測定中...")
    features, avg_time, std_time = measure_bert_inference_time(
        BENCHMARK_TEXTS, device, warmup_runs, test_runs
    )

    # ピークメモリ使用量を取得
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    else:
        peak_memory = 0.0

    print(f"平均推論時間: {avg_time:.3f}±{std_time:.3f}秒")
    print(f"ピークメモリ使用量: {peak_memory:.2f}MB")

    return features, avg_time, std_time, peak_memory


def run_benchmark(
    device: str = "cuda",
    warmup_runs: int = 3,
    test_runs: int = 5,
) -> None:
    """
    ベンチマークを実行する。
    """
    print("=" * 80)
    print("BERT 8bit 量子化精度評価テスト (実用的測定)")
    print("=" * 80)
    print(f"デバイス: {device}")
    print(f"ウォーミングアップ実行回数: {warmup_runs}")
    print(f"測定実行回数: {test_runs}")
    print(f"テストテキスト数: {len(BENCHMARK_TEXTS)}")
    print("=" * 80)

    # テスト設定
    configs: list[BertConfig] = [
        {
            "name": "FP32 (baseline)",
            "use_fp16": False,
            "use_int8": False,
        },
        {
            "name": "FP16",
            "use_fp16": True,
            "use_int8": False,
        },
        {
            "name": "INT8 (threshold=6.0, no skip)",
            "use_fp16": False,
            "use_int8": True,
            "llm_int8_threshold": 6.0,
            "llm_int8_skip_modules": None,
        },
        {
            "name": "INT8 (threshold=6.0, default skip)",
            "use_fp16": False,
            "use_int8": True,
            "llm_int8_threshold": 6.0,
            "llm_int8_skip_modules": ["embeddings", "LayerNorm"],
        },
        {
            "name": "INT8 (threshold=6.0, projection skip)",
            "use_fp16": False,
            "use_int8": True,
            "llm_int8_threshold": 6.0,
            "llm_int8_skip_modules": ["embeddings", "LayerNorm", "embed_proj", "dense"],
        },
        {
            "name": "INT8 (threshold=6.0, attention skip)",
            "use_fp16": False,
            "use_int8": True,
            "llm_int8_threshold": 6.0,
            "llm_int8_skip_modules": [
                "embeddings",
                "LayerNorm",
                "query_proj",
                "key_proj",
                "value_proj",
            ],
        },
        {
            "name": "INT8 (threshold=6.0, position skip)",
            "use_fp16": False,
            "use_int8": True,
            "llm_int8_threshold": 6.0,
            "llm_int8_skip_modules": [
                "embeddings",
                "LayerNorm",
                "pos_key_proj",
                "pos_query_proj",
                "position_embeddings",
            ],
        },
        {
            "name": "INT8 (threshold=5.0, default skip)",
            "use_fp16": False,
            "use_int8": True,
            "llm_int8_threshold": 5.0,
            "llm_int8_skip_modules": ["embeddings", "LayerNorm"],
        },
        {
            "name": "INT8 (threshold=4.0, default skip)",
            "use_fp16": False,
            "use_int8": True,
            "llm_int8_threshold": 4.0,
            "llm_int8_skip_modules": ["embeddings", "LayerNorm"],
        },
        {
            "name": "INT8 (threshold=3.0, default skip)",
            "use_fp16": False,
            "use_int8": True,
            "llm_int8_threshold": 3.0,
            "llm_int8_skip_modules": ["embeddings", "LayerNorm"],
        },
    ]

    results = {}
    baseline_features = None

    # 各設定でテストを実行
    for config in configs:
        try:
            features, avg_time, std_time, peak_memory = test_configuration(
                name=config["name"],
                use_fp16=config["use_fp16"],
                use_int8=config["use_int8"],
                llm_int8_threshold=config.get("llm_int8_threshold", 6.0),
                llm_int8_skip_modules=config.get("llm_int8_skip_modules", None),
                device=device,
                warmup_runs=warmup_runs,
                test_runs=test_runs,
            )

            # 結果を保存
            results[config["name"]] = {
                "features": features,
                "avg_time": avg_time,
                "std_time": std_time,
                "peak_memory": peak_memory,
            }

            # ベースライン特徴量を保存
            if "baseline" in config["name"]:
                baseline_features = features

        except Exception as ex:
            logger.exception(f"設定 '{config['name']}' でエラーが発生しました: {ex}")
            continue

    # すべてのモデルをアンロード
    if bert_models.is_model_loaded(Languages.JP):
        bert_models.unload_model(Languages.JP)

    # 結果の分析と表示
    print("\n" + "=" * 100)
    print("総合結果")
    print("=" * 100)
    print(
        f"{'設定':<35} {'推論時間(秒)':<15} {'メモリ(MB)':<12} {'類似度(min/avg/max)'}"
    )
    print("-" * 100)

    for name, result in results.items():
        avg_time = result["avg_time"]
        std_time = result["std_time"]
        peak_memory = result["peak_memory"]

        # コサイン類似度を計算
        if baseline_features and name != "FP32 (baseline)":
            similarities = compute_cosine_similarity_batch(
                baseline_features, result["features"]
            )
            min_sim = min(similarities)
            avg_sim = np.mean(similarities)
            max_sim = max(similarities)
            sim_str = f"{min_sim:.4f}/{avg_sim:.4f}/{max_sim:.4f}"
        else:
            sim_str = "1.0000/1.0000/1.0000 (baseline)"

        print(
            f"{name:<35} {avg_time:.3f}±{std_time:.3f}     {peak_memory:>8.1f}    {sim_str}"
        )

    print("\n" + "=" * 100)
    print("分析:")
    print("- 推論時間: ウォーミングアップ後の安定した性能")
    print("- コサイン類似度: 最小/平均/最大値")
    print("- 0.999以上: 実用上問題なし")
    print("- 0.995-0.999: わずかな劣化だが多くの場合許容範囲")
    print("- 0.995未満: 明確な劣化、要検討")
    print()

    # GPU アーキテクチャ別のアドバイス
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        if "GTX" in gpu_name or "Pascal" in gpu_name:
            print(
                "- Pascal アーキテクチャでは INT8 テンソルコアがないため量子化の速度向上は期待できません"
            )
            print("- RTX シリーズ（Turing 以降）では INT8 演算が大幅に高速化されます")
        elif "RTX" in gpu_name or "Ampere" in gpu_name or "Ada" in gpu_name:
            print("- このGPUでは INT8 テンソルコアにより量子化の速度向上が期待できます")

    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BERT 8bit 量子化精度評価テスト (実用的測定)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="推論に使用するデバイス (default: cuda)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="ウォーミングアップ実行回数 (default: 3)",
    )
    parser.add_argument(
        "--test-runs",
        type=int,
        default=5,
        help="測定実行回数 (default: 5)",
    )

    args = parser.parse_args()

    set_random_seeds()

    # CPU では 8bit 量子化は使えないので警告
    if args.device == "cpu":
        print("警告: 8bit 量子化は GPU (CUDA) でのみ動作します。")
        print("CPU モードでは FP32 と FP16 の比較のみ行います。")

    try:
        run_benchmark(
            device=args.device,
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs,
        )
    except KeyboardInterrupt:
        print("\nベンチマークが中断されました。")
    except Exception as ex:
        logger.exception(f"ベンチマーク実行中にエラーが発生しました: {ex}")


if __name__ == "__main__":
    main()
