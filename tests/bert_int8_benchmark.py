#!/usr/bin/env python3
"""
BERT 8bit 量子化の音声合成品質への影響評価テストスクリプト

このスクリプトは実際のユースケースに近い形で BERT モデルの 8bit 量子化が音声合成品質に与える影響を評価する。
各精度設定で一度だけモデルをロードし、複数のテキストで音声合成を実行してウォーミングアップ後の性能を測定する。

使用方法:
    .venv/bin/python -m tests.bert_int8_benchmark [--device cuda] [--model koharune-ami] [--warmup-runs 2] [--test-runs 3]
"""

import argparse
import time
from pathlib import Path
from typing import NotRequired, TypedDict

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
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder

from .utils import save_benchmark_audio, set_random_seeds


class TTSBertConfig(TypedDict):
    """TTS BERT設定の型定義。"""

    name: str
    use_fp16: bool
    use_int8: bool
    llm_int8_threshold: NotRequired[float]
    llm_int8_skip_modules: NotRequired[list[str] | None]


# 測定用サンプルテキスト
BENCHMARK_TEXTS = [
    {
        "text": "こんにちは、今日はいい天気ですね。",
        "description": "Short_greeting",
    },
    {
        "text": "人工知能の発展により、私たちの生活は大きく変わりつつあります。",
        "description": "Medium_AI_topic",
    },
    {
        "text": "春の桜、夏の花火、秋の紅葉、冬の雪景色。日本の四季は本当に美しいです。",
        "description": "Seasons_poetic",
    },
]


def measure_tts_inference_time(
    model: TTSModel,
    texts: list[str],
    warmup_runs: int = 2,
    test_runs: int = 3,
) -> tuple[list[NDArray[np.float32]], float, float]:
    """
    TTS 推論時間を測定する（モデルロード済み前提）。

    Returns:
        tuple: (音声データリスト, 平均推論時間, 標準偏差)
    """
    net_g = model.net_g
    assert net_g is not None
    style_vec = model.get_style_vector(0, 1.0)

    audio_results = []
    inference_times = []

    # ウォーミングアップ実行
    for i in range(warmup_runs):
        for text in texts:
            with torch.inference_mode():
                _ = infer(
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
                    device=model.device,
                    use_fp16=True,
                )
        print(f"    ウォーミングアップ {i + 1}/{warmup_runs} 完了")

    # 本測定
    for run in range(test_runs):
        run_audio_results = []
        run_start_time = time.perf_counter()

        for text in texts:
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
                    device=model.device,
                    use_fp16=True,
                )
                run_audio_results.append(audio_data)

        run_end_time = time.perf_counter()
        run_time = run_end_time - run_start_time
        inference_times.append(run_time)

        if run == 0:  # 最初の実行の音声データを保存
            audio_results = run_audio_results

        print(f"    測定実行 {run + 1}/{test_runs}: {run_time:.3f}秒")

    avg_time = float(np.mean(inference_times))
    std_time = float(np.std(inference_times))

    return audio_results, avg_time, std_time


def test_bert_configuration(
    model: TTSModel,
    name: str,
    use_fp16: bool = False,
    use_int8: bool = False,
    llm_int8_threshold: float = 6.0,
    llm_int8_skip_modules: list[str] | None = None,
    device: str = "cuda",
    warmup_runs: int = 2,
    test_runs: int = 3,
    output_dir: Path | None = None,
    fix_seed: bool = False,
) -> tuple[list[NDArray[np.float32]], float, float, float]:
    """
    指定された BERT 設定でテストを実行する。

    Returns:
        tuple: (音声データリスト, 平均推論時間, 標準偏差, ピークメモリ使用量)
    """
    print(f"\n{'=' * 60}")
    print(f"BERT 設定: {name}")
    print(f"{'=' * 60}")

    # ランダムシード固定が有効な場合のみ各設定開始時にリセット
    if fix_seed:
        set_random_seeds()

    # 既存の BERT モデルをアンロード
    if bert_models.is_model_loaded(Languages.JP):
        bert_models.unload_model(Languages.JP)
        if device == "cuda":
            torch.cuda.empty_cache()

    # メモリ統計をリセット
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    print("BERT モデルロード中...")
    load_start_time = time.perf_counter()

    # BERT モデルをロード
    bert_models.load_model(
        Languages.JP,
        device_map=device,
        use_fp16=use_fp16,
        use_int8=use_int8,
        llm_int8_threshold=llm_int8_threshold,
        llm_int8_skip_modules=llm_int8_skip_modules,
    )

    load_time = time.perf_counter() - load_start_time
    print(f"BERT ロード完了: {load_time:.3f}秒")

    # TTS 推論時間を測定
    print("TTS 推論性能測定中...")
    texts = [case["text"] for case in BENCHMARK_TEXTS]
    audio_results, avg_time, std_time = measure_tts_inference_time(
        model, texts, warmup_runs, test_runs
    )

    # 音声ファイルを保存
    if output_dir:
        for case, audio_data in zip(BENCHMARK_TEXTS, audio_results):
            suffix = (
                name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("=", "_")
                .replace(",", "_")
            )
            save_benchmark_audio(
                audio_data,
                model.hyper_parameters.data.sampling_rate,
                case["description"],
                "bert_int8_benchmark",
                suffix,
            )

    # ピークメモリ使用量を取得
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    else:
        peak_memory = 0.0

    print(f"平均TTS推論時間: {avg_time:.3f}±{std_time:.3f}秒")
    print(f"ピークメモリ使用量: {peak_memory:.2f}MB")

    return audio_results, avg_time, std_time, peak_memory


def run_benchmark(
    device: str = "cuda",
    model_name: str = "koharune-ami",
    warmup_runs: int = 2,
    test_runs: int = 3,
    fix_seed: bool = False,
) -> None:
    """
    ベンチマークを実行する。
    """
    # ランダムシード固定が指定された場合のみ初期化
    if fix_seed:
        set_random_seeds()

    print("=" * 80)
    print("BERT 8bit 量子化の音声合成品質への影響評価テスト (実用的測定)")
    print("=" * 80)
    print(f"デバイス: {device}")
    print(f"モデル: {model_name}")
    print(f"ウォーミングアップ実行回数: {warmup_runs}")
    print(f"測定実行回数: {test_runs}")
    print(f"テストテキスト数: {len(BENCHMARK_TEXTS)}")
    print(f"ランダムシード固定: {'有効' if fix_seed else '無効'}")
    print("=" * 80)

    # モデルホルダーを初期化
    model_holder = TTSModelHolder(
        BASE_DIR / "model_assets",
        device,
        onnx_providers=[],
        ignore_onnx=True,
        use_fp16=True,  # TTS モデルは FP16 で実行
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

    # TTS モデルをロード
    model = model_holder.get_model(model_name, model_file)
    model.load()

    # 出力ディレクトリを作成
    output_dir = Path("tests/wavs/bert_int8_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    # BERT 設定
    bert_configs: list[TTSBertConfig] = [
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
    ]

    results = {}

    # 各 BERT 設定でテストを実行
    for config in bert_configs:
        try:
            audio_results, avg_time, std_time, peak_memory = test_bert_configuration(
                model,
                name=config["name"],
                use_fp16=config["use_fp16"],
                use_int8=config["use_int8"],
                llm_int8_threshold=config.get("llm_int8_threshold", 6.0),
                llm_int8_skip_modules=config.get("llm_int8_skip_modules", None),
                device=device,
                warmup_runs=warmup_runs,
                test_runs=test_runs,
                output_dir=output_dir,
                fix_seed=fix_seed,
            )

            # 結果を保存
            results[config["name"]] = {
                "audio_results": audio_results,
                "avg_time": avg_time,
                "std_time": std_time,
                "peak_memory": peak_memory,
            }

        except Exception as ex:
            logger.exception(
                f"BERT設定 '{config['name']}' でエラーが発生しました: {ex}"
            )
            continue

    # TTS モデルをアンロード
    model.unload()

    # すべての BERT モデルをアンロード
    if bert_models.is_model_loaded(Languages.JP):
        bert_models.unload_model(Languages.JP)

    # 結果の分析と表示
    print("\n" + "=" * 100)
    print("総合結果")
    print("=" * 100)
    print(
        f"{'BERT設定':<25} {'TTS推論時間(秒)':<18} {'メモリ(MB)':<12} {'音声長合計(秒)'}"
    )
    print("-" * 100)

    for name, result in results.items():
        avg_time = result["avg_time"]
        std_time = result["std_time"]
        peak_memory = result["peak_memory"]

        # 生成音声の合計長を計算
        total_audio_duration = sum(
            len(audio) / model.hyper_parameters.data.sampling_rate
            for audio in result["audio_results"]
        )

        print(
            f"{name:<25} {avg_time:.3f}±{std_time:.3f}        {peak_memory:>8.1f}    {total_audio_duration:>12.2f}"
        )

    print("\n" + "=" * 100)
    print("分析:")
    print("- TTS推論時間: ウォーミングアップ後の安定した性能（BERT + TTS の合計時間）")
    print("- メモリ使用量: BERT + TTS モデルの合計メモリ使用量")
    print(f"- 生成された音声ファイルは {output_dir} に保存されています")
    print("- 音声品質の違いは聴感での確認が必要です")
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
        description="BERT 8bit 量子化の音声合成品質への影響評価テスト (実用的測定)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="推論に使用するデバイス (default: cuda)",
    )
    parser.add_argument(
        "--model",
        default="koharune-ami",
        help="使用するモデル名 (default: koharune-ami)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="ウォーミングアップ実行回数 (default: 2)",
    )
    parser.add_argument(
        "--test-runs",
        type=int,
        default=3,
        help="測定実行回数 (default: 3)",
    )
    parser.add_argument(
        "--fix-seed",
        action="store_true",
        help="ランダムシードを固定して再現性を確保する",
    )

    args = parser.parse_args()

    # CPU では 8bit 量子化は使えないので警告
    if args.device == "cpu":
        print("警告: 8bit 量子化は GPU (CUDA) でのみ動作します。")
        print("CPU モードでは FP32 と FP16 の比較のみ行います。")

    try:
        run_benchmark(
            device=args.device,
            model_name=args.model,
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs,
            fix_seed=args.fix_seed,
        )
    except KeyboardInterrupt:
        print("\nベンチマークが中断されました。")
    except Exception as ex:
        logger.exception(f"ベンチマーク実行中にエラーが発生しました: {ex}")


if __name__ == "__main__":
    main()
