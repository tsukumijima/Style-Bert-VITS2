#!/usr/bin/env python3
"""
Style-Bert-VITS2 長文一括推論のパフォーマンス測定スクリプト

このスクリプトは infer.py に実装されている infer() 関数のパフォーマンスを測定し、
長いテキストでの処理時間とピークメモリ使用量をベンチマークする。
測定時の推論処理はすべて FP16 で行う。

測定項目:
- 総処理時間
- ピークメモリ使用量 (VRAM)
- 生成音声の長さごとの効果

使用方法:
    .venv/bin/python -m tests.long_inference_benchmark [--device cuda] [--model koharune-ami] [--runs 5]
"""

import argparse
import time
from typing import Any

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


# 測定用サンプルテキスト (300文字以上)
BENCHMARK_TEXTS = [
    {
        # 初回ロード用（ダミー）
        "text": "あああ",
        "description": "短文（ダミー）",
    },
    {
        "text": "あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。またそのなかでいっしょになったたくさんのひとたち、ファゼーロとロザーロ、羊飼のミーロや、顔の赤いこどもたち、地主のテーモ、山猫博士のボーガント・デストゥパーゴなど、いまこの暗い巨きな家にはたったひとりがいません。",
        "description": "中文",
    },
    {
        "text": "あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。またそのなかでいっしょになったたくさんのひとたち、ファゼーロとロザーロ、羊飼のミーロや、顔の赤いこどもたち、地主のテーモ、山猫博士のボーガント・デストゥパーゴなど、いまこの暗い巨きな家にはたったひとりがいません。夏の夕暮れどきに白い月がのぼり、草の波から出る霧がそのへんをいつそう白くしたころ、たくさんのおじいさんやおばあさんが、木の机の前にすわって、青い蝋燭をたてて、なにやら熱心に読書をしています。そこにはまた、青い服を着たこどもたちが、たくさん集まって、なにやら楽しげに遊んでいます。",
        "description": "長文",
    },
    {
        "text": "あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。またそのなかでいっしょになったたくさんのひとたち、ファゼーロとロザーロ、羊飼のミーロや、顔の赤いこどもたち、地主のテーモ、山猫博士のボーガント・デストゥパーゴなど、いまこの暗い巨きな家にはたったひとりがいません。夏の夕暮れどきに白い月がのぼり、草の波から出る霧がそのへんをいつそう白くしたころ、たくさんのおじいさんやおばあさんが、木の机の前にすわって、青い蝋燭をたてて、なにやら熱心に読書をしています。そこにはまた、青い服を着たこどもたちが、たくさん集まって、なにやら楽しげに遊んでいます。そこにはまた、青い服を着たこどもたちが、たくさん集まって、なにやら楽しげに遊んでいます。そこにはまた、青い服を着たこどもたちが、たくさん集まって、なにやら楽しげに遊んでいます。そこにはまた、青い服を着たこどもたちが、たくさん集まって、なにやら楽しげに遊んでいます。そこにはまた、青い服を着たこどもたちが、たくさん集まって、なにやら楽しげに遊んでいます。",
        "description": "長文",
    },
]


def measure_infer_performance(
    model: TTSModel,
    text: str,
    device: str,
    **infer_kwargs: Any,
) -> tuple[float, float, float, NDArray[np.float32]]:
    """
    infer() 関数のパフォーマンスを測定する。

    Returns:
        tuple: (総処理時間, ピークメモリ使用量(MB), 生成音声長(秒), 音声データ)
    """
    # メモリ統計をリセット
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        print("CUDA キャッシュをクリアしました")

    start_time = time.perf_counter()

    net_g = model.net_g
    assert net_g is not None

    style_vec = model.get_style_vector(0, 1.0)  # デフォルトスタイル

    # 推論を実行
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
        **infer_kwargs,
    )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # ピークメモリ使用量を取得
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    else:
        peak_memory = 0.0  # CPUでは測定しない

    # 生成音声の長さを計算
    audio_duration = len(audio_data) / model.hyper_parameters.data.sampling_rate

    return total_time, peak_memory, audio_duration, audio_data


def run_benchmark(
    device: str = "cpu",
    model_name: str = "koharune-ami",
    num_runs: int = 5,
    use_fp16: bool = True,
) -> None:
    """
    ベンチマークを実行する。
    """
    print("=" * 80)
    print("Style-Bert-VITS2 長文一括推論パフォーマンス測定")
    print("=" * 80)
    print(f"デバイス: {device}")
    print(f"モデル: {model_name}")
    print(f"測定回数: {num_runs}")
    print(f"FP16: {use_fp16}")
    print("=" * 80)

    # モデルホルダーを初期化
    model_holder = TTSModelHolder(
        BASE_DIR / "model_assets",
        device,
        onnx_providers=[
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
        ],
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

    # 結果を保存するリスト
    results = []

    # 各テキストでベンチマークを実行
    for i, test_case in enumerate(BENCHMARK_TEXTS):
        text = test_case["text"]
        description = test_case["description"]

        print(f"測定中: {description}")
        print(f"テキスト: {text[:50]}... (全{len(text)}文字)")

        # 複数回実行して平均を取る
        infer_times = []
        peak_memories = []
        infer_durations = []

        for run in range(num_runs):
            try:
                # 推論を測定
                infer_time, peak_memory, infer_duration, _ = measure_infer_performance(
                    model,
                    text,
                    device,
                    use_fp16=use_fp16,
                )
                infer_times.append(infer_time)
                peak_memories.append(peak_memory)
                infer_durations.append(infer_duration)

                print(
                    f"  実行{run + 1}: 時間 {infer_time:.3f}秒, ピークメモリ {peak_memory:.2f}MB"
                )

            except Exception as ex:
                logger.exception(f"測定中にエラーが発生しました: {ex}")
                continue

        if not infer_times:
            print("  測定に失敗しました。")
            continue

        # 初回はロードが入るため捨てる
        if i == 0:
            continue

        # 平均値を計算
        avg_infer_time = np.mean(infer_times)
        avg_peak_memory = np.mean(peak_memories)
        avg_infer_duration = np.mean(infer_durations)

        # 結果を保存
        result = {
            "text": text,
            "description": description,
            "actual_duration": avg_infer_duration,
            "infer_time": avg_infer_time,
            "peak_memory": avg_peak_memory,
        }
        results.append(result)

        # 個別結果を表示
        print(f"  平均音声長: {avg_infer_duration:.2f}秒")
        print(f"  平均推論時間: {avg_infer_time:.3f}秒")
        print(f"  平均ピークメモリ: {avg_peak_memory:.2f}MB")
        print("=" * 80)

    # モデルをアンロード
    model.unload()

    # 総合結果を表示
    print("総合結果")
    print("=" * 80)
    print(
        f"{'説明':<20} {'音声長(秒)':<12} {'推論時間(秒)':<12} {'ピークメモリ(MB)':<15}"
    )
    print("-" * 60)

    for result in results:
        print(
            f"{result['description']:<20} "
            f"{result['actual_duration']:<12.2f} "
            f"{result['infer_time']:<12.3f} "
            f"{result['peak_memory']:<15.2f}"
        )

    print("=" * 80)
    print("分析:")
    avg_time = np.mean([r["infer_time"] for r in results])
    avg_memory = np.mean([r["peak_memory"] for r in results])
    print(f"- 全体平均推論時間: {avg_time:.3f}秒")
    print(f"- 全体平均ピークメモリ: {avg_memory:.2f}MB")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Style-Bert-VITS2 長文一括推論パフォーマンス測定"
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
        "--runs",
        type=int,
        default=5,
        help="各テストケースの実行回数 (default: 5)",
    )
    parser.add_argument(
        "--fp16",
        dest="use_fp16",
        action="store_true",
        help="FP16 で推論を行う (default)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="use_fp16",
        action="store_false",
        help="FP16 を無効化する",
    )
    parser.set_defaults(use_fp16=True)

    args = parser.parse_args()

    try:
        # Preload BERT model
        logger.info("Preloading BERT model...")
        bert_models.load_model(
            Languages.JP, device_map=args.device, use_fp16=args.use_fp16
        )
        logger.info("BERT model preloaded successfully")

        run_benchmark(
            device=args.device,
            model_name=args.model,
            num_runs=args.runs,
            use_fp16=args.use_fp16,
        )
    except KeyboardInterrupt:
        print("\nベンチマークが中断されました。")
    except Exception as ex:
        logger.exception(f"ベンチマーク実行中にエラーが発生しました: {ex}")


if __name__ == "__main__":
    main()
