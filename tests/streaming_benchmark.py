#!/usr/bin/env python3
"""
Style-Bert-VITS2 ストリーミング推論のパフォーマンス測定スクリプト

このスクリプトは infer.py に実装されている infer() 関数と infer_stream() 関数のパフォーマンスを比較し、
https://qiita.com/__dAi00/items/970f0fe66286510537dd の結果と同様の測定を行う。
測定時の推論処理はすべて FP16 で行う。

測定項目:
- 初回チャンク生成までの時間（ストリーミング版のレイテンシ）
- 全音声生成完了までの時間（総処理時間）
- 生成音声の長さごとの効果の変化

使用方法:
    .venv/bin/python -m tests.streaming_benchmark [--device cuda] [--model koharune-ami] [--runs 3]
"""

import argparse
import time
from typing import Any, cast

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
from style_bert_vits2.models.infer import infer, infer_stream
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder

from .utils import save_benchmark_audio, set_random_seeds


# 測定用サンプルテキスト
BENCHMARK_TEXTS = [
    {
        # 初回ロード用（ダミー）
        "text": "あああ",
        "estimated_duration": 1.1,
        "description": "短文（約1秒）",
    },
    {
        "text": "こんにちは",
        "estimated_duration": 1.1,
        "description": "短文（約1秒）",
    },
    {
        "text": "ウオウオフィッシュライフ",
        "estimated_duration": 1.4,
        "description": "短文（約1.4秒）",
    },
    {
        "text": "東京特許許可局",
        "estimated_duration": 1.8,
        "description": "短文（約1.8秒）",
    },
    {
        "text": "毒を食らわば皿まで",
        "estimated_duration": 2.0,
        "description": "中文（約2秒）",
    },
    {
        "text": "あのイーハトーヴォのすきとおった風",
        "estimated_duration": 2.5,
        "description": "中文（約2.5秒）",
    },
    {
        "text": "えー！なるっちの担当箇所がバグだらけ！？",
        "estimated_duration": 3.2,
        "description": "中文（約3.2秒）",
    },
    {
        "text": "あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら",
        "estimated_duration": 5.4,
        "description": "長文（約5.4秒）",
    },
    {
        "text": "あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市",
        "estimated_duration": 7.7,
        "description": "長文（約7.7秒）",
    },
    {
        "text": "あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。",
        "estimated_duration": 10.4,
        "description": "長文（約10.4秒）",
    },
]


def measure_infer_performance(
    model: TTSModel,
    text: str,
    **infer_kwargs: Any,
) -> tuple[float, float, NDArray[np.float32]]:
    """
    通常の infer() 関数のパフォーマンスを測定する。

    Returns:
        tuple: (総処理時間, 生成音声長(秒), 音声データ)
    """
    start_time = time.perf_counter()

    net_g = model.net_g
    assert net_g is not None

    style_vec = model.get_style_vector(0, 1.0)  # デフォルトスタイル

    # 比較のために低レベル API で推論を実行
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
            **infer_kwargs,
        )

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # 生成音声の長さを計算
        audio_duration = len(audio_data) / model.hyper_parameters.data.sampling_rate

        return total_time, audio_duration, audio_data


def measure_infer_stream_performance(
    model_holder: TTSModelHolder,
    model_name: str,
    model_file: str,
    text: str,
    device: str,
    **infer_kwargs: Any,
) -> tuple[float, float, float, list[NDArray[np.float32]]]:
    """
    ストリーミング版 infer_stream() 関数のパフォーマンスを測定する。

    Returns:
        tuple: (初回チャンク時間, 総処理時間, 生成音声長(秒), 音声チャンクリスト)
    """
    # モデルをロード
    model = model_holder.get_model(model_name, model_file)
    model.load()

    # 低レベル API でストリーミング推論を実行
    net_g = model.net_g
    assert net_g is not None

    style_vec = model.get_style_vector(0, 1.0)  # デフォルトスタイル

    start_time = time.perf_counter()
    first_chunk_time = None
    audio_chunks = []

    # ストリーミング推論を実行
    with torch.inference_mode():
        audio_generator = infer_stream(
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
            **infer_kwargs,
        )

        for i, audio_chunk in enumerate(audio_generator):
            if i == 0:
                # 初回チャンクが生成された時刻を記録
                first_chunk_time = time.perf_counter() - start_time
            audio_chunks.append(audio_chunk)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # 全音声チャンクを結合
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            audio_duration = len(full_audio) / model.hyper_parameters.data.sampling_rate
        else:
            audio_duration = 0.0

        return first_chunk_time or 0.0, total_time, audio_duration, audio_chunks


def run_benchmark(
    device: str = "cpu",
    model_name: str = "koharune-ami",
    num_runs: int = 3,
    use_fp16: bool = True,
    fix_seed: bool = False,
) -> None:
    """
    ベンチマークを実行する。
    """
    # ランダムシード固定
    if fix_seed:
        set_random_seeds()

    print("=" * 80)
    print("Style-Bert-VITS2 ストリーミング推論パフォーマンス測定")
    print("=" * 80)
    print(f"デバイス: {device}")
    print(f"モデル: {model_name}")
    print(f"測定回数: {num_runs}")
    print(f"FP16: {use_fp16}")
    print(f"ランダムシード固定: {'有効' if fix_seed else '無効'}")
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

    # 結果を保存するリスト
    results = []

    # 各テキストでベンチマークを実行
    for i, test_case in enumerate(BENCHMARK_TEXTS):
        text = cast(str, test_case["text"])
        estimated_duration = test_case["estimated_duration"]
        description = test_case["description"]

        print(f"測定中: {description}")
        print(f"テキスト: {text}")

        # 複数回実行して平均を取る
        infer_times = []
        infer_durations = []
        stream_first_times = []
        stream_total_times = []
        stream_durations = []

        # 最後の実行の音声データを保存用に記録
        last_normal_audio = None
        last_stream_audio = None
        last_sample_rate = None

        for run in range(num_runs):
            try:
                # 通常の infer() を測定
                infer_time, infer_duration, normal_audio = measure_infer_performance(
                    model,
                    text,
                    use_fp16=use_fp16,
                    clear_cuda_cache=True,
                )
                infer_times.append(infer_time)
                infer_durations.append(infer_duration)

                # 最後の実行の音声データを保存
                if run == num_runs - 1:
                    last_normal_audio = normal_audio
                    last_sample_rate = model.hyper_parameters.data.sampling_rate

                # ストリーミング版 infer_stream() を測定
                stream_first_time, stream_total_time, stream_duration, stream_chunks = (
                    measure_infer_stream_performance(
                        model_holder,
                        model_name,
                        model_file,
                        text,
                        device,
                        use_fp16=use_fp16,
                        clear_cuda_cache=True,
                    )
                )
                stream_first_times.append(stream_first_time)
                stream_total_times.append(stream_total_time)
                stream_durations.append(stream_duration)

                # 最後の実行のストリーミング音声データを保存
                if run == num_runs - 1 and stream_chunks:
                    last_stream_audio = np.concatenate(stream_chunks)

                print("=" * 80)

            except Exception as ex:
                logger.exception(f"測定中にエラーが発生しました: {ex}")
                continue

        if not infer_times:
            print("  測定に失敗しました。")
            continue

        # 初回はロードが入るため捨てる
        if i == 0:
            continue

        # 音声ファイルを保存（初回のダミーは除く）
        if (
            i > 0  # 初回はダミー
            and last_normal_audio is not None
            and last_stream_audio is not None
            and last_sample_rate is not None
        ):
            save_benchmark_audio(
                last_normal_audio,
                last_sample_rate,
                text,
                "streaming_benchmark",
                "normal",
            )
            save_benchmark_audio(
                last_stream_audio,
                last_sample_rate,
                text,
                "streaming_benchmark",
                "streaming",
            )

        # 平均値を計算
        avg_infer_time = np.mean(infer_times)
        avg_infer_duration = np.mean(infer_durations)
        avg_stream_first_time = np.mean(stream_first_times)
        avg_stream_total_time = np.mean(stream_total_times)
        avg_stream_duration = np.mean(stream_durations)

        # 結果を保存
        result = {
            "text": text,
            "description": description,
            "estimated_duration": estimated_duration,
            "actual_duration": avg_infer_duration,
            "infer_time": avg_infer_time,
            "stream_first_time": avg_stream_first_time,
            "stream_total_time": avg_stream_total_time,
            "efficiency_infer": avg_infer_duration / avg_infer_time,
            "efficiency_stream": avg_stream_duration / avg_stream_total_time,
            "latency_improvement": avg_infer_time - avg_stream_first_time,
        }
        results.append(result)

        # 個別結果を表示
        print(f"  実際の音声長: {avg_infer_duration:.2f}秒")
        print(f"  通常推論時間: {avg_infer_time:.3f}秒")
        print(f"  ストリーミング初回: {avg_stream_first_time:.3f}秒")
        print(f"  ストリーミング総時間: {avg_stream_total_time:.3f}秒")
        print(f"  レイテンシ改善: {result['latency_improvement']:.3f}秒")
        print("=" * 80)

    # モデルをアンロード
    model.unload()

    # 総合結果を表示
    print("総合結果")
    print("=" * 80)
    print(
        f"{'音声長(秒)':<12} {'通常推論(秒)':<12} {'ストリーミング初回(秒)':<18} {'改善度':<10}"
    )
    print("-" * 60)

    for result in results:
        improvement = "BETTER" if result["latency_improvement"] > 0 else "WORSE"
        print(
            f"{result['actual_duration']:<12.2f} "
            f"{result['infer_time']:<12.3f} "
            f"{result['stream_first_time']:<18.3f} "
            f"{improvement:<10}"
        )

    print("=" * 80)
    print("分析:")
    # 2秒以上のケースでの改善効果を確認
    long_audio_results = [r for r in results if r["actual_duration"] >= 2.0]
    if long_audio_results:
        avg_improvement = np.mean(
            [r["latency_improvement"] for r in long_audio_results]
        )
        print(f"- 2秒以上の音声でのレイテンシ改善: 平均 {avg_improvement:.3f}秒")

    # 効率の比較
    avg_infer_efficiency = np.mean([r["efficiency_infer"] for r in results])
    avg_stream_efficiency = np.mean([r["efficiency_stream"] for r in results])
    print(f"- 通常推論効率: {avg_infer_efficiency:.2f} (音声長/推論時間)")
    print(f"- ストリーミング効率: {avg_stream_efficiency:.2f} (音声長/推論時間)")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Style-Bert-VITS2 ストリーミング推論パフォーマンス測定"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="推論に使用するデバイス (default: cpu)",
    )
    parser.add_argument(
        "--model",
        default="koharune-ami",
        help="使用するモデル名 (default: koharune-ami)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="各テストケースの実行回数 (default: 3)",
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
    parser.add_argument(
        "--fix-seed",
        action="store_true",
        help="ランダムシードを固定して再現性を確保する",
    )
    parser.set_defaults(use_fp16=True)

    args = parser.parse_args()

    try:
        run_benchmark(
            device=args.device,
            model_name=args.model,
            num_runs=args.runs,
            use_fp16=args.use_fp16,
            fix_seed=args.fix_seed,
        )
    except KeyboardInterrupt:
        print("\nベンチマークが中断されました。")
    except Exception as ex:
        logger.exception(f"ベンチマーク実行中にエラーが発生しました: {ex}")


if __name__ == "__main__":
    main()
