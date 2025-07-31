#!/usr/bin/env python3
"""
Usage: .venv/bin/python -m tests.debug_memory_profiling [--device cuda] [--models 3] [--warmup 5] [--iterations 10]

高精度メモリプロファイリングスクリプト

PyTorch Profiler とメモリスナップショットを使用して、
実際の VRAM 使用量とボトルネックを詳細に分析する。

機能:
- torch.profiler による詳細メモリトレース
- ランダム性完全排除による再現可能な測定
- 複数モデル・テキストでの分析
- 長時間実行による断片化進行分析
- BERT 単体 vs VITS2 単体 vs 統合処理の比較
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

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
from style_bert_vits2.nlp import (
    bert_models,
    clean_text_with_given_phone_tone,
    extract_bert_feature,
)
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder

from .utils import set_random_seeds


# 様々な長さのテストテキスト（断片化パターン分析用）
TEST_TEXTS = [
    # 短文（8-16音素）
    "こんにちは。",
    "今日は晴れです。",
    "ありがとう。",
    # 中文（32-64音素）
    "今日はとてもいい天気ですね。空が青くて気持ちがいいです。",
    "人工知能の技術が急速に発展しています。",
    "桜の季節になりました。公園で花見をしましょう。",
    # やや長文（96-128音素）
    "昨日は雨が降っていましたが、今日は晴れてとても気持ちがいいです。公園を散歩してから、カフェでコーヒーを飲みました。",
    "最近の機械学習技術の進歩は目覚ましく、特に大規模言語モデルの登場により自然言語処理分野が大きく変化しています。",
    # 長文（192-256音素）
    "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては、人一倍に敏感であった。",
    # 非常に長文（384-512音素）
    "あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。またそのなかでいっしょになったたくさんのひとたち、ファゼーロとロザーロ、羊飼のミーロや、顔の赤いこどもたち、地主のテーモ、山猫博士のボーガント・デストゥパーゴなど、いまこの暗い巨きな家にはたったひとりがいません。夏の夕暮れどきに白い月がのぼり、草の波から出る霧がそのへんをいつそう白くしたころ、たくさんのおじいさんやおばあさんが、木の机の前にすわって、青い蝋燭をたてて、なにやら熱心に読書をしています。",
]


class MemoryProfiler:
    """高精度メモリプロファイリングクラス"""

    def __init__(self, device: str):
        self.device = device
        self.snapshots: list[dict[str, Any]] = []
        self.profiler_traces: list[str] = []

    def reset_memory_stats(self) -> None:
        """メモリ統計をリセット"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

    def get_memory_info(self) -> dict[str, Any]:
        """詳細なメモリ情報を取得"""
        if self.device != "cuda":
            return {
                "allocated": 0.0,
                "reserved": 0.0,
                "max_allocated": 0.0,
                "fragmentation": 0.0,
            }

        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        fragmentation = reserved - allocated

        return {
            "allocated": allocated,
            "reserved": reserved,
            "max_allocated": max_allocated,
            "fragmentation": fragmentation,
        }

    def snapshot(
        self, label: str, additional_info: dict[str, Any] | None = None
    ) -> None:
        """メモリスナップショットを記録"""
        memory_info = self.get_memory_info()
        memory_info["label"] = label
        memory_info["timestamp"] = time.time()

        if additional_info:
            memory_info.update(additional_info)

        self.snapshots.append(memory_info)

        logger.info(
            f"{label}: Allocated={memory_info['allocated']:.3f}GB, "
            f"Reserved={memory_info['reserved']:.3f}GB, "
            f"Fragmentation={memory_info['fragmentation']:.3f}GB"
        )

    def save_profiler_trace(self, trace_path: str) -> None:
        """プロファイラートレースを保存"""
        self.profiler_traces.append(trace_path)
        logger.info(f"Profiler trace saved: {trace_path}")


def profile_bert_only(
    profiler: MemoryProfiler,
    texts: list[str],
    device: str,
    use_fp16: bool = True,
    warmup_runs: int = 3,
    measurement_runs: int = 5,
) -> dict[str, Any]:
    """BERT単体のメモリ・処理時間プロファイリング"""

    logger.info("=== BERT単体プロファイリング開始 ===")

    profiler.reset_memory_stats()
    profiler.snapshot("BERT_start")

    # BERTモデルロード
    with record_function("bert_model_load"):
        bert_models.load_model(Languages.JP, device_map=device, use_fp16=use_fp16)

    profiler.snapshot("BERT_model_loaded")

    # ウォーミングアップ
    logger.info(f"BERT ウォーミングアップ実行中 ({warmup_runs}回)")
    for i in range(warmup_runs):
        for text in texts[:3]:  # 短いテキストでウォーミングアップ
            with record_function(f"bert_warmup_{i}"):
                norm_text, _, _, word2ph, sep_text, _, _ = (
                    clean_text_with_given_phone_tone(
                        text, Languages.JP, use_jp_extra=True, raise_yomi_error=False
                    )
                )
                _ = extract_bert_feature(
                    norm_text, word2ph, Languages.JP, device, sep_text=sep_text
                )

    profiler.snapshot("BERT_after_warmup")

    # 本測定
    results: dict[str, Any] = {"texts": [], "total_times": [], "individual_times": []}

    logger.info(f"BERT 本測定開始 ({measurement_runs}回 × {len(texts)}テキスト)")

    for run in range(measurement_runs):
        run_times = []
        run_start = time.perf_counter()

        for text_idx, text in enumerate(texts):
            with record_function(f"bert_inference_run{run}_text{text_idx}"):
                text_start = time.perf_counter()

                norm_text, _, _, word2ph, sep_text, _, _ = (
                    clean_text_with_given_phone_tone(
                        text, Languages.JP, use_jp_extra=True, raise_yomi_error=False
                    )
                )

                bert_feature = extract_bert_feature(
                    norm_text, word2ph, Languages.JP, device, sep_text=sep_text
                )

                text_time = time.perf_counter() - text_start
                run_times.append(text_time)

                # メモリ情報を記録
                if run == 0:  # 最初の実行のみ詳細記録
                    profiler.snapshot(
                        f"BERT_text_{text_idx}_len{len(text)}",
                        {"text_length": len(text), "inference_time": text_time},
                    )

        run_total = time.perf_counter() - run_start
        results["total_times"].append(run_total)
        results["individual_times"].append(run_times)

        logger.info(f"  Run {run + 1}: 総時間 {run_total:.3f}s")

    profiler.snapshot("BERT_measurement_complete")

    # 統計計算
    avg_total_time = np.mean(results["total_times"])
    std_total_time = np.std(results["total_times"])

    avg_individual_times = np.mean(results["individual_times"], axis=0)

    results["avg_total_time"] = avg_total_time
    results["std_total_time"] = std_total_time
    results["avg_individual_times"] = avg_individual_times.tolist()
    results["texts"] = texts

    logger.info(f"BERT平均総処理時間: {avg_total_time:.3f}±{std_total_time:.3f}s")

    return results


def profile_full_tts(
    profiler: MemoryProfiler,
    models: list[TTSModel],
    texts: list[str],
    device: str,
    use_fp16: bool = True,
    warmup_runs: int = 3,
    measurement_runs: int = 5,
) -> dict[str, Any]:
    """完全TTS推論のメモリ・処理時間プロファイリング"""

    logger.info("=== 完全TTS推論プロファイリング開始 ===")

    results = {"models": [], "texts": [], "times": [], "memory_peaks": []}

    for model_idx, model in enumerate(models):
        logger.info(f"モデル {model_idx + 1}/{len(models)}: {model.model_path}")

        profiler.reset_memory_stats()
        profiler.snapshot(f"TTS_model_{model_idx}_start")

        model.load()
        profiler.snapshot(f"TTS_model_{model_idx}_loaded")

        # ウォーミングアップ
        logger.info(f"  ウォーミングアップ実行中 ({warmup_runs}回)")
        for i in range(warmup_runs):
            with record_function(f"tts_warmup_model{model_idx}_{i}"):
                assert model.net_g is not None
                style_vec = model.get_style_vector(0, 1.0)
                _ = infer(
                    text=texts[0],  # 短いテキストでウォーミングアップ
                    style_vec=style_vec,
                    sdp_ratio=DEFAULT_SDP_RATIO,
                    noise_scale=DEFAULT_NOISE,
                    noise_scale_w=DEFAULT_NOISEW,
                    length_scale=DEFAULT_LENGTH,
                    sid=0,
                    language=Languages.JP,
                    hps=model.hyper_parameters,
                    net_g=model.net_g,
                    device=device,
                    use_fp16=use_fp16,
                    clear_cuda_cache=False,
                    enable_tensor_padding=False,
                )

        profiler.snapshot(f"TTS_model_{model_idx}_after_warmup")

        # 本測定
        model_results = {"text_times": [], "text_peaks": []}

        for run in range(measurement_runs):
            run_times = []
            run_peaks = []

            for text_idx, text in enumerate(texts):
                profiler.reset_memory_stats()

                with record_function(
                    f"tts_inference_model{model_idx}_run{run}_text{text_idx}"
                ):
                    start_time = time.perf_counter()

                    assert model.net_g is not None
                    style_vec = model.get_style_vector(0, 1.0)
                    audio = infer(
                        text=text,
                        style_vec=style_vec,
                        sdp_ratio=DEFAULT_SDP_RATIO,
                        noise_scale=DEFAULT_NOISE,
                        noise_scale_w=DEFAULT_NOISEW,
                        length_scale=DEFAULT_LENGTH,
                        sid=0,
                        language=Languages.JP,
                        hps=model.hyper_parameters,
                        net_g=model.net_g,
                        device=device,
                        use_fp16=use_fp16,
                        clear_cuda_cache=False,
                        enable_tensor_padding=False,
                    )

                    inference_time = time.perf_counter() - start_time
                    run_times.append(inference_time)

                    memory_info = profiler.get_memory_info()
                    run_peaks.append(memory_info["max_allocated"])

                    if run == 0:  # 最初の実行のみ詳細記録
                        profiler.snapshot(
                            f"TTS_model{model_idx}_text{text_idx}_len{len(text)}",
                            {
                                "text_length": len(text),
                                "inference_time": inference_time,
                                "audio_length": len(audio)
                                / model.hyper_parameters.data.sampling_rate,
                            },
                        )

            model_results["text_times"].append(run_times)
            model_results["text_peaks"].append(run_peaks)

            logger.info(f"  Run {run + 1}: 平均時間 {np.mean(run_times):.3f}s")

        # 統計計算
        avg_times = np.mean(model_results["text_times"], axis=0)
        avg_peaks = np.mean(model_results["text_peaks"], axis=0)

        results["models"].append(model.model_path.name)
        results["times"].append(avg_times.tolist())
        results["memory_peaks"].append(avg_peaks.tolist())

        model.unload()
        profiler.snapshot(f"TTS_model_{model_idx}_unloaded")

    results["texts"] = texts

    return results


def analyze_memory_fragmentation(profiler: MemoryProfiler) -> dict[str, Any]:
    """メモリ断片化の詳細分析"""

    if not profiler.snapshots:
        return {}

    analysis: dict[str, Any] = {
        "fragmentation_progression": [],
        "peak_fragmentation": 0.0,
        "avg_fragmentation": 0.0,
        "memory_efficiency": 0.0,
    }

    fragmentations = []
    allocations = []

    for snapshot in profiler.snapshots:
        frag = snapshot.get("fragmentation", 0.0)
        alloc = snapshot.get("allocated", 0.0)

        fragmentations.append(float(frag))
        allocations.append(float(alloc))

        analysis["fragmentation_progression"].append(
            {
                "label": snapshot["label"],
                "fragmentation_gb": frag,
                "allocated_gb": alloc,
                "timestamp": snapshot.get("timestamp", 0),
            }
        )

    if fragmentations:
        analysis["peak_fragmentation"] = float(max(fragmentations))
        analysis["avg_fragmentation"] = float(np.mean(fragmentations))

        if allocations:
            # メモリ効率 = 実使用量 / (実使用量 + 断片化量)
            total_allocated = np.mean(allocations)
            avg_fragmentation = analysis["avg_fragmentation"]
            if total_allocated + avg_fragmentation > 0:
                analysis["memory_efficiency"] = float(
                    total_allocated / (total_allocated + avg_fragmentation)
                )

    return analysis


def save_results_to_json(results: dict[str, Any], output_path: Path) -> None:
    """結果をJSONファイルに保存"""

    # NumPy配列をリストに変換
    def convert_numpy(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj

    def recursive_convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(v) for v in obj]
        else:
            return convert_numpy(obj)

    converted_results = recursive_convert(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted_results, f, indent=2, ensure_ascii=False)

    logger.info(f"結果を保存しました: {output_path}")


def run_comprehensive_profiling(
    device: str = "cuda",
    num_models: int = 3,
    warmup_runs: int = 5,
    measurement_runs: int = 10,
    use_fp16: bool = True,
    save_traces: bool = True,
) -> None:
    """包括的メモリプロファイリングを実行"""

    # ランダムシード固定
    set_random_seeds()

    logger.info("=" * 80)
    logger.info("Style-Bert-VITS2 包括的メモリプロファイリング")
    logger.info("=" * 80)
    logger.info(f"デバイス: {device}")
    logger.info(f"対象モデル数: {num_models}")
    logger.info(f"ウォーミングアップ実行回数: {warmup_runs}")
    logger.info(f"測定実行回数: {measurement_runs}")
    logger.info(f"FP16使用: {use_fp16}")
    logger.info(f"プロファイラートレース保存: {save_traces}")
    logger.info("=" * 80)

    profiler = MemoryProfiler(device)
    results = {}

    # 出力ディレクトリ作成
    output_dir = Path("tests/profiling_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = int(time.time())

    # PyTorch Profilerの設定
    profiler_activities = [ProfilerActivity.CPU]
    if device == "cuda":
        profiler_activities.append(ProfilerActivity.CUDA)

    # 1. BERT単体プロファイリング
    logger.info("Phase 1: BERT単体プロファイリング")

    if save_traces:
        trace_path = output_dir / f"bert_trace_{timestamp}.json"
        with profile(
            activities=profiler_activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            bert_results = profile_bert_only(
                profiler, TEST_TEXTS, device, use_fp16, warmup_runs, measurement_runs
            )

        prof.export_chrome_trace(str(trace_path))
        profiler.save_profiler_trace(str(trace_path))
    else:
        bert_results = profile_bert_only(
            profiler, TEST_TEXTS, device, use_fp16, warmup_runs, measurement_runs
        )

    results["bert_only"] = bert_results

    # 2. TTSモデル選択とロード
    logger.info("Phase 2: TTSモデル準備")

    model_holder = TTSModelHolder(
        BASE_DIR / "model_assets",
        device,
        onnx_providers=[],
        ignore_onnx=True,
        use_fp16=use_fp16,
    )

    if len(model_holder.models_info) == 0:
        logger.error("音声合成モデルが見つかりませんでした")
        return

    # 利用可能なモデルからランダムに選択
    available_models = []
    for info in model_holder.models_info[:num_models]:
        safetensors_files = [
            f
            for f in info.files
            if f.endswith(".safetensors") and not f.startswith(".")
        ]
        if safetensors_files:
            model = model_holder.get_model(info.name, safetensors_files[0])
            available_models.append(model)

    if not available_models:
        logger.error("利用可能なsafetensorsモデルが見つかりませんでした")
        return

    logger.info(f"選択されたモデル: {[m.model_path.name for m in available_models]}")

    # 3. 完全TTS推論プロファイリング
    logger.info("Phase 3: 完全TTS推論プロファイリング")

    if save_traces:
        trace_path = output_dir / f"tts_trace_{timestamp}.json"
        with profile(
            activities=profiler_activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            tts_results = profile_full_tts(
                profiler,
                available_models,
                TEST_TEXTS,
                device,
                use_fp16,
                warmup_runs,
                measurement_runs,
            )

        prof.export_chrome_trace(str(trace_path))
        profiler.save_profiler_trace(str(trace_path))
    else:
        tts_results = profile_full_tts(
            profiler,
            available_models,
            TEST_TEXTS,
            device,
            use_fp16,
            warmup_runs,
            measurement_runs,
        )

    results["full_tts"] = tts_results

    # 4. メモリ断片化分析
    logger.info("Phase 4: メモリ断片化分析")
    fragmentation_analysis = analyze_memory_fragmentation(profiler)
    results["fragmentation_analysis"] = fragmentation_analysis

    # 5. 全体統計
    results["metadata"] = {
        "device": device,
        "num_models": len(available_models),
        "warmup_runs": warmup_runs,
        "measurement_runs": measurement_runs,
        "use_fp16": use_fp16,
        "timestamp": timestamp,
        "test_texts": TEST_TEXTS,
        "profiler_traces": profiler.profiler_traces,
    }

    results["memory_snapshots"] = profiler.snapshots

    # 6. 結果保存
    result_path = output_dir / f"profiling_results_{timestamp}.json"
    save_results_to_json(results, result_path)

    # 7. サマリー出力
    print_profiling_summary(results)

    logger.info("=" * 80)
    logger.info("プロファイリング完了")
    logger.info(f"結果ファイル: {result_path}")
    if save_traces:
        logger.info(f"プロファイラートレース: {profiler.profiler_traces}")
    logger.info("=" * 80)


def print_profiling_summary(results: dict[str, Any]) -> None:
    """プロファイリング結果のサマリーを出力"""

    print("\n" + "=" * 80)
    print("プロファイリング結果サマリー")
    print("=" * 80)

    # BERT結果
    bert_results = results.get("bert_only", {})
    if bert_results:
        print("\nBERT単体処理:")
        print(
            f"  平均総処理時間: {bert_results.get('avg_total_time', 0):.3f}±{bert_results.get('std_total_time', 0):.3f}s"
        )

        individual_times = bert_results.get("avg_individual_times", [])
        texts = bert_results.get("texts", [])

        if individual_times and texts:
            print("  テキスト別平均時間:")
            for i, (time_val, text) in enumerate(zip(individual_times, texts)):
                text_preview = text[:30] + "..." if len(text) > 30 else text
                print(
                    f"    {i + 1}. {time_val:.3f}s - {text_preview} ({len(text)}文字)"
                )

    # TTS結果
    tts_results = results.get("full_tts", {})
    if tts_results:
        print("\n完全TTS推論:")
        models = tts_results.get("models", [])
        times = tts_results.get("times", [])
        peaks = tts_results.get("memory_peaks", [])
        texts = tts_results.get("texts", [])

        if models and times:
            for i, (model, model_times, model_peaks) in enumerate(
                zip(models, times, peaks)
            ):
                print(f"  モデル {i + 1}: {model}")
                print(f"    平均推論時間: {np.mean(model_times):.3f}s")
                print(f"    平均ピークメモリ: {np.mean(model_peaks):.3f}GB")

                if i == 0 and texts:  # 最初のモデルでテキスト別詳細表示
                    print("    テキスト別時間:")
                    for j, (time_val, text) in enumerate(zip(model_times, texts)):
                        text_preview = text[:20] + "..." if len(text) > 20 else text
                        print(f"      {j + 1}. {time_val:.3f}s - {text_preview}")

    # 断片化分析
    frag_analysis = results.get("fragmentation_analysis", {})
    if frag_analysis:
        print("\nメモリ断片化分析:")
        print(f"  ピーク断片化: {frag_analysis.get('peak_fragmentation', 0):.3f}GB")
        print(f"  平均断片化: {frag_analysis.get('avg_fragmentation', 0):.3f}GB")
        print(f"  メモリ効率: {frag_analysis.get('memory_efficiency', 0):.1%}")

    # メタデータ
    metadata = results.get("metadata", {})
    if metadata:
        print("\n測定環境:")
        print(f"  デバイス: {metadata.get('device')}")
        print(f"  対象モデル数: {metadata.get('num_models')}")
        print(f"  測定回数: {metadata.get('measurement_runs')}")
        print(f"  FP16: {metadata.get('use_fp16')}")


def main():
    parser = argparse.ArgumentParser(description="高精度メモリプロファイリング")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="使用デバイス (default: cuda)",
    )
    parser.add_argument(
        "--models",
        type=int,
        default=3,
        help="プロファイリング対象モデル数 (default: 3)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="ウォーミングアップ実行回数 (default: 5)"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="測定実行回数 (default: 10)"
    )
    parser.add_argument(
        "--fp16", dest="use_fp16", action="store_true", help="FP16推論を使用 (default)"
    )
    parser.add_argument(
        "--no-fp16", dest="use_fp16", action="store_false", help="FP16推論を無効化"
    )
    parser.add_argument(
        "--no-traces",
        dest="save_traces",
        action="store_false",
        help="プロファイラートレースの保存を無効化",
    )
    parser.set_defaults(use_fp16=True, save_traces=True)

    args = parser.parse_args()

    try:
        run_comprehensive_profiling(
            device=args.device,
            num_models=args.models,
            warmup_runs=args.warmup,
            measurement_runs=args.iterations,
            use_fp16=args.use_fp16,
            save_traces=args.save_traces,
        )
    except KeyboardInterrupt:
        print("\nプロファイリングが中断されました。")
    except Exception as ex:
        logger.exception(f"プロファイリング実行中にエラーが発生しました: {ex}")


if __name__ == "__main__":
    main()
