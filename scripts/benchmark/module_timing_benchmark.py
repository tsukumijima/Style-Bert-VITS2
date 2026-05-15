"""
Usage: uv run python -m scripts.benchmark.module_timing_benchmark [--device cuda] [--model koharune-ami] [--runs 5]

各モジュール（TextEncoder, DurationPredictor, Flow, Generator）の実行時間を計測し、
入力長に対するスケーリングを分析するベンチマークスクリプト。

目的:
- 入力長が長くなると一番時間がかかるモジュールを特定する
- torch.compile の適用効果を検証する
"""

import argparse
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from style_bert_vits2.constants import (
    DEFAULT_LENGTH,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons
from style_bert_vits2.models.infer import get_text
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder
from style_bert_vits2.utils.paths import get_paths_config

from ..utils import set_random_seeds


@dataclass
class ModuleTimings:
    """各モジュールの実行時間を保持するデータクラス"""

    text_encoder_ms: float
    duration_predictor_ms: float
    flow_ms: float
    generator_ms: float
    total_ms: float
    output_length: int  # 生成されたフレーム数


# テスト用テキスト（長さ別）
BENCHMARK_TEXTS = [
    {"text": "こんにちは。", "description": "Very Short (6 chars)"},
    {"text": "今日はとても良い天気ですね。", "description": "Short (14 chars)"},
    {
        "text": "音声合成は、機械学習を活用してテキストから人の声を生成する技術です。",
        "description": "Medium (34 chars)",
    },
    {
        "text": "音声合成は、機械学習を活用してテキストから人の声を生成する技術です。このテキストは、モデルの推論速度とメモリ消費を比較するためのベンチマークとして使用されます。",
        "description": "Long (80 chars)",
    },
    {
        "text": "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては、人一倍に敏感であった。",
        "description": "Very Long (104 chars)",
    },
]


def measure_module_timings(
    model: TTSModel,
    text: str,
    device: str,
    use_fp16: bool = True,
) -> ModuleTimings:
    """
    各モジュールの実行時間を個別に計測する。

    Args:
        model: TTSModel インスタンス
        text: 合成するテキスト
        device: 使用するデバイス
        use_fp16: FP16 を使用するかどうか

    Returns:
        ModuleTimings: 各モジュールの実行時間
    """

    # このベンチマークは JP-Extra モデル専用
    # net_g の型をキャストして pyright の型チェックを満たす
    net_g = model.net_g
    assert net_g is not None
    assert isinstance(net_g, SynthesizerTrnJPExtra), (
        "This benchmark only supports JP-Extra models"
    )
    hps = model.hyper_parameters

    # テキストを処理して入力テンソルを準備
    # JP-Extra モデルでは ja_bert のみ使用する (bert_ori, en_bert は使用しない)
    _bert_ori, ja_bert, _en_bert, phones, tones, lang_ids, _token_types = get_text(
        text,
        Languages.JP,
        hps,
        device,
    )

    # テンソルを GPU に転送
    x = phones.to(device).unsqueeze(0)
    x_lengths = torch.tensor([phones.size(0)], dtype=torch.long, device=device)
    tone = tones.to(device).unsqueeze(0)
    language = lang_ids.to(device).unsqueeze(0)
    bert = ja_bert.to(device).unsqueeze(0)
    sid = torch.tensor([0], dtype=torch.long, device=device)
    style_vec = torch.from_numpy(model.get_style_vector(0, 1.0)).to(device).unsqueeze(0)

    # ウォームアップ（初回のCUDAカーネル起動オーバーヘッドを除外）
    with torch.inference_mode():
        _ = net_g.infer(
            x,
            x_lengths,
            sid,
            tone,
            language,
            bert,
            style_vec,
            noise_scale=DEFAULT_NOISE,
            length_scale=DEFAULT_LENGTH,
            noise_scale_w=DEFAULT_NOISEW,
            sdp_ratio=DEFAULT_SDP_RATIO,
            use_fp16=use_fp16,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # 各モジュールの時間を個別に計測
    with torch.inference_mode():
        # === Speaker Embedding ===
        if net_g.n_speakers > 0:
            g = net_g.emb_g(sid).unsqueeze(-1)
        else:
            raise RuntimeError("Reference encoder not supported in this benchmark")

        # BERT 特徴量の型変換
        bert_input = bert.float() if use_fp16 else bert

        # === TextEncoder (enc_p) ===
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()

        x_enc, m_p, logs_p, x_mask = net_g.enc_p(
            x,
            x_lengths,
            tone,
            language,
            bert_input,
            style_vec,
            g=g,
            use_fp16=use_fp16,
        )

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        text_encoder_time = (time.perf_counter() - start) * 1000  # ms

        # === Duration Predictor (sdp + dp) ===
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()

        logw = net_g.sdp(
            x_enc, x_mask, g=g, reverse=True, noise_scale=DEFAULT_NOISEW
        ) * DEFAULT_SDP_RATIO + net_g.dp(x_enc, x_mask, g=g) * (1 - DEFAULT_SDP_RATIO)

        w = torch.exp(logw) * x_mask * DEFAULT_LENGTH
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p_expanded = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )
        logs_p_expanded = torch.matmul(
            attn.squeeze(1), logs_p.transpose(1, 2)
        ).transpose(1, 2)

        z_p = (
            m_p_expanded
            + torch.randn_like(m_p_expanded)
            * torch.exp(logs_p_expanded)
            * DEFAULT_NOISE
        )

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        duration_predictor_time = (time.perf_counter() - start) * 1000  # ms

        # === Flow ===
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()

        z = net_g.flow(z_p, y_mask, g=g, reverse=True)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        flow_time = (time.perf_counter() - start) * 1000  # ms

        output_length = z.size(2)

        # === Generator (dec) ===
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()

        if use_fp16:
            z_input = (z * y_mask).half()
            device_type = z_input.device.type
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=True,
            ):
                _o = net_g.dec(z_input, g=g.half())
        else:
            _o = net_g.dec(z * y_mask, g=g)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        generator_time = (time.perf_counter() - start) * 1000  # ms

    total_time = (
        text_encoder_time + duration_predictor_time + flow_time + generator_time
    )

    return ModuleTimings(
        text_encoder_ms=text_encoder_time,
        duration_predictor_ms=duration_predictor_time,
        flow_ms=flow_time,
        generator_ms=generator_time,
        total_ms=total_time,
        output_length=output_length,
    )


def run_benchmark(
    device: str = "cuda",
    model_name: str = "koharune-ami",
    num_runs: int = 5,
    use_fp16: bool = True,
) -> None:
    """
    ベンチマークを実行する。
    """

    set_random_seeds()

    print("=" * 90)
    print("Style-Bert-VITS2 モジュール別実行時間ベンチマーク")
    print("=" * 90)
    print(f"デバイス: {device}")
    print(f"モデル: {model_name}")
    print(f"測定回数: {num_runs}")
    print(f"FP16: {use_fp16}")
    print("=" * 90)

    # BERT モデルをロード
    if device.startswith("cuda"):
        logger.info("Loading BERT model...")
        bert_models.load_model(Languages.JP, device_map=device, use_fp16=use_fp16)

    # モデルホルダーを初期化
    model_holder = TTSModelHolder(
        get_paths_config().assets_root,
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

    # 結果を保存
    results: list[dict[str, Any]] = []

    # ウォームアップ（初回実行）
    print("ウォームアップ実行中...")
    _ = measure_module_timings(model, "テスト", device, use_fp16)
    print()

    # 各テキストでベンチマークを実行
    for test_case in BENCHMARK_TEXTS:
        text = test_case["text"]
        description = test_case["description"]

        print(f"測定中: {description}")
        print(f"テキスト: {text[:40]}{'...' if len(text) > 40 else ''}")

        # 複数回実行して平均を取る
        timings_list: list[ModuleTimings] = []

        for run in range(num_runs):
            try:
                timings = measure_module_timings(model, text, device, use_fp16)
                timings_list.append(timings)

                print(
                    f"  Run {run + 1}: "
                    f"TextEnc: {timings.text_encoder_ms:.2f}ms, "
                    f"DurPred: {timings.duration_predictor_ms:.2f}ms, "
                    f"Flow: {timings.flow_ms:.2f}ms, "
                    f"Gen: {timings.generator_ms:.2f}ms, "
                    f"Total: {timings.total_ms:.2f}ms "
                    f"(frames: {timings.output_length})"
                )

            except Exception as ex:
                logger.exception(f"測定中にエラーが発生しました: {ex}")
                continue

        if not timings_list:
            print("  測定に失敗しました。")
            continue

        # 平均値を計算
        avg_text_encoder = np.mean([t.text_encoder_ms for t in timings_list])
        avg_duration_predictor = np.mean(
            [t.duration_predictor_ms for t in timings_list]
        )
        avg_flow = np.mean([t.flow_ms for t in timings_list])
        avg_generator = np.mean([t.generator_ms for t in timings_list])
        avg_total = np.mean([t.total_ms for t in timings_list])
        avg_output_length = np.mean([t.output_length for t in timings_list])

        result = {
            "description": description,
            "text_length": len(text),
            "output_frames": avg_output_length,
            "text_encoder_ms": avg_text_encoder,
            "duration_predictor_ms": avg_duration_predictor,
            "flow_ms": avg_flow,
            "generator_ms": avg_generator,
            "total_ms": avg_total,
        }
        results.append(result)

        print(f"  平均: Total: {avg_total:.2f}ms, frames: {avg_output_length:.0f}")
        print("-" * 60)

    # モデルをアンロード
    model.unload()

    # 総合結果を表示
    print()
    print("=" * 120)
    print("総合結果")
    print("=" * 120)
    print(
        f"{'説明':<25} {'文字数':>6} {'出力長':>8} {'TextEnc':>10} {'DurPred':>10} "
        f"{'Flow':>10} {'Generator':>12} {'Total':>10} {'Gen比率':>8}"
    )
    print("-" * 120)

    for result in results:
        gen_ratio = result["generator_ms"] / result["total_ms"] * 100
        print(
            f"{result['description']:<25} "
            f"{result['text_length']:>6} "
            f"{result['output_frames']:>8.0f} "
            f"{result['text_encoder_ms']:>10.2f} "
            f"{result['duration_predictor_ms']:>10.2f} "
            f"{result['flow_ms']:>10.2f} "
            f"{result['generator_ms']:>12.2f} "
            f"{result['total_ms']:>10.2f} "
            f"{gen_ratio:>7.1f}%"
        )

    print("=" * 120)

    # 分析サマリー
    print()
    print("分析:")
    if results:
        avg_gen_ratio = np.mean(
            [r["generator_ms"] / r["total_ms"] * 100 for r in results]
        )
        print(f"- Generator の平均時間比率: {avg_gen_ratio:.1f}%")

        # 出力長との相関を分析
        output_lengths = [r["output_frames"] for r in results]
        gen_times = [r["generator_ms"] for r in results]
        flow_times = [r["flow_ms"] for r in results]

        # 出力長が最小と最大の場合のスケーリングを計算
        if len(results) >= 2:
            min_idx = np.argmin(output_lengths)
            max_idx = np.argmax(output_lengths)
            length_ratio = output_lengths[max_idx] / output_lengths[min_idx]
            gen_time_ratio = gen_times[max_idx] / gen_times[min_idx]
            flow_time_ratio = flow_times[max_idx] / flow_times[min_idx]

            print(f"- 出力長の増加比率: {length_ratio:.2f}x")
            print(f"- Generator 時間の増加比率: {gen_time_ratio:.2f}x")
            print(f"- Flow 時間の増加比率: {flow_time_ratio:.2f}x")

    print("=" * 120)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Style-Bert-VITS2 モジュール別実行時間ベンチマーク"
    )
    parser.add_argument(
        "--device",
        default="cuda",
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
