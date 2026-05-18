"""
Usage: uv run python -m scripts.benchmark.torch_compile_benchmark [--device cuda] [--model koharune-ami] [--runs 10]

Generator モジュールに torch.compile を適用して、パフォーマンス改善効果を測定するベンチマークスクリプト。

目的:
- torch.compile の各モード (default, reduce-overhead, max-autotune) の効果を比較する
- コンパイルオーバーヘッドと推論時間の改善率を測定する
"""

import argparse
import gc
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
class BenchmarkResult:
    """ベンチマーク結果を保持するデータクラス"""

    mode: str  # "baseline", "default", "reduce-overhead", "max-autotune"
    compile_time_ms: float  # コンパイル時間（baseline の場合は 0）
    warmup_time_ms: float  # ウォームアップ時間
    avg_inference_ms: float  # 平均推論時間
    std_inference_ms: float  # 標準偏差
    min_inference_ms: float  # 最小推論時間
    max_inference_ms: float  # 最大推論時間


# テスト用テキスト
BENCHMARK_TEXTS = [
    {"text": "こんにちは。", "description": "Short"},
    {
        "text": "音声合成は、機械学習を活用してテキストから人の声を生成する技術です。",
        "description": "Medium",
    },
    {
        "text": "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては、人一倍に敏感であった。",
        "description": "Long",
    },
]


def measure_generator_time(
    net_g: Any,
    z: torch.Tensor,
    y_mask: torch.Tensor,
    g: torch.Tensor,
    use_fp16: bool,
    device: str,
) -> float:
    """
    Generator モジュールの実行時間を計測する。

    Args:
        net_g: モデルのネットワーク
        z: 入力 latent テンソル
        y_mask: マスクテンソル
        g: グローバルコンディショニングテンソル
        use_fp16: FP16 を使用するかどうか
        device: 使用するデバイス

    Returns:
        float: 実行時間（ミリ秒）
    """

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
            _ = net_g.dec(z_input, g=g.half())
    else:
        _ = net_g.dec(z * y_mask, g=g)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000  # ms


def prepare_inputs(
    model: TTSModel,
    text: str,
    device: str,
    use_fp16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generator に渡す入力テンソルを事前に準備する。

    Args:
        model: TTSModel インスタンス
        text: 合成するテキスト
        device: 使用するデバイス
        use_fp16: FP16 を使用するかどうか

    Returns:
        tuple: (z, y_mask, g)
    """

    # このベンチマークは JP-Extra モデル専用
    net_g = model.net_g
    assert net_g is not None
    assert isinstance(net_g, SynthesizerTrnJPExtra), (
        "This benchmark only supports JP-Extra models"
    )
    hps = model.hyper_parameters

    # テキストを処理して入力テンソルを準備
    # JP-Extra モデルでは ja_bert のみ使用する (bert_ori, en_bert は使用しない)
    _bert_ori, ja_bert, _en_bert, phones, tones, lang_ids, _duration_symbol_type_ids = (
        get_text(
            text,
            Languages.JP,
            hps,
            device,
        )
    )

    # テンソルを GPU に転送
    x = phones.to(device).unsqueeze(0)
    x_lengths = torch.tensor([phones.size(0)], dtype=torch.long, device=device)
    tone = tones.to(device).unsqueeze(0)
    language = lang_ids.to(device).unsqueeze(0)
    bert = ja_bert.to(device).unsqueeze(0)
    sid = torch.tensor([0], dtype=torch.long, device=device)
    style_vec = torch.from_numpy(model.get_style_vector(0, 1.0)).to(device).unsqueeze(0)

    with torch.inference_mode():
        # Speaker Embedding
        if net_g.n_speakers > 0:
            g = net_g.emb_g(sid).unsqueeze(-1)
        else:
            raise RuntimeError("Reference encoder not supported in this benchmark")

        # BERT 特徴量の型変換
        bert_input = bert.float() if use_fp16 else bert

        # TextEncoder
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

        # Duration Predictor
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

        # Flow
        z = net_g.flow(z_p, y_mask, g=g, reverse=True)

    return z, y_mask, g


def run_torch_compile_benchmark(
    device: str = "cuda",
    model_name: str = "koharune-ami",
    num_runs: int = 10,
    num_warmup: int = 3,
    use_fp16: bool = True,
) -> None:
    """
    torch.compile のベンチマークを実行する。

    Args:
        device: 使用するデバイス
        model_name: 使用するモデル名
        num_runs: 各テストケースの実行回数
        num_warmup: ウォームアップ実行回数
        use_fp16: FP16 を使用するかどうか
    """

    set_random_seeds()

    print("=" * 100)
    print("Style-Bert-VITS2 torch.compile ベンチマーク")
    print("=" * 100)
    print(f"デバイス: {device}")
    print(f"PyTorch バージョン: {torch.__version__}")
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"  # type: ignore[attr-defined]
    print(f"CUDA バージョン: {cuda_version}")
    print(f"モデル: {model_name}")
    print(f"測定回数: {num_runs}")
    print(f"ウォームアップ回数: {num_warmup}")
    print(f"FP16: {use_fp16}")
    print("=" * 100)

    # torch.compile が利用可能かチェック
    if not hasattr(torch, "compile"):
        print("エラー: torch.compile は PyTorch 2.0 以降でのみ利用可能です。")
        return

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

    # GPU の CUDA Capability をチェック
    cuda_capability = None
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(device)
        cuda_capability = props.major + props.minor / 10
        print(f"CUDA Capability: {cuda_capability}")
        print()

    # テストするモードを決定
    # Triton バックエンドは CUDA Capability 7.0 以上が必要
    if cuda_capability is not None and cuda_capability >= 7.0:
        # Triton バックエンドが使用可能
        compile_modes = [
            ("baseline", None, None),  # (name, mode, backend)
            ("default", "default", None),
            ("reduce-overhead", "reduce-overhead", None),
            # ("max-autotune", "max-autotune", None),  # 非常に時間がかかるため、デフォルトでは無効
        ]
    else:
        # Triton バックエンドが使用不可能な場合、eager 系バックエンドを使用
        print(
            "注意: CUDA Capability < 7.0 のため、Triton バックエンドは使用できません。"
        )
        print("代替として eager/aot_eager/cudagraphs バックエンドを使用します。")
        print()
        compile_modes = [
            ("baseline", None, None),  # (name, mode, backend)
            ("eager", None, "eager"),
            ("aot_eager", None, "aot_eager"),
            ("cudagraphs", None, "cudagraphs"),  # CUDA Graph を使用
        ]

    # 各テキストについてベンチマークを実行
    for test_case in BENCHMARK_TEXTS:
        text = test_case["text"]
        description = test_case["description"]

        print()
        print("=" * 100)
        print(f"テスト: {description}")
        print(f"テキスト: {text[:50]}{'...' if len(text) > 50 else ''}")
        print("=" * 100)

        results: list[BenchmarkResult] = []

        for mode_name, compile_mode, compile_backend in compile_modes:
            print()
            print(f"--- {mode_name} モード ---")

            # 毎回モデルを新しくロードする（前のコンパイル結果をリセット）
            model = model_holder.get_model(model_name, model_file)
            model.load()
            net_g = model.net_g
            assert net_g is not None

            # 入力を準備
            z, y_mask, g = prepare_inputs(model, text, device, use_fp16)
            output_frames = z.size(2)
            print(f"出力フレーム数: {output_frames}")

            # torch.compile を適用
            compile_time_ms = 0.0
            is_compile_enabled = compile_mode is not None or compile_backend is not None
            if is_compile_enabled:
                backend_info = (
                    f"backend: {compile_backend}"
                    if compile_backend
                    else f"mode: {compile_mode}"
                )
                print(f"torch.compile を適用中 ({backend_info})...")

                if device.startswith("cuda"):
                    torch.cuda.synchronize()

                # Generator (dec) を compile
                # dynamic=False: 固定形状を想定（動的形状は別途検討）
                # fullgraph=False: サブグラフ分割を許可（複雑なモデル向け）
                try:
                    compile_kwargs: dict[str, Any] = {
                        "dynamic": False,
                        "fullgraph": False,
                    }
                    if compile_mode is not None:
                        compile_kwargs["mode"] = compile_mode
                    if compile_backend is not None:
                        compile_kwargs["backend"] = compile_backend

                    # torch.compile は元のモジュールをラップした OptimizedModule を返す
                    # pyright は型が合わないと警告するが、実行時には問題ない
                    net_g.dec = torch.compile(  # type: ignore[assignment]
                        net_g.dec,
                        **compile_kwargs,
                    )
                except Exception as ex:
                    print(f"torch.compile に失敗しました: {ex}")
                    model.unload()
                    del model
                    gc.collect()
                    continue

                # コンパイルは初回実行時に行われるため、ここでは時間計測しない
                # 代わりにウォームアップ時にコンパイル時間を含める

            # ウォームアップ（コンパイル + JIT 最適化を含む）
            print(f"ウォームアップ実行中 ({num_warmup} 回)...")
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            warmup_start = time.perf_counter()

            with torch.inference_mode():
                for _ in range(num_warmup):
                    _ = measure_generator_time(net_g, z, y_mask, g, use_fp16, device)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            warmup_time_ms = (time.perf_counter() - warmup_start) * 1000
            if is_compile_enabled is True:
                # torch.compile は初回実行時にコンパイルされるため、
                # warmup 時間を compile_time として記録する。
                compile_time_ms = warmup_time_ms

            # 本番測定
            print(f"本番測定実行中 ({num_runs} 回)...")
            inference_times: list[float] = []

            with torch.inference_mode():
                for _run in range(num_runs):
                    time_ms = measure_generator_time(
                        net_g, z, y_mask, g, use_fp16, device
                    )
                    inference_times.append(time_ms)

            # 統計計算
            avg_time = float(np.mean(inference_times))
            std_time = float(np.std(inference_times))
            min_time = float(np.min(inference_times))
            max_time = float(np.max(inference_times))

            result = BenchmarkResult(
                mode=mode_name,
                compile_time_ms=compile_time_ms,
                warmup_time_ms=warmup_time_ms,
                avg_inference_ms=avg_time,
                std_inference_ms=std_time,
                min_inference_ms=min_time,
                max_inference_ms=max_time,
            )
            results.append(result)

            print(f"  ウォームアップ時間: {warmup_time_ms:.2f}ms")
            print(
                f"  平均推論時間: {avg_time:.2f}ms (±{std_time:.2f}ms, "
                f"min: {min_time:.2f}ms, max: {max_time:.2f}ms)"
            )

            # モデルをアンロード
            model.unload()
            del model
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        # 結果の比較表を出力
        print()
        print("-" * 100)
        print(f"結果比較 ({description})")
        print("-" * 100)
        print(
            f"{'モード':<20} {'ウォームアップ':>14} {'平均推論':>12} {'標準偏差':>10} "
            f"{'最小':>10} {'最大':>10} {'改善率':>10}"
        )
        print("-" * 100)

        baseline_time = None
        for result in results:
            if result.mode == "baseline":
                baseline_time = result.avg_inference_ms

            improvement = "-"
            if baseline_time is not None and result.mode != "baseline":
                speedup = baseline_time / result.avg_inference_ms
                improvement = f"{speedup:.2f}x"

            print(
                f"{result.mode:<20} "
                f"{result.warmup_time_ms:>12.2f}ms "
                f"{result.avg_inference_ms:>10.2f}ms "
                f"{result.std_inference_ms:>8.2f}ms "
                f"{result.min_inference_ms:>8.2f}ms "
                f"{result.max_inference_ms:>8.2f}ms "
                f"{improvement:>10}"
            )

    print()
    print("=" * 100)
    print("ベンチマーク完了")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Style-Bert-VITS2 torch.compile ベンチマーク"
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
        default=10,
        help="各テストケースの実行回数 (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="ウォームアップ実行回数 (default: 3)",
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
        run_torch_compile_benchmark(
            device=args.device,
            model_name=args.model,
            num_runs=args.runs,
            num_warmup=args.warmup,
            use_fp16=args.use_fp16,
        )
    except KeyboardInterrupt:
        print("\nベンチマークが中断されました。")
    except Exception as ex:
        logger.exception(f"ベンチマーク実行中にエラーが発生しました: {ex}")


if __name__ == "__main__":
    main()
