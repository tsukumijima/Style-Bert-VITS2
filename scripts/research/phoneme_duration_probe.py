"""
Usage: uv run python -m scripts.research.phoneme_duration_probe [--model koharune-ami] [--text "今日はいい天気ですね。"] [--language JP]
    [--speaker-id 0] [--style Neutral] [--style-weight 1.0] [--length-scale 1.0] [--sdp-ratio 0.0] [--noise-scale-w 0.8]
    [--device cpu] [--runs 1] [--seed 1234] [--max-tokens 80] [--dump-json output.json]

Style-Bert-VITS2 モデルのトークンレベルの持続時間（ブランクトークンを含む）を推定するスクリプト。

このスクリプトはアライメント生成ステップまでモデルを実行し、各入力トークンに割り当てられた
メルフレーム数を報告する。`add_blank`（挿入）の動作を実証的に調査し、以下のトークンに
どれだけの持続時間が割り当てられているかを確認することを目的とする。
--sdp-ratio を 0.0 から変更すると、音素長のランダム変化幅が大きくなる。

- 挿入されたブランクトークン（`add_blank` 後の偶数インデックス）
- G2P からの元の PAD トークン（先頭/末尾に追加された "_"；奇数インデックス）
- 通常の音素トークンと句読点トークン
"""

from __future__ import annotations

import argparse
import json
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from style_bert_vits2.constants import BASE_DIR, Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models.infer import prepare_inference_data
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.nlp.symbols import PUNCTUATIONS, SYMBOLS
from style_bert_vits2.tts_model import TTSModelHolder
from style_bert_vits2.utils.paths import get_paths_config


def _percentiles(
    values: NDArray[np.float32], percentiles: list[float]
) -> dict[str, float]:
    """
    空でない配列のパーセンタイルを計算する。

    Args:
        values (np.ndarray): 値の 1 次元配列。
        percentiles (list[float]): [0, 100] の範囲のパーセンタイル。

    Returns:
        dict[str, float]: {"p50": 1.0, ...} のようなマッピング。
    """

    if values.size == 0:
        return {f"p{int(p)}": float("nan") for p in percentiles}
    computed = np.percentile(values, percentiles)
    return {f"p{int(p)}": float(v) for p, v in zip(percentiles, computed, strict=False)}


def _summarize(values: NDArray[np.float32]) -> dict[str, float]:
    """
    数値の 1 次元配列を要約する。

    Args:
        values (np.ndarray): 値の 1 次元配列。

    Returns:
        dict[str, float]: 要約統計情報。
    """

    if values.size == 0:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            **_percentiles(values, [10, 50, 90, 99]),
        }
    return {
        "count": float(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        **_percentiles(values, [10, 50, 90, 99]),
    }


def _decode_symbols(token_ids: list[int]) -> list[str]:
    """
    モデルの `SYMBOLS` テーブルを使用してトークン ID をシンボル文字列にデコードする。

    Args:
        token_ids (list[int]): トークン ID。

    Returns:
        list[str]: デコードされたシンボル文字列。
    """

    symbols: list[str] = []
    for token_id in token_ids:
        if 0 <= token_id < len(SYMBOLS):
            symbols.append(SYMBOLS[token_id])
        else:
            symbols.append(f"<UNK_ID:{token_id}>")
    return symbols


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VITS2 モデルのトークンレベルの持続時間（ブランクを含む）を調査する。",
    )
    parser.add_argument(
        "--model",
        default="koharune-ami",
        help="使用するモデル名 (default: koharune-ami)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="今日はいい天気ですね。",
        help="持続時間を調査するために合成するテキスト。",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="JP",
        choices=["JP", "ZH", "EN"],
        help="言語コード。",
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="話者 ID (sid)。",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="Neutral",
        help="スタイル名。",
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        default=1.0,
        help="スタイルの強さ。",
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        default=1.0,
        help="長さスケール（グローバルな話速）。",
    )
    parser.add_argument(
        "--sdp-ratio",
        type=float,
        default=0.0,
        help="SDP 比率。0.0 は DP のみを使用し、>0 は持続時間に確率性を導入する可能性がある。",
    )
    parser.add_argument(
        "--noise-scale-w",
        type=float,
        default=0.8,
        help="SDP のノイズスケール（持続時間の確率性に影響）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch 推論に使用するデバイス（例: cpu, cuda）。",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="実行回数（sdp_ratio > 0 の場合に有用）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="ベースランダムシード。実行 i は seed + i を使用する。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="プレビューテーブルとして出力する最大トークン数。",
    )
    parser.add_argument(
        "--dump-json",
        type=str,
        default=None,
        help="結果を JSON ファイルに保存する（ファイル名のみ指定、出力先は scripts/research/phoneme_duration_probe/ 固定）。",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    # モデルホルダーを初期化
    model_holder = TTSModelHolder(
        get_paths_config().assets_root,
        device=args.device,
        onnx_providers=[],
        ignore_onnx=True,
        use_fp16=False,
    )

    if len(model_holder.models_info) == 0:
        logger.error("音声合成モデルが見つかりませんでした。")
        return

    # 指定されたモデルを検索
    model_info = None
    for info in model_holder.models_info:
        if info.name == args.model:
            model_info = info
            break

    if model_info is None:
        logger.error(f'モデル "{args.model}" が見つかりませんでした。')
        logger.info("利用可能なモデル:")
        for info in model_holder.models_info:
            logger.info(f"  - {info.name}")
        return

    # Safetensors 形式のモデルファイルを検索
    model_files = [
        f
        for f in model_info.files
        if f.endswith(".safetensors") and not f.startswith(".")
    ]
    if len(model_files) == 0:
        logger.error(
            f'モデル "{args.model}" の .safetensors ファイルが見つかりませんでした。'
        )
        return

    model_file = model_files[0]
    logger.info(f"使用するモデルファイル: {model_file}")

    # モデルを取得
    tts_model = model_holder.get_model(args.model, model_file)
    tts_model.load()
    if tts_model.net_g is None:
        raise RuntimeError("Failed to load net_g")

    language = Languages(args.language)
    hps = tts_model.hyper_parameters
    hop_length = int(hps.data.hop_length)
    sampling_rate = int(hps.data.sampling_rate)
    seconds_per_frame = hop_length / sampling_rate
    is_add_blank_enabled = bool(hps.data.add_blank)

    logger.info("Loading model for duration probing")
    logger.info(f"model_path: {tts_model.model_path}")
    logger.info(f"config_path: {tts_model.config_path}")
    logger.info(f"style_vec_path: {tts_model.style_vec_path}")
    logger.info(f"device: {args.device}")

    if args.style not in tts_model.style2id:
        logger.warning(
            f"Style '{args.style}' not found. Falling back to the first available style."
        )
        style_name = next(iter(tts_model.style2id.keys()))
    else:
        style_name = args.style
    style_id = tts_model.style2id[style_name]
    style_vec = tts_model.get_style_vector(style_id, args.style_weight)

    per_run_records: list[dict[str, Any]] = []

    for run_index in range(int(args.runs)):
        seed = int(args.seed) + run_index
        torch.manual_seed(seed)
        np.random.seed(seed)

        with torch.inference_mode():
            (
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                zh_bert,
                ja_bert,
                en_bert,
                style_vec_tensor,
                _duration_symbol_type_ids,
            ) = prepare_inference_data(
                args.text,
                style_vec=style_vec,
                sid=int(args.speaker_id),
                language=language,
                hps=hps,
                device=args.device,
                given_phone=None,
                given_tone=None,
                enable_tensor_padding=False,
            )

            net_g = tts_model.net_g
            assert net_g is not None
            is_jp_extra = hps.is_jp_extra_like_model()
            if is_jp_extra:
                _z, _y_mask, _g, attn, *_ = cast(
                    SynthesizerTrnJPExtra, net_g
                ).infer_input_feature(
                    x=x_tst,
                    x_lengths=x_tst_lengths,
                    sid=sid_tensor,
                    tone=tones,
                    language=lang_ids,
                    bert=ja_bert,
                    style_vec=style_vec_tensor,
                    noise_scale=0.667,
                    length_scale=float(args.length_scale),
                    noise_scale_w=float(args.noise_scale_w),
                    sdp_ratio=float(args.sdp_ratio),
                    y=None,
                    use_fp16=False,
                )
            else:
                _z, _y_mask, _g, attn, *_ = cast(
                    SynthesizerTrn, net_g
                ).infer_input_feature(
                    x=x_tst,
                    x_lengths=x_tst_lengths,
                    sid=sid_tensor,
                    tone=tones,
                    language=lang_ids,
                    bert=zh_bert,
                    ja_bert=ja_bert,
                    en_bert=en_bert,
                    style_vec=style_vec_tensor,
                    noise_scale=0.667,
                    length_scale=float(args.length_scale),
                    noise_scale_w=float(args.noise_scale_w),
                    sdp_ratio=float(args.sdp_ratio),
                    y=None,
                    use_fp16=False,
                )

            token_ids = x_tst[0].detach().cpu().to(torch.int64).tolist()
            token_symbols = _decode_symbols(token_ids)
            token_count = len(token_ids)

            # attn: [B, 1, T_y, T_x] -> トークンごとの持続時間: [T_x]
            durations_frames = (
                attn[0, 0].sum(dim=0).detach().cpu().to(torch.float32).numpy()
            )
            durations_seconds = durations_frames * seconds_per_frame

            is_pad_id: NDArray[np.bool_] = np.array(
                [token_id == 0 for token_id in token_ids], dtype=bool
            )
            is_even_index: NDArray[np.bool_] = np.array(
                [(token_index % 2) == 0 for token_index in range(token_count)],
                dtype=bool,
            )
            is_inserted_blank: NDArray[np.bool_] = np.logical_and(
                is_add_blank_enabled,
                is_pad_id & is_even_index,
            )
            is_original_pad: NDArray[np.bool_] = is_pad_id & ~is_inserted_blank
            is_punctuation: NDArray[np.bool_] = np.array(
                [symbol in PUNCTUATIONS for symbol in token_symbols],
                dtype=bool,
            )

            total_frames = float(durations_frames.sum())
            total_seconds = float(durations_seconds.sum())

            record: dict[str, Any] = {
                "run_index": run_index,
                "seed": seed,
                "text": args.text,
                "model": str(tts_model.model_path),
                "config": str(tts_model.config_path),
                "style": style_name,
                "style_weight": float(args.style_weight),
                "length_scale": float(args.length_scale),
                "sdp_ratio": float(args.sdp_ratio),
                "noise_scale_w": float(args.noise_scale_w),
                "sampling_rate": sampling_rate,
                "hop_length": hop_length,
                "seconds_per_frame": seconds_per_frame,
                "add_blank": is_add_blank_enabled,
                "token_count": token_count,
                "total_frames": total_frames,
                "total_seconds": total_seconds,
                "stats": {
                    "inserted_blank_frames": _summarize(
                        durations_frames[is_inserted_blank]
                    ),
                    "original_pad_frames": _summarize(
                        durations_frames[is_original_pad]
                    ),
                    "punctuation_frames": _summarize(durations_frames[is_punctuation]),
                    "non_pad_frames": _summarize(durations_frames[~is_pad_id]),
                    "all_frames": _summarize(durations_frames),
                    "inserted_blank_share": float(
                        durations_frames[is_inserted_blank].sum() / total_frames
                    )
                    if total_frames > 0
                    else float("nan"),
                    "original_pad_share": float(
                        durations_frames[is_original_pad].sum() / total_frames
                    )
                    if total_frames > 0
                    else float("nan"),
                },
                "tokens": [
                    {
                        "index": int(token_index),
                        "symbol": token_symbols[token_index],
                        "token_id": int(token_ids[token_index]),
                        "duration_frames": float(durations_frames[token_index]),
                        "duration_seconds": float(durations_seconds[token_index]),
                        "is_pad_id": bool(is_pad_id[token_index]),
                        "is_inserted_blank": bool(is_inserted_blank[token_index]),
                        "is_original_pad": bool(is_original_pad[token_index]),
                        "is_punctuation": bool(is_punctuation[token_index]),
                    }
                    for token_index in range(token_count)
                ],
            }
            per_run_records.append(record)

            logger.info(
                "Run completed. "
                f"run_index: {run_index}, tokens: {token_count}, "
                f"total_frames: {total_frames:.1f}, total_seconds: {total_seconds:.3f}"
            )

            preview_len = min(int(args.max_tokens), token_count)
            logger.info(f"Preview table (first {preview_len} tokens):")
            for token_index in range(preview_len):
                symbol = token_symbols[token_index]
                duration_f = float(durations_frames[token_index])
                duration_s = float(durations_seconds[token_index])
                flags: list[str] = []
                if is_inserted_blank[token_index]:
                    flags.append("inserted_blank")
                if is_original_pad[token_index]:
                    flags.append("original_pad")
                if is_punctuation[token_index]:
                    flags.append("punctuation")
                flag_str = ",".join(flags) if len(flags) > 0 else "-"
                logger.info(
                    f"idx: {token_index:3d}, id: {token_ids[token_index]:3d}, "
                    f"sym: {symbol:<6s}, dur_frames: {duration_f:6.1f}, "
                    f"dur_sec: {duration_s:7.4f}, flags: {flag_str}"
                )

            stats: dict[str, Any] = record["stats"]
            logger.info(
                f"Summary (frames): inserted_blank: {stats['inserted_blank_frames']}"
            )
            logger.info(
                f"Summary (frames): original_pad: {stats['original_pad_frames']}"
            )
            logger.info(f"Summary (frames): non_pad: {stats['non_pad_frames']}")
            inserted_blank_share = stats.get("inserted_blank_share", float("nan"))
            original_pad_share = stats.get("original_pad_share", float("nan"))
            logger.info(
                "Share: "
                f"inserted_blank: {float(inserted_blank_share):.4f}, "
                f"original_pad: {float(original_pad_share):.4f}"
            )

    # JSON ダンプ処理
    if args.dump_json is not None:
        # ファイル名に .json 拡張子がない場合は追加
        json_filename = args.dump_json
        if not json_filename.endswith(".json"):
            json_filename = f"{json_filename}.json"

        dump_path = (
            BASE_DIR / "scripts" / "research" / "phoneme_duration_probe" / json_filename
        )
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "records": per_run_records,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"JSON dump saved to {dump_path}")


if __name__ == "__main__":
    main()
