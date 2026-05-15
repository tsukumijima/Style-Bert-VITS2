"""
Usage: uv run python -m scripts.research.vowel_stretch_experiment --text '今日はいい天気ですね。'

指定テキストに対して、母音トークンの推定 duration のみを一定倍率で引き延ばした音声を生成し、
通常生成時と比較して WAV 長がどれだけ変化するかを調査するスクリプト。

このスクリプトは、Style-Bert-VITS2 の内部表現である「メルフレーム単位の duration」を直接操作する。
`add_blank` により挿入されるブランクトークンは音素遷移（境界）の表現として学習されている可能性が高く、
本実験では音質劣化を避けるため、ブランクトークンの duration は変更しない。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons
from style_bert_vits2.models.infer import prepare_inference_data
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.nlp.japanese.mora_list import VOWELS
from style_bert_vits2.nlp.symbols import SYMBOLS
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder
from style_bert_vits2.utils.paths import get_paths_config


def _decode_symbols(token_ids: list[int]) -> list[str]:
    """
    モデルの `SYMBOLS` テーブルを使用してトークン ID をシンボル文字列にデコードする。

    Args:
        token_ids (list[int]): トークン ID。

    Returns:
        list[str]: デコードされたシンボル文字列。
    """

    decoded: list[str] = []
    for token_id in token_ids:
        if 0 <= token_id < len(SYMBOLS):
            decoded.append(SYMBOLS[token_id])
        else:
            decoded.append(f"<UNK_ID:{token_id}>")
    return decoded


def _is_vowel_symbol(symbol: str) -> bool:
    """
    シンボル文字列が母音（もしくは「ン」）に該当するかを判定する。

    Args:
        symbol (str): `SYMBOLS` 上のシンボル。

    Returns:
        bool: 母音相当なら True。
    """

    if symbol in VOWELS:
        return True

    # JP-Extra のシンボル表現では長母音が `a:` のように表現されることがある
    if symbol.endswith(":") and symbol[:-1] in {"a", "i", "u", "e", "o"}:
        return True

    return False


def _safe_filename(text: str, max_len: int = 64) -> str:
    """
    テキストをファイル名に埋め込めるように安全化する。

    Args:
        text (str): 元のテキスト。
        max_len (int): 最大長。

    Returns:
        str: 安全化済み文字列。
    """

    sanitized = re.sub(r"\s+", "_", text.strip())
    sanitized = re.sub(r"[^0-9A-Za-zぁ-んァ-ン一-龯_\-\(\)\[\]\.]+", "_", sanitized)
    if len(sanitized) == 0:
        return "empty"
    return sanitized[:max_len]


def _save_wav_int16(
    wav_path: Path,
    sampling_rate: int,
    wave_int16: NDArray[np.int16],
) -> None:
    """
    int16 PCM の WAV を保存する。

    Args:
        wav_path (Path): 出力先。
        sampling_rate (int): サンプリングレート。
        wave_int16 (NDArray[np.int16]): 1 次元の int16 PCM。
    """

    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # scipy があればそれを使い、なければ標準ライブラリで書き出す
    try:
        from scipy.io import wavfile

        wavfile.write(str(wav_path), sampling_rate, wave_int16)
        return
    except Exception as ex:
        logger.warning(
            "Failed to write WAV via scipy. Falling back to wave module.", exc_info=ex
        )

    import wave

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sampling_rate)
        wf.writeframes(wave_int16.tobytes())


def _infer_jp_extra_with_custom_durations(
    net_g: SynthesizerTrnJPExtra,
    x_tst: torch.Tensor,
    x_tst_lengths: torch.Tensor,
    sid_tensor: torch.Tensor,
    tones: torch.Tensor,
    lang_ids: torch.Tensor,
    ja_bert: torch.Tensor,
    style_vec_tensor: torch.Tensor,
    noise_scale: float,
    length_scale: float,
    noise_scale_w: float,
    sdp_ratio: float,
    vowel_stretch_ratio: float,
    token_symbols: list[str],
    is_add_blank_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    JP-Extra の推論を duration 操作付きで実行する。

    Args:
        net_g (SynthesizerTrnJPExtra): JP-Extra モデル。
        x_tst (torch.Tensor): 入力トークン ID（shape: [B, T_x]）。
        x_tst_lengths (torch.Tensor): 入力長（shape: [B]）。
        sid_tensor (torch.Tensor): 話者 ID（shape: [B]）。
        tones (torch.Tensor): アクセント（shape: [B, T_x]）。
        lang_ids (torch.Tensor): 言語 ID（shape: [B, T_x]）。
        ja_bert (torch.Tensor): 日本語 BERT 特徴（shape: [B, 1024, T_x]）。
        style_vec_tensor (torch.Tensor): スタイルベクトル（shape: [B, style_dim]）。
        noise_scale (float): 生成ノイズ。
        length_scale (float): 話速スケール。
        noise_scale_w (float): duration ノイズ。
        sdp_ratio (float): SDP 比率。
        vowel_stretch_ratio (float): 母音トークンの duration を伸ばす倍率（例: 3.0）。
        token_symbols (list[str]): トークンをデコードしたシンボル列（長さ T_x）。
        is_add_blank_enabled (bool): `add_blank` が有効かどうか。

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - audio tensor（shape: [B, 1, T_samples]）
            - durations_frames（shape: [T_x]、float32）
    """

    # Speaker conditioning
    if net_g.n_speakers > 0:
        g = net_g.emb_g(sid_tensor).unsqueeze(-1)
    else:
        raise ValueError(
            "This experiment expects a multi-speaker model (n_speakers > 0)."
        )

    # Encoder
    x, m_p, logs_p, x_mask = net_g.enc_p(
        x_tst,
        x_tst_lengths,
        tones,
        lang_ids,
        ja_bert,
        style_vec_tensor,
        g=g,
        use_fp16=False,
    )

    # Duration predictor (DP/SDP mix)
    logw = net_g.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
        sdp_ratio
    ) + net_g.dp(x, x_mask, g=g) * (1 - sdp_ratio)

    w = torch.exp(logw) * x_mask * float(length_scale)
    w_ceil = torch.ceil(w)

    # Apply vowel-only stretching while keeping blank/pad tokens intact
    token_ids = x_tst[0].detach().cpu().to(torch.int64).tolist()
    is_pad_id = np.array([token_id == 0 for token_id in token_ids], dtype=bool)
    if len(token_symbols) != len(token_ids):
        raise ValueError("token_symbols length mismatch")

    should_stretch = np.array(
        [
            (not is_pad_id[i]) and _is_vowel_symbol(token_symbols[i])
            for i in range(len(token_ids))
        ],
        dtype=bool,
    )

    if is_add_blank_enabled is True:
        # inserted blanks are at even indices after intersperse, and are PAD id (=0)
        # (this condition exists mainly for readability / defensive clarity)
        inserted_blank = (
            np.array([(i % 2) == 0 for i in range(len(token_ids))], dtype=bool)
            & is_pad_id
        )
        should_stretch = should_stretch & ~inserted_blank

    w_ceil_stretched = w_ceil.clone()
    if vowel_stretch_ratio != 1.0:
        indices = np.where(should_stretch)[0].tolist()
        if len(indices) > 0:
            w_ceil_stretched[:, :, indices] = torch.ceil(
                w_ceil_stretched[:, :, indices] * float(vowel_stretch_ratio)
            )

    # Alignment with stretched durations
    y_lengths = torch.clamp_min(torch.sum(w_ceil_stretched, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil_stretched, attn_mask)

    m_p_exp = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p_exp = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

    z_p = m_p_exp + torch.randn_like(m_p_exp) * torch.exp(logs_p_exp) * float(
        noise_scale
    )
    z = net_g.flow(z_p, y_mask, g=g, reverse=True)
    audio = net_g.dec(z * y_mask, g=g)

    durations_frames = attn[0, 0].sum(dim=0).detach().cpu().to(torch.float32)
    return audio, durations_frames


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate WAVs with vowel-only duration stretching and compare lengths.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="今日はいい天気ですね。",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="JP",
        choices=["JP", "ZH", "EN"],
        help="Language code.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="koharune-ami",
        help="Comma-separated model names under assets_root.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="Neutral",
        help="Style name.",
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        default=1.0,
        help="Style weight.",
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Speaker ID.",
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        default=1.0,
        help="Global length scale.",
    )
    parser.add_argument(
        "--sdp-ratio",
        type=float,
        default=0.0,
        help="SDP ratio.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.667,
        help="Noise scale for z sampling.",
    )
    parser.add_argument(
        "--noise-scale-w",
        type=float,
        default=0.8,
        help="Noise scale for SDP.",
    )
    parser.add_argument(
        "--vowel-stretch",
        type=float,
        default=3.0,
        help="Duration multiplier applied only to vowel tokens.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    model_names = [
        name.strip() for name in args.models.split(",") if name.strip() != ""
    ]
    if len(model_names) == 0:
        raise ValueError("No model names provided")

    assets_root = get_paths_config().assets_root
    model_holder = TTSModelHolder(
        assets_root,
        device=args.device,
        onnx_providers=[],
        ignore_onnx=True,
        use_fp16=False,
    )
    if len(model_holder.models_info) == 0:
        raise RuntimeError(f"No models found under {assets_root}")

    language = Languages(args.language)
    output_dir = Path(__file__).resolve().parent / "vowel_stretch_experiment"
    text_tag = _safe_filename(args.text)

    logger.info("Starting vowel stretch experiment")
    logger.info(f"text: {args.text}")
    logger.info(f"vowel_stretch: {float(args.vowel_stretch):.3f}")
    logger.info(
        f"length_scale: {float(args.length_scale):.3f}, sdp_ratio: {float(args.sdp_ratio):.3f}"
    )

    for model_name in model_names:
        model_info = next(
            (info for info in model_holder.models_info if info.name == model_name), None
        )
        if model_info is None:
            logger.warning(f"Model not found. name: {model_name}")
            continue

        safetensors_files = [
            f
            for f in model_info.files
            if f.endswith(".safetensors") and not f.startswith(".")
        ]
        if len(safetensors_files) == 0:
            logger.warning(f"No .safetensors found for model. name: {model_name}")
            continue

        model_file = safetensors_files[0]
        logger.info(f"Loading model. name: {model_name}, file: {model_file}")
        tts_model: TTSModel = model_holder.get_model(model_name, model_file)
        tts_model.load()
        if tts_model.net_g is None:
            logger.warning(f"net_g is not loaded. name: {model_name}")
            continue

        if not tts_model.hyper_parameters.is_jp_extra_like_model():
            logger.warning(
                f"Skipping non JP-Extra model. name: {model_name}, version: {tts_model.hyper_parameters.version}"
            )
            continue

        if args.style not in tts_model.style2id:
            logger.warning(
                f"Style '{args.style}' not found for model. Falling back to the first available style. name: {model_name}"
            )
            style_name = next(iter(tts_model.style2id.keys()))
        else:
            style_name = args.style

        style_id = tts_model.style2id[style_name]
        style_vec = tts_model.get_style_vector(style_id, args.style_weight)

        with torch.inference_mode():
            (
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                _zh_bert,
                ja_bert,
                _en_bert,
                style_vec_tensor,
                _token_types,
            ) = prepare_inference_data(
                args.text,
                style_vec=style_vec,
                sid=int(args.speaker_id),
                language=language,
                hps=tts_model.hyper_parameters,
                device=args.device,
                given_phone=None,
                given_tone=None,
                enable_tensor_padding=False,
            )

            token_ids = x_tst[0].detach().cpu().to(torch.int64).tolist()
            token_symbols = _decode_symbols(token_ids)

            net_g = cast(SynthesizerTrnJPExtra, tts_model.net_g)

            # Normal (no stretching)
            audio_normal, durations_normal = _infer_jp_extra_with_custom_durations(
                net_g=net_g,
                x_tst=x_tst,
                x_tst_lengths=x_tst_lengths,
                sid_tensor=sid_tensor,
                tones=tones,
                lang_ids=lang_ids,
                ja_bert=ja_bert,
                style_vec_tensor=style_vec_tensor,
                noise_scale=float(args.noise_scale),
                length_scale=float(args.length_scale),
                noise_scale_w=float(args.noise_scale_w),
                sdp_ratio=float(args.sdp_ratio),
                vowel_stretch_ratio=1.0,
                token_symbols=token_symbols,
                is_add_blank_enabled=bool(tts_model.hyper_parameters.data.add_blank),
            )

            # Vowel stretched
            audio_stretched, durations_stretched = (
                _infer_jp_extra_with_custom_durations(
                    net_g=net_g,
                    x_tst=x_tst,
                    x_tst_lengths=x_tst_lengths,
                    sid_tensor=sid_tensor,
                    tones=tones,
                    lang_ids=lang_ids,
                    ja_bert=ja_bert,
                    style_vec_tensor=style_vec_tensor,
                    noise_scale=float(args.noise_scale),
                    length_scale=float(args.length_scale),
                    noise_scale_w=float(args.noise_scale_w),
                    sdp_ratio=float(args.sdp_ratio),
                    vowel_stretch_ratio=float(args.vowel_stretch),
                    token_symbols=token_symbols,
                    is_add_blank_enabled=bool(
                        tts_model.hyper_parameters.data.add_blank
                    ),
                )
            )

        wave_normal_f32 = audio_normal[0, 0].detach().cpu().to(torch.float32).numpy()
        wave_stretched_f32 = (
            audio_stretched[0, 0].detach().cpu().to(torch.float32).numpy()
        )

        wave_normal_i16 = tts_model.convert_to_16_bit_wav(wave_normal_f32)
        wave_stretched_i16 = tts_model.convert_to_16_bit_wav(wave_stretched_f32)

        sr = int(tts_model.hyper_parameters.data.sampling_rate)
        normal_sec = float(len(wave_normal_i16) / sr)
        stretched_sec = float(len(wave_stretched_i16) / sr)
        ratio = stretched_sec / normal_sec if normal_sec > 0 else float("nan")

        model_tag = _safe_filename(model_name, max_len=48)
        wav_normal_path = output_dir / f"{model_tag}_{text_tag}_normal.wav"
        wav_stretched_path = (
            output_dir
            / f"{model_tag}_{text_tag}_vowelx{float(args.vowel_stretch):g}.wav"
        )

        _save_wav_int16(wav_normal_path, sr, cast(NDArray[np.int16], wave_normal_i16))
        _save_wav_int16(
            wav_stretched_path, sr, cast(NDArray[np.int16], wave_stretched_i16)
        )

        total_frames_normal = float(durations_normal.sum().item())
        total_frames_stretched = float(durations_stretched.sum().item())
        frame_ratio = (
            total_frames_stretched / total_frames_normal
            if total_frames_normal > 0
            else float("nan")
        )

        logger.info(
            "Result. "
            f"name: {model_name}, sr: {sr}, "
            f"normal_sec: {normal_sec:.3f}, stretched_sec: {stretched_sec:.3f}, ratio: {ratio:.3f}, "
            f"frames_normal: {total_frames_normal:.1f}, frames_stretched: {total_frames_stretched:.1f}, frame_ratio: {frame_ratio:.3f}"
        )
        logger.info(f"WAV saved. normal: {wav_normal_path}")
        logger.info(f"WAV saved. stretched: {wav_stretched_path}")


if __name__ == "__main__":
    main()
