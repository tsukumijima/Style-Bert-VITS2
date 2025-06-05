from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pyopenjtalk import OpenJTalk
from torch.overrides import TorchFunctionMode
from torch.utils import _device

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons, utils
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.nlp import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
    extract_bert_feature,
)
from style_bert_vits2.nlp.symbols import SYMBOLS


class EmptyInitOnDevice(TorchFunctionMode):
    def __init__(self, device=None):  # type: ignore
        self.device = device

    def __torch_function__(self, func, types, args=(), kwargs=None):  # type: ignore
        kwargs = kwargs or {}
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            else:
                return args[0]
        if (
            self.device is not None
            and func in _device._device_constructors()  # type: ignore
            and kwargs.get("device") is None
        ):  # type: ignore
            kwargs["device"] = self.device
        return func(*args, **kwargs)


def get_net_g(
    model_path: str,
    version: str,
    device: str,
    hps: HyperParameters,
    use_fp16: bool = False,
) -> SynthesizerTrn | SynthesizerTrnJPExtra:
    with EmptyInitOnDevice(device):
        if version.endswith("JP-Extra"):
            logger.info("Using JP-Extra model")
            net_g = SynthesizerTrnJPExtra(
                n_vocab=len(SYMBOLS),
                spec_channels=hps.data.filter_length // 2 + 1,
                segment_size=hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                # hps.model 以下のすべての値を引数に渡す
                use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
                use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
                use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
                use_duration_discriminator=hps.model.use_duration_discriminator,
                use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
                inter_channels=hps.model.inter_channels,
                hidden_channels=hps.model.hidden_channels,
                filter_channels=hps.model.filter_channels,
                n_heads=hps.model.n_heads,
                n_layers=hps.model.n_layers,
                kernel_size=hps.model.kernel_size,
                p_dropout=hps.model.p_dropout,
                resblock=hps.model.resblock,
                resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
                resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
                upsample_rates=hps.model.upsample_rates,
                upsample_initial_channel=hps.model.upsample_initial_channel,
                upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
                n_layers_q=hps.model.n_layers_q,
                use_spectral_norm=hps.model.use_spectral_norm,
                gin_channels=hps.model.gin_channels,
                slm=hps.model.slm,
            ).to(device)
        else:
            logger.info("Using normal model")
            net_g = SynthesizerTrn(
                n_vocab=len(SYMBOLS),
                spec_channels=hps.data.filter_length // 2 + 1,
                segment_size=hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                # hps.model 以下のすべての値を引数に渡す
                use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
                use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
                use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
                use_duration_discriminator=hps.model.use_duration_discriminator,
                use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
                inter_channels=hps.model.inter_channels,
                hidden_channels=hps.model.hidden_channels,
                filter_channels=hps.model.filter_channels,
                n_heads=hps.model.n_heads,
                n_layers=hps.model.n_layers,
                kernel_size=hps.model.kernel_size,
                p_dropout=hps.model.p_dropout,
                resblock=hps.model.resblock,
                resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
                resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
                upsample_rates=hps.model.upsample_rates,
                upsample_initial_channel=hps.model.upsample_initial_channel,
                upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
                n_layers_q=hps.model.n_layers_q,
                use_spectral_norm=hps.model.use_spectral_norm,
                gin_channels=hps.model.gin_channels,
                slm=hps.model.slm,
            ).to(device)

    net_g.eval()
    if model_path.endswith(".pth") or model_path.endswith(".pt"):
        _ = utils.checkpoints.load_checkpoint(
            model_path, net_g, None, skip_optimizer=True, device=device
        )
    elif model_path.endswith(".safetensors") or model_path.endswith(".aivm"):
        _ = utils.safetensors.load_safetensors(model_path, net_g, True, device=device)
    else:
        raise ValueError(f"Unknown model format: {model_path}")

    # 一番実行速度の遅い Generator (Decoder) のみを FP16 に変換
    # それ以外のモジュールはほとんどが精度センシティブな処理で、FP32 でなければ精度や数値安定性の問題で動作しない
    if use_fp16 and device != "cpu":
        net_g.dec.half()
        logger.info("Generator module converted to FP16 for selective mixed precision")

    return net_g


def get_text(
    text: str,
    language_str: Languages,
    hps: HyperParameters,
    device: str,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    given_phone: list[str] | None = None,
    given_tone: list[int] | None = None,
    jtalk: OpenJTalk | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    use_jp_extra = hps.version.endswith("JP-Extra")
    norm_text, phone, tone, word2ph, sep_text, _, _ = clean_text_with_given_phone_tone(
        text,
        language_str,
        given_phone=given_phone,
        given_tone=given_tone,
        use_jp_extra=use_jp_extra,
        # 推論時のみ呼び出されるので、raise_yomi_error は False に設定
        raise_yomi_error=False,
        jtalk=jtalk,
    )
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = extract_bert_feature(
        norm_text,
        word2ph,
        language_str,
        device,
        assist_text,
        assist_text_weight,
        sep_text,  # clean_text_with_given_phone_tone() の中間生成物を再利用して効率向上を図る
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == Languages.ZH:
        zh_bert = bert_ori
        ja_bert = torch.zeros(1024, len(phone), device=device)
        en_bert = torch.zeros(1024, len(phone), device=device)
    elif language_str == Languages.JP:
        zh_bert = torch.zeros(1024, len(phone), device=device)
        ja_bert = bert_ori
        en_bert = torch.zeros(1024, len(phone), device=device)
    elif language_str == Languages.EN:
        zh_bert = torch.zeros(1024, len(phone), device=device)
        ja_bert = torch.zeros(1024, len(phone), device=device)
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert zh_bert.shape[-1] == len(phone), (
        f"Bert seq len {zh_bert.shape[-1]} != {len(phone)}"
    )

    phone = torch.LongTensor(phone).to(device)
    tone = torch.LongTensor(tone).to(device)
    language = torch.LongTensor(language).to(device)
    return zh_bert, ja_bert, en_bert, phone, tone, language


def prepare_inference_data(
    text: str,
    style_vec: NDArray[Any],
    sid: int,  # In the original Bert-VITS2, its speaker_name: str, but here it's id
    language: Languages,
    hps: HyperParameters,
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    given_phone: list[str] | None = None,
    given_tone: list[int] | None = None,
    jtalk: OpenJTalk | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    推論に必要なデータの前処理を行う共通関数。
    infer() と infer_stream() で共通に使用される。

    Returns:
        tuple: (x_tst, x_tst_lengths, sid_tensor, tones, lang_ids, zh_bert, ja_bert, en_bert, style_vec_tensor)
    """
    # テキストから BERT 特徴量・音素列・アクセント列・言語 ID を取得
    # zh_bert, ja_bert, en_bert のうち、指定された言語に対応する1つのみが実際の特徴量を持ち、残りの2つは空のテンソルになる
    zh_bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        device,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_phone=given_phone,
        given_tone=given_tone,
        jtalk=jtalk,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        zh_bert = zh_bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        zh_bert = zh_bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]

    x_tst = phones.unsqueeze(0)
    tones = tones.unsqueeze(0)
    lang_ids = lang_ids.unsqueeze(0)
    zh_bert = zh_bert.unsqueeze(0)
    ja_bert = ja_bert.unsqueeze(0)
    en_bert = en_bert.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
    style_vec_tensor = torch.from_numpy(style_vec).to(device).unsqueeze(0)

    del phones
    sid_tensor = torch.LongTensor([sid]).to(device)

    return (
        x_tst,
        x_tst_lengths,
        sid_tensor,
        tones,
        lang_ids,
        zh_bert,
        ja_bert,
        en_bert,
        style_vec_tensor,
    )


def infer(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,  # In the original Bert-VITS2, its speaker_name: str, but here it's id
    language: Languages,
    hps: HyperParameters,
    net_g: SynthesizerTrn | SynthesizerTrnJPExtra,
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    given_phone: list[str] | None = None,
    given_tone: list[int] | None = None,
    jtalk: OpenJTalk | None = None,
    use_fp16_inference: bool = False,
    clear_cuda_cache: bool = True,
) -> NDArray[np.float32]:
    """
    PyTorch 版音声合成モデルの推論を実行する関数。
    """
    is_jp_extra = hps.version.endswith("JP-Extra")

    # 推論データの前処理（共通処理）
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
        ) = prepare_inference_data(
            text,
            style_vec=style_vec,
            sid=sid,
            language=language,
            hps=hps,
            device=device,
            skip_start=skip_start,
            skip_end=skip_end,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            given_phone=given_phone,
            given_tone=given_tone,
            jtalk=jtalk,
        )

        if is_jp_extra:
            output = cast(SynthesizerTrnJPExtra, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                ja_bert,
                style_vec=style_vec_tensor,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                use_fp16_inference=use_fp16_inference,
            )
        else:
            output = cast(SynthesizerTrn, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                zh_bert,
                ja_bert,
                en_bert,
                style_vec=style_vec_tensor,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                use_fp16_inference=use_fp16_inference,
            )

        audio = output[0][0, 0].data.cpu().float().numpy()

        del (
            x_tst,
            x_tst_lengths,
            sid_tensor,
            tones,
            lang_ids,
            zh_bert,
            ja_bert,
            en_bert,
            style_vec,
        )

        # CUDA メモリを解放する (デフォルトでは True)
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio


def infer_stream(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,  # In the original Bert-VITS2, its speaker_name: str, but here it's id
    language: Languages,
    hps: HyperParameters,
    net_g: SynthesizerTrn | SynthesizerTrnJPExtra,
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    given_phone: list[str] | None = None,
    given_tone: list[int] | None = None,
    jtalk: OpenJTalk | None = None,
    use_fp16_inference: bool = False,
    clear_cuda_cache: bool = True,
    chunk_size: int = 65,  # 下記記事を参考に最適な値を調整
    overlap_size: int = 22,  # 下記記事を参照 (L=11, 11+11=22)
) -> Iterator[NDArray[np.float32]]:
    """
    PyTorch 版音声合成モデルのストリーミング推論を実行する関数。
    Generator 部分のみストリーミング処理を行い、音声チャンクを逐次 yield する。
    ref: https://qiita.com/__dAi00/items/970f0fe66286510537dd
    """
    is_jp_extra = hps.version.endswith("JP-Extra")

    # 推論データの前処理（共通処理）
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
        ) = prepare_inference_data(
            text,
            style_vec=style_vec,
            sid=sid,
            language=language,
            hps=hps,
            device=device,
            skip_start=skip_start,
            skip_end=skip_end,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            given_phone=given_phone,
            given_tone=given_tone,
            jtalk=jtalk,
        )

        # Generator 実行前の共通処理を実行
        if is_jp_extra:
            z, y_mask, g, attn, z_p, m_p, logs_p = cast(
                SynthesizerTrnJPExtra, net_g
            ).infer_input_feature(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                ja_bert,
                style_vec=style_vec_tensor,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                use_fp16_inference=use_fp16_inference,
            )
        else:
            z, y_mask, g, attn, z_p, m_p, logs_p = cast(
                SynthesizerTrn, net_g
            ).infer_input_feature(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                zh_bert,
                ja_bert,
                en_bert,
                style_vec=style_vec_tensor,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                use_fp16_inference=use_fp16_inference,
            )

        # Generator 部分のストリーミング処理
        z_input = z * y_mask
        total_length = z_input.shape[2]  # 入力特徴量の総フレーム数

        # torch.autocast() 用のデバイスタイプを取得
        device_obj = torch.device(device)
        device_type = (
            device_obj.type
            if hasattr(device_obj, "type")
            else str(device).split(":")[0]
        )

        # 全体のアップサンプリング率を計算
        # hps.model.upsample_rates の積が Decoder の総アップサンプリング率
        total_upsample_factor = np.prod(hps.model.upsample_rates).item()
        # overlap_size は入力特徴量空間でのオーバーラップフレーム数 (e.g., 22)
        # margin_frames は片側のマージンフレーム数 (e.g., 11)
        margin_frames = overlap_size // 2

        for start_idx in range(0, total_length, chunk_size - overlap_size):
            end_idx = min(start_idx + chunk_size, total_length)
            # 現在処理する入力特徴量のチャンク
            chunk = z_input[:, :, start_idx:end_idx]

            # FP16 推論の処理
            if use_fp16_inference is True:
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.float16,
                ):
                    # Generator への入力を FP16 に変換
                    # chunk_output は音声波形チャンク (B, 1, T_samples)
                    chunk_output = net_g.dec(chunk.half(), g=g.half())
            else:
                # FP16 を使わない場合は通常通り実行
                chunk_output = net_g.dec(chunk, g=g)

            # オーバーラップ処理: 音声サンプル単位でトリミング
            current_output_length_samples = chunk_output.shape[2]

            trim_left_samples = 0
            # 最初のチャンクでない場合、左マージンに対応するサンプル数を計算してトリム
            if start_idx != 0:
                trim_left_samples = margin_frames * total_upsample_factor

            trim_right_samples = 0
            # 最後のチャンクでない場合、右マージンに対応するサンプル数を計算してトリム
            if end_idx != total_length:
                trim_right_samples = margin_frames * total_upsample_factor

            # 有効な音声部分の開始・終了インデックス（サンプル単位）
            start_slice_idx = trim_left_samples
            end_slice_idx = current_output_length_samples - trim_right_samples

            if start_slice_idx < end_slice_idx:
                # 有効な音声部分をスライス
                valid_audio_chunk_tensor = chunk_output[
                    :, :, start_slice_idx:end_slice_idx
                ]
                # 音声チャンクを numpy 配列に変換して yield
                audio_chunk = valid_audio_chunk_tensor[0, 0].data.cpu().float().numpy()
                if audio_chunk.size > 0:
                    yield audio_chunk
            else:
                # 有効な音声部分がない場合は何も yield しない
                pass

        del (
            x_tst,
            x_tst_lengths,
            sid_tensor,
            tones,
            lang_ids,
            zh_bert,
            ja_bert,
            en_bert,
            style_vec,
            z,
            y_mask,
            g,
            z_input,
        )

        # CUDA メモリを解放する (デフォルトでは True)
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
