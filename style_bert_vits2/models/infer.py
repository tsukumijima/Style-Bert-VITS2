from contextlib import nullcontext
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

    # FP16 変換を適用
    if use_fp16 is True:
        logger.info("Converting model to FP16 for inference")
        net_g = _convert_model_to_fp16(net_g)

    return net_g


def _convert_model_to_fp16(
    model: SynthesizerTrn | SynthesizerTrnJPExtra,
) -> SynthesizerTrn | SynthesizerTrnJPExtra:
    """
    モデルを FP16 推論用に変換する。
    torch.autocast との組み合わせで使用することを前提とし、
    数値安定性のため、一部の層は FP32 のまま保持する。

    Args:
        model: 変換対象のモデル

    Returns:
        FP16 に変換されたモデル
    """

    # 精度を維持すべき層の型を定義
    fp32_layer_types = (
        torch.nn.LayerNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.GroupNorm,
        torch.nn.Embedding,
    )

    # まずモデル全体を FP16 に変換
    model = model.half()

    # 特定の層を FP32 に戻す
    for name, module in model.named_modules():
        if isinstance(module, fp32_layer_types):
            logger.debug(
                f"Keeping {name} ({type(module).__name__}) in FP32 for numerical stability"
            )
            module.float()

    return model


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
        bert = bert_ori
        ja_bert = torch.zeros(1024, len(phone), device=device)
        en_bert = torch.zeros(1024, len(phone), device=device)
    elif language_str == Languages.JP:
        bert = torch.zeros(1024, len(phone), device=device)
        ja_bert = bert_ori
        en_bert = torch.zeros(1024, len(phone), device=device)
    elif language_str == Languages.EN:
        bert = torch.zeros(1024, len(phone), device=device)
        ja_bert = torch.zeros(1024, len(phone), device=device)
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert bert.shape[-1] == len(phone), (
        f"Bert seq len {bert.shape[-1]} != {len(phone)}"
    )

    phone = torch.LongTensor(phone).to(device)
    tone = torch.LongTensor(tone).to(device)
    language = torch.LongTensor(language).to(device)
    return bert, ja_bert, en_bert, phone, tone, language


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
    clear_cuda_cache: bool = True,
) -> NDArray[np.float32]:
    is_jp_extra = hps.version.endswith("JP-Extra")
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
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
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]

    # FP16 が有効かどうかを判定（モデルの最初のパラメータの dtype で判定）
    use_fp16_inference = next(net_g.parameters()).dtype == torch.float16

    with torch.inference_mode():
        x_tst = phones.unsqueeze(0)
        tones = tones.unsqueeze(0)
        lang_ids = lang_ids.unsqueeze(0)
        bert = bert.unsqueeze(0)
        ja_bert = ja_bert.unsqueeze(0)
        en_bert = en_bert.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)

        # スタイルベクトルは精度劣化を避けるため FP16 には変換せず、元の精度（通常 FP32）を保持する
        style_vec_tensor = torch.from_numpy(style_vec).to(device).unsqueeze(0)
        del phones
        sid_tensor = torch.LongTensor([sid]).to(device)

        # FP16 推論の場合は torch.autocast を使用して安全な mixed precision を実現
        autocast_context = (
            torch.autocast(
                device_type=device.split(":")[0] if ":" in device else device,
                dtype=torch.float16,
            )
            if use_fp16_inference and device != "cpu"
            else nullcontext()
        )

        with autocast_context:
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
                )
            else:
                output = cast(SynthesizerTrn, net_g).infer(
                    x_tst,
                    x_tst_lengths,
                    sid_tensor,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    en_bert,
                    style_vec=style_vec_tensor,
                    length_scale=length_scale,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                )

        audio = output[0][0, 0].data.cpu().float().numpy()

        del (
            x_tst,
            tones,
            lang_ids,
            bert,
            x_tst_lengths,
            sid_tensor,
            ja_bert,
            en_bert,
            style_vec,
        )  # , emo

        # CUDA メモリを解放する (デフォルトでは True)
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio
