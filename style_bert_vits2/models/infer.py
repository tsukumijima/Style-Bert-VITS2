import traceback
from typing import Any, cast

import torch
import torch_tensorrt
from numpy.typing import NDArray
from torch.overrides import TorchFunctionMode

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
            and func in torch.utils._device._device_constructors()  # type: ignore
            and kwargs.get("device") is None
        ):  # type: ignore
            kwargs["device"] = self.device
        return func(*args, **kwargs)


class SynthesizerTrnWrapperForTensorRT(torch.nn.Module):
    def __init__(
        self,
        synthesizer_model: SynthesizerTrn | SynthesizerTrnJPExtra,
        is_jp_extra: bool,
    ):
        super().__init__()
        self.synthesizer_model = synthesizer_model
        self.is_jp_extra = is_jp_extra

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor,
        tones: torch.Tensor,
        lang_ids: torch.Tensor,
        bert_or_ja_bert: torch.Tensor,
        style_vec: torch.Tensor,
        ja_bert_normal: torch.Tensor | None = None,
        en_bert_normal: torch.Tensor | None = None,
        length_scale: float = 1.0,
        sdp_ratio: float = 0.0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
    ):
        if self.is_jp_extra:
            return cast(SynthesizerTrnJPExtra, self.synthesizer_model).infer(
                x,
                x_lengths,
                sid,
                tones,
                lang_ids,
                bert_or_ja_bert,
                style_vec=style_vec,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
            )
        else:
            if ja_bert_normal is None or en_bert_normal is None:
                raise ValueError(
                    "ja_bert_normal and en_bert_normal must be provided for non-JP-Extra model in TensorRT wrapper."
                )
            return cast(SynthesizerTrn, self.synthesizer_model).infer(
                x,
                x_lengths,
                sid,
                tones,
                lang_ids,
                bert_or_ja_bert,
                ja_bert_normal,
                en_bert_normal,
                style_vec=style_vec,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
            )


def get_net_g(
    model_path: str,
    version: str,
    device: str,
    hps: HyperParameters,
    use_tensorrt: bool = True,
) -> SynthesizerTrn | SynthesizerTrnJPExtra | torch.jit.ScriptModule:
    with EmptyInitOnDevice(device):
        is_jp_extra_model = version.endswith("JP-Extra")
        if is_jp_extra_model:
            logger.info("Using JP-Extra model")
            net_g_pytorch = SynthesizerTrnJPExtra(
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
            net_g_pytorch = SynthesizerTrn(
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

    net_g_pytorch.eval()
    if model_path.endswith(".pth") or model_path.endswith(".pt"):
        _ = utils.checkpoints.load_checkpoint(
            model_path, net_g_pytorch, None, skip_optimizer=True, device=device
        )
    elif model_path.endswith(".safetensors") or model_path.endswith(".aivm"):
        _ = utils.safetensors.load_safetensors(
            model_path, net_g_pytorch, True, device=device
        )
    else:
        raise ValueError(f"Unknown model format: {model_path}")

    if use_tensorrt and device.startswith("cuda"):
        try:
            logger.info(
                f"Attempting to compile the model with Torch-TensorRT for device: {device}"
            )
            infer_wrapper = (
                SynthesizerTrnWrapperForTensorRT(net_g_pytorch, is_jp_extra_model)
                .eval()
                .to(device)
            )

            seq_len_min, seq_len_opt, seq_len_max = 30, 250, 1200
            style_vec_dim = 256

            x_input = torch_tensorrt.Input(
                min_shape=[1, seq_len_min],
                opt_shape=[1, seq_len_opt],
                max_shape=[1, seq_len_max],
                dtype=torch.long,
            )
            x_lengths_input = torch_tensorrt.Input(shape=[1], dtype=torch.long)
            sid_input = torch_tensorrt.Input(shape=[1], dtype=torch.long)
            tones_input = torch_tensorrt.Input(
                min_shape=[1, seq_len_min],
                opt_shape=[1, seq_len_opt],
                max_shape=[1, seq_len_max],
                dtype=torch.long,
            )
            lang_ids_input = torch_tensorrt.Input(
                min_shape=[1, seq_len_min],
                opt_shape=[1, seq_len_opt],
                max_shape=[1, seq_len_max],
                dtype=torch.long,
            )
            bert_tensor_input = torch_tensorrt.Input(
                min_shape=[1, 1024, seq_len_min],
                opt_shape=[1, 1024, seq_len_opt],
                max_shape=[1, 1024, seq_len_max],
                dtype=torch.float,
            )
            style_vec_input = torch_tensorrt.Input(
                shape=[1, style_vec_dim], dtype=torch.float
            )

            compile_inputs = [
                x_input,
                x_lengths_input,
                sid_input,
                tones_input,
                lang_ids_input,
            ]
            if is_jp_extra_model:
                compile_inputs.append(bert_tensor_input)
            else:
                compile_inputs.append(bert_tensor_input)
                compile_inputs.append(bert_tensor_input)
                compile_inputs.append(bert_tensor_input)
            compile_inputs.append(style_vec_input)

            enabled_precisions = {torch.float, torch.half}

            logger.info("Compiling the model with Torch-TensorRT...")
            trt_model = cast(
                torch.jit.ScriptModule,
                torch_tensorrt.compile(
                    infer_wrapper,
                    inputs=compile_inputs,
                    enabled_precisions=enabled_precisions,  # type: ignore
                    workspace_size=1 << 30,
                ),
            )
            logger.info("Model compiled successfully with Torch-TensorRT.")
            return trt_model
        except Exception as e:
            logger.error(f"Failed to compile the model with Torch-TensorRT: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error("Falling back to the original PyTorch model.")
            return net_g_pytorch
    return net_g_pytorch


def get_text(
    text: str,
    language_str: Languages,
    hps: HyperParameters,
    device: str,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    given_phone: list[str] | None = None,
    given_tone: list[int] | None = None,
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
    net_g: SynthesizerTrn | SynthesizerTrnJPExtra | torch.jit.ScriptModule,
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    given_phone: list[str] | None = None,
    given_tone: list[int] | None = None,
) -> NDArray[Any]:
    is_tensorrt_model = isinstance(net_g, torch.jit.ScriptModule)

    if is_tensorrt_model:
        is_jp_extra_net = (
            net_g.is_jp_extra
            if hasattr(net_g, "is_jp_extra")
            else hps.version.endswith("JP-Extra")
        )
    else:
        is_jp_extra_net = hps.version.endswith("JP-Extra")

    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        device,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_phone=given_phone,
        given_tone=given_tone,
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

    if is_tensorrt_model:
        logger.debug("Converting float tensors to half for TensorRT model.")
        if bert.dtype == torch.float32:
            bert = bert.half()
        if ja_bert.dtype == torch.float32:
            ja_bert = ja_bert.half()
        if en_bert.dtype == torch.float32:
            en_bert = en_bert.half()

    with torch.no_grad():
        x_tst = phones.unsqueeze(0)
        tones = tones.unsqueeze(0)
        lang_ids = lang_ids.unsqueeze(0)
        bert = bert.unsqueeze(0)
        ja_bert = ja_bert.unsqueeze(0)
        en_bert = en_bert.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        style_vec_tensor = torch.from_numpy(style_vec).to(device).unsqueeze(0)
        del phones
        sid_tensor = torch.LongTensor([sid]).to(device)

        if is_tensorrt_model and style_vec_tensor.dtype == torch.float32:
            style_vec_tensor = style_vec_tensor.half()

        if is_tensorrt_model:
            logger.debug("Using TensorRT model for inference.")
            forward_args_pos = [
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
            ]
            if is_jp_extra_net:
                forward_args_pos.append(ja_bert)
            else:
                forward_args_pos.append(bert)
                forward_args_pos.append(ja_bert)
                forward_args_pos.append(en_bert)

            output = net_g(
                *forward_args_pos,
                style_vec=style_vec_tensor,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
            )
        else:
            logger.debug("Using PyTorch model for inference.")
            if is_jp_extra_net:
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio
