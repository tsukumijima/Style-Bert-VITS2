import traceback
from typing import Any, cast

import torch


# torch_tensorrt をインポート (失敗してもエラーにしない)
try:
    import torch_tensorrt

    _torch_tensorrt_available = True
except ImportError:
    _torch_tensorrt_available = False
    torch_tensorrt = None  # モジュールが存在しない場合 None を設定

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


# TensorRT が利用可能かどうかのフラグ
_is_tensorrt_available = False
_compiled_dec = None  # コンパイル済み Generator をキャッシュする変数

# torch_tensorrt がインポートでき、かつ CUDA が利用可能かチェック
if _torch_tensorrt_available and torch.cuda.is_available():
    _is_tensorrt_available = True
    logger.info("Torch-TensorRT is available.")
elif _torch_tensorrt_available:
    logger.warning("Torch-TensorRT is installed but CUDA is not available.")
else:
    logger.info("Torch-TensorRT is not installed. Falling back to PyTorch.")


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


def get_net_g(
    model_path: str, version: str, device: str, hps: HyperParameters
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
    net_g: SynthesizerTrn | SynthesizerTrnJPExtra,
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    given_phone: list[str] | None = None,
    given_tone: list[int] | None = None,
) -> NDArray[Any]:
    global _is_tensorrt_available
    global _compiled_dec
    global torch_tensorrt  # グローバル変数を参照

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

        # --- Generator (self.dec) の TensorRT 化 ---
        # net_g.dec をローカル変数にコピー
        dec_module = net_g.dec

        # TensorRT が利用可能で、まだコンパイルされていない場合
        if _is_tensorrt_available and _compiled_dec is None:
            try:
                logger.info(
                    "Attempting to compile the Generator (dec) with Torch-TensorRT..."
                )
                # Generator への入力形状を定義 (time 次元を可変に)
                # time の最小・最適・最大値を設定 (必要に応じて調整)
                seq_len_min, seq_len_opt, seq_len_max = 30, 250, 1200
                # z の形状: [batch, inter_channels, time]
                # torch_tensorrt.Input を使用するには torch_tensorrt が None でないことを確認
                assert torch_tensorrt is not None
                input_z_shape = torch_tensorrt.Input(
                    min_shape=[1, hps.model.inter_channels, seq_len_min],
                    opt_shape=[1, hps.model.inter_channels, seq_len_opt],
                    max_shape=[1, hps.model.inter_channels, seq_len_max],
                    dtype=torch.float32,  # 必要に応じて dtype を確認・変更
                )
                # g の形状: [batch, gin_channels, 1] (通常固定長)
                input_g_shape = torch_tensorrt.Input(
                    shape=[1, hps.model.gin_channels, 1],
                    dtype=torch.float32,  # 必要に応じて dtype を確認・変更
                )

                _compiled_dec = torch_tensorrt.compile(
                    dec_module,
                    inputs=[input_z_shape, input_g_shape],  # z と g を入力として指定
                    enabled_precisions={torch.float},  # type: ignore
                    # workspace_size=1 << 30,  # 1GB
                    # pass_through_build_failures=True,  # ビルド失敗時に PyTorch にフォールバック
                    ir="dynamo",  # 必要であれば指定
                    # assume_dynamic_shape_support=True, # 必要であれば指定
                )
                logger.info("Generator compiled successfully with Torch-TensorRT.")
            except Exception as ex:
                logger.warning("Failed to compile Generator with Torch-TensorRT:")
                # トレースバックを出力
                traceback.print_exc()
                logger.warning("Falling back to PyTorch for Generator.")
                _is_tensorrt_available = False  # コンパイル失敗時は TensorRT を無効化
                _compiled_dec = (
                    dec_module  # 元のモジュールをキャッシュ (フォールバック用)
                )
        elif _is_tensorrt_available and _compiled_dec is not None:
            # コンパイル済み Generator を使用
            dec_module = _compiled_dec
        elif not _is_tensorrt_available and _compiled_dec is None:
            # TensorRT が利用不可の場合、元のモジュールをキャッシュ
            _compiled_dec = dec_module
        elif not _is_tensorrt_available and _compiled_dec is not None:
            # キャッシュされた元のモジュールを使用 (フォールバック後)
            dec_module = _compiled_dec
        # else: # _is_tensorrt_available is False and _compiled_dec is not None は上の elif でカバーされる

        # --- 推論実行 ---
        if is_jp_extra:
            # JP-Extra モデルの推論
            _o, _attn, _y_mask, _hidden_states = cast(
                SynthesizerTrnJPExtra, net_g
            ).infer(
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
            # Generator の呼び出し部分を差し替えるのではなく、infer 全体を呼び出した上で
            # dec の部分だけ TensorRT 化されたモジュールを使うアプローチは難しい
            # infer メソッド内部のロジックをここに展開する必要がある

            # --- infer メソッド内のロジックを展開 ---
            if net_g.n_speakers > 0:
                g = net_g.emb_g(sid_tensor).unsqueeze(-1)
            else:
                # このケースは現状の infer 関数呼び出しでは y が渡されないため考慮不要
                # g = net_g.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                raise NotImplementedError(
                    "Reference encoder case is not handled here yet."
                )

            x, m_p, logs_p, x_mask = net_g.enc_p(
                x_tst,
                x_tst_lengths,
                tones,
                lang_ids,
                ja_bert,
                style_vec=style_vec_tensor,
                g=g,
            )
            logw = net_g.sdp(
                x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w
            ) * (sdp_ratio) + net_g.dp(x, x_mask, g=g) * (1 - sdp_ratio)
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
                x_mask.dtype
            )
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = commons.generate_path(w_ceil, attn_mask)

            m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
                1, 2
            )

            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
            z = net_g.flow(z_p, y_mask, g=g, reverse=True)  # flow は元の net_g を使う

            # --- TensorRT 化された (または元の) Generator を使用 ---
            o = dec_module((z * y_mask)[:, :, :None], g=g)  # max_len は None (制限なし)
            # -----------------------------------------
            output = (o, attn, y_mask, (z, z_p, m_p, logs_p))

        else:
            # 通常モデルの推論
            _o, _attn, _y_mask, _hidden_states = cast(SynthesizerTrn, net_g).infer(
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
            # --- infer メソッド内のロジックを展開 ---
            if net_g.n_speakers > 0:
                g = net_g.emb_g(sid_tensor).unsqueeze(-1)
            else:
                raise NotImplementedError(
                    "Reference encoder case is not handled here yet."
                )

            x, m_p, logs_p, x_mask = net_g.enc_p(
                x_tst,
                x_tst_lengths,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                style_vec=style_vec_tensor,
                sid=sid_tensor,
                g=g,
            )

            logw = net_g.sdp(
                x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w
            ) * (sdp_ratio) + net_g.dp(x, x_mask, g=g) * (1 - sdp_ratio)
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
                x_mask.dtype
            )
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = commons.generate_path(w_ceil, attn_mask)

            m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
                1, 2
            )

            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
            z = net_g.flow(z_p, y_mask, g=g, reverse=True)  # flow は元の net_g を使う

            # --- TensorRT 化された (または元の) Generator を使用 ---
            o = dec_module((z * y_mask)[:, :, :None], g=g)  # max_len は None (制限なし)
            # -----------------------------------------
            output = (o, attn, y_mask, (z, z_p, m_p, logs_p))

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
            style_vec_tensor,  # style_vec ではなく style_vec_tensor を削除
            o,
            attn,
            y_mask,
            z,
            z_p,
            m_p,
            logs_p,  # 追加の変数を削除
            x,
            logw,
            w,
            w_ceil,
            y_lengths,
            attn_mask,
            g,  # 追加の変数を削除
            _o,
            _attn,
            _y_mask,
            _hidden_states,  # 元の infer の戻り値を削除
        )  # , emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio
