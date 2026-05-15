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
from style_bert_vits2.models.infer_onnx import TokenDurationsResult
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.models.models_nanairo import (
    SynthesizerTrn as SynthesizerTrnNanairo,
)
from style_bert_vits2.models.tensor_padding import pad_sequence_tensor
from style_bert_vits2.nlp import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
    extract_bert_feature,
)
from style_bert_vits2.nlp.symbols import NANAIRO_SYMBOLS, SYMBOLS


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
) -> SynthesizerTrn | SynthesizerTrnJPExtra | SynthesizerTrnNanairo:
    is_jp_extra_like_model = hps.is_jp_extra_like_model()
    is_nanairo_like_model = hps.is_nanairo_like_model()
    with EmptyInitOnDevice(device):
        if is_nanairo_like_model is True:
            logger.info("Using Nanairo model")
            net_g = SynthesizerTrnNanairo(
                n_vocab=len(NANAIRO_SYMBOLS),
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
                use_transformer_duration_predictor=hps.model.use_transformer_duration_predictor,
                duration_filter_channels=hps.model.duration_filter_channels,
                use_duration_token_type_embedding=hps.model.use_duration_token_type_embedding,
                duration_token_type_count=hps.model.duration_token_type_count,
                use_speaker_adapter=hps.model.use_speaker_adapter,
                speaker_adapter_input_dim=hps.model.speaker_adapter_input_dim,
                speaker_adapter_bottleneck_dim=hps.model.speaker_adapter_bottleneck_dim,
            ).to(device)
        elif is_jp_extra_like_model is True:
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
    if use_fp16 is True:
        net_g.dec.half()
        logger.info("Generator module converted to FP16 for selective mixed precision")

    # Generator (Decoder) の推論最適化: weight_norm を取り除く
    # 学習完了後の推論時には weight_norm は不要なオーバーヘッドとなるため除去しておく
    # 精度に影響はなく、単に計算効率が向上する
    net_g.dec.remove_weight_norm()
    logger.info(
        "Generator module weight normalization removed for inference optimization"
    )

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
    is_jp_extra_like_model = hps.is_jp_extra_like_model()
    is_nanairo_like_model = hps.is_nanairo_like_model()
    norm_text, phone, tone, word2ph, sep_text, _, _ = clean_text_with_given_phone_tone(
        text,
        language_str,
        given_phone=given_phone,
        given_tone=given_tone,
        use_jp_extra=is_jp_extra_like_model,
        use_nanairo=is_nanairo_like_model,
        # 推論時のみ呼び出されるので、raise_yomi_error は False に設定
        raise_yomi_error=False,
        jtalk=jtalk,
    )
    phone, tone, language = cleaned_text_to_sequence(
        phone,
        tone,
        language_str,
        use_nanairo=is_nanairo_like_model,
    )

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
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        sep_text=sep_text,  # clean_text_with_given_phone_tone() の中間生成物を再利用して効率向上を図る
        use_nanairo=is_nanairo_like_model,
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if is_jp_extra_like_model is True:
        # 日本語のみに対応した JP-Extra モデルでは ja_bert のみが推論時に参照され、他言語の特徴量は推論時には一切利用されない
        # 空テンソルは CPU 上で作成し、GPU 転送を避けることで VRAM 使用量とメモリ断片化を削減
        empty_tensor = torch.empty(0, 0)  # CPU 上で作成
        if language_str == Languages.JP:
            zh_bert = empty_tensor
            ja_bert = bert_ori
            en_bert = empty_tensor
        elif language_str == Languages.ZH:
            zh_bert = bert_ori
            ja_bert = empty_tensor
            en_bert = empty_tensor
        elif language_str == Languages.EN:
            zh_bert = empty_tensor
            ja_bert = empty_tensor
            en_bert = bert_ori
        else:
            raise ValueError("language_str should be ZH, JP or EN")
    else:
        # 通常モデルでは全言語の BERT 特徴量が必要なため、GPU 上でゼロテンソルを作成
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

    for bert_name, bert_tensor in [("zh", zh_bert), ("ja", ja_bert), ("en", en_bert)]:
        assert bert_tensor.shape[-1] == len(phone) or bert_tensor.numel() == 0, (
            f"{bert_name}_bert seq len {bert_tensor.shape[-1]} != {len(phone)}"
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
    enable_tensor_padding: bool = False,
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

    # テンソルパディング処理
    if enable_tensor_padding is True:
        # x_tst とその関連テンソルをパディング
        x_tst, _ = pad_sequence_tensor(
            x_tst,
            length_dim=1,
            pool_type="vits2_sequence",
            use_pool=True,
        )
        tones, _ = pad_sequence_tensor(
            tones,
            length_dim=1,
            pool_type="vits2_tones",
            use_pool=True,
        )
        lang_ids, _ = pad_sequence_tensor(
            lang_ids,
            length_dim=1,
            pool_type="vits2_lang_ids",
            use_pool=True,
        )
        # BERT 特徴量もパディング（空でない場合のみ）
        if zh_bert.numel() > 0:
            zh_bert, _ = pad_sequence_tensor(
                zh_bert,
                length_dim=2,
                pool_type="vits2_zh_bert",
                use_pool=True,
            )
        if ja_bert.numel() > 0:
            ja_bert, _ = pad_sequence_tensor(
                ja_bert,
                length_dim=2,
                pool_type="vits2_ja_bert",
                use_pool=True,
            )
        if en_bert.numel() > 0:
            en_bert, _ = pad_sequence_tensor(
                en_bert,
                length_dim=2,
                pool_type="vits2_en_bert",
                use_pool=True,
            )

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


def predict_token_durations(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale_w: float,
    sid: int,
    language: Languages,
    hps: HyperParameters,
    net_g: SynthesizerTrn | SynthesizerTrnJPExtra | SynthesizerTrnNanairo,
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    given_phone: list[str] | None = None,
    given_tone: list[int] | None = None,
    jtalk: OpenJTalk | None = None,
    use_fp16: bool = False,
    enable_tensor_padding: bool = False,
) -> TokenDurationsResult:
    """
    PyTorch 版音声合成モデルの duration（内部トークン列単位）を推定する関数。
    Decoder (Generator) は実行せず、Encoder + DP/SDP のみを実行する。
    返り値の durations_frames は add_blank 適用後の内部トークン列に対応する。

    重要：
    - ここでいう duration は「メルスペクトログラムのフレーム数」であり、秒ではない。
      秒への換算は `seconds_per_frame = hop_length / sampling_rate` に基づいて行う。
    - VITS 系では、音声波形を直接 44.1kHz のサンプル単位で長さ制御するのではなく、
      まず潜在表現（メルフレーム空間）を音素列にアライメントし、その後に Decoder がアップサンプリングして波形を生成する。
      したがって「フレーム単位」が長さ制御の最小単位になり、秒指定を内部でフレームに丸める必要がある。
    - この関数は「長さ推定だけ高速に欲しい」用途を想定しており、Decoder 以降の計算 (flow / generator) を意図的に省略する。
      そのため `noise_scale` (Decoder 側のサンプリングノイズなど) は関与せず、duration 推定に影響する `noise_scale_w`
      (SDP: Stochastic Duration Predictor のノイズ) だけを受け取る。
    - 返り値は常に `length_scale=1.0`（話速スケール未適用）を前提とした duration になる。

    Returns:
        TokenDurationsResult: 予測されたトークンの長さ（メルフレーム数）を秒単位で返す
    """

    is_jp_extra_like_model = hps.is_jp_extra_like_model()
    is_nanairo_like_model = hps.is_nanairo_like_model()

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
            enable_tensor_padding=enable_tensor_padding,
        )

        # 通常は multi-speaker 前提で、sid から話者埋め込み g を作る
        # n_speakers <= 0 のモデルは事前学習モデル時点で存在しないはずだが、念のため明示的にエラーにする
        if net_g.n_speakers <= 0:
            raise ValueError(
                "predict_token_durations does not support n_speakers <= 0 models"
            )
        g = net_g.emb_g(sid_tensor).unsqueeze(-1)

        # Encoder 入力の組み立てのみ、JP-Extra と通常モデルでシグネチャが異なるため分岐する
        # それ以外（SDP/DP の呼び出しや後続計算）は同一 API のため共通化する
        if is_jp_extra_like_model:
            if use_fp16 is True:
                # JP-Extra では ja_bert のみを参照するため、ここだけ float32 に戻せばよい
                # （prepare_inference_data() 側で既に float32 になっているはずだが、念のため明示的に float32 に戻す）
                ja_bert = ja_bert.float()

            if is_nanairo_like_model is True:
                x, _m_p, _logs_p, x_mask = cast(SynthesizerTrnNanairo, net_g).enc_p(
                    x_tst,
                    x_tst_lengths,
                    tones,
                    lang_ids,
                    ja_bert,
                    style_vec_tensor,
                    g=g,
                    use_fp16=use_fp16,
                )
            else:
                x, _m_p, _logs_p, x_mask = cast(SynthesizerTrnJPExtra, net_g).enc_p(
                    x_tst,
                    x_tst_lengths,
                    tones,
                    lang_ids,
                    ja_bert,
                    style_vec_tensor,
                    g=g,
                    use_fp16=use_fp16,
                )
        else:
            if use_fp16 is True:
                # 通常モデルは多言語対応で、zh/ja/en の各 BERT を参照し得るため、まとめて float32 に正規化する
                zh_bert = zh_bert.float()
                ja_bert = ja_bert.float()
                en_bert = en_bert.float()

            x, _m_p, _logs_p, x_mask = cast(SynthesizerTrn, net_g).enc_p(
                x_tst,
                x_tst_lengths,
                tones,
                lang_ids,
                zh_bert,
                ja_bert,
                en_bert,
                style_vec_tensor,
                sid_tensor,
                g=g,
                use_fp16=use_fp16,
            )

        # Duration Predictor の推定値 logw を、DP と SDP の比率で混合して使う
        # SDP（stochastic）は noise_scale_w により揺らぎが入りやすい
        # DP（deterministic）は揺らぎが少なく、比較的安定しやすい
        net_g_duration = cast(Any, net_g)
        logw = net_g_duration.sdp(
            x,
            x_mask,
            g=g,
            reverse=True,
            noise_scale=noise_scale_w,
        ) * (sdp_ratio) + net_g_duration.dp(x, x_mask, g=g) * (1 - sdp_ratio)

        # 推定 duration（フレーム数）は exp(logw) を基に作る
        # さらに x_mask を掛けることで padding を除外する
        # ここでは話速 (length_scale) を適用せず、常に length_scale=1.0 での基準 duration を返す
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w)

        durations_frames = w_ceil[0, 0].detach().cpu().to(torch.int64).numpy()
        token_ids = x_tst[0].detach().cpu().to(torch.int64).numpy()

        seconds_per_frame = float(hps.data.hop_length) / float(hps.data.sampling_rate)
        durations_seconds = durations_frames.astype(np.float32) * float(
            seconds_per_frame
        )

        return TokenDurationsResult(
            sampling_rate=int(hps.data.sampling_rate),
            hop_length=int(hps.data.hop_length),
            token_ids=token_ids.tolist(),
            durations_frames=durations_frames.tolist(),
            durations_seconds=durations_seconds.tolist(),
        )


def _prepare_duration_frames_override_from_given_phone_length(
    given_phone: list[str] | None,
    given_phone_length: list[float | None] | None,
    expected_token_length: int,
    skip_start: bool,
    skip_end: bool,
    length_scale: float,
    hps: HyperParameters,
    device: str,
    enable_tensor_padding: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    音素長（秒）の指定を、モデル内部の duration 上書きテンソル（フレーム数）に変換して返す。

    この関数では以下を一括で行う。
    1) 入力 (given_phone_length / given_phone) の整合性チェック
    2) 秒 → フレームへの換算 (`ceil(seconds / seconds_per_frame)`)
    3) `add_blank` への対応（内部トークン列は 2N+1 になる）
    4) `skip_start / skip_end` のスライスを、duration 上書きテンソルにも同じように適用
    5) 最終的に、推論に使われる `x_tst`（内部トークン列）と長さが一致することを検証

    VITS 系の `add_blank` は「音素遷移（境界）」の表現としてモデルが学習している可能性が高く、
    実測でも blank トークンが総フレームのかなりの割合を占める。
    そのため、API 側が blank を意識しない設計であっても、ここで blank の duration を雑に固定したり、
    一律 0 に潰したりすると推論品質が劣化し得る。
    そこで本実装では、呼び出し側が音素長（子音/母音など）を指定した場合でも、
    blank トークン（`add_blank` により挿入されるトークン）については「未指定」として扱い、
    duration 推定器 (DP/SDP) が出した値をそのまま採用する。
    具体的には、`add_blank=True` のとき、元音素に対応する odd index のみをマスク True にし、
    even index（blank）をマスク False のままにする。

    Returns:
        tuple[torch.Tensor | None, torch.Tensor | None]:
            - durations_frames_override: shape [1, 1, T_x]（未指定なら None）
            - durations_frames_override_mask: shape [1, 1, T_x]（未指定なら None）
    """

    # 音素長（秒）の最大値を 10 秒とする
    MAX_PHONE_LENGTH_SECONDS = 10.0

    # given_phone_length が未指定の場合は、duration 上書きを行わない
    if given_phone_length is None:
        return None, None

    # given_phone のバリデーション
    if given_phone is None:
        raise ValueError("given_phone_length requires given_phone")
    if len(given_phone_length) != len(given_phone):
        raise ValueError(
            "Length of given_phone_length must match length of given_phone. "
            f"given_phone_length: {len(given_phone_length)}, given_phone: {len(given_phone)}"
        )
    phone_count = len(given_phone_length)
    if phone_count == 0:
        raise ValueError("given_phone_length must not be empty")

    # length_scale のバリデーション
    if float(length_scale) <= 0.0:
        raise ValueError("length_scale must be positive")

    # seconds_per_frame のバリデーション
    seconds_per_frame = float(hps.data.hop_length) / float(hps.data.sampling_rate)
    if seconds_per_frame <= 0.0:
        raise ValueError("seconds_per_frame must be positive")

    # まずは add_blank 適用前（given_phone と同じ長さ）の単位で、秒→フレームに変換する
    # ここで「未指定」は 0 フレーム + mask False として表現し、後段で `torch.where(mask, override, predicted)` にかける
    durations_frames_list: list[float] = []
    durations_mask_list: list[bool] = []
    for duration_seconds in given_phone_length:
        if duration_seconds is None or duration_seconds <= 0.0:
            durations_frames_list.append(0.0)
            durations_mask_list.append(False)
            continue

        duration_seconds_float = float(duration_seconds)
        if not np.isfinite(duration_seconds_float):
            raise ValueError("given_phone_length must be finite")
        if duration_seconds_float > MAX_PHONE_LENGTH_SECONDS:
            raise ValueError(
                "given_phone_length is too large. "
                f"max: {MAX_PHONE_LENGTH_SECONDS}, actual: {duration_seconds_float}"
            )

        # given_phone_length は「length_scale=1.0 基準の秒」として受け取り、
        # モデル内部の duration（フレーム数）は、推論時点の length_scale に合わせてスケールして上書きする
        scaled_seconds = duration_seconds_float * float(length_scale)
        frames = int(np.ceil(scaled_seconds / seconds_per_frame))
        if frames < 1:
            # 0 フレームは音が消える/アライメントが破綻する可能性があるため避ける
            # 「最小 1 フレーム」は最終的な秒精度よりも、推論の安定性を優先する設計
            frames = 1
        durations_frames_list.append(float(frames))
        durations_mask_list.append(True)

    # add_blank=True の場合、モデル内部のトークン列は intersperse により `2N+1` へ変換される
    # 例 (N=3):
    #   phone: [a, b, c]
    #   token: [0, a, 0, b, 0, c, 0]
    # このうち even index（0,2,4,6...）は blank であり、ここは外部に公開するのが難しいため、
    # ここでは「未指定」としてマスク False のままにする（= duration 推定器の出力を採用する）
    if bool(hps.data.add_blank) is True:
        token_count = phone_count * 2 + 1
        durations_frames_token: list[float] = [0.0] * token_count
        durations_mask_token: list[bool] = [False] * token_count
        for phone_index, (frames, mask) in enumerate(
            zip(durations_frames_list, durations_mask_list, strict=True)
        ):
            token_index = phone_index * 2 + 1
            durations_frames_token[token_index] = frames
            durations_mask_token[token_index] = mask
    else:
        token_count = phone_count
        durations_frames_token = durations_frames_list
        durations_mask_token = durations_mask_list

    durations_frames_override = (
        torch.tensor(
            durations_frames_token,
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    durations_frames_override_mask = (
        torch.tensor(
            durations_mask_token,
            dtype=torch.bool,
            device=device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    assert durations_frames_override.shape[2] == token_count
    assert durations_frames_override_mask.shape[2] == token_count

    # `prepare_inference_data()` が skip_start / skip_end により x_tst をスライスしている場合、
    # duration 上書きテンソル側も同じようにスライスしないと、内部トークン長が一致しない
    # ここでのスライス値（先頭 3 / 末尾 2）は prepare_inference_data の実装と揃えることが必須
    if skip_start:
        durations_frames_override = durations_frames_override[:, :, 3:]
        durations_frames_override_mask = durations_frames_override_mask[:, :, 3:]
    if skip_end:
        durations_frames_override = durations_frames_override[:, :, :-2]
        durations_frames_override_mask = durations_frames_override_mask[:, :, :-2]

    if int(durations_frames_override.shape[2]) != int(expected_token_length):
        # enable_tensor_padding=True の場合、x_tst 側が右側に PAD で拡張されている可能性がある
        # その場合は、duration 上書きテンソルも右側に 0（未指定）を詰めて長さを合わせる
        # enable_tensor_padding=False の場合は想定外の不一致なので、早期にエラーで止める
        current_len = int(durations_frames_override.shape[2])
        expected_len = int(expected_token_length)
        if current_len < expected_len and enable_tensor_padding is True:
            pad_len = expected_len - current_len
            pad_frames = torch.zeros(
                (1, 1, pad_len),
                dtype=durations_frames_override.dtype,
                device=durations_frames_override.device,
            )
            pad_mask = torch.zeros(
                (1, 1, pad_len),
                dtype=durations_frames_override_mask.dtype,
                device=durations_frames_override_mask.device,
            )
            durations_frames_override = torch.cat(
                (durations_frames_override, pad_frames),
                dim=2,
            )
            durations_frames_override_mask = torch.cat(
                (durations_frames_override_mask, pad_mask),
                dim=2,
            )
        else:
            raise ValueError(
                "durations_frames_override token length mismatch. "
                f"expected: {expected_len}, actual: {current_len}"
            )

    return durations_frames_override, durations_frames_override_mask


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
    net_g: SynthesizerTrn | SynthesizerTrnJPExtra | SynthesizerTrnNanairo,
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    given_phone: list[str] | None = None,
    given_phone_length: list[float | None] | None = None,
    given_tone: list[int] | None = None,
    jtalk: OpenJTalk | None = None,
    use_fp16: bool = False,
    clear_cuda_cache: bool = True,
    enable_tensor_padding: bool = False,
    speaker_embedding: NDArray[Any] | torch.Tensor | None = None,
    g_adjust: NDArray[Any] | torch.Tensor | None = None,
) -> NDArray[np.float32]:
    """
    PyTorch 版音声合成モデルの推論を実行する関数。
    Nanairo（`use_speaker_adapter`）では、`speaker_embedding` に anime-speaker-embedding 由来のベクトル（例: `.spk.npy`）を渡す。
    """
    is_jp_extra_like_model = hps.is_jp_extra_like_model()
    is_nanairo_like_model = hps.is_nanairo_like_model()

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
            enable_tensor_padding=enable_tensor_padding,
        )

        # given_phone_length による duration 指定（オプション）
        # 呼び出し側が「特定音素だけ延ばす/縮める」用途を想定しているため、未指定 (None or <=0.0) は推定値を採用する
        # blank トークンは外部に公開するのが難しいため、基本は推定値を維持する（= マスク False のまま）
        durations_frames_override, durations_frames_override_mask = (
            _prepare_duration_frames_override_from_given_phone_length(
                given_phone=given_phone,
                given_phone_length=given_phone_length,
                expected_token_length=int(x_tst.shape[1]),
                skip_start=skip_start,
                skip_end=skip_end,
                length_scale=length_scale,
                hps=hps,
                device=device,
                enable_tensor_padding=enable_tensor_padding,
            )
        )

        if is_nanairo_like_model is False and (
            speaker_embedding is not None or g_adjust is not None
        ):
            raise ValueError(
                "speaker_embedding or g_adjust is only supported for Nanairo."
            )

        if is_jp_extra_like_model:
            if isinstance(speaker_embedding, np.ndarray):
                speaker_embedding = torch.from_numpy(speaker_embedding)
            if isinstance(g_adjust, np.ndarray):
                g_adjust = torch.from_numpy(g_adjust)
            if isinstance(speaker_embedding, torch.Tensor):
                speaker_embedding = speaker_embedding.to(device).float()
            if isinstance(g_adjust, torch.Tensor):
                g_adjust = g_adjust.to(device).float()
            if is_nanairo_like_model is True:
                output = cast(SynthesizerTrnNanairo, net_g).infer(
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
                    use_fp16=use_fp16,
                    durations_frames_override=durations_frames_override,
                    durations_frames_override_mask=durations_frames_override_mask,
                    speaker_embedding=speaker_embedding,
                    g_adjust=g_adjust,
                )
            else:
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
                    use_fp16=use_fp16,
                    durations_frames_override=durations_frames_override,
                    durations_frames_override_mask=durations_frames_override_mask,
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
                use_fp16=use_fp16,
                durations_frames_override=durations_frames_override,
                durations_frames_override_mask=durations_frames_override_mask,
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
            style_vec_tensor,
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
    net_g: SynthesizerTrn | SynthesizerTrnJPExtra | SynthesizerTrnNanairo,
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    given_phone: list[str] | None = None,
    given_phone_length: list[float | None] | None = None,
    given_tone: list[int] | None = None,
    jtalk: OpenJTalk | None = None,
    use_fp16: bool = False,
    clear_cuda_cache: bool = True,
    chunk_size: int = 65,  # 下記記事を参考に最適な値を調整
    overlap_size: int = 22,  # 下記記事を参照 (L=11, 11+11=22)
    enable_tensor_padding: bool = False,
) -> Iterator[NDArray[np.float32]]:
    """
    PyTorch 版音声合成モデルのストリーミング推論を実行する関数。
    Generator 部分のみストリーミング処理を行い、音声チャンクを逐次 yield する。
    ref: https://qiita.com/__dAi00/items/970f0fe66286510537dd
    """
    assert chunk_size > overlap_size, (
        f"chunk_size ({chunk_size}) must be larger than overlap_size ({overlap_size}) to avoid infinite loop."
    )
    assert chunk_size > 0 and overlap_size > 0, (
        "chunk_size and overlap_size must be positive."
    )
    assert overlap_size % 2 == 0, (
        "overlap_size must be even for proper margin calculation."
    )
    is_jp_extra_like_model = hps.is_jp_extra_like_model()
    is_nanairo_like_model = hps.is_nanairo_like_model()

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
            enable_tensor_padding=enable_tensor_padding,
        )

        # given_phone_length による duration 指定（オプション）
        # 呼び出し側が「特定音素だけ延ばす/縮める」用途を想定しているため、未指定 (None or <=0.0) は推定値を採用する
        # blank トークンは外部に公開するのが難しいため、基本は推定値を維持する（= マスク False のまま）
        durations_frames_override, durations_frames_override_mask = (
            _prepare_duration_frames_override_from_given_phone_length(
                given_phone=given_phone,
                given_phone_length=given_phone_length,
                expected_token_length=int(x_tst.shape[1]),
                skip_start=skip_start,
                skip_end=skip_end,
                length_scale=length_scale,
                hps=hps,
                device=device,
                enable_tensor_padding=enable_tensor_padding,
            )
        )

        # Generator 実行前の共通処理を実行
        if is_jp_extra_like_model:
            if is_nanairo_like_model is True:
                z, y_mask, g, attn, z_p, m_p, logs_p = cast(
                    SynthesizerTrnNanairo, net_g
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
                    use_fp16=use_fp16,
                    durations_frames_override=durations_frames_override,
                    durations_frames_override_mask=durations_frames_override_mask,
                )
            else:
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
                    use_fp16=use_fp16,
                    durations_frames_override=durations_frames_override,
                    durations_frames_override_mask=durations_frames_override_mask,
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
                use_fp16=use_fp16,
                durations_frames_override=durations_frames_override,
                durations_frames_override_mask=durations_frames_override_mask,
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
            if use_fp16 is True:
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
            style_vec_tensor,
            z,
            y_mask,
            g,
            z_input,
            attn,
            z_p,
            m_p,
            logs_p,
        )

        # CUDA メモリを解放する (デフォルトでは True)
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
