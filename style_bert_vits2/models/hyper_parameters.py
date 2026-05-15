"""
Style-Bert-VITS2 モデルのハイパーパラメータを表す Pydantic モデル。
デフォルト値は configs/config_jp_extra.json 内の定義と概ね同一で、
万が一ロードした config.json に存在しないキーがあった際のフェイルセーフとして適用される。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from style_bert_vits2.logging import logger


class HyperParametersTrain(BaseModel):
    log_interval: int = 200
    eval_interval: int = 1000
    seed: int = 42
    epochs: int = 1000
    learning_rate: float = 0.0001
    betas: tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    batch_size: int = 2
    bf16_run: bool = False
    fp16_run: bool = False
    lr_decay: float = 0.99996
    segment_size: int = 16384
    init_lr_ratio: int = 1
    warmup_epochs: int = 0
    c_mel: int = 45
    c_kl: float = 1.0
    c_commit: int = 100
    c_teacher: float = 0.1
    c_delta_l2: float = 0.01
    skip_optimizer: bool = False
    freeze_ZH_bert: bool = False
    freeze_JP_bert: bool = False
    freeze_EN_bert: bool = False
    freeze_emo: bool = False
    freeze_style: bool = False
    freeze_decoder: bool = False
    train_speaker_adapter_only: bool = False
    disable_discriminators_for_adapter: bool = True
    # 学習中に Adapter 出力分散比がこの値を下回ったら警告ログを出す閾値 (early stopping のシグナル)
    adapter_min_variance_ratio_warning: float = 0.5
    # 主成分別分散モニタで保持する PC 数 (PCA 上位 k 個の分散比を TensorBoard に記録)
    pc_variance_monitor_k: int = 8


class HyperParametersData(BaseModel):
    # use_jp_extra フィールドが存在しない旧モデルとの互換性のために False をデフォルト値とする
    use_jp_extra: bool = False
    # use_nanairo フィールドが存在しない旧モデルとの互換性のために False をデフォルト値とする
    use_nanairo: bool = False
    training_files: str = "Data/Dummy/train.list"
    validation_files: str = "Data/Dummy/val.list"
    max_wav_value: float = 32768.0
    sampling_rate: int = 44100
    filter_length: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    n_mel_channels: int = 128
    mel_fmin: float = 0.0
    mel_fmax: float | None = None
    add_blank: bool = True
    n_speakers: int = 1
    cleaned_text: bool = True
    spk2id: dict[str, int] = {
        "Dummy": 0,
    }
    num_styles: int = 1
    style2id: dict[str, int] = {
        "Neutral": 0,
    }
    use_speaker_embedding: bool = False


class HyperParametersModelSLM(BaseModel):
    model: str = "./slm/wavlm-base-plus"
    sr: int = 16000
    hidden: int = 768
    nlayers: int = 13
    initial_channel: int = 64


class HyperParametersModel(BaseModel):
    use_spk_conditioned_encoder: bool = True
    use_noise_scaled_mas: bool = True
    use_mel_posterior_encoder: bool = False
    use_duration_discriminator: bool = False
    use_wavlm_discriminator: bool = True
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = [3, 7, 11]
    resblock_dilation_sizes: list[list[int]] = [
        [1, 3, 5],
        [1, 3, 5],
        [1, 3, 5],
    ]
    upsample_rates: list[int] = [8, 8, 2, 2, 2]
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list[int] = [16, 16, 8, 2, 2]
    n_layers_q: int = 3
    use_spectral_norm: bool = False
    gin_channels: int = 512
    use_transformer_duration_predictor: bool = False
    duration_filter_channels: int = 192
    use_duration_token_type_embedding: bool = False
    duration_token_type_count: int = 4
    use_speaker_adapter: bool = False
    speaker_adapter_input_dim: int = 384
    speaker_adapter_bottleneck_dim: int = 96
    slm: HyperParametersModelSLM = HyperParametersModelSLM()


class HyperParameters(BaseModel):
    model_name: str = "Dummy"
    version: str = "2.0-JP-Extra"
    train: HyperParametersTrain = HyperParametersTrain()
    data: HyperParametersData = HyperParametersData()
    model: HyperParametersModel = HyperParametersModel()

    # model_ 以下を Pydantic の保護対象から除外する
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="after")
    def normalize_model_flags(self) -> HyperParameters:
        """
        Nanairo / JP-Extra の設定整合性を正規化する。

        Returns:
            HyperParameters: 正規化済みのハイパーパラメータ
        """

        # Nanairo は JP-Extra ベースなので、Nanairo 互換なら自動的に JP-Extra 互換とする
        if self.is_nanairo_like_model() is True and self.data.use_jp_extra is not True:
            logger.warning(
                "Nanairo-compatible model detected while use_jp_extra is disabled. Forcing use_jp_extra to True."
            )
            self.data.use_jp_extra = True

        # version 文字列が Nanairo を示しているのに data.use_nanairo が有効でないと、bert_gen / DataLoader / 推論で語彙や NLP の経路が不一致になる
        # この不整合を放置すると学習がサイレントに壊れる可能性があるため、警告を出したうえで True に補正する
        if self.version.endswith("Nanairo") and self.data.use_nanairo is not True:
            logger.warning(
                "Model version string ends with 'Nanairo' but data.use_nanairo is false. "
                "Forcing data.use_nanairo to True so bert_gen, training, and inference agree on Nanairo NLP and vocabulary."
            )
            self.data.use_nanairo = True

        return self

    def is_jp_extra_like_model(self) -> bool:
        """
        JP-Extra 互換モデルかどうかを判定する。

        Returns:
            bool: JP-Extra 互換モデルかどうか
        """

        # 基本的にはハイパーパラメータのこれが明示的に True かで判断できるはず
        if self.data.use_jp_extra is True:
            return True

        # 当該エントリが存在しない場合にも、バージョン側で JP-Extra と識別可能であれば True とする
        return self.version.endswith("JP-Extra") or self.version.endswith("Nanairo")

    def is_nanairo_like_model(self) -> bool:
        """
        Nanairo 互換モデルかどうかを判定する。

        Returns:
            bool: Nanairo 互換モデルかどうか
        """

        # 基本的にはハイパーパラメータのこれが明示的に True かで判断できるはず
        if self.data.use_nanairo is True:
            return True

        # 当該エントリが存在しない場合にも、バージョン側で Nanairo と識別可能であれば True とする
        return self.version.endswith("Nanairo")

    @staticmethod
    def load_from_json(json_path: str | Path) -> HyperParameters:
        """
        与えられた JSON ファイルからハイパーパラメータを読み込む。

        Args:
            json_path (str | Path): JSON ファイルのパス

        Returns:
            HyperParameters: ハイパーパラメータ
        """

        with open(json_path, encoding="utf-8") as f:
            return HyperParameters.model_validate_json(f.read())
