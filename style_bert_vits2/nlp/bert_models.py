"""
Style-Bert-VITS2 の学習・推論に必要な各言語ごとの BERT モデルをロード/取得するためのモジュール。

オリジナルの Bert-VITS2 では各言語ごとの BERT モデルが初回インポート時にハードコードされたパスから「暗黙的に」ロードされているが、
場合によっては多重にロードされて非効率なほか、BERT モデルのロード元のパスがハードコードされているためライブラリ化ができない。

そこで、ライブラリの利用前に、音声合成に利用する言語の BERT モデルだけを「明示的に」ロードできるようにした。
一度 load_model/tokenizer() で当該言語の BERT モデルがロードされていれば、ライブラリ内部のどこからでもロード済みのモデル/トークナイザーを取得できる。
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Literal

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DebertaV2Model,
    DebertaV2TokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATHS, Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import onnx_bert_models


if TYPE_CHECKING:
    import torch


# 各言語ごとのロード済みの BERT モデルを格納する辞書
__loaded_models: dict[Languages, PreTrainedModel | DebertaV2Model] = {}

# 各言語ごとのロード済みモデルの精度情報を格納する辞書
__loaded_model_dtypes: dict[Languages, Literal["fp32", "fp16", "int8"]] = {}

# 各言語ごとのロード済みの BERT トークナイザーを格納する辞書
__loaded_tokenizers: dict[
    Languages,
    PreTrainedTokenizer | PreTrainedTokenizerFast | DebertaV2TokenizerFast,
] = {}


def load_model(
    language: Languages,
    pretrained_model_name_or_path: str | None = None,
    device_map: str
    | dict[str, int | str | torch.device]
    | int
    | torch.device
    | None = None,
    cache_dir: str | None = None,
    revision: str = "main",
    use_fp16: bool = False,
    use_int8: bool = False,
    llm_int8_threshold: float = 6.0,
    llm_int8_skip_modules: list[str] | None = [
        "embeddings",
        "LayerNorm",
        "embed_proj",
        "dense",
    ],
) -> PreTrainedModel | DebertaV2Model:
    """
    指定された言語の BERT モデルをロードし、ロード済みの BERT モデルを返す。
    一度ロードされていれば、ロード済みの BERT モデルを即座に返す。
    ライブラリ利用時は常に必ず pretrain_model_name_or_path (Hugging Face のリポジトリ名 or ローカルのファイルパス) を指定する必要がある。
    ロードにはそれなりに時間がかかるため、ライブラリ利用前に明示的に pretrained_model_name_or_path を指定してロードしておくべき。
    device_map は既に指定された言語の BERT モデルがロードされている場合は効果がない。
    cache_dir と revision は pretrain_model_name_or_path がリポジトリ名の場合のみ有効。
    use_fp16 を True にすると torch_dtype=torch.float16 でロードされ、メモリ使用量を大幅に削減し推論を高速化できる。
    最新の GPU では FP16 による精度低下はほとんどないため、実用的な選択肢である。
    use_int8 を True にすると bitsandbytes による INT8 量子化でロードされ、さらなるメモリ削減が可能。
    ただし、8bit 量子化は GPU 必須で、わずかな精度低下が発生する可能性がある。

    Style-Bert-VITS2 では、BERT モデルに下記の 3 つが利用されている。
    これ以外の BERT モデルを指定した場合は正常に動作しない可能性が高い。
    - 日本語: ku-nlp/deberta-v2-large-japanese-char-wwm
    - 英語: microsoft/deberta-v3-large
    - 中国語: hfl/chinese-roberta-wwm-ext-large

    Args:
        language (Languages): ロードする学習済みモデルの対象言語
        pretrained_model_name_or_path (str | None): ロードする学習済みモデルの名前またはパス。指定しない場合はデフォルトのパスが利用される (デフォルト: None)
        device_map (str | None): accelerate を使用して高速にデバイスにモデルをロードするためのデバイスマップ。
            指定しない場合は通常のモデルロード処理になる (デフォルト: None)
            ref: https://huggingface.co/docs/accelerate/usage_guides/big_modeling
        cache_dir (str | None): モデルのキャッシュディレクトリ。指定しない場合はデフォルトのキャッシュディレクトリが利用される (デフォルト: None)
        revision (str): モデルの Hugging Face 上の Git リビジョン。指定しない場合は最新の main ブランチの内容が利用される (デフォルト: None)
        use_fp16 (bool): FP16 (半精度) でモデルをロードするかどうか。True の場合、メモリ使用量を削減し推論を高速化する (デフォルト: False)
        use_int8 (bool): INT8 (8bit) 量子化でモデルをロードするかどうか。True の場合、bitsandbytes を使用してメモリ使用量を大幅に削減する。GPU 必須 (デフォルト: False)
        llm_int8_threshold (float): LLM.int8 の外れ値判定しきい値。値を下げると精度が向上するが速度が低下する (デフォルト: 6.0)
        llm_int8_skip_modules (list[str] | None): 8bit 量子化をスキップするモジュール名のリスト。埋め込み層や LayerNorm など重要層を指定 (デフォルト: ["embeddings", "LayerNorm", "embed_proj", "dense"])

    Returns:
        PreTrainedModel | DebertaV2Model: ロード済みの BERT モデル
    """

    import torch

    # すでにロード済みの場合はそのまま返す
    if language in __loaded_models:
        return __loaded_models[language]

    # use_fp16 と use_int8 が同時に指定されている場合はエラー
    if use_fp16 and use_int8:
        raise ValueError("use_fp16 and use_int8 cannot be True at the same time")

    # 8bit 量子化は CUDA が必須
    if use_int8 and not torch.cuda.is_available():
        raise RuntimeError("8bit quantization requires GPU (CUDA) support")

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        assert DEFAULT_BERT_MODEL_PATHS[language].exists(), \
            f"The default {language.name} BERT model does not exist on the file system. Please specify the path to the pre-trained model."  # fmt: skip
        pretrained_model_name_or_path = str(DEFAULT_BERT_MODEL_PATHS[language])

    # 量子化設定
    if use_int8 is True:
        # use_int8 が True の場合のみ BitsAndBytesConfig をインポートする
        from transformers import BitsAndBytesConfig

        # 8bit 量子化時は torch_dtype が必ず float16 でなければならない
        torch_dtype = torch.float16
        # デフォルトのスキップモジュール設定
        if llm_int8_skip_modules is None:
            llm_int8_skip_modules = ["embeddings", "LayerNorm"]
        # 8bit 量子化の設定
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=llm_int8_threshold,
            llm_int8_skip_modules=llm_int8_skip_modules,
            llm_int8_enable_fp32_cpu_offload=False,
        )
    elif use_fp16 is True:
        # FP16 量子化時は torch_dtype が必ず float16 でなければならない
        torch_dtype = torch.float16
        # BitsAndBytesConfig は不要
        quantization_config = None
    else:
        # FP32 推論時は torch_dtype を指定しない
        torch_dtype = None
        # BitsAndBytesConfig は不要
        quantization_config = None

    # BERT モデルをロードし、辞書に格納して返す
    ## 日本語または英語のみ DebertaV2Model でロードする必要がある
    start_time = time.time()
    if language == Languages.JP or language == Languages.EN:
        __loaded_models[language] = DebertaV2Model.from_pretrained(
            pretrained_model_name_or_path,
            device_map=device_map,
            cache_dir=cache_dir,
            revision=revision,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,  # 常に True にしてメモリ効率を向上
        )
    else:
        __loaded_models[language] = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name_or_path,
            device_map=device_map,
            cache_dir=cache_dir,
            revision=revision,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,  # 常に True にしてメモリ効率を向上
        )

    # ロード済みモデルの精度情報を記録
    if use_int8:
        current_dtype_key = "int8"
    elif use_fp16:
        current_dtype_key = "fp16"
    else:
        current_dtype_key = "fp32"
    __loaded_model_dtypes[language] = current_dtype_key

    if use_int8:
        precision_info = " (INT8)"
    elif use_fp16:
        precision_info = " (FP16)"
    else:
        precision_info = " (default precision)"

    logger.info(
        f"Loaded the {language.name} BERT model from {pretrained_model_name_or_path}{precision_info} ({time.time() - start_time:.2f}s)"
    )

    return __loaded_models[language]


def load_tokenizer(
    language: Languages,
    pretrained_model_name_or_path: str | None = None,
    cache_dir: str | None = None,
    revision: str = "main",
) -> PreTrainedTokenizer | PreTrainedTokenizerFast | DebertaV2TokenizerFast:
    """
    指定された言語の BERT トークナイザーをロードし、ロード済みの BERT トークナイザーを返す。
    一度ロードされていれば、ロード済みの BERT トークナイザーを即座に返す。
    ライブラリ利用時は常に必ず pretrain_model_name_or_path (Hugging Face のリポジトリ名 or ローカルのファイルパス) を指定する必要がある。
    ロードにはそれなりに時間がかかるため、ライブラリ利用前に明示的に pretrained_model_name_or_path を指定してロードしておくべき。
    cache_dir と revision は pretrain_model_name_or_path がリポジトリ名の場合のみ有効。

    Style-Bert-VITS2 では、BERT モデルに下記の 3 つが利用されている。
    これ以外の BERT モデルを指定した場合は正常に動作しない可能性が高い。
    - 日本語: ku-nlp/deberta-v2-large-japanese-char-wwm
    - 英語: microsoft/deberta-v3-large
    - 中国語: hfl/chinese-roberta-wwm-ext-large

    Args:
        language (Languages): ロードする学習済みモデルの対象言語
        pretrained_model_name_or_path (str | None): ロードする学習済みモデルの名前またはパス。指定しない場合はデフォルトのパスが利用される (デフォルト: None)
        cache_dir (str | None): モデルのキャッシュディレクトリ。指定しない場合はデフォルトのキャッシュディレクトリが利用される (デフォルト: None)
        revision (str): モデルの Hugging Face 上の Git リビジョン。指定しない場合は最新の main ブランチの内容が利用される (デフォルト: None)

    Returns:
        PreTrainedTokenizer | PreTrainedTokenizerFast | DebertaV2TokenizerFast: ロード済みの BERT トークナイザー
    """

    # すでにロード済みの場合はそのまま返す
    if language in __loaded_tokenizers:
        return __loaded_tokenizers[language]

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        # ライブラリ利用時、特例的にこの状況で ONNX 版 BERT トークナイザーがロードされている場合はそのまま返す
        ## ONNX 版 BERT トークナイザー単独で g2p 処理を行うために必要 (各言語の g2p.py はこの関数に依存している)
        ## 設計的には微妙だがこの方が差異を吸収できて手っ取り早い
        if DEFAULT_BERT_MODEL_PATHS[language].exists() is False and onnx_bert_models.is_tokenizer_loaded(language):  # fmt: skip
            return onnx_bert_models.load_tokenizer(language)
        assert DEFAULT_BERT_MODEL_PATHS[language].exists(), \
            f"The default {language.name} BERT tokenizer does not exist on the file system. Please specify the path to the pre-trained model."  # fmt: skip
        pretrained_model_name_or_path = str(DEFAULT_BERT_MODEL_PATHS[language])

    # BERT トークナイザーをロードし、辞書に格納して返す
    ## 英語のみ DebertaV2TokenizerFast でロードする必要がある
    if language == Languages.EN:
        __loaded_tokenizers[language] = DebertaV2TokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
        )
    else:
        __loaded_tokenizers[language] = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            use_fast=True,  # デフォルトで True だが念のため明示的に指定
        )
    logger.info(
        f"Loaded the {language.name} BERT tokenizer from {pretrained_model_name_or_path}"
    )

    return __loaded_tokenizers[language]


def transfer_model(language: Languages, device: str) -> None:
    """
    指定された言語の BERT モデルを、指定されたデバイスに移動する。
    モデルのロード後に推論デバイスを変更したい場合に利用する。
    既に指定されたデバイスにモデルがロードされている場合は何も行われない。

    Args:
        language (Languages): モデルを移動する言語
        device (str): モデルを移動するデバイス
    """

    if language not in __loaded_models:
        raise ValueError(f"BERT model for {language.name} is not loaded.")

    # 既に指定されたデバイスにモデルがロードされている場合は何もしない
    # ex: current_device="cuda:0", device="cuda" → 何もしない
    # ex: current_device="cuda:0", device="cpu" → モデルを CPU に移動
    current_device = str(__loaded_models[language].device)
    if current_device.startswith(device):
        return

    __loaded_models[language].to(device)  # type: ignore
    logger.info(
        f"Transferred the {language.name} BERT model from {current_device} to {device}"
    )


def is_model_loaded(language: Languages) -> bool:
    """
    指定された言語の BERT モデルがロード済みかどうかを返す。
    """

    return language in __loaded_models


def is_tokenizer_loaded(language: Languages) -> bool:
    """
    指定された言語の BERT トークナイザーがロード済みかどうかを返す。
    """

    return language in __loaded_tokenizers


def get_model_dtype(language: Languages) -> Literal["fp32", "fp16", "int8"] | None:
    """
    指定された言語の BERT モデルの精度情報を返す。
    モデルがロードされていない場合は None を返す。

    Args:
        language (Languages): 精度情報を取得する言語

    Returns:
        Literal["fp32", "fp16", "int8"] | None: 'fp32', 'fp16', 'int8' (モデルがロードされていない場合は None)
    """

    return __loaded_model_dtypes.get(language)


def unload_model(language: Languages) -> None:
    """
    指定された言語の BERT モデルをアンロードする。

    Args:
        language (Languages): アンロードする BERT モデルの言語
    """

    import torch

    if language in __loaded_models:
        del __loaded_models[language]
        __loaded_model_dtypes.pop(language, None)  # 精度情報をクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Unloaded the {language.name} BERT model")


def unload_tokenizer(language: Languages) -> None:
    """
    指定された言語の BERT トークナイザーをアンロードする。

    Args:
        language (Languages): アンロードする BERT トークナイザーの言語
    """

    if language in __loaded_tokenizers:
        del __loaded_tokenizers[language]
        gc.collect()
        logger.info(f"Unloaded the {language.name} BERT tokenizer")


def unload_all_models() -> None:
    """
    すべての BERT モデルをアンロードする。
    """

    for language in list(__loaded_models.keys()):
        unload_model(language)
    __loaded_model_dtypes.clear()  # 念のため、全ての精度情報もクリア
    logger.info("Unloaded all BERT models")


def unload_all_tokenizers() -> None:
    """
    すべての BERT トークナイザーをアンロードする。
    """

    for language in list(__loaded_tokenizers.keys()):
        unload_tokenizer(language)
    logger.info("Unloaded all BERT tokenizers")
