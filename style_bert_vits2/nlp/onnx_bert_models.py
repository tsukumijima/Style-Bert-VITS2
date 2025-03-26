"""
Style-Bert-VITS2 の ONNX 推論に必要な各言語ごとの ONNX 版 BERT モデルをロード/取得するためのモジュール。
このモジュールは style_bert_vits2.nlp.bert_models での実装を ONNX 推論向けに変更したもの。

オリジナルの Bert-VITS2 では各言語ごとの BERT モデルが初回インポート時にハードコードされたパスから「暗黙的に」ロードされているが、
場合によっては多重にロードされて非効率なほか、BERT モデルのロード元のパスがハードコードされているためライブラリ化ができない。

そこで、ライブラリの利用前に、音声合成に利用する言語の BERT モデルだけを「明示的に」ロードできるようにした。
一度 load_model/tokenizer() で当該言語の BERT モデルがロードされていれば、ライブラリ内部のどこからでもロード済みのモデル/トークナイザーを取得できる。
"""

import gc
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import onnxruntime
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    DebertaV2TokenizerFast,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from style_bert_vits2.constants import DEFAULT_ONNX_BERT_MODEL_PATHS, Languages
from style_bert_vits2.logging import logger


# 各言語ごとのロード済みの BERT モデルを格納する辞書
__loaded_models: dict[Languages, onnxruntime.InferenceSession] = {}

# 各言語ごとのロード済みの BERT トークナイザーを格納する辞書
__loaded_tokenizers: dict[
    Languages,
    Union[PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaV2TokenizerFast],
] = {}


def load_model(
    language: Languages,
    pretrained_model_name_or_path: Optional[str] = None,
    onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]] = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})],
    cache_dir: Optional[str] = None,
    revision: str = "main",
    enable_cpu_mem_arena: bool | None = None,
) -> onnxruntime.InferenceSession:  # fmt: skip
    """
    指定された言語の ONNX 版 BERT モデルをロードし、ロード済みの ONNX 版 BERT モデルを返す。
    一度ロードされていれば、ロード済みの ONNX 版 BERT モデルを即座に返す。
    ライブラリ利用時は常に必ず pretrain_model_name_or_path (Hugging Face のリポジトリ名 or ローカルのファイルパス) を指定する必要がある。
    ロードにはそれなりに時間がかかるため、ライブラリ利用前に明示的に pretrained_model_name_or_path を指定してロードしておくべき。
    cache_dir と revision は pretrain_model_name_or_path がリポジトリ名の場合のみ有効。

    Style-Bert-VITS2 では、ONNX 版 BERT モデルに下記の 3 つが利用されている。
    これ以外の ONNX 版 BERT モデルを指定した場合は正常に動作しない可能性が高い。
    - 日本語: tsukumijima/deberta-v2-large-japanese-char-wwm-onnx

    Args:
        language (Languages): ロードする学習済みモデルの対象言語
        pretrained_model_name_or_path (Optional[str]): ロードする学習済みモデルの名前またはパス。指定しない場合はデフォルトのパスが利用される (デフォルト: None)
        onnx_providers (list[str]): ONNX 推論で利用する ExecutionProvider (CPUExecutionProvider, CUDAExecutionProvider など)
        cache_dir (Optional[str]): モデルのキャッシュディレクトリ。指定しない場合はデフォルトのキャッシュディレクトリが利用される (デフォルト: None)
        revision (str): モデルの Hugging Face 上の Git リビジョン。指定しない場合は最新の main ブランチの内容が利用される (デフォルト: None)
        enable_cpu_mem_arena (bool | None): CPU 推論時にもメモリアリーナを有効化するかどうか。デフォルトでは GPU 推論時のみ有効化される (デフォルト: None)

    Returns:
        onnxruntime.InferenceSession: ロード済みの BERT モデル
    """

    # すでにロード済みの場合はそのまま返す
    if language in __loaded_models:
        return __loaded_models[language]

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        assert DEFAULT_ONNX_BERT_MODEL_PATHS[language].exists(), \
            f"The default {language.name} ONNX BERT model does not exist on the file system. Please specify the path to the pre-trained model."  # fmt: skip
        pretrained_model_name_or_path = str(DEFAULT_ONNX_BERT_MODEL_PATHS[language])

    # pretrained_model_name_or_path に Hugging Face のリポジトリ名が指定された場合 (aaaa/bbbb のフォーマットを想定):
    # 指定された revision の ONNX 版 BERT モデルを cache_dir にダウンロードする (既にダウンロード済みの場合は何も行われない)
    if len(pretrained_model_name_or_path.split("/")) == 2:
        model_path = Path(
            hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="model_fp16.onnx",
                cache_dir=cache_dir,
                revision=revision,
            )
        )
        # 英語用 BERT のみ、spm.model もダウンロードする
        # Fast 版の BERT トークナイザーでは不要なはずだが、念のため
        if language == Languages.EN:
            hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="spm.model",
                cache_dir=cache_dir,
                revision=revision,
            )
    # pretrained_model_name_or_path にファイルパスが指定された場合:
    # 既にダウンロード済みという前提のもと、モデルへのローカルパスを model_path に格納する
    else:
        model_path = Path(pretrained_model_name_or_path).resolve() / "model_fp16.onnx"

    # 推論時に一番優先される ExecutionProvider の名前を取得
    assert len(onnx_providers) > 0
    first_provider_name = (
        onnx_providers[0] if type(onnx_providers[0]) is str else onnx_providers[0][0]
    )

    # 推論セッションの設定
    sess_options = onnxruntime.SessionOptions()
    ## ONNX モデルの作成時にすでに onnxsim により最適化されていることから、ロード高速化のため最適化を無効にする
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL  # fmt: skip
    ## エラー以外のログを出力しない
    ## 本来は log_severity_level = 3 だけで効くはずだが、なぜか CUDA 系のログが抑制できないので set_default_logger_severity() も呼び出している
    sess_options.log_severity_level = 3
    onnxruntime.set_default_logger_severity(3)

    # CPU 推論時のみ enable_cpu_mem_arena を無効化し、BERT モデルの推論セッションより富豪的なメモリ消費を防止する
    ## 既に RunOptions の memory.enable_memory_arena_shrinkage や、ProviderOptions の "arena_extend_strategy": "kSameAsRequested" を指定して
    ## InferenceSession が構築するメモリアリーナを推論後に縮小するよう構成し、メモリアリーナによるメモリ消費量が漸進的に増加することを防いでいる
    ## しかし、CPU 推論時の BERT モデルに関しては入力長や入力内容次第では依然大量のメモリが確保される傾向にあるため、CPU 推論時のみメモリアリーナ自体を無効化する
    ## BERT 特徴量の抽出処理が 0.数秒遅くなるトレードオフがあるが、元々 CPU 推論は CUDA 推論よりかなり遅いこと、
    ## BERT 特徴量の抽出処理自体は音声合成処理よりも遥かに軽量なこと、低メモリ環境での OOM エラー回避の観点から有益だと判断した
    ## メモリアリーナを無効化することで、若干の速度低下と引き換えに、多量の推論処理を行ってもメモリリークのような挙動が発生しなくなる
    ## なお CUDA 推論時は独自に VRAM 管理が行われているようで、CPU 推論時のように過剰に VRAM が消費されることはない
    ## 明示的に enable_cpu_mem_arena が指定されている場合は、指定された値を利用する
    if enable_cpu_mem_arena is not None:
        sess_options.enable_cpu_mem_arena = enable_cpu_mem_arena
    ## 明示的に enable_cpu_mem_arena が指定されていない場合は、推論セッションが CPUExecutionProvider の場合のみメモリアリーナを無効化する
    elif first_provider_name == "CPUExecutionProvider":
        sess_options.enable_cpu_mem_arena = False

    # BERT モデルをロードし、辞書に格納して返す
    start_time = time.time()
    __loaded_models[language] = onnxruntime.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=onnx_providers,
    )
    logger.info(
        f"Loaded the {language.name} ONNX BERT model from {pretrained_model_name_or_path} ({time.time() - start_time:.2f}s)"
    )

    return __loaded_models[language]


def load_tokenizer(
    language: Languages,
    pretrained_model_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaV2TokenizerFast]:
    """
    指定された言語の ONNX 版 BERT トークナイザーをロードし、ロード済みの ONNX 版 BERT トークナイザーを返す。
    一度ロードされていれば、ロード済みの ONNX 版 BERT トークナイザーを即座に返す。
    ライブラリ利用時は常に必ず pretrain_model_name_or_path (Hugging Face のリポジトリ名 or ローカルのファイルパス) を指定する必要がある。
    ロードにはそれなりに時間がかかるため、ライブラリ利用前に明示的に pretrained_model_name_or_path を指定してロードしておくべき。
    cache_dir と revision は pretrain_model_name_or_path がリポジトリ名の場合のみ有効。

    Style-Bert-VITS2 では、ONNX 版 BERT モデルに下記の 3 つが利用されている。
    これ以外の ONNX 版 BERT モデルを指定した場合は正常に動作しない可能性が高い。
    - 日本語: tsukumijima/deberta-v2-large-japanese-char-wwm-onnx

    Args:
        language (Languages): ロードする学習済みモデルの対象言語
        pretrained_model_name_or_path (Optional[str]): ロードする学習済みモデルの名前またはパス。指定しない場合はデフォルトのパスが利用される (デフォルト: None)
        cache_dir (Optional[str]): モデルのキャッシュディレクトリ。指定しない場合はデフォルトのキャッシュディレクトリが利用される (デフォルト: None)
        revision (str): モデルの Hugging Face 上の Git リビジョン。指定しない場合は最新の main ブランチの内容が利用される (デフォルト: None)

    Returns:
        Union[PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaV2TokenizerFast]: ロード済みの BERT トークナイザー
    """

    # すでにロード済みの場合はそのまま返す
    if language in __loaded_tokenizers:
        return __loaded_tokenizers[language]

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        assert DEFAULT_ONNX_BERT_MODEL_PATHS[language].exists(), \
            f"The default {language.name} BERT tokenizer does not exist on the file system. Please specify the path to the pre-trained model."  # fmt: skip
        pretrained_model_name_or_path = str(DEFAULT_ONNX_BERT_MODEL_PATHS[language])

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
        f"Loaded the {language.name} ONNX BERT tokenizer from {pretrained_model_name_or_path}"
    )

    return __loaded_tokenizers[language]


def is_model_loaded(language: Languages) -> bool:
    """
    指定された言語の ONNX 版 BERT モデルがロード済みかどうかを返す。
    """

    return language in __loaded_models


def is_tokenizer_loaded(language: Languages) -> bool:
    """
    指定された言語の ONNX 版 BERT トークナイザーがロード済みかどうかを返す。
    """

    return language in __loaded_tokenizers


def unload_model(language: Languages) -> None:
    """
    指定された言語の ONNX 版 BERT モデルをアンロードする。

    Args:
        language (Languages): アンロードする BERT モデルの言語
    """

    if language in __loaded_models:
        del __loaded_models[language]
        gc.collect()
        logger.info(f"Unloaded the {language.name} ONNX BERT model")


def unload_tokenizer(language: Languages) -> None:
    """
    指定された言語の ONNX 版 BERT トークナイザーをアンロードする。

    Args:
        language (Languages): アンロードする BERT トークナイザーの言語
    """

    if language in __loaded_tokenizers:
        del __loaded_tokenizers[language]
        gc.collect()
        logger.info(f"Unloaded the {language.name} ONNX BERT tokenizer")


def unload_all_models() -> None:
    """
    すべての ONNX 版 BERT モデルをアンロードする。
    """

    for language in list(__loaded_models.keys()):
        unload_model(language)
    logger.info("Unloaded all ONNX BERT models")


def unload_all_tokenizers() -> None:
    """
    すべての ONNX 版 BERT トークナイザーをアンロードする。
    """

    for language in list(__loaded_tokenizers.keys()):
        unload_tokenizer(language)
    logger.info("Unloaded all ONNX BERT tokenizers")
