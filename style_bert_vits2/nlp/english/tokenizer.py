from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from sentencepiece import sentencepiece_model_pb2
from tokenizers import SentencePieceUnigramTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def load_deberta_v3_sentencepiece_tokenizer(
    pretrained_model_name_or_path: str,
    cache_dir: str | None = None,
    revision: str = "main",
) -> PreTrainedTokenizerFast:
    """
    DeBERTa v3 の `spm.model` から v4 互換の Fast Tokenizer をロードする

    Args:
        pretrained_model_name_or_path (str): Hugging Face リポジトリ名または `spm.model` を含むローカルディレクトリ
        cache_dir (str | None): Hugging Face から取得する場合のキャッシュディレクトリ
        revision (str): Hugging Face 上の Git リビジョン

    Returns:
        PreTrainedTokenizerFast: SentencePiece の正規化を保持した Fast Tokenizer
    """

    # `tokenizers` はトップレベルの `sentencepiece_model_pb2` を要求するため、`sentencepiece` 同梱版を見せる
    sys.modules.setdefault("sentencepiece_model_pb2", sentencepiece_model_pb2)

    # Hugging Face リポジトリ名が渡された場合も `spm.model` を取得し、ローカルディレクトリ指定と同じ処理に流す
    if len(pretrained_model_name_or_path.split("/")) == 2:
        sentencepiece_model_path = Path(
            hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="spm.model",
                cache_dir=cache_dir,
                revision=revision,
            )
        )
    else:
        sentencepiece_model_path = Path(pretrained_model_name_or_path) / "spm.model"

    tokenizer = SentencePieceUnigramTokenizer.from_spm(str(sentencepiece_model_path))

    # DeBERTa v3 の特殊トークン ID は `spm.model` 内の ID と一致させる
    tokenizer.add_special_tokens(["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"])
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
    )
