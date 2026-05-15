"""
Nanairo duration トークン種別の回帰テスト。
"""

import torch

from style_bert_vits2.models.models_nanairo import SynthesizerTrn
from style_bert_vits2.nlp import phone_symbols_to_duration_token_types
from style_bert_vits2.nlp.symbols import (
    DURATION_TOKEN_TYPE_BLANK,
    DURATION_TOKEN_TYPE_COMMA,
    DURATION_TOKEN_TYPE_CONTENT,
    DURATION_TOKEN_TYPE_COUNT,
    DURATION_TOKEN_TYPE_ELLIPSIS_DOT,
    DURATION_TOKEN_TYPE_EVENT_EMOJI,
    DURATION_TOKEN_TYPE_EXCLAMATION,
    DURATION_TOKEN_TYPE_HYPHEN,
    DURATION_TOKEN_TYPE_PERIOD,
    DURATION_TOKEN_TYPE_PROSODY_MARKER_EMOJI,
    DURATION_TOKEN_TYPE_QUESTION,
    DURATION_TOKEN_TYPE_QUOTE_BOUNDARY,
    NANAIRO_SYMBOLS,
)
from training.data_utils import TextAudioSpeakerCollate


def test_phone_symbols_to_duration_token_types_classifies_boundaries() -> None:
    """句読点・境界記号・絵文字を duration 用の粗い種別に変換する。"""

    phones = [
        "_",
        "s",
        "o",
        ",",
        ".",
        ".",
        ".",
        "?",
        "!",
        "'",
        "-",
        "💋",
        "😆",
        "_",
    ]

    assert phone_symbols_to_duration_token_types(phones) == [
        DURATION_TOKEN_TYPE_BLANK,
        DURATION_TOKEN_TYPE_CONTENT,
        DURATION_TOKEN_TYPE_CONTENT,
        DURATION_TOKEN_TYPE_COMMA,
        DURATION_TOKEN_TYPE_ELLIPSIS_DOT,
        DURATION_TOKEN_TYPE_ELLIPSIS_DOT,
        DURATION_TOKEN_TYPE_ELLIPSIS_DOT,
        DURATION_TOKEN_TYPE_QUESTION,
        DURATION_TOKEN_TYPE_EXCLAMATION,
        DURATION_TOKEN_TYPE_QUOTE_BOUNDARY,
        DURATION_TOKEN_TYPE_HYPHEN,
        DURATION_TOKEN_TYPE_EVENT_EMOJI,
        DURATION_TOKEN_TYPE_PROSODY_MARKER_EMOJI,
        DURATION_TOKEN_TYPE_BLANK,
    ]


def test_phone_symbols_to_duration_token_types_keeps_single_period_separate() -> None:
    """単独の `.` と連続する `.` を別種別として扱う。"""

    phones = ["_", "a", ".", "i", ".", ".", "_"]

    assert phone_symbols_to_duration_token_types(phones) == [
        DURATION_TOKEN_TYPE_BLANK,
        DURATION_TOKEN_TYPE_CONTENT,
        DURATION_TOKEN_TYPE_PERIOD,
        DURATION_TOKEN_TYPE_CONTENT,
        DURATION_TOKEN_TYPE_ELLIPSIS_DOT,
        DURATION_TOKEN_TYPE_ELLIPSIS_DOT,
        DURATION_TOKEN_TYPE_BLANK,
    ]


def test_phone_symbols_to_duration_token_types_inserts_blank_types() -> None:
    """`add_blank` 有効時は音素 ID と同じ規則で blank 種別を挿入する。"""

    token_types = phone_symbols_to_duration_token_types(
        ["_", "a", "."],
        add_blank=True,
    )

    assert token_types == [
        DURATION_TOKEN_TYPE_BLANK,
        DURATION_TOKEN_TYPE_BLANK,
        DURATION_TOKEN_TYPE_BLANK,
        DURATION_TOKEN_TYPE_CONTENT,
        DURATION_TOKEN_TYPE_BLANK,
        DURATION_TOKEN_TYPE_PERIOD,
        DURATION_TOKEN_TYPE_BLANK,
    ]


def test_nanairo_duration_token_type_count_default_matches_symbols() -> None:
    """`SynthesizerTrn` 直接生成時も duration トークン種別数の既定値を共有する。"""

    model = SynthesizerTrn(
        n_vocab=len(NANAIRO_SYMBOLS),
        spec_channels=5,
        segment_size=4,
        n_speakers=1,
        inter_channels=8,
        hidden_channels=16,
        filter_channels=16,
        n_heads=2,
        n_layers=3,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]],
        upsample_rates=[2],
        upsample_initial_channel=16,
        upsample_kernel_sizes=[4],
        gin_channels=8,
        use_duration_token_type_embedding=True,
    )

    assert model.duration_token_type_emb is not None
    assert model.duration_token_type_emb.num_embeddings == DURATION_TOKEN_TYPE_COUNT


def test_text_audio_speaker_collate_keeps_jp_extra_batch_shape_without_opt_in() -> None:
    """Nanairo 用の追加情報を明示しない限り、JP-Extra 互換の戻り値を維持する。"""

    phones = torch.arange(5, dtype=torch.long)
    spec = torch.zeros(3, 7)
    wav = torch.zeros(1, 20)
    sid = torch.tensor([0])
    tone = torch.zeros(5, dtype=torch.long)
    language = torch.zeros(5, dtype=torch.long)
    ja_bert = torch.zeros(1024, 5)
    style_vec = torch.zeros(256)
    collate = TextAudioSpeakerCollate(
        use_jp_extra=True,
        use_speaker_embedding=False,
        return_duration_token_types=False,
    )

    batch = collate([(phones, spec, wav, sid, tone, language, ja_bert, style_vec)])

    assert len(batch) == 11


def test_text_audio_speaker_collate_adds_duration_token_types_only_with_opt_in() -> (
    None
):
    """Nanairo 用の追加情報は `return_duration_token_types` 有効時だけ末尾に追加する。"""

    phones = torch.arange(5, dtype=torch.long)
    spec = torch.zeros(3, 7)
    wav = torch.zeros(1, 20)
    sid = torch.tensor([0])
    tone = torch.zeros(5, dtype=torch.long)
    language = torch.zeros(5, dtype=torch.long)
    ja_bert = torch.zeros(1024, 5)
    style_vec = torch.zeros(256)
    token_types = torch.arange(5, dtype=torch.long)
    collate = TextAudioSpeakerCollate(
        use_jp_extra=True,
        use_speaker_embedding=False,
        return_duration_token_types=True,
    )

    batch = collate(
        [(phones, spec, wav, sid, tone, language, ja_bert, style_vec, token_types)]
    )

    assert len(batch) == 12
    assert torch.equal(batch[-1][0, :5], token_types)
