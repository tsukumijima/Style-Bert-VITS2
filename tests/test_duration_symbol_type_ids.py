"""
Nanairo duration 記号種別の回帰テスト。
"""

import torch

from style_bert_vits2.models.models_nanairo import SynthesizerTrn
from style_bert_vits2.models.tensor_padding import pad_sequence_tensor
from style_bert_vits2.nlp import phone_symbols_to_duration_symbol_type_ids
from style_bert_vits2.nlp.symbols import (
    DURATION_SYMBOL_TYPE_BLANK,
    DURATION_SYMBOL_TYPE_COMMA,
    DURATION_SYMBOL_TYPE_CONTENT,
    DURATION_SYMBOL_TYPE_COUNT,
    DURATION_SYMBOL_TYPE_ELLIPSIS_DOT,
    DURATION_SYMBOL_TYPE_EVENT_EMOJI,
    DURATION_SYMBOL_TYPE_EXCLAMATION,
    DURATION_SYMBOL_TYPE_HYPHEN,
    DURATION_SYMBOL_TYPE_PERIOD,
    DURATION_SYMBOL_TYPE_PROSODY_MARKER_EMOJI,
    DURATION_SYMBOL_TYPE_QUESTION,
    DURATION_SYMBOL_TYPE_QUOTE_BOUNDARY,
    NANAIRO_SYMBOLS,
)
from training.data_utils import TextAudioSpeakerCollate


def test_phone_symbols_to_duration_symbol_type_ids_classifies_boundaries() -> None:
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

    assert phone_symbols_to_duration_symbol_type_ids(phones) == [
        DURATION_SYMBOL_TYPE_BLANK,
        DURATION_SYMBOL_TYPE_CONTENT,
        DURATION_SYMBOL_TYPE_CONTENT,
        DURATION_SYMBOL_TYPE_COMMA,
        DURATION_SYMBOL_TYPE_ELLIPSIS_DOT,
        DURATION_SYMBOL_TYPE_ELLIPSIS_DOT,
        DURATION_SYMBOL_TYPE_ELLIPSIS_DOT,
        DURATION_SYMBOL_TYPE_QUESTION,
        DURATION_SYMBOL_TYPE_EXCLAMATION,
        DURATION_SYMBOL_TYPE_QUOTE_BOUNDARY,
        DURATION_SYMBOL_TYPE_HYPHEN,
        DURATION_SYMBOL_TYPE_EVENT_EMOJI,
        DURATION_SYMBOL_TYPE_PROSODY_MARKER_EMOJI,
        DURATION_SYMBOL_TYPE_BLANK,
    ]


def test_phone_symbols_to_duration_symbol_type_ids_keeps_single_period_separate() -> (
    None
):
    """単独の `.` と連続する `.` を別種別として扱う。"""

    phones = ["_", "a", ".", "i", ".", ".", "_"]

    assert phone_symbols_to_duration_symbol_type_ids(phones) == [
        DURATION_SYMBOL_TYPE_BLANK,
        DURATION_SYMBOL_TYPE_CONTENT,
        DURATION_SYMBOL_TYPE_PERIOD,
        DURATION_SYMBOL_TYPE_CONTENT,
        DURATION_SYMBOL_TYPE_ELLIPSIS_DOT,
        DURATION_SYMBOL_TYPE_ELLIPSIS_DOT,
        DURATION_SYMBOL_TYPE_BLANK,
    ]


def test_phone_symbols_to_duration_symbol_type_ids_inserts_blank_types() -> None:
    """`add_blank` 有効時は音素 ID と同じ規則で blank 種別を挿入する。"""

    duration_symbol_type_ids = phone_symbols_to_duration_symbol_type_ids(
        ["_", "a", "."],
        add_blank=True,
    )

    assert duration_symbol_type_ids == [
        DURATION_SYMBOL_TYPE_BLANK,
        DURATION_SYMBOL_TYPE_BLANK,
        DURATION_SYMBOL_TYPE_BLANK,
        DURATION_SYMBOL_TYPE_CONTENT,
        DURATION_SYMBOL_TYPE_BLANK,
        DURATION_SYMBOL_TYPE_PERIOD,
        DURATION_SYMBOL_TYPE_BLANK,
    ]


def test_nanairo_duration_symbol_type_count_default_matches_symbols() -> None:
    """`SynthesizerTrn` 直接生成時も duration 記号種別数の既定値を共有する。"""

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
        use_duration_symbol_type_embedding=True,
    )

    assert model.duration_symbol_type_emb is not None
    assert model.duration_symbol_type_emb.num_embeddings == DURATION_SYMBOL_TYPE_COUNT
    assert torch.count_nonzero(model.duration_symbol_type_emb.weight).item() == 0


def test_duration_symbol_type_padding_uses_blank_id() -> None:
    """推論用の固定長 padding では右側の補助系列を blank 種別で埋める。"""

    duration_symbol_type_ids = torch.tensor(
        [[DURATION_SYMBOL_TYPE_CONTENT] * 7],
        dtype=torch.long,
    )

    padded, actual_length = pad_sequence_tensor(
        duration_symbol_type_ids,
        length_dim=1,
        pool_type="test_duration_symbol_type_ids",
        use_pool=False,
        padding_value=DURATION_SYMBOL_TYPE_BLANK,
    )

    assert actual_length == 7
    assert padded.shape == (1, 8)
    assert torch.equal(padded[:, :7], duration_symbol_type_ids)
    assert torch.equal(
        padded[:, 7:],
        torch.full((1, 1), DURATION_SYMBOL_TYPE_BLANK, dtype=torch.long),
    )


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
        return_duration_symbol_type_ids=False,
    )

    batch = collate([(phones, spec, wav, sid, tone, language, ja_bert, style_vec)])

    assert len(batch) == 11


def test_text_audio_speaker_collate_adds_duration_symbol_type_ids_only_with_opt_in() -> (
    None
):
    """Nanairo 用の追加情報は `return_duration_symbol_type_ids` 有効時だけ末尾に追加する。"""

    phones = torch.arange(5, dtype=torch.long)
    spec = torch.zeros(3, 7)
    wav = torch.zeros(1, 20)
    sid = torch.tensor([0])
    tone = torch.zeros(5, dtype=torch.long)
    language = torch.zeros(5, dtype=torch.long)
    ja_bert = torch.zeros(1024, 5)
    style_vec = torch.zeros(256)
    duration_symbol_type_ids = torch.arange(5, dtype=torch.long)
    collate = TextAudioSpeakerCollate(
        use_jp_extra=True,
        use_speaker_embedding=False,
        return_duration_symbol_type_ids=True,
    )

    batch = collate(
        [
            (
                phones,
                spec,
                wav,
                sid,
                tone,
                language,
                ja_bert,
                style_vec,
                duration_symbol_type_ids,
            )
        ]
    )

    assert len(batch) == 12
    assert torch.equal(batch[-1][0, :5], duration_symbol_type_ids)
