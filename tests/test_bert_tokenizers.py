from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerFast

from convert_bert_onnx import convert_japanese_character_tokenizer
from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATHS, Languages
from style_bert_vits2.nlp import bert_models, onnx_bert_models


JP_BERT_MODEL_PATH = DEFAULT_BERT_MODEL_PATHS[Languages.JP]

JP_TOKENIZER_CASES = [
    ("今日はいい天気ですね", [1, 268, 55, 18, 9, 9, 455, 197, 13, 23, 305, 2]),
    (
        "Style-Bert-VITS2です",
        [
            1,
            129,
            78,
            306,
            122,
            56,
            102,
            227,
            56,
            91,
            78,
            102,
            360,
            172,
            160,
            129,
            34,
            13,
            23,
            2,
        ],
    ),
    ("ABC 123!?", [1, 118, 227, 141, 26, 34, 69, 173, 344, 2]),
    ("ＡＢＣ１２３", [1, 118, 227, 141, 26, 34, 69, 2]),
    ("髙﨑🍣", [1, 2770, 2842, 8245, 2]),
    ("こんにちは、世界。", [1, 37, 57, 8, 105, 18, 7, 267, 379, 10, 2]),
    ("空 白\t改行\nテスト", [1, 462, 623, 517, 114, 93, 29, 35, 2]),
    ("[MASK]今日は[MASK]", [1, 4, 268, 55, 18, 4, 2]),
    ("未知文字🧪テスト", [1, 785, 280, 284, 652, 10459, 93, 29, 35, 2]),
    (
        "カタカナ・ひらがな・漢字",
        [1, 81, 66, 81, 128, 45, 498, 33, 20, 21, 45, 1345, 652, 2],
    ),
    ("“引用”と—ダッシュ", [1, 849, 441, 116, 713, 16, 1671, 137, 48, 67, 136, 2]),
    ("🙂🙃", [1, 5186, 7809, 2]),
]

EN_TOKENIZER_CASES = [
    ("This is a test.", [1, 329, 269, 266, 1010, 260, 2]),
    (
        "Style-Bert-VITS2 is useful.",
        [1, 6780, 271, 90294, 271, 1989, 44681, 445, 269, 1772, 260, 2],
    ),
    ("Hello, world!", [1, 5365, 261, 447, 300, 2]),
    (
        "I can't believe it's 2026.",
        [1, 273, 295, 280, 297, 770, 278, 280, 268, 39119, 260, 2],
    ),
    (
        "New York-based AI costs $12.50.",
        [1, 485, 920, 271, 1173, 5536, 1294, 419, 1432, 260, 1794, 260, 2],
    ),
    ("Multiple   spaces\tand\nnewlines.", [1, 10189, 3654, 263, 353, 17480, 260, 2]),
    ("[MASK] token stays special.", [1, 128000, 10704, 7213, 779, 260, 2]),
    (
        "Café naïve façade coöperate.",
        [1, 16709, 34972, 29411, 1376, 16846, 4183, 5588, 260, 2],
    ),
    ("日本語 and English mixed.", [1, 507, 100868, 89387, 263, 1342, 3230, 260, 2]),
    (
        "email@example.com / https://example.com",
        [1, 871, 1683, 28748, 260, 549, 840, 3597, 294, 320, 320, 28748, 260, 549, 2],
    ),
    ("🙂 emoji test", [1, 11799, 30151, 1010, 2]),
    ("ＡＢＣ１２３ fullwidth", [1, 6783, 17319, 540, 29978, 2]),
]

ZH_TOKENIZER_CASES = [
    ("今天天气很好。", [101, 791, 1921, 1921, 3698, 2523, 1962, 511, 102]),
    (
        "Style-Bert-VITS2很好用。",
        [
            101,
            8969,
            118,
            8815,
            8716,
            118,
            10138,
            8723,
            8144,
            2523,
            1962,
            4500,
            511,
            102,
        ],
    ),
    ("你好，世界！", [101, 872, 1962, 8024, 686, 4518, 8013, 102]),
    ("今天 123 ABC。", [101, 791, 1921, 8604, 8425, 511, 102]),
    ("[MASK]天气[MASK]", [101, 103, 1921, 3698, 103, 102]),
    ("繁體字與简体字", [101, 5246, 7768, 2099, 5645, 5042, 860, 2099, 102]),
    ("空 白\t换行\n测试", [101, 4958, 4635, 2940, 6121, 3844, 6407, 102]),
    ("email@example.com", [101, 8307, 137, 9577, 8608, 10383, 119, 8134, 102]),
    ("🙂表情测试", [101, 100, 6134, 2658, 3844, 6407, 102]),
    ("ＡＢＣ１２３全角", [101, 8051, 12641, 10675, 8939, 8929, 9089, 1059, 6235, 102]),
    ("价格是12.50元。", [101, 817, 3419, 3221, 8110, 119, 8145, 1039, 511, 102]),
    ("中文-English混合。", [101, 704, 3152, 118, 8899, 3921, 1394, 511, 102]),
]


@pytest.fixture(autouse=True)
def unload_bert_tokenizers() -> None:
    """
    各テストで `bert_models` / `onnx_bert_models` のキャッシュ状態が混ざらないようにする。
    """

    for language in Languages:
        bert_models.unload_tokenizer(language)
        onnx_bert_models.unload_tokenizer(language)


@pytest.fixture
def japanese_fast_tokenizer(tmp_path: Path) -> PreTrainedTokenizerFast:
    """
    日本語 char-wwm 用 Fast Tokenizer JSON を生成してロードする。

    Args:
        tmp_path (Path): pytest が用意する一時ディレクトリ

    Returns:
        PreTrainedTokenizerFast: 検証対象の Fast Tokenizer
    """

    tokenizer_json_path = tmp_path / "tokenizer.json"
    convert_japanese_character_tokenizer(JP_BERT_MODEL_PATH).save(
        str(tokenizer_json_path)
    )
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json_path),
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )


@pytest.mark.parametrize(("text", "expected_input_ids"), JP_TOKENIZER_CASES)
def test_japanese_character_fast_tokenizer_matches_slow_tokenizer(
    japanese_fast_tokenizer: PreTrainedTokenizerFast,
    text: str,
    expected_input_ids: list[int],
) -> None:
    """
    日本語 char-wwm 用 Fast Tokenizer JSON が Slow Tokenizer と同じ ID 列を返すことを検証する。

    Args:
        japanese_fast_tokenizer (PreTrainedTokenizerFast): 検証対象の Fast Tokenizer
        text (str): 比較対象の入力テキスト
        expected_input_ids (list[int]): 既存挙動として期待する入力 ID 列
    """

    slow_tokenizer = bert_models.load_tokenizer(Languages.JP)
    slow_encoded = slow_tokenizer(text, return_token_type_ids=True)
    fast_encoded = japanese_fast_tokenizer(text, return_token_type_ids=True)

    assert slow_encoded.input_ids == expected_input_ids
    assert fast_encoded.input_ids == expected_input_ids
    assert fast_encoded.token_type_ids == [0] * len(expected_input_ids)
    assert japanese_fast_tokenizer.tokenize(text) == slow_tokenizer.tokenize(text)


def test_japanese_character_fast_tokenizer_pair_token_type_ids(
    japanese_fast_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """
    日本語 char-wwm 用 Fast Tokenizer JSON が文ペアの `token_type_ids` を保持することを検証する。

    Args:
        japanese_fast_tokenizer (PreTrainedTokenizerFast): 検証対象の Fast Tokenizer
    """

    encoded = japanese_fast_tokenizer(
        "文A",
        "文B",
        return_token_type_ids=True,
    )

    assert encoded.input_ids == [1, 284, 118, 2, 284, 227, 2]
    assert encoded.token_type_ids == [0, 0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize(("text", "expected_input_ids"), EN_TOKENIZER_CASES)
def test_english_tokenizer_ids_are_stable(
    text: str,
    expected_input_ids: list[int],
) -> None:
    """
    英語 BERT トークナイザーの代表入力 ID 列が変化していないことを検証する。

    Args:
        text (str): 比較対象の入力テキスト
        expected_input_ids (list[int]): 既存挙動として期待する入力 ID 列
    """

    for tokenizer in [
        bert_models.load_tokenizer(Languages.EN),
        onnx_bert_models.load_tokenizer(Languages.EN),
    ]:
        encoded = tokenizer(text, return_token_type_ids=True)
        assert encoded.input_ids == expected_input_ids
        assert encoded.token_type_ids == [0] * len(expected_input_ids)


def test_english_tokenizer_pair_token_type_ids() -> None:
    """
    英語 BERT トークナイザーが文ペアの `token_type_ids` を保持することを検証する。
    """

    for tokenizer in [
        bert_models.load_tokenizer(Languages.EN),
        onnx_bert_models.load_tokenizer(Languages.EN),
    ]:
        encoded = tokenizer("Sentence A", "Sentence B", return_token_type_ids=True)
        assert encoded.input_ids == [1, 42530, 336, 2, 42530, 736, 2]
        assert encoded.token_type_ids == [0, 0, 0, 0, 1, 1, 1]


def test_english_tokenizer_loads_from_sentencepiece_without_tokenizer_json(
    tmp_path: Path,
) -> None:
    """
    HF 直参照と同じ `spm.model` のみの構成でも v4 互換の ID 列になることを検証する

    Args:
        tmp_path (Path): pytest が用意する一時ディレクトリ
    """

    model_path = tmp_path / "deberta-v3-large"
    model_path.mkdir()
    shutil.copy(DEFAULT_BERT_MODEL_PATHS[Languages.EN] / "spm.model", model_path)

    for tokenizer in [
        bert_models.load_tokenizer(Languages.EN, str(model_path)),
        onnx_bert_models.load_tokenizer(Languages.EN, str(model_path)),
    ]:
        encoded = tokenizer("ＡＢＣ１２３ fullwidth", return_token_type_ids=True)
        assert encoded.input_ids == [1, 6783, 17319, 540, 29978, 2]
        assert encoded.token_type_ids == [0, 0, 0, 0, 0, 0]
        assert tokenizer.tokenize("[MASK] token stays special.") == [
            "[MASK]",
            "▁token",
            "▁stays",
            "▁special",
            ".",
        ]


@pytest.mark.parametrize(("text", "expected_input_ids"), ZH_TOKENIZER_CASES)
def test_chinese_tokenizer_ids_are_stable(
    text: str,
    expected_input_ids: list[int],
) -> None:
    """
    中国語 BERT トークナイザーの代表入力 ID 列が変化していないことを検証する。

    Args:
        text (str): 比較対象の入力テキスト
        expected_input_ids (list[int]): 既存挙動として期待する入力 ID 列
    """

    for tokenizer in [
        bert_models.load_tokenizer(Languages.ZH),
        onnx_bert_models.load_tokenizer(Languages.ZH),
    ]:
        encoded = tokenizer(text, return_token_type_ids=True)
        assert encoded.input_ids == expected_input_ids
        assert encoded.token_type_ids == [0] * len(expected_input_ids)


def test_chinese_tokenizer_pair_token_type_ids() -> None:
    """
    中国語 BERT トークナイザーが文ペアの `token_type_ids` を保持することを検証する。
    """

    for tokenizer in [
        bert_models.load_tokenizer(Languages.ZH),
        onnx_bert_models.load_tokenizer(Languages.ZH),
    ]:
        encoded = tokenizer("文A", "文B", return_token_type_ids=True)
        assert encoded.input_ids == [101, 3152, 143, 102, 3152, 144, 102]
        assert encoded.token_type_ids == [0, 0, 0, 0, 1, 1, 1]
