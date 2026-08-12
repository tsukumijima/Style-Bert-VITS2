"""
tsqyomi 有効時の日本語 g2p 回帰テスト。
"""

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from pyopenjtalk.tsqyomi import is_model_loaded, load_model, unload_model

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import clean_text_with_given_phone_tone
from style_bert_vits2.nlp.japanese.g2p import g2p
from style_bert_vits2.nlp.japanese.normalizer import normalize_text


@pytest.fixture(scope="module")
def tsqyomi_model_fixture() -> Iterator[None]:
    load_model(onnx_providers=["CPUExecutionProvider"])
    yield
    unload_model()


@dataclass(frozen=True)
class _Case:
    text: str
    expected_kana_normal: str
    expected_kana_tsqyomi: str


_TSQYOMI_READING_COMPARISON_CASES: tuple[_Case, ...] = (
    _Case(
        text="深夜の路地は人気が無くて怖い。",
        expected_kana_normal="シンヤノロジワニンキガナクテコワイ.",
        expected_kana_tsqyomi="シンヤノロジワヒトケガナクテコワイ.",
    ),
    _Case(
        text="辛いことだが仕方がない。",
        expected_kana_normal="ツライコトダガシカタガナイ.",
        expected_kana_tsqyomi="ツライコトダガシカタガナイ.",
    ),
    _Case(
        text="とても辛いカレーを食べた。",
        expected_kana_normal="トテモツライカレーヲタベタ.",
        expected_kana_tsqyomi="トテモカライカレーヲタベタ.",
    ),
    _Case(
        text="彼に敬意を表します。",
        expected_kana_normal="カレニケーイヲアラワシマス.",
        expected_kana_tsqyomi="カレニケーイヲヒョーシマス.",
    ),
    _Case(
        text="新しく金が発見された地に赴くにも金がかかる。",
        expected_kana_normal="アタラシクカネガハッケンサレタチニオモムクニモカネガカカル.",
        expected_kana_tsqyomi="アタラシクキンガハッケンサレタチニオモムクニモカネガカカル.",
    ),
    _Case(
        text="泥を被るという被害を被った。",
        expected_kana_normal="ドロヲコームルトイウヒガイヲコームッタ.",
        expected_kana_tsqyomi="ドロヲカブルトイウヒガイヲコームッタ.",
    ),
    _Case(
        text="カブトムシの立派な角に止まった小さな虫を、指で軽く弾く。",
        expected_kana_normal="カブトムシノリッパナカドニトマッタチーサナムシヲ,ユビデカルクヒク.",
        expected_kana_tsqyomi="カブトムシノリッパナツノニトマッタチーサナムシヲ,ユビデカルクハジク.",
    ),
    _Case(
        text="庭に植えた紅葉の木が立派に育ってきた。",
        expected_kana_normal="ニワニウエタコーヨーノキガリッパニソダッテキタ.",
        expected_kana_tsqyomi="ニワニウエタモミジノキガリッパニソダッテキタ.",
    ),
    _Case(
        text="診療は月・水・金です。",
        expected_kana_normal="シンリョーワツキ,ミズ,カネデス.",
        expected_kana_tsqyomi="シンリョーワゲツ,スイ,キンデス.",
    ),
    _Case(
        text="会議は火・木に開きます。",
        expected_kana_normal="カイギワヒ,キニヒラキマス.",
        expected_kana_tsqyomi="カイギワカ,モクニヒラキマス.",
    ),
)


def _extract_joined_sep_kata(text: str, *, use_tsqyomi: bool) -> str:
    _, _, _, _, _, sep_kata, _ = clean_text_with_given_phone_tone(
        text=text,
        language=Languages.JP,
        use_jp_extra=True,
        use_tsqyomi=use_tsqyomi,
        raise_yomi_error=False,
    )
    assert sep_kata is not None
    return "".join(sep_kata)


def test_g2p_requires_explicit_tsqyomi_model_loading() -> None:
    """tsqyomi 利用時は、呼び出し元プロセスでの明示的なモデルロードを要求する。"""

    if is_model_loaded() is True:
        unload_model()

    with pytest.raises(RuntimeError, match="tsqyomi model is not loaded"):
        g2p(
            normalize_text("人気の店です。"),
            use_tsqyomi=True,
        )


@pytest.mark.parametrize(
    "case",
    _TSQYOMI_READING_COMPARISON_CASES,
    ids=lambda case: case.text[:16],
)
def test_g2p_tsqyomi_changes_reading_from_dictionary_baseline(
    case: _Case,
    tsqyomi_model_fixture: None,
) -> None:
    """辞書のみ g2p と tsqyomi 有効 g2p が、文脈に応じて期待どおりに読み分ける。"""

    joined_sep_kata_normal = _extract_joined_sep_kata(case.text, use_tsqyomi=False)
    assert joined_sep_kata_normal == case.expected_kana_normal, (
        f"Unexpected dictionary-only reading. text: {case.text}, "
        f"actual: {joined_sep_kata_normal}, expected: {case.expected_kana_normal}"
    )

    joined_sep_kata_tsqyomi = _extract_joined_sep_kata(case.text, use_tsqyomi=True)
    assert joined_sep_kata_tsqyomi == case.expected_kana_tsqyomi, (
        f"Unexpected tsqyomi reading. text: {case.text}, "
        f"actual: {joined_sep_kata_tsqyomi}, expected: {case.expected_kana_tsqyomi}"
    )
