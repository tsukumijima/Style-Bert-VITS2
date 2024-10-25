import re

import jaconv


# 事前に正規表現パターンをコンパイル
__YOMI_PATTERN = re.compile(r"..*/..*")


def apply_yomikata_diff(word_list: list[str], kata_list: list[str]) -> list[str]:
    """
    yomikata ライブラリによる文脈を考慮した読み仮名解決処理を適用し、
    単語単位の単語のカタカナ読みを改善する。特に同形異音語 (heteronyms) の正確な読み方を決定するのに役立つ。

    yomikata を利用するには yomikata パッケージをインストールする必要がある（ただし依存関係の競合が面倒…）。
    ref: https://github.com/passaglia/yomikata

    Args:
        word_list (list[str]): 単語単位の単語のリスト
        kata_list (list[str]): 単語単位の単語のカタカナ読みのリスト

    Returns:
        list[str]: yomikata によって修正された kata_list
    """

    try:
        from yomikata.dbert import dBert
    except ImportError:
        raise ImportError("yomikata package is not installed.")

    norm_text = "".join(word_list)

    reader = dBert()
    out_text: str = reader.furigana(norm_text)

    # 読みを振ったほうが良い文字が "{何/なに}が{何/なん}でも" のような形式になる
    # "{}/|" は正規化で消えるため text に混ざることはない
    # 正規化処理を変更したときはこの処理も合わせて変更する必要がある

    out_text = out_text.replace("{", "|")
    out_text = out_text.replace("}", "|")
    out_list = out_text.split("|")

    convert_list_text: list[str] = []
    convert_list_kana: list[str] = []

    for i in out_list:
        if __YOMI_PATTERN.fullmatch(i):

            word = i.split("/")
            convert_list_text.append(word[0])
            # 読みをカタカナからひらがなに変更
            kana = jaconv.hira2kata(word[1])
            convert_list_kana.append(kana)

    for i in range(0, len(word_list)):
        for ii in range(0, len(convert_list_text)):
            # 読みを上書きすべき文字があったら
            if word_list[i] == convert_list_text[ii]:
                kata_list[i] = convert_list_kana[ii]

    return kata_list
