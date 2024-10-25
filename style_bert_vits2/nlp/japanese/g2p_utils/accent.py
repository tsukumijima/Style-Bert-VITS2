import re

from style_bert_vits2.nlp.japanese.g2p import kata_to_phoneme_list


# 事前に正規表現パターンをコンパイル
__YOUON_PATTERN = re.compile("[ァィゥェォャュョヮ]+")


def convert_accent2hl(kana: str, accent: str) -> list[str]:
    """
    語句ごとのカタカナの読みとアクセント情報から、各音素に対応する High/Low アクセントのリストを生成する。
    アクセントは各音素に対して '0' (Low) または '1' (High) の文字列として割り当てられる。

    Args:
        kana (str): 語句ごとのカタカナの読み。拗音を含む場合は特別な処理が適用される。
        accent (str): 語句ごとのアクセント情報。以下のいずれかの値:
            - 'ALL_H': すべての音素を High ('1') に設定
            - 'ALL_L': すべての音素を Low ('0') に設定
            - '0': 平板型アクセント
            - '1': 頭高型アクセント
            - その他の数値: アクセント核の位置（1始まり）

    Returns:
        list[str]: 各音素に対応するアクセントのリスト。
            各要素は '0' (Low) または '1' (High) の文字列。
            リストの長さは kata_to_phoneme_list(kana) の返り値の長さと一致する。
    """

    # 0 or 1 のトーンのリスト
    accent_hl_list: list[str] = []

    mora = len(kana)

    # アクセントトーンをすべて 1 にする
    if accent == "ALL_H":
        for phone in kata_to_phoneme_list(kana):
            accent_hl_list.append("1")
        return accent_hl_list

    # アクセントトーンをすべて 0 にする
    elif accent == "ALL_L":
        for phone in kata_to_phoneme_list(kana):
            accent_hl_list.append("0")
        return accent_hl_list

    int_accent: int = int(accent)

    # 一文字の場合
    if mora == 1:
        if int_accent == 1:
            # 返した音素のリストの数だけ実行
            for phone in kata_to_phoneme_list(kana):
                accent_hl_list.append("1")
        else:
            for phone in kata_to_phoneme_list(kana):
                accent_hl_list.append("0")

    # 二文字で拗音が続く場合
    elif mora == 2 and __YOUON_PATTERN.fullmatch(kana[1]):
        if int_accent == 1:
            for phone in kata_to_phoneme_list(kana):
                accent_hl_list.append("1")
        else:
            for phone in kata_to_phoneme_list(kana):
                accent_hl_list.append("0")

    # アクセント核が平型の場合、平型にする
    elif int_accent == 0:
        # 2文字目に拗音が続く場合
        # 例　"キャ" など
        if __YOUON_PATTERN.fullmatch(kana[1]):
            # 先頭を追加
            for phone in kata_to_phoneme_list(kana[:2]):
                accent_hl_list.append("0")

            for phone in kata_to_phoneme_list(kana[2:]):
                accent_hl_list.append("1")

        # 拗音が続かない場合
        else:
            # 先頭を追加
            for phone in kata_to_phoneme_list(kana[0]):
                accent_hl_list.append("0")

            for phone in kata_to_phoneme_list(kana[1:]):
                accent_hl_list.append("1")

    # アクセント核が先頭の場合、先頭を先に音素に変換しアクセントを設定する
    # 例　"ア"  =>　"a" -> ("a", "1")
    elif int_accent == 1:
        # 2文字目に拗音が続く場合
        # 例　"キャ" など
        if __YOUON_PATTERN.fullmatch(kana[1]):
            # 2文字目までを追加
            for phone in kata_to_phoneme_list(kana[:2]):
                accent_hl_list.append("1")

            # 3文字目以降を追加
            for phone in kata_to_phoneme_list(kana[2:]):
                accent_hl_list.append("0")

        # 拗音が続かない場合
        else:
            # 1文字目を追加
            for phone in kata_to_phoneme_list(kana[0]):
                accent_hl_list.append("1")

            # 2文字目以降を追加
            for phone in kata_to_phoneme_list(kana[1:]):
                accent_hl_list.append("0")

    # アクセント核が先端と終端の間に位置する場合、先頭を先に音素に変換しアクセントを設定する
    elif int_accent < mora:
        # acc: 0スタートに直した泊で数えたアクセント数
        acc = int_accent - 1

        # 2文字目に拗音が続く場合でアクセント核が3文字目の場合
        # 例　"フェニックス" など
        if __YOUON_PATTERN.fullmatch(kana[1]) and int_accent == 2:
            # 2文字目までを追加
            for phone in kata_to_phoneme_list(kana[:2]):
                accent_hl_list.append("0")

            # 3文字目以降を追加
            for phone in kata_to_phoneme_list(kana[2]):
                accent_hl_list.append("1")

            for phone in kata_to_phoneme_list(kana[3:]):
                accent_hl_list.append("0")

        # アクセント核の2文字目に拗音が続く場合
        # 例　"インフェルノ" など
        elif __YOUON_PATTERN.fullmatch(kana[acc + 1]):
            # アクセント核までを追加
            for phone in kata_to_phoneme_list(kana[:acc]):
                accent_hl_list.append("0")

            # アクセント核以降を追加
            for phone in kata_to_phoneme_list(kana[acc : int_accent + 1]):
                accent_hl_list.append("1")

            for phone in kata_to_phoneme_list(kana[int_accent + 1 :]):
                accent_hl_list.append("0")

        else:
            # 先頭からアクセント核まで
            for phone in kata_to_phoneme_list(kana[:acc]):
                accent_hl_list.append("0")

            # アクセント核を追加
            for phone in kata_to_phoneme_list(kana[acc]):
                accent_hl_list.append("1")

            # アクセント核の一個先から終端まで
            for phone in kata_to_phoneme_list(kana[int_accent:]):
                accent_hl_list.append("0")

    # アクセント核が終端の場合、先頭を先に音素に変換しアクセントを設定する
    elif int_accent == mora:
        # acc: 0スタートに直した泊で数えたアクセント数
        acc = int_accent - 1

        for phone in kata_to_phoneme_list(kana[:acc]):
            accent_hl_list.append("0")

        # 終端を追加
        for phone in kata_to_phoneme_list(kana[acc]):
            accent_hl_list.append("1")

    return accent_hl_list
