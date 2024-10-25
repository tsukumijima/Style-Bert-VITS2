import os
import re
from pathlib import Path


def analyze_text_with_fugashi(
    text: str, dict_path: Path | None = None, user_dict_path: Path | None = None
) -> tuple[list[str], list[str], list[str | list[str]], list[str]]:
    """
    fugashi でテキストを解析し、語句ごとの単語、カタカナ読み、アクセント、品詞を取得する。

    Args:
        text (str): テキスト
        dict_path (Path | None, optional): 辞書のパス。Defaults to None.
        user_dict_path (Path | None, optional): ユーザー辞書のパス。Defaults to None.

    Returns:
        tuple[list[str], list[str], list[str | list[str]], list[str]]:
        語句ごとの単語のリスト、カタカナ読みのリスト、アクセントのリスト、品詞のリスト
    """

    try:
        from fugashi import GenericTagger, Tagger, try_import_unidic
    except ImportError:
        raise ImportError("fugashi package is not installed.")

    # ユーザー辞書が指定されている場合
    if user_dict_path is not None:

        # ユーザー辞書の保存先ディレクトリ
        user_dict_path_str: str = str(user_dict_path)

        # システム辞書が指定されていない場合、unidic パッケージに含まれているシステム辞書を使う
        if dict_path is None:
            dict_path_str = try_import_unidic()
            dicrc_path = Path(dict_path_str) / "dicrc"
        else:
            dict_path_str = str(dict_path)
            dicrc_path = dict_path / "dicrc"

        # Windows 環境だと \ が途中でエスケープされるバグがあるため二重にする
        if os.name == "nt":
            user_dict_path_str: str = str(user_dict_path).replace("\\", "\\\\")
            dict_path_str: str = dict_path_str.replace("\\", "\\\\")
            dicrc_path_str: str = str(dicrc_path).replace("\\", "\\\\")
            # ユーザー辞書を読ませる場合はシステム辞書も引数で読ませる必要がある
            tagger = GenericTagger(
                f"-r {dicrc_path_str} -Owakati -d {dict_path_str} -u {user_dict_path_str}"
            )
        else:
            tagger = GenericTagger(
                f"-r {dicrc_path!s} -Owakati -d {dict_path_str!s} -u {user_dict_path!s}"
            )

    # ユーザー辞書は指定されていないが、システム辞書は指定されている場合
    elif dict_path is not None:
        dicrc_path = dict_path / "dicrc"
        # Windows 環境だと \ が途中でエスケープされるバグがあるため二重にする
        if os.name == "nt":
            dict_path_str: str = str(dict_path).replace("\\", "\\\\")
            dicrc_path_str: str = str(dicrc_path).replace("\\", "\\\\")
            tagger = GenericTagger(f"-r {dicrc_path_str} -Owakati -d {dict_path_str}")
        else:
            tagger = GenericTagger(f"-r {dicrc_path!s} -Owakati -d {dict_path!s}")

    # システム辞書もユーザー辞書も指定されていない場合
    else:
        tagger = Tagger("-Owakati")

    word_list: list[str] = []
    kana_list: list[str] = []
    accent_list: list[str | list[str]] = []
    pos_list: list[str] = []

    # 解析
    for word in tagger(text):
        feature = word.feature_raw

        # アクセント核が二つある場合「"*,*"」という風に記述されているので、「,」を「/」に変更し「"」を消す
        if re.search(r"\".*,.*\"", feature):
            accStart = re.search(r"\".*,.*\"", feature).start()  # type: ignore
            accEnd = re.search(r"\".*,.*\"", feature).end()  # type: ignore

            accent = feature[accStart:accEnd].replace(",", "/")

            feature = feature[:accStart] + accent.replace('"', "") + feature[accEnd:]

        feature = feature.split(",")

        """
        "feature" は Unidic の特徴データを named tuple として表現したもの。
        "feature_raw" はその語句の生の特徴情報。

        UniDic から得られる分類情報についてのメモ
        - 0 から数えて 0 番目 (CSV 形式: 0 から数えて 4 番目) => 品詞分類1
        - 0 から数えて 9 番目 (CSV 形式: 0 から数えて 13 番目) => 発音系
        - 0 から数えて 24 番目 (CSV 形式: 0 から数えて 28 番目) => アクセントタイプ
        - 0 から数えて 25 番目 (CSV 形式: 0 から数えて 29 番目) => アクセント結合型
        """

        # 辞書にある場合
        if len(feature) == 29:
            pos1: str = feature[0]
            kana: str = feature[9]
            accent = feature[24]

            # 読みがない場合
            if kana == "*":
                kana = "'"

            # 感嘆符か疑問符の場合
            if re.match(r"[!?]+", str(word)):
                kana = str(word)

            word_list.append(str(word))
            kana_list.append(kana)
            accent_list.append(accent)
            pos_list.append(pos1)

        # 辞書にない場合の処理
        elif re.match(r"[ァ-ロワ-ヴぁ-ろわ-ん－a-zA-Zａ-ｚＡ-Ｚ]+", str(word)):
            word, kana = text_to_sep_kata(str(word), raise_yomi_error=False)  # type: ignore
            word_list += word
            kana_list += kana

            # Xbox などの語句は、fugashi 解析時につながっていても pyopenjtalk では x box に分かれてしまう
            # そのため、分かれた数だけアクセントを追加する
            for i in range(len(word)):
                accent_list.append("0")
                pos_list.append("未分類")

        else:
            kana = "'"
            accent = "*"
            pos1 = "未分類"

            word_list.append(str(word))
            kana_list.append(kana)
            accent_list.append(accent)
            pos_list.append(pos1)

    # fugashi からアクセントを取得できなかった場合、0 に設定
    for i in range(len(accent_list)):
        if str(accent_list[i]) == "*":
            accent_list[i] = "0"

        # アクセントの種類が 2 つ以上の場合、先頭のものを使う
        elif len(accent_list[i]) == 3:
            accent_list[i] = str(accent_list[i]).split("/")[0]

    return word_list, kana_list, accent_list, pos_list
