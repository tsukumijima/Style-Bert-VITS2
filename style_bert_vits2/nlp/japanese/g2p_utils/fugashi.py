import re
from pathlib import Path

from fugashi import Tagger, try_import_unidic

from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata


# 事前に正規表現パターンをコンパイル
__FEATURE_PATTERN = re.compile(r"\".*,.*\"")


def analyze_text_with_fugashi(
    text: str,
    dict_dir_path: Path | None = None,
    user_dict_dir_path: Path | None = None,
) -> tuple[list[str], list[str], list[str | list[str]], list[str]]:
    """
    fugashi (fugashi-plus) でテキストを解析し、語句ごとの単語、カタカナ読み、アクセント、品詞を取得する。
    fugashi-plus でないと Windows 環境で辞書へのパスを正しく指定できない。

    Args:
        text (str): テキスト
        dict_path (Path | None, optional): fugashi のシステム辞書のパス。
            未指定時は unidic / unidic-lite パッケージから取得する。Defaults to None.
        user_dict_path (Path | None, optional): fugashi のユーザー辞書のパス。
            Defaults to None.

    Returns:
        tuple[list[str], list[str], list[str | list[str]], list[str]]:
        語句ごとの単語のリスト、カタカナ読みのリスト、アクセントのリスト、品詞のリスト
    """

    # システム辞書が指定されていない場合、unidic / unidic-lite パッケージがインストールされていれば、
    # 内蔵されているシステム辞書を使う
    if dict_dir_path is None:
        dict_dir_path_str = try_import_unidic()
        if dict_dir_path_str is None:
            raise ValueError("fugashi system dictionary is not found.")
        dict_dir_path = Path(dict_dir_path_str)

    # システム辞書ディレクトリのパス・システム辞書の mecabrc ファイルのパスを取得
    # 念のため、パス区切り文字は OS に関わらず通常のスラッシュとしている
    dict_dir_path_str = dict_dir_path.as_posix()
    dicrc_path_str = (dict_dir_path / "dicrc").as_posix()

    # ユーザー辞書が指定されている場合
    if user_dict_dir_path is not None:
        user_dict_dir_path_str = user_dict_dir_path.as_posix()
        # パスをダブルクオートで囲わないと、空白の入ったパスでエラーになる
        tagger = Tagger(f'-Owakati -r "{dicrc_path_str}" -d "{dict_dir_path_str}" -u "{user_dict_dir_path_str}"')  # fmt: skip

    # ユーザー辞書が指定されていない場合
    else:
        # パスをダブルクオートで囲わないと、空白の入ったパスでエラーになる
        tagger = Tagger(f'-Owakati -r "{dicrc_path_str}" -d "{dict_dir_path_str}"')

    word_list: list[str] = []
    kana_list: list[str] = []
    accent_list: list[str | list[str]] = []
    pos_list: list[str] = []

    # 解析
    for word in tagger(text):
        feature = word.feature_raw

        # アクセント核が二つある場合「"*,*"」という風に記述されているので、「,」を「/」に変更し「"」を消す
        if __FEATURE_PATTERN.search(feature):
            accent_start = __FEATURE_PATTERN.search(feature).start()  # type: ignore
            accent_end = __FEATURE_PATTERN.search(feature).end()  # type: ignore

            accent = feature[accent_start:accent_end].replace(",", "/")

            feature = (
                feature[:accent_start] + accent.replace('"', "") + feature[accent_end:]
            )

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
