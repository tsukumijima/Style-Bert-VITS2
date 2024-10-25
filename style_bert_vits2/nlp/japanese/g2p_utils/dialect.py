import re
from typing import Literal


# 実装されている日本語方言/喋り方ルール名の型
DialectRule = Literal[
    "tokyo",
    "kinki",
    "kyusyu",
    "convert2b2v",
    "convert2t2ts",
    "convert2d2r",
    "convert2r2d",
    "convert2s2z_sh2j",
    "1st_mora_tyouon",
    "1st_mora_sokuon",
    "1st_mora_remove",
    "1st_mora_renboin",
    "last_mora_acc_h",
    "last_word_acc_1",
    "add_youon_a",
    "add_youon_i",
    "add_youon_e",
    "add_youon_o",
    "hatuonbin",
    "youjigo_like",
]


# 事前に正規表現パターンをコンパイル
__KYUSHU_HATSUON_PATTERN = re.compile("[ヌニムモミ]+")
__YOUON_PATTERN = re.compile("[ァィゥェォャュョヮ]+")
__A_DAN_PATTERN = re.compile("[アカサタナハマヤラワガダバパ]|[ャヮ]+")
__I_DAN_PATTERN = re.compile("[イキシチニヒミリギジビピ]|ィ+")
__E_DAN_PATTERN = re.compile("[エケセテネヘメレゲデベペ]|ェ+")
__O_DAN_PATTERN = re.compile("[オコソトノホモヨロゴゾドボポ]|[ョォ]+")


def apply_dialect_diff(
    kata_list: list[str],
    accent_list: list[str],
    pos_list: list[str],
    dialect_rules: list[DialectRule],
) -> tuple[list[str], list[str]]:
    """
    NHK 日本語アクセント辞典を参考に、日本語方言特有の訛り・アクセントの差分を適用する。
    区分は付録 NHK 日本語アクセント辞典 125p を参照した。
    持っていない人のためにも、細かくコメントを残しておく。

    Args:
        kata_list (list[str]): 単語単位の単語のカタカナ読みのリスト
        accent_list (list[str]): 単語単位の単語のアクセントのリスト
        pos_list (list[str]): 単語単位の単語の品詞 (Part-Of-Speech) のリスト
        dialect_rules (list[DialectRule]): 適用対象の方言ルールのリスト

    Returns:
        tuple[list[str], list[str]]: 修正された kata_list と accent_list
    """

    """
    日本語方言の区分は以下の通り
    - 本土方言
        - 八丈方言
        - 東部方言
        - 西部方言
            - 近畿方言 => "kinki"
        - 九州方言 => "kyusyu"

    以下、厳密でない方言もしくは喋り方の実装
    - モーラ "b" を "v" に変換する: convert2b2v
    - モーラ "t" を "ts" に変換する: convert2t2ts
    - モーラ "d" を "r" に変換し、アクセントを頭高型にする: convert2d2r

    - 文章の1モーラ目を長音化しアクセントを頭高型に: 1st_mora_tyouon / "やはり、" => "やーはり" (HLLL)
    - 文章の1モーラ目を撥音化しアクセントを頭高型に: 1st_mora_sokuon / "やはり、" => "やっはり" (HLLL)
    - 文章の1モーラ目をしアクセントを平型に "っ" に変換: 1st_mora_remove / "やはり、" => "っはり" (LHH)

    - 最後の単語の終端をアクセント核にする: last_mora_acc_h
    - 最後のアクセントを頭高型にする: last_word_acc_1
    - 単語の先頭以外のの "no", "ra", "ru" を "N" に変換する: hatuonbin

    - "s" を "ch" に変換する: youjigo_like
      (幼児語のネイティブ話者つまり幼児の喋る幼児語でなく、我々大人の喋る(イメージする)幼児語である)

    - 各単語に最初にあ行がでてきた時 "ァ" をつけ "ァ" をアクセント核にする: add_youon_a /
      "そうさ、ボクの仕業さ。悪く思うなよ" => "そうさぁ。ボクの仕業さぁ。わぁるく思うなぁよ"
      ("ァ" は "ア" に置き換えられるので "ー" でも "ア" でもよいが、わかりやすくするため "ァ" とした)
    - 各単語に最初にい行がでてきた時 "ィ" をつけアクセントを頭高型にする: add_youon_i /
      "しまった。にげられた。" => "しぃまった。にぃげられた"
    - 各単語に最初にえ行がでてきた時 "ェ" をつけ "ェ" をアクセント核にする: add_youon_e /
      "へえ、それで" => "へェえ、それェでェ"
    - 各単語に最初にお行がでてきた時 "ぉ" をつけアクセントを頭高型にする: add_youon_i /
      "ようこそ。" => "よぉうこぉそ。"
    - 各単語の最初を連母音にしアクセントを頭高型にする。"e" は "ei", "o" は "ou" になる: 1st_mora_renboin /
      "俺のターン。" => "おぅれのターン" / "先生。" => "せぃんせい"
    """

    if "kyusyu" in dialect_rules:
        for i in range(len(kata_list)):
            # 九州のほぼ全域で "e" を "ye" と発音する: 付録 131p
            kata_list[i] = kata_list[i].replace("エ", "イェ")

            # 九州のほぼ全域で "s e" を "sh e", "z e" を "j e" と発音する: 付録 132p
            kata_list[i] = kata_list[i].replace("セ", "シェ")
            kata_list[i] = kata_list[i].replace("ゼ", "ジェ")

            # 発音化: 語末の "ヌ", "ニ", "ム", "モ, "ミ" などが発音 "ンN" で表される: 付録 132p
            num = len(kata_list[i])
            if __KYUSHU_HATSUON_PATTERN.fullmatch(kata_list[i][num - 1]):
                kata_list[i] = kata_list[i][: num - 1] + "ン"

    if "kinki" in dialect_rules:
        for i in range(len(kata_list)):
            # 1泊の名詞を長音化し2泊で発音する
            if pos_list[i] == "名詞" and len(kata_list[i]) == 1:
                if kata_list[i] == "!" or "?" or "'":
                    kata_list[i] = kata_list[i] + "ー"

    # ここから特に参考資料はないが表現の幅が広がったり、話者の特性を再現できそうなもの
    if "convert2b2v" in dialect_rules:
        for i in range(len(kata_list)):
            if "バ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("バ", "ヴァ")
            if "ビ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ビ", "ヴィ")
            if "ブ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ブ", "ヴ")
            if "ベ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ベ", "ヴェ")
            if "ボ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ボ", "ヴォ")

    if "convert2t2ts" in dialect_rules:
        for i in range(len(kata_list)):
            if "タ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("タ", "ツァ")
            if "チ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("チ", "ツィ")
            if "テ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("テ", "ツェ")
            if "ト" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ト", "ツォ")

    if "convert2d2r" in dialect_rules:
        for i in range(len(kata_list)):
            if "ダ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ダ", "ラ")
            if "デ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("デ", "レ")
            if "ド" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ド", "ロ")
            # アクセントを平型に変更
            accent_list[0] = "0"

    if "convert2r2d" in dialect_rules:
        for i in range(len(kata_list)):
            if "ラ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ラ", "ダ")
            if "レ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("レ", "デ")
            if "ロ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ロ", "ド")
            # アクセントを頭高型に変更
            accent_list[0] = "1"

    if "convert2s2z_sh2j" in dialect_rules:
        for i in range(len(kata_list)):
            if "サ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("サ", "ザ")
            if "スィ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("スィ", "ズィ")
            if "ス" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ス", "ズ")
            if "セ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("セ", "ゼ")
            if "ソ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ソ", "ゾ")
            if "シャ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("シャ", "ジャ")
            if "シ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("シ", "ジ")
            if "シュ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("シュ", "ジュ")
            if "シェ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("シェ", "ジェ")
            if "ショ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ショ", "ジョ")
            # アクセントを頭高型に変更
            accent_list[0] = "1"

    if "hatuonbin" in dialect_rules:
        for i in range(len(kata_list)):
            # 1文字以外の時
            if len(str(kata_list[i])) != 1:
                # 各単語先頭と終端は置き換えない
                if "ナ" in str(kata_list[i][1:-1]):
                    kata_list[i] = kata_list[i].replace("ナ", "ン")
                elif "ノ" in str(kata_list[i][1:-1]):
                    kata_list[i] = kata_list[i].replace("ノ", "ン")
                # 一種ずつしか撥音化しない
                elif "ル" in str(kata_list[i][1:-1]):
                    kata_list[i] = kata_list[i].replace("ル", "ン")
                elif "ラ" in str(kata_list[i][1:-1]):
                    kata_list[i] = kata_list[i].replace("ラ", "ン")

    if "youjigo_like" in dialect_rules:
        for i in range(len(kata_list)):
            if "サ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("サ", "チャ")
            if "シ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("シ", "チ")
            if "ス" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ス", "チュ")
            if "セ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("セ", "チェ")
            if "ソ" in str(kata_list[i]):
                kata_list[i] = kata_list[i].replace("ソ", "チョ")

    if "add_youon_a" in dialect_rules:
        for i in range(len(kata_list)):
            pos = __A_DAN_PATTERN.search(str(kata_list[i]))
            if pos:
                kata_list[i] = (
                    kata_list[i][: pos.end()] + "ァ" + kata_list[i][pos.end() :]
                )
                if type(pos.end()) == int:
                    # ァがアクセント核になる
                    accent_list[i] = str(pos.end())

    if "add_youon_i" in dialect_rules:
        for i in range(len(kata_list)):
            pos = __I_DAN_PATTERN.search(str(kata_list[i]))
            if pos:
                # マッチした語が最後の以外で　シャ　等　マッチした文字の後に拗音が来ない場合
                if (
                    len(str(kata_list[i])) > pos.end()
                    and kata_list[i][pos.end()] != "ャ"
                ):
                    kata_list[i] = (
                        kata_list[i][: pos.end()] + "ィ" + kata_list[i][pos.end() :]
                    )
                    if type(pos.end()) == int:
                        # アクセントを頭高型にする。
                        accent_list[i] = "1"
                # 上記以外のャが入っていない条件
                elif "ャ" not in kata_list[i]:
                    kata_list[i] = (
                        kata_list[i][: pos.end()] + "ィ" + kata_list[i][pos.end() :]
                    )
                    if type(pos.end()) == int:
                        # アクセントを頭高型にする
                        accent_list[i] = "1"

    if "add_youon_e" in dialect_rules:
        for i in range(len(kata_list)):
            pos = __E_DAN_PATTERN.search(str(kata_list[i]))
            if pos:
                kata_list[i] = (
                    kata_list[i][: pos.end()] + "ェ" + kata_list[i][pos.end() :]
                )
                # ェがアクセント核になる
                if type(pos.end()) == int:
                    accent_list[i] = str(pos.end())

    if "add_youon_o" in dialect_rules:
        for i in range(len(kata_list)):
            pos = __O_DAN_PATTERN.search(str(kata_list[i]))
            if pos:
                kata_list[i] = (
                    kata_list[i][: pos.end()] + "ォ" + kata_list[i][pos.end() :]
                )
                # アクセントを頭高型にする。
                if type(pos.end()) == int:
                    accent_list[i] = "1"

    if "1st_mora_tyouon" in dialect_rules:
        pos = __YOUON_PATTERN.search(str(kata_list[0]))
        if pos:
            # マッチしたパターンが二文字目から(一文字文字以内の場合)
            if pos.start() == 1:
                kata_list[0] = (
                    kata_list[0][: pos.end()] + "ー" + kata_list[0][pos.end() :]
                )
        else:
            kata_list[0] = kata_list[0][0] + "ー" + kata_list[0][1:]
        # アクセントを頭高型に変更
        accent_list[0] = "1"

    if "1st_mora_sokuon" in dialect_rules:
        pos = __YOUON_PATTERN.search(str(kata_list[0]))
        if pos:
            # マッチしたパターンが二文字目から (一文字文字以内の場合)
            if pos.start() == 1:
                kata_list[0] = (
                    kata_list[0][: pos.end()] + "ッ" + kata_list[0][pos.end() :]
                )
        else:
            kata_list[0] = kata_list[0][0] + "ッ" + kata_list[0][1:]
        # アクセントを頭高型に変更
        accent_list[0] = "1"

    if "1st_mora_remove" in dialect_rules:
        pos = __YOUON_PATTERN.search(str(kata_list[0]))
        if pos:
            # マッチしたパターンが二文字目からでかつ伸ばす必要がある (一文字文字以内の場合)
            if pos.start() == 1:
                kata_list[0] = "ッ" + kata_list[0][pos.end() :]
        else:
            kata_list[0] = "ッ" + kata_list[0][1:]

    if "1st_mora_renboin" in dialect_rules:
        pos = __O_DAN_PATTERN.search(str(kata_list[0]))
        if pos:
            kata_list[0] = kata_list[0][: pos.end()] + "ゥ" + kata_list[0][pos.end() :]
            # アクセントを頭高型に
            if type(pos.end()) == int:
                accent_list[0] = "1"
        else:
            pos = __E_DAN_PATTERN.search(str(kata_list[0]))
            if pos:
                kata_list[0] = (
                    kata_list[0][: pos.end()] + "ィ" + kata_list[0][pos.end() :]
                )
                # アクセントを頭高型に
                if type(pos.end()) == int:
                    accent_list[0] = "1"

    if "last_mora_acc_h" in dialect_rules:
        # 最後の単語の終端をアクセント核にする
        last_word = kata_list[len(kata_list) - 1]
        accent_list[len(accent_list) - 1] = str(len(last_word))

    if "last_word_acc_1" in dialect_rules:
        # 最後のアクセントを頭高型に
        accent_list[len(accent_list) - 1] = "1"

    return kata_list, accent_list


def apply_keihan_accent_diff(
    kata_list: list[str],
    accent_list: list[str],
    pos_list: list[str],
) -> list[str]:
    """
    NHK 日本語アクセント辞典を参考に、京阪式アクセントの差分を適用する。
    東京式と京阪式の対応表は付録 146p を参照した。
    持っていない人のためにも、細かくコメントを残しておく。

    Args:
        kata_list (list[str]): 単語単位の単語のカタカナ読みのリスト
        accent_list (list[str]): 単語単位の単語のアクセントのリスト
        pos_list (list[str]): 単語単位の単語の品詞 (Part-Of-Speech) のリスト
    Returns:
        accent_list (list[str]): 修正された accent_list
    """

    for i in range(len(pos_list)):
        # 分類が名詞の場合
        if pos_list[i] == "名詞":
            # 一音の場合(長音可で2泊化)されている
            if kata_list[i][1] == "ー":
                # 平型の場合頭高型に
                if accent_list[i] == "0":
                    accent_list[i] = "1"
                # 頭高型の場合全て低く
                if accent_list[i] == "ALL_L":
                    accent_list[i] = "0"
            # ニ音の場合
            elif len(kata_list[i]) == 2:
                # 平型の場合全て高く
                if accent_list[i] == "0":
                    accent_list[i] = "ALL_H"
                # 尾高型の場合頭高型に
                if accent_list[i] == "2":
                    accent_list[i] = "1"

        # 分類が動詞の場合
        elif pos_list[i] == "動詞":
            # ニ音の場合
            if len(kata_list[i]) == 2:
                # 平型の場合全て高く
                if accent_list[i] == "0":
                    accent_list[i] = "ALL_H"
                # 頭高型の場合頭高型に
                if accent_list[i] == "1":
                    accent_list[i] = "2"
            # 三音の場合
            if len(kata_list[i]) == 3:
                # 平型の場合全て高く
                if accent_list[i] == "0":
                    accent_list[i] = "ALL_H"
                # 中高型の場合尾高型に
                if accent_list[i] == "2":
                    accent_list[i] = "3"

        # 分類が形容詞の場合
        elif pos_list[i] == "形容詞":
            # ニ音の場合
            if len(kata_list[i]) == 2:
                # 頭高型の場合頭高型に
                if accent_list[i] == "1":
                    accent_list[i] = "2"
            # 三音の場合
            if len(kata_list[i]) == 3:
                # 平型の場合頭高に
                if accent_list[i] == "0":
                    accent_list[i] = "1"
                # 中高型の場合頭高に
                if accent_list[i] == "2":
                    accent_list[i] = "1"

    return accent_list
