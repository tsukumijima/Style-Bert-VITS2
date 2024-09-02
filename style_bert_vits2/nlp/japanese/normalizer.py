import re
import sys
import unicodedata
from datetime import datetime

from num2words import num2words

from style_bert_vits2.nlp.japanese.katakana_map import KATAKANA_MAP
from style_bert_vits2.nlp.symbols import PUNCTUATIONS


# 記号類の正規化マップ
__SYMBOL_REPLACE_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "/": ".",
    "／": ".",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    # NFKC 正規化後のハイフン・ダッシュの変種を全て通常半角ハイフン - \u002d に変換
    "\u02d7": "\u002d",  # ˗, Modifier Letter Minus Sign
    "\u2010": "\u002d",  # ‐, Hyphen,
    # "\u2011": "\u002d",  # ‑, Non-Breaking Hyphen, NFKC により \u2010 に変換される
    "\u2012": "\u002d",  # ‒, Figure Dash
    "\u2013": "\u002d",  # –, En Dash
    "\u2014": "\u002d",  # —, Em Dash
    "\u2015": "\u002d",  # ―, Horizontal Bar
    "\u2043": "\u002d",  # ⁃, Hyphen Bullet
    "\u2212": "\u002d",  # −, Minus Sign
    "\u23af": "\u002d",  # ⎯, Horizontal Line Extension
    "\u23e4": "\u002d",  # ⏤, Straightness
    "\u2500": "\u002d",  # ─, Box Drawings Light Horizontal
    "\u2501": "\u002d",  # ━, Box Drawings Heavy Horizontal
    "\u2e3a": "\u002d",  # ⸺, Two-Em Dash
    "\u2e3b": "\u002d",  # ⸻, Three-Em Dash
    # "～": "-",  # これは長音記号「ー」として扱うよう変更
    # "~": "-",  # これも長音記号「ー」として扱うよう変更
    "「": "'",
    "」": "'",
}
# 記号類の正規化パターン
__SYMBOL_REPLACE_PATTERN = re.compile(
    "|".join(re.escape(p) for p in __SYMBOL_REPLACE_MAP)
)

# 記号などの読み正規化マップ
__SYMBOL_YOMI_MAP = {
    "+": "プラス",
    "＋": "プラス",
    "➕": "プラス",
    "➖": "マイナス",  # 絵文字以外のハイフンは伸ばす棒と区別がつかないので記述していない
    "×": "掛ける",
    "✖": "掛ける",
    "÷": "割る",
    "➗": "割る",
    "=": "イコール",
    "＝": "イコール",
    "≠": "ノットイコール",
    "≒": "ニアリーイコール",
    "±": "プラスマイナス",
    "%": "パーセント",
    "‰": "パーミル",
    "′": "プライム",
    "″": "ダブルプライム",
    "°": "度",
    "℃": "度",
    "℉": "度",
    "@": "アットマーク",
    "＠": "アットマーク",
    "#": "ハッシュ",
    "＃": "ハッシュ",
    "#️⃣": "ハッシュ",
    "&": "アンド",
    "＆": "アンド",
    "*": "アスタリスク",
    "＊": "アスタリスク",
    "♯": "シャープ",
    "♭": "フラット",
    "♮": "ナチュラル",
    "<": "未満",
    ">": "より大きい",
    "≤": "以下",
    "≥": "以上",
    "∧": "かつ",
    "∨": "または",
    "√": "ルート",
    "∞": "無限大",
    "♾️": "無限大",
    "π": "パイ",
    "∑": "シグマ",
    "∫": "インテグラル",
    "∂": "パーシャル",
    "∇": "ナブラ",
    "∝": "比例",
    "∈": "属する",
    "∉": "属さない",
    "∪": "和集合",
    "∩": "共通部分",
    "⊂": "部分集合",
    "⊃": "上位集合",
    "≡": "合同",
    "∥": "平行",
    "⊥": "垂直",
    "∠": "角",
    "∧": "論理積",
    "∨": "論理和",
    "∩": "共通部分",
    "∪": "和集合",
    "∅": "空集合",
    "⊕": "排他的論理和",
    "⊗": "テンソル積",
    "💲": "ドル",
}
# 記号類の読み正規化パターン
__SYMBOL_YOMI_PATTERN = re.compile("|".join(re.escape(p) for p in __SYMBOL_YOMI_MAP))

# 単位の正規化マップ
# 単位は OpenJTalk 側でも変換してくれるので、単位が1文字で読み間違いが発生しやすい L, m, g, B とその関連単位のみ変換する
__UNIT_MAP = {
    "kL": "キロリットル",
    "L": "リットル",
    "dL": "デシリットル",
    "mL": "ミリリットル",
    "km": "キロメートル",
    "m": "メートル",
    "cm": "センチメートル",
    "mm": "ミリメートル",
    "kg": "キログラム",
    "g": "グラム",
    "mg": "ミリグラム",
    "PB": "ペタバイト",
    "PiB": "ペビバイト",
    "TB": "テラバイト",
    "TiB": "テビバイト",
    "GB": "ギガバイト",
    "GiB": "ギビバイト",
    "MB": "メガバイト",
    "MiB": "メビバイト",
    "KB": "キロバイト",
    "kB": "キロバイト",
    "KiB": "キビバイト",
    "B": "バイト",
}
# 単位の正規化パターン
__UNIT_PATTERN = re.compile(
    r"([0-9.]*[0-9])\s*((k|d|m)?L|(k|c|m)?m|(k|m)?g|PB|PiB|TB|TiB|GB|GiB|MB|MiB|KB|kB|KiB|B)(?=[^a-zA-Z]|$)"
)

# 句読点等の正規化パターン
__PUNCTUATION_CLEANUP_PATTERN = re.compile(
    # ↓ ひらがな、カタカナ、漢字
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    # ↓ 半角アルファベット（大文字と小文字）
    + r"\u0041-\u005A\u0061-\u007A"
    # ↓ 全角アルファベット（大文字と小文字）
    + r"\uFF21-\uFF3A\uFF41-\uFF5A"
    # ↓ ギリシャ文字
    + r"\u0370-\u03FF\u1F00-\u1FFF"
    # ↓ "!", "?", "…", ",", ".", "'", "-", 但し`…`はすでに`...`に変換されている
    + "".join(PUNCTUATIONS) + r"]+",  # fmt: skip
)

# 数字・通貨記号の正規化パターン
__CURRENCY_MAP = {
    "$": "ドル",
    "¥": "円",
    "€": "ユーロ",
    "£": "ポンド",
    "₩": "ウォン",
    "₹": "ルピー",  # インド・ルピー
    "₽": "ルーブル",
    "₺": "リラ",  # トルコ・リラ
    "฿": "バーツ",
    "₱": "ペソ",  # フィリピン・ペソ
    "₴": "フリヴニャ",
    "₫": "ドン",
    "₪": "シェケル",  # イスラエル・新シェケル
    "₦": "ナイラ",
    "₡": "コロン",  # コスタリカ・コロン
    "₿": "ビットコイン",
    "﷼": "リヤル",  # サウジアラビア・リヤル
    "₠": "ECU",  # European Currency Unit (廃止)
    "₢": "クルザード",  # ブラジル・クルザード (廃止)
    "₣": "フランスフラン",  # フランス・フラン (廃止)
    "₤": "リラ",  # イタリア・リラ (廃止)
    "₥": "ミル",  # アメリカ・ミル (廃止)
    "₧": "ペセタ",  # スペイン・ペセタ (廃止)
    "₨": "ルピー",  # パキスタン・ルピー
    "₭": "キープ",  # ラオス・キープ
    "₮": "トゥグルグ",  # モンゴル・トゥグルグ
    "₯": "ドラクマ",  # ギリシャ・ドラクマ (廃止)
    "₰": "ドイツペニヒ",  # ドイツ・ペニヒ (廃止)
    "₲": "グアラニー",  # パラグアイ・グアラニー
    "₳": "アウストラール",  # アルゼンチン・アウストラール (廃止)
    "₵": "セディ",  # ガーナ・セディ
    "₶": "リヴルトゥールヌワ",  # フランス・リヴルトゥールヌワ (廃止)
    "₷": "スペルリング",  # マルタ・スペルリング (廃止)
    "₸": "テンゲ",  # カザフスタン・テンゲ
    "₻": "マナト",  # トルクメニスタン・マナト
    "₼": "アゼルバイジャンマナト",
    "₾": "ラリ",  # ジョージア・ラリ
}
__CURRENCY_PATTERN = re.compile(
    r"([$¥€£₩₹₽₺฿₱₴₫₪₦₡₿﷼₠₢₣₤₥₧₨₭₮₯₰₲₳₵₶₷₸₻₼₾])([0-9.]*[0-9])"
)
__NUMBER_PATTERN = re.compile(r"[0-9]+(\.[0-9]+)?")
__NUMBER_WITH_SEPARATOR_PATTERN = re.compile("[0-9]{1,3}(,[0-9]{3})+")

# __replace_symbols() で使う正規表現パターン
__NUMBER_RANGE_PATTERN = re.compile(r"(\d+)\s*[〜~～]\s*(\d+)")
__NUMBER_MATH_PATTERN = re.compile(r"(\d+)\s*([+\-×÷])\s*(\d+)\s*=\s*(\d+)")
__DATE_EXPAND_PATTERN = re.compile(r"\d{2}[-/]\d{1,2}[-/]\d{1,2}")
__DATE_PATTERN = re.compile(
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}/\d{1,2}"
)
__FRACTION_PATTERN = re.compile(r"(\d+)/(\d+)")
__EXPONENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)[eE]([-+]?\d+)")

# __convert_english_to_katakana() で使う正規表現パターン
__SUB_WORDS_PATTERN = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\d|\W|$)|\d+")
__ENGLISH_WORD_PATTERN = re.compile(r"[a-zA-Z0-9]")


def normalize_text(text: str) -> str:
    """
    日本語のテキストを正規化する。
    結果は、ちょうど次の文字のみからなる：
    - ひらがな
    - カタカナ（全角長音記号「ー」が入る！）
    - 漢字
    - 半角アルファベット（大文字と小文字）
    - ギリシャ文字
    - `.` （句点`。`や`…`の一部や改行等）
    - `,` （読点`、`や`:`等）
    - `?` （疑問符`？`）
    - `!` （感嘆符`！`）
    - `'` （`「`や`」`等）
    - `-` （`―`（ダッシュ、長音記号ではない）や`-`等）

    注意点:
    - 三点リーダー`…`は`...`に変換される（`なるほど…。` → `なるほど....`）
    - 数字は漢字に変換される（`1,100円` → `千百円`、`52.34` → `五十二点三四`）
    - 読点や疑問符等の位置・個数等は保持される（`??あ、、！！！` → `??あ,,!!!`）

    Args:
        text (str): 正規化するテキスト

    Returns:
        str: 正規化されたテキスト
    """

    # 一番先に記号を変換
    # 最初でないと ℃ が unicodedata.normalize() で分割されてしまう
    res = __replace_symbols(text)

    res = unicodedata.normalize("NFKC", res)  # ここでアルファベットは半角になる
    res = __convert_numbers_to_words(res)  # 「100円」→「百円」等
    # 「～」と「〜」と「~」も長音記号として扱う
    res = res.replace("~", "ー")
    res = res.replace("～", "ー")
    res = res.replace("〜", "ー")

    res = __convert_english_to_katakana(res)  # 英単語をカタカナに変換

    res = replace_punctuation(res)  # 句読点等正規化、読めない文字を削除

    # 結合文字の濁点・半濁点を削除
    # 通常の「ば」等はそのままのこされる、「あ゛」は上で「あ゙」になりここで「あ」になる
    res = res.replace("\u3099", "")  # 結合文字の濁点を削除、る゙ → る
    res = res.replace("\u309A", "")  # 結合文字の半濁点を削除、な゚ → な
    return res


def __replace_symbols(text: str) -> str:
    """
    記号類の読みを適切に変換する。

    Args:
        text (str): 正規化するテキスト

    Returns:
        str: 正規化されたテキスト
    """

    # 数字と数字に挟まれた「〜」を「から」に置換
    text = __NUMBER_RANGE_PATTERN.sub(lambda m: f"{m.group(1)}から{m.group(2)}", text)

    # 数式の読み方を改善
    text = __NUMBER_MATH_PATTERN.sub(
        lambda m: f'{num2words(m.group(1), lang="ja")}{__SYMBOL_YOMI_MAP.get(m.group(2), m.group(2))}{num2words(m.group(3), lang="ja")}イコール{num2words(m.group(4), lang="ja")}',
        text,
    )

    def date_to_words(match: re.Match[str]) -> str:
        date_str = match.group(0)
        try:
            # 2桁の年を4桁に拡張する処理 (Y/m/d or Y-m-d の時のみ)
            if __DATE_EXPAND_PATTERN.match(date_str):
                if len(date_str.split("/")[0]) == 2 or len(date_str.split("-")[0]) == 2:
                    date_str = "20" + date_str

            # Y/m/d, Y-m-d, m/d のパターンを試す
            for fmt in ["%Y/%m/%d", "%Y-%m-%d", "%m/%d"]:
                try:
                    date = datetime.strptime(date_str, fmt)
                    if fmt == "%m/%d":
                        return f"{date.month}月{date.day}日"
                    return f"{date.year}年{date.month}月{date.day}日"
                except ValueError:
                    continue
            # どのパターンにも一致しない場合は元の文字列を返す
            return date_str
        except Exception:
            # エラーが発生した場合は元の文字列を返す
            return date_str

    # 日付パターンの変換
    text = __DATE_PATTERN.sub(date_to_words, text)

    # 分数の処理
    text = __FRACTION_PATTERN.sub(
        lambda m: f'{num2words(m.group(1), lang="ja")}分の{num2words(m.group(2), lang="ja")}',
        text,
    )

    # 指数表記の処理
    text = __EXPONENT_PATTERN.sub(
        lambda m: f'{num2words(float(m.group(0)), lang="ja")}', text
    )

    # 記号類を辞書で置換
    text = __SYMBOL_YOMI_PATTERN.sub(lambda x: __SYMBOL_YOMI_MAP[x.group()], text)

    return text


def __convert_numbers_to_words(text: str) -> str:
    """
    記号や数字を日本語の文字表現に変換する。

    Args:
        text (str): 変換するテキスト

    Returns:
        str: 変換されたテキスト
    """

    res = __UNIT_PATTERN.sub(lambda m: m[1] + __UNIT_MAP.get(m[2], m[2]), text)
    res = __NUMBER_WITH_SEPARATOR_PATTERN.sub(lambda m: m[0].replace(",", ""), res)
    res = __CURRENCY_PATTERN.sub(lambda m: m[2] + __CURRENCY_MAP.get(m[1], m[1]), res)
    res = __NUMBER_PATTERN.sub(lambda m: num2words(m[0], lang="ja"), res)

    return res


def __convert_english_to_katakana(text: str) -> str:
    """
    テキスト中の英単語をカタカナに変換する。
    Language-Specific のような複数の英単語がハイフンで連結されている単語や、
    ApplePencil のようなキャメルケースの複合語にも対応する。
    英単語は大文字・小文字を区別せず、変換マップに存在しない場合は元の単語をそのまま返す。

    Args:
        text (str): 変換するテキスト (例: "なぜかVSCodeのSyntax Highlightingが効かない")

    Returns:
        str: 変換されたテキスト (例: "なぜかブイエスコードのシンタックスハイライティングが効かない")
    """

    def process_english_word(word: str) -> str:
        """
        英単語をカタカナに変換する。
        ハイフンで連結された単語や、キャメルケースの複合語に対応する。

        Args:
            word (str): 変換する英単語

        Returns:
            str: カタカナに変換された単語
        """

        # まず、単語全体でカタカナ変換を試みる
        katakana_word = KATAKANA_MAP.get(word.lower())
        if katakana_word:
            return katakana_word

        # ハイフンで分割して処理
        if "-" in word:
            sub_words = word.split("-")
            katakana_sub_words = [
                KATAKANA_MAP.get(sub.lower(), sub) for sub in sub_words
            ]
            return "-".join(katakana_sub_words)

        # キャメルケースの複合語を分割して処理
        sub_words = __SUB_WORDS_PATTERN.findall(word)
        if len(sub_words) > 1:
            katakana_sub_words = []
            for sub in sub_words:
                katakana_sub = KATAKANA_MAP.get(sub.lower())
                if katakana_sub:
                    katakana_sub_words.append(katakana_sub)
                else:
                    return word  # 一つでも変換できない部分があれば、元の単語を返す
            return "".join(katakana_sub_words)

        # 上記のいずれにも該当しない場合は元の単語を返す
        return word

    words = []
    current_word = ""

    for char in text:
        # 英数字であれば current_word に追加
        if __ENGLISH_WORD_PATTERN.match(char) is not None or char in "-.'+":
            current_word += char
        else:
            # 英単語が終了したらカタカナに変換して words に追加
            if current_word:
                words.append(process_english_word(current_word))
                current_word = ""
            words.append(char)

    # 最後の単語を処理
    if current_word:
        words.append(process_english_word(current_word))

    return "".join(words)


def replace_punctuation(text: str) -> str:
    """
    句読点等を「.」「,」「!」「?」「'」「-」に正規化し、OpenJTalk で読みが取得できるもののみ残す：
    漢字・平仮名・カタカナ、アルファベット、ギリシャ文字

    Args:
        text (str): 正規化するテキスト

    Returns:
        str: 正規化されたテキスト
    """

    # 句読点を辞書で置換
    replaced_text = __SYMBOL_REPLACE_PATTERN.sub(
        lambda x: __SYMBOL_REPLACE_MAP[x.group()], text
    )

    # 上述以外の文字を削除
    replaced_text = __PUNCTUATION_CLEANUP_PATTERN.sub("", replaced_text)

    return replaced_text


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m style_bert_vits2.nlp.japanese.normalizer <text>")
        sys.exit(1)
    print(normalize_text(sys.argv[1]))
