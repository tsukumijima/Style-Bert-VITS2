import re
import sys
import unicodedata
from datetime import datetime

from num2words import num2words

from style_bert_vits2.nlp.japanese.katakana_map import KATAKANA_MAP
from style_bert_vits2.nlp.japanese.romkan import to_katakana
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
    # 算術演算子
    "+": "プラス",
    "＋": "プラス",
    "➕": "プラス",
    "➖": "マイナス",  # 絵文字以外のハイフンは伸ばす棒と区別がつかないので記述していない
    "×": "掛ける",
    "✖": "掛ける",
    "⨯": "掛ける",
    "÷": "割る",
    "➗": "割る",
    # 等号・不等号
    "=": "イコール",
    "＝": "イコール",
    "≠": "ノットイコール",
    "≒": "ニアリーイコール",
    "≈": "ニアリーイコール",
    "≅": "合同",
    "≡": "合同",
    "≢": "合同でない",
    # 比較演算子
    "<": "未満",
    "＜": "未満",
    ">": "より大きい",
    "＞": "より大きい",
    "≤": "以下",
    "≦": "以下",
    "≥": "以上",
    "≧": "以上",
    # 単位・数値記号
    "%": "パーセント",
    "％": "パーセント",
    "٪": "パーセント",
    "﹪": "パーセント",
    "‰": "パーミル",
    "‱": "パーミリアド",
    "′": "プライム",
    "″": "ダブルプライム",
    "‴": "トリプルプライム",
    "°": "度",
    "℃": "度",
    "℉": "度",
    "±": "プラスマイナス",
    "∓": "マイナスプラス",
    "№": "ナンバー",
    "℡": "テレフォン",
    "℠": "エスエム",
    "™": "ティーエム",
    "©": "コピーライト",
    "®": "アールマーク",
    "💲": "ドル",
    # 一般記号
    "@": "アットマーク",
    "＠": "アットマーク",
    "#": "ハッシュ",
    "＃": "ハッシュ",
    "#️⃣": "ハッシュ",
    "&": "アンド",
    "＆": "アンド",
    "*": "アスタリスク",
    "＊": "アスタリスク",
    "†": "ダガー",
    "‡": "ダブルダガー",
    "§": "セクション",
    "¶": "パラグラフ",
    # 音楽記号
    "♯": "シャープ",
    "♭": "フラット",
    "♮": "ナチュラル",
    # "♩": "音符",
    # "♪": "音符",
    # "♫": "音符",
    # "♬": "音符",
    # 数学記号
    "∧": "かつ",
    "∨": "または",
    "¬": "ノット",
    "⊕": "排他的論理和",
    "⊗": "テンソル積",
    "√": "ルート",
    "∛": "立方根",
    "∜": "四乗根",
    "∞": "無限大",
    "♾️": "無限大",
    "π": "パイ",
    "∑": "シグマ",
    "∏": "パイ積分",
    "∫": "インテグラル",
    "∬": "二重積分",
    "∭": "三重積分",
    "∮": "周回積分",
    "∯": "面積分",
    "∰": "体積分",
    "∂": "パーシャル",
    "∇": "ナブラ",
    "∝": "比例",
    # 集合記号
    "∈": "属する",
    "∉": "属さない",
    "∋": "含む",
    "∌": "含まない",
    "∪": "和集合",
    "∩": "共通部分",
    "⊂": "部分集合",
    "⊃": "上位集合",
    "⊄": "部分集合でない",
    "⊅": "上位集合でない",
    "⊆": "部分集合または等しい",
    "⊇": "上位集合または等しい",
    "∅": "空集合",
    "∖": "差集合",
    "∆": "対称差",
    # 幾何記号
    "∥": "平行",
    "⊥": "垂直",
    "∠": "角",
    "∟": "直角",
    "∡": "測定角",
    "∢": "球面角",
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
    # ↓ 半角数字
    + r"\u0030-\u0039"
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
__FRACTION_PATTERN = re.compile(r"(\d+)[/／](\d+)")
__ASPECT_PATTERN = re.compile(r"(\d+)[:：](\d+)")
__EXPONENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)[eE]([-+]?\d+)")

# __convert_english_to_katakana() で使う正規表現パターン
__ENGLISH_WORD_PATTERN = re.compile(r"[a-zA-Z0-9]")
__ENGLISH_WORD_WITH_NUMBER_PATTERN = re.compile(r"^([a-zA-Z]+)([0-9]{1,2})$")
__ALPHABET_PATTERN = re.compile(r"[a-zA-Z]")


def normalize_text(text: str) -> str:
    """
    日本語のテキストを正規化する。
    結果は、ちょうど次の文字のみからなる：
    - ひらがな
    - カタカナ（全角長音記号「ー」が入る！）
    - 漢字
    - 半角数字
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
    - 読点や疑問符等の位置・個数等は保持される（`??あ、、！！！` → `??あ,,!!!`）

    Args:
        text (str): 正規化するテキスト

    Returns:
        str: 正規化されたテキスト
    """

    # 一番先に記号を変換
    # 最初でないと ℃ が unicodedata.normalize() で分割されてしまう
    res = __replace_symbols(text)

    # 自然な日本語テキスト読み上げのために、全角スペースは句点に変換
    # 半角スペースが入る箇所で止めて読むかはケースバイケースなため、変換は行わない
    # Unicode 正規化でスペースが全て半角に変換される前に実行する必要がある
    res = res.replace("\u3000", "。")

    res = unicodedata.normalize("NFKC", res)  # ここでアルファベットは半角になる

    res = __convert_english_to_katakana(res)  # 英単語をカタカナに変換

    res = __convert_numbers_to_words(res)  # 「100円」→「百円」等
    # 「～」と「〜」と「~」も長音記号として扱う
    res = res.replace("~", "ー")
    res = res.replace("～", "ー")
    res = res.replace("〜", "ー")

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
        lambda m: f"{m.group(1)}{__SYMBOL_YOMI_MAP.get(m.group(2), m.group(2))}{m.group(3)}イコール{m.group(4)}",
        text,
    )

    def date_to_words(match: re.Match[str]) -> str:
        date_str = match.group(0)
        try:
            # 2桁の年を4桁に拡張する処理 (Y/m/d or Y-m-d の時のみ)
            if __DATE_EXPAND_PATTERN.match(date_str):
                # スラッシュまたはハイフンで分割して年部分を取得
                year_str = (
                    date_str.split("/")[0]
                    if "/" in date_str
                    else date_str.split("-")[0]
                )
                if len(year_str) == 2:
                    # 50 以降は 1900 年代、49 以前は 2000 年代として扱う
                    # 98/04/11 → 1998/04/11 / 36-01-01 → 2036-01-01
                    year_prefix = "19" if int(year_str) >= 50 else "20"
                    date_str = year_prefix + date_str

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
        lambda m: f'{num2words(m.group(2), lang="ja")}分の{num2words(m.group(1), lang="ja")}',
        text,
    )

    # アスペクト比の処理
    text = __ASPECT_PATTERN.sub(
        lambda m: f'{num2words(m.group(1), lang="ja")}たい{num2words(m.group(2), lang="ja")}',
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
    記号を日本語の文字表現に変換する。
    以前は数字を漢数字表現に変換していたが、pyopenjtalk 側の変換処理の方が優秀なため撤去した。

    Args:
        text (str): 変換するテキスト

    Returns:
        str: 変換されたテキスト
    """

    res = __UNIT_PATTERN.sub(lambda m: m[1] + __UNIT_MAP.get(m[2], m[2]), text)
    res = __NUMBER_WITH_SEPARATOR_PATTERN.sub(lambda m: m[0].replace(",", ""), res)
    res = __CURRENCY_PATTERN.sub(lambda m: m[2] + __CURRENCY_MAP.get(m[1], m[1]), res)

    return res


def __convert_english_to_katakana(text: str) -> str:
    """
    テキスト中の英単語をカタカナに変換する。
    複合語や略語、記号を含む単語など、様々なパターンに対応する。
    ただし、誤変換を防ぐため、確実に変換できるパターンのみを処理する。

    Args:
        text (str): 変換するテキスト

    Returns:
        str: 変換されたテキスト
    """

    def split_camel_case(word: str) -> list[str]:
        """
        CamelCase の単語を分割する。
        大文字が連続する場合はそれを一つの部分として扱う。

        Args:
            word (str): 分割する単語

        Returns:
            list[str]: 分割された部分文字列のリスト
        """

        parts = []
        current = word[0]
        prev_is_upper = word[0].isupper()

        for char in word[1:]:
            is_upper = char.isupper()

            # 小文字から大文字への変化、または大文字から小文字への変化を検出
            if (is_upper and not prev_is_upper) or (
                not is_upper and prev_is_upper and len(current) > 1
            ):
                parts.append(current)
                current = char
            else:
                current += char

            prev_is_upper = is_upper

        if current:
            parts.append(current)

        return parts

    def process_english_word(word: str, enable_romaji: bool = False) -> str:
        """
        英単語をカタカナに変換する。確実に変換できるパターンのみを処理し、
        不確実な場合は元の単語をそのまま返す (pyopenjtalk 側でアルファベット読みされる)。

        Args:
            word (str): 変換する英単語
            enable_romaji (bool): ローマ字変換を有効にするかどうか
        Returns:
            str: カタカナに変換された単語
        """

        # 英単語の末尾に2桁以下の数字がつく場合の処理
        number_match = __ENGLISH_WORD_WITH_NUMBER_PATTERN.match(word)
        if number_match:
            base_word = number_match.group(1)
            number = number_match.group(2)
            # まず base_word をカタカナに変換できるか確認
            base_katakana = KATAKANA_MAP.get(base_word.lower())
            if base_katakana:
                # 数字を英語表現に変換し、それをカタカナに変換
                number_in_english = num2words(int(number), lang="en")
                number_katakana = process_english_word(number_in_english)
                if number_katakana:
                    return base_katakana + number_katakana

        # 1. 完全一致での変換を試みる（最も信頼できる変換）
        # 1.1 まず元の文字列で試す（辞書に大文字で登録されている頭字語はここで変換される）
        katakana_word = KATAKANA_MAP.get(word)
        if katakana_word:
            return katakana_word
        # 1.2 小文字に変換した上で試す
        katakana_word = KATAKANA_MAP.get(word.lower())
        if katakana_word:
            return katakana_word

        # 2. 末尾のピリオドを除去して再試行
        if word.endswith("."):
            katakana_word = KATAKANA_MAP.get(word[:-1].lower())
            if katakana_word:
                return katakana_word

        # 3. 所有格の処理（確実なパターン）
        if word.lower().endswith(("'s", "’s")):
            base_word = word[:-2]
            katakana_word = KATAKANA_MAP.get(base_word.lower())
            if katakana_word:
                return katakana_word + "ズ"

        # 4. 複数形の処理
        if word.endswith("s"):
            base_word = word[:-1]
            katakana_word = KATAKANA_MAP.get(base_word.lower())
            if katakana_word:
                return katakana_word + "ズ"

        # 5. 記号で区切られた複合語の処理（部分的な変換を許可）
        for separator, join_word in [
            ("&", "アンド"),
            ("-", ""),
            (".", ""),
            ("+", "プラス"),
        ]:
            if separator in word:
                # "." の場合は、小数点かどうかをチェック
                if separator == ".":
                    parts = word.split(".")
                    # 隣接する部分が両方数字の場合は次のセパレータへ
                    should_skip = False
                    for i in range(len(parts) - 1):
                        if (
                            parts[i]
                            and parts[i][-1].isdigit()
                            and parts[i + 1]
                            and parts[i + 1][0].isdigit()
                        ):
                            should_skip = True
                            break
                    if should_skip:
                        continue

                sub_words = word.split(separator)
                katakana_sub_words = []

                for sub in sub_words:
                    # 辞書にある場合はカタカナに変換、ない場合は元の単語をそのまま使用
                    sub_katakana = KATAKANA_MAP.get(sub.lower(), sub)
                    katakana_sub_words.append(sub_katakana)

                return join_word.join(katakana_sub_words)

        # 6. の処理を行う前に、先行して単位系の変換を終わらせておく
        # さもなければ「MiB」が分割されてしまう
        word = __UNIT_PATTERN.sub(lambda m: m[1] + __UNIT_MAP.get(m[2], m[2]), word)

        # 6. CamelCase の複合語を処理
        if any(c.isupper() for c in word[1:]):  # 2文字目以降に大文字が含まれる
            parts = split_camel_case(word)
            result_parts = []

            for part in parts:
                # 大文字のみで構成される部分
                # 辞書になければそのまま、pyopenjtalk でアルファベット読みされる
                if all(c.isupper() for c in part):
                    result_parts.append(KATAKANA_MAP.get(part, part))
                else:
                    # それ以外は辞書で変換を試みる
                    converted = process_english_word(part)
                    result_parts.append(converted)

            # ここでは戻らず、値の上書きのみにとどめる
            word = "".join(result_parts)

        # 7. 数字（小数点含む）が含まれる場合、数字部分とそれ以外の部分に分割して処理
        if any(c.isdigit() for c in word):

            # 数字（小数点含む）とそれ以外の部分を分割
            parts = []
            last_end = 0

            for match in __NUMBER_PATTERN.finditer(word):
                # 数字の前の部分を処理
                if match.start() > last_end:
                    non_number = word[last_end : match.start()]
                    parts.append(process_english_word(non_number))

                # 数字部分をそのまま追加
                parts.append(match.group())
                last_end = match.end()

            # 最後の非数字部分を処理
            if last_end < len(word):
                non_number = word[last_end:]
                parts.append(process_english_word(non_number))

            return "".join(parts)

        # 8. アルファベットが含まれる場合、ローマ字 -> カタカナ変換を試みる
        # 2文字以上の場合のみ変換を試みる (I -> イ のような1文字変換を防ぐ)
        if (
            len(word) >= 2
            and any(__ALPHABET_PATTERN.match(c) for c in word)
            and enable_romaji
        ):
            katakana = to_katakana(word)
            # 全文字を完全にカタカナに変換できた場合のみ採用
            if not any(__ALPHABET_PATTERN.match(c) for c in katakana):
                return katakana

        # 上記以外は元の単語を返す (pyopenjtalk 側でアルファベット読みされる)
        return word

    words = []
    current_word = ""
    prev_char = ""

    for i, char in enumerate(text):
        next_char = text[i + 1] if i < len(text) - 1 else ""

        # 英数字または特定の記号であれば current_word に追加
        if __ENGLISH_WORD_PATTERN.match(char) is not None or char in "-&+'":
            current_word += char
        # ピリオドの特別処理
        elif char == ".":
            # 前後が英数字の場合は単語の一部として扱う (例: Node.js)
            if (
                current_word
                and next_char
                and (
                    __ENGLISH_WORD_PATTERN.match(prev_char) is not None
                    and __ENGLISH_WORD_PATTERN.match(next_char) is not None
                )
            ):
                current_word += char
            # それ以外は文の区切りとして扱う (例: I'm fine.)
            else:
                if current_word:
                    words.append(process_english_word(current_word, enable_romaji=True))
                    current_word = ""
                words.append(char)
        else:
            # 英単語が終了したらカタカナに変換して words に追加
            if current_word:
                words.append(process_english_word(current_word, enable_romaji=True))
                current_word = ""
            words.append(char)

        prev_char = char

    # 最後の単語を処理
    if current_word:
        words.append(process_english_word(current_word, enable_romaji=True))

    return "".join(words)


def replace_punctuation(text: str) -> str:
    """
    句読点等を「.」「,」「!」「?」「'」「-」に正規化し、OpenJTalk で読みが取得できるもののみ残す：
    漢字・平仮名・カタカナ、数字、アルファベット、ギリシャ文字

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
