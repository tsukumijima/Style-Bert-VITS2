import re
import sys
import unicodedata
from datetime import datetime

from num2words import num2words

from style_bert_vits2.logging import logger
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
    "/": "/",  # スラッシュは pyopenjtalk での形態素解析処理で重要なので正規化後も残す
    "／": "/",  # スラッシュは pyopenjtalk での形態素解析処理で重要なので正規化後も残す
    "\\": ".",
    "＼": ".",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    ":": ",",
    ";": ",",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "「": "'",
    "」": "'",
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
}
# 記号類の正規化パターン
__SYMBOL_REPLACE_PATTERN = re.compile(
    "|".join(re.escape(p) for p in __SYMBOL_REPLACE_MAP)
)

# 記号などの読み正規化マップ
# 一度リストアップしたがユースケース上不要と判断した記号はコメントアウトされている
__SYMBOL_YOMI_MAP = {
    # 一般記号
    # "@": "アット",  # カタカナに変換するとユーザー辞書登録が効かなくなり不便なためコメントアウト
    # "＠": "アット",  # カタカナに変換するとユーザー辞書登録が効かなくなり不便なためコメントアウト
    # "&": "アンド",  # カタカナに変換するとユーザー辞書登録が効かなくなり不便なためコメントアウト
    # "＆": "アンド",  # カタカナに変換するとユーザー辞書登録が効かなくなり不便なためコメントアウト
    # "*": "アスタリスク",  # 発音されると不都合な場合があるためコメントアウト
    # "＊": "アスタリスク",  # 発音されると不都合な場合があるためコメントアウト
    "#": "シャープ",
    "＃": "シャープ",
    "#️⃣": "シャープ",
    "†": "ダガー",
    "‡": "ダブルダガー",
    "§": "セクション",
    "¶": "パラグラフ",
    # 算術演算子
    "+": "プラス",
    "＋": "プラス",
    "➕": "プラス",
    "➖": "マイナス",  # 絵文字以外のハイフンは伸ばす棒と区別がつかないので記述していない
    "×": "かける",
    "✖": "かける",
    "⨯": "かける",
    "÷": "わる",
    "➗": "わる",
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
    # 山括弧は装飾的に使われることも多いため、別途数式や比較演算子として使われる場合のみ読み上げる
    "≤": "小なりイコール",
    "≦": "小なりイコール",
    "⩽": "小なりイコール",
    "≥": "大なりイコール",
    "≧": "大なりイコール",
    "⩾": "大なりイコール",
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
    "℡": "電話番号",
    "〒": "郵便番号",
    "〶": "郵便番号",
    "㏍": "株式会社",
    "℠": "エスエム",
    # "™": "ティーエム",
    "©": "コピーライト",
    # "®": "アールマーク",
    "💲": "$",  # __convert_numbers_to_words() で「100ドル」のように読み上げできるように
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
    # 囲み付き・丸付き文字 (Unicode 正規化で問題なく変換される「㈦」「㊥」などを除く)
    # ref: https://ja.wikipedia.org/wiki/%E5%9B%B2%E3%81%BFCJK%E6%96%87%E5%AD%97%E3%83%BB%E6%9C%88
    "㈱": "株式会社",
    "㈲": "有限会社",
    "㈳": "社団法人",
    "㈴": "合名会社",
    "㈵": "特殊法人",
    "㈶": "財団法人",
    "㈷": "祝日",
    "㈸": "労働組合",
    "㈹": "代表電話",
    "㈺": "呼出し電話",
    "㈻": "学校法人",
    "㈼": "監査法人",
    "㈽": "企業組合",
    "㈾": "合資会社",
    "㈿": "協同組合",
    "㉀": "祭日",
    "㉁": "休日",
    "㉅": "幼稚園",
    "㊑": "株式会社",
    "㊒": "有限会社",
    "㊓": "社団法人",
    "㊔": "合名会社",
    "㊕": "特殊法人",
    "㊖": "財団法人",
    "㊗": "祝日",
    "㊘": "労働組合",
    "㊙": "マル秘",
    "㊝": "マル優",
    "㊡": "休日",
    "㊢": "写し",
    "㊩": "医療法人",
    "㊪": "宗教法人",
    "㊫": "学校法人",
    "㊬": "監査法人",
    "㊭": "企業組合",
    "㊮": "合資会社",
    "㊯": "協同組合",
}
# 記号類の読み正規化パターン
__SYMBOL_YOMI_PATTERN = re.compile("|".join(re.escape(p) for p in __SYMBOL_YOMI_MAP))

# 単位の正規化マップ
# 単位は OpenJTalk 側で変換してくれるものもあるため、単位が1文字で読み間違いが発生しやすい L, m, g, B と、
# OpenJTalk では変換できない単位のみ変換する
__UNIT_MAP = {
    "kL": "キロリットル",
    "dL": "デシリットル",
    "mL": "ミリリットル",
    "L": "リットル",
    "km": "キロメートル",
    "km2": "平方キロメートル",
    "km3": "立方キロメートル",
    "m2": "平方メートル",
    "m3": "立方メートル",
    "cm": "センチメートル",
    "cm2": "平方センチメートル",
    "cm3": "立方センチメートル",
    "mm": "ミリメートル",
    "mm2": "平方ミリメートル",
    "mm3": "立方ミリメートル",
    "m": "メートル",
    "kg": "キログラム",
    "mg": "ミリグラム",
    "g": "グラム",
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
    "t": "トン",
    "d": "日",
    "h": "時間",
    "s": "秒",
    "ms": "ミリ秒",
    "μs": "マイクロ秒",
    "ns": "ナノ秒",
}
# 単位の正規化パターン
__UNIT_PATTERN = re.compile(
    r"([0-9.]*[0-9](?:[eE][-+]?[0-9]+)?)\s*((k|d|m)?L|(?:k|c|m)m[23]?|m[23]?|m(?![a-zA-Z])|(?:k|m)?g|PB|PiB|TB|TiB|GB|GiB|MB|MiB|KB|kB|KiB|B|t|d|h|s|ms|μs|ns)(?=[^a-zA-Z]|$)"
)

# 正規化後に残す文字種を表すパターン
__PUNCTUATION_CLEANUP_PATTERN = re.compile(
    # ↓ ひらがな、カタカナ、漢字
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    # ↓ 半角数字
    + r"\u0030-\u0039"
    # ↓ 全角数字
    + r"\uFF10-\uFF19"
    # ↓ 半角アルファベット（大文字と小文字）
    + r"\u0041-\u005A\u0061-\u007A"
    # ↓ 全角アルファベット（大文字と小文字）
    + r"\uFF21-\uFF3A\uFF41-\uFF5A"
    # ↓ ギリシャ文字
    + r"\u0370-\u03FF\u1F00-\u1FFF"
    # ↓ "!", "?", "…", ",", ".", "'", "-", 但し`…`はすでに`...`に変換されている
    # スラッシュは pyopenjtalk での形態素解析処理で重要なので、例外的に正規化後も残す (g2p 処理内で "." に変換される)
    + "".join(re.escape(p) for p in (PUNCTUATIONS + ["/"])) + r"]+"  # fmt: skip
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
    r"([$¥€£₩₹₽₺฿₱₴₫₪₦₡₿﷼₠₢₣₤₥₧₨₭₮₯₰₲₳₵₶₷₸₻₼₾])([0-9.]*[0-9])|([0-9.]*[0-9])([$¥€£₩₹₽₺฿₱₴₫₪₦₡₿﷼₠₢₣₤₥₧₨₭₮₯₰₲₳₵₶₷₸₻₼₾])"
)
__NUMBER_PATTERN = re.compile(r"[0-9]+(\.[0-9]+)?")
__NUMBER_WITH_SEPARATOR_PATTERN = re.compile("[0-9]{1,3}(,[0-9]{3})+")

# __replace_symbols() で使う正規表現パターン
__DATE_ZERO_PADDING_PATTERN = re.compile(r"(?<!\d)0(\d)(?=月|日|時|分|秒)")
__TIME_PATTERN = re.compile(r"(\d+)時(\d+)分(?:(\d+)秒)?")
__ASPECT_PATTERN = re.compile(r"(\d+)[:：](\d+)(?:[:：](\d+))?")
__WEEKDAY_PATTERN = re.compile(
    r"("  # 日付部分をキャプチャ開始
    r"(?:\d{4}年\s*)?"  # 4桁の年 + 年（省略可）
    r"(?:\d{1,2}月\s*)?"  # 1-2桁の月 + 月（省略可）
    r"\d{1,2}日"  # 1-2桁の日 + 日（必須）
    r")"  # 日付部分をキャプチャ終了
    r"\s*[（(]([月火水木金土日])[)）]"  # 全角/半角括弧で囲まれた曜日漢字
    r"|"  # または
    r"("  # 日付部分をキャプチャ開始
    r"(?:\d{4}[-/]\s*)?"  # 4桁の年 + 区切り（省略可）
    r"(?:\d{1,2}[-/]\s*)?"  # 1-2桁の月 + 区切り（省略可）
    r"\d{1,2}"  # 1-2桁の日（必須）
    r")"  # 日付部分をキャプチャ終了
    r"\s*[（(]([月火水木金土日])[)）]"  # 全角/半角括弧で囲まれた曜日漢字
)
__URL_PATTERN = re.compile(
    r"https?://[-a-zA-Z0-9.]+(?:/[-a-zA-Z0-9._~:/?#\[\]@!$&\'()*+,;=]*)?"
)
__EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
__NUMBER_RANGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?(?:\s*[a-zA-Z]+)?)\s*[〜~～ー]\s*(\d+(?:\.\d+)?(?:\s*[a-zA-Z]+)?)"
)
__NUMBER_MATH_PATTERN = re.compile(
    r"(\d+)\s*([+＋➕\-−－ー➖×✖⨯÷➗*＊])\s*(\d+)\s*=\s*(\d+)"
)
__NUMBER_COMPARISON_PATTERN = re.compile(r"(\d+)\s*([<＜>＞])\s*(\d+)")
__YEAR_MONTH_PATTERN = re.compile(r"(?<!\d)(18|19|20|21|22)(\d{2})/([0-1]?\d)(?!\d)")
__DATE_EXPAND_PATTERN = re.compile(r"\d{2}[-/\.]\d{1,2}[-/\.]\d{1,2}")
__DATE_PATTERN = re.compile(
    r"(?<!\d)(?:\d{4}[-/\.][0-9]{1,2}[-/\.][0-9]{1,2}|\d{2}[-/\.][0-9]{1,2}[-/\.][0-9]{1,2}|[0-9]{1,2}/[0-9]{1,2}|\d{4}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))(?!\d)"
)
__FRACTION_PATTERN = re.compile(r"(\d+)[/／](\d+)")
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
    - `/` （スラッシュは pyopenjtalk での形態素解析処理で重要なので、例外的に正規化後も残し、g2p 処理内で "." に変換される）

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

    # ゼロ幅スペースを削除
    res = res.replace("\u200b", "")

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
    この関数は正規化処理の最初に実行する必要がある（さもなければ英数字のカタカナ変換処理の影響を受けてしまう）。
    処理順序によって結果が変わるので無闇に並び替えてはいけない。

    Args:
        text (str): 正規化するテキスト

    Returns:
        str: 正規化されたテキスト
    """

    # 年月日のゼロ埋めを除去
    text = __DATE_ZERO_PADDING_PATTERN.sub(r"\1", text)

    # 括弧内の曜日表記を変換（日付の後にある場合のみ）
    text = __WEEKDAY_PATTERN.sub(
        lambda m: f"{m.group(1) or m.group(3)}{m.group(2) or m.group(4)}曜日", text
    )

    def convert_url_symbols(match: re.Match[str]) -> str:
        url = match.group(0)
        # 記号を日本語に変換
        # コンマの位置は実際に読み上げた際にちょうど良いテンポ感になるように意図的につけたりつけなかったりしている
        url = url.replace("https://", "エイチティーティーピーエス,")
        url = url.replace("http://", "エイチティーティーピー,")
        url = url.replace(".com", "ドットコム,")
        url = url.replace(".net", "ドットネット,")
        url = url.replace(".org", "ドットオーグ,")
        url = url.replace(".info", "ドットインフォ,")
        url = url.replace(".co.jp", "ドットシーオードットジェイピー,")
        url = url.replace(".jp", "ドットジェイピー,")
        url = url.replace(".", "ドット,")
        url = url.replace("/", ",スラッシュ,")
        url = url.replace("?", ",クエスチョン,")
        url = url.replace("&", ",アンド,")
        url = url.replace("=", "イコール")
        url = url.replace("_", "アンダーバー")
        url = url.replace("-", "ハイフン")
        url = url.replace("#", "シャープ")
        url = url.replace("@", ",アットマーク,")
        url = url.replace(":", "コロン")
        url = url.replace("~", "チルダ")
        url = url.replace("+", "プラス")
        return url.rstrip(",").replace(",,", ",")

    # URL パターンの処理
    text = __URL_PATTERN.sub(convert_url_symbols, text)

    def convert_email_symbols(match: re.Match[str]) -> str:
        email = match.group(0)
        # 記号を日本語に変換
        # コンマの位置は実際に読み上げた際にちょうど良いテンポ感になるように意図的につけたりつけなかったりしている
        email = email.replace("@", ",アットマーク,")
        email = email.replace(".com", "ドットコム")
        email = email.replace(".net", "ドットネット")
        email = email.replace(".org", "ドットオーグ")
        email = email.replace(".info", "ドットインフォ")
        email = email.replace(".co.jp", "ドットシーオードットジェイピー")
        email = email.replace(".jp", "ドットジェイピー")
        email = email.replace(".", "ドット")
        email = email.replace("-", "ハイフン")
        email = email.replace("_", "アンダーバー")
        email = email.replace("+", "プラス")
        return email.rstrip(",").replace(",,", ",")

    # メールアドレスパターンの処理
    text = __EMAIL_PATTERN.sub(convert_email_symbols, text)

    # 数字の範囲を処理
    def convert_range(match: re.Match[str]) -> str:
        start = match.group(1)
        end = match.group(2)
        # 単位を含む場合は単位も含めて変換
        # __UNIT_MAP の単位に対応
        for unit_abbr, unit_full in __UNIT_MAP.items():
            if unit_abbr in start or unit_full in start:
                # 省略形から完全な形に変換
                converted_start = start.replace(unit_abbr, unit_full)
                converted_end = end.replace(unit_abbr, unit_full)
                return f"{converted_start}から{converted_end}"
        return f"{start}から{end}"

    text = __NUMBER_RANGE_PATTERN.sub(convert_range, text)

    def get_symbol_yomi(symbol: str) -> str:
        # 読み間違いを防ぐため、数式の間に挟まれた場合にのみ下記の通り読み上げる
        if symbol in ("-", "−", "－", "ー"):
            return "マイナス"
        if symbol in ("*", "＊"):
            return "かける"
        return __SYMBOL_YOMI_MAP.get(symbol, symbol)

    def get_comparison_yomi(symbol: str) -> str:
        # 比較演算子の読み方を定義
        if symbol in ("<", "＜"):
            return "小なり"
        if symbol in (">", "＞"):
            return "大なり"
        return symbol

    # 数式の処理（括弧内の数式も含む）
    def process_math_expression(text: str) -> str:
        # 数式を処理
        text = __NUMBER_MATH_PATTERN.sub(
            lambda m: f"{m.group(1)}{get_symbol_yomi(m.group(2))}{m.group(3)}イコール{m.group(4)}",
            text,
        )
        # 比較演算子を処理
        text = __NUMBER_COMPARISON_PATTERN.sub(
            lambda m: f"{m.group(1)}{get_comparison_yomi(m.group(2))}{m.group(3)}", text
        )
        return text

    # 括弧内の数式を処理
    text = re.sub(
        r"\(([^()]*)\)", lambda m: f"'{process_math_expression(m.group(1))}'", text
    )
    # 括弧外の数式も処理
    text = process_math_expression(text)

    def date_to_words(match: re.Match[str]) -> str:
        date_str = match.group(0)
        try:
            # 連続した数字形式（YYYYMMDD）の場合
            if len(date_str) == 8 and date_str.isdigit():
                try:
                    date = datetime.strptime(date_str, "%Y%m%d")
                    return f"{date.year}年{date.month}月{date.day}日"
                except ValueError:
                    pass

            # 2桁の年を4桁に拡張する処理 (Y/m/d or Y-m-d or Y.m.d の時のみ)
            if __DATE_EXPAND_PATTERN.match(date_str):
                # スラッシュまたはハイフンまたはドットで分割して年部分を取得
                year_str = (
                    date_str.split("/")[0]
                    if "/" in date_str
                    else (
                        date_str.split("-")[0]
                        if "-" in date_str
                        else date_str.split(".")[0]
                    )
                )
                if len(year_str) == 2:
                    # 50 以降は 1900 年代、49 以前は 2000 年代として扱う
                    # 98/04/11 → 1998/04/11 / 36-01-01 → 2036-01-01
                    year_prefix = "19" if int(year_str) >= 50 else "20"
                    date_str = year_prefix + date_str

            # Y/m/d, Y-m-d, Y.m.d, m/d のパターンを試す
            for fmt in ["%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d", "%m/%d"]:
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

    # 年/月形式の処理（1800-2200年の範囲で、かつ月が1-12の場合のみ）
    def convert_year_month(match: re.Match[str]) -> str:
        year = int(f"{match.group(1)}{match.group(2)}")
        month = int(match.group(3))
        # 月が1-12の範囲外の場合は分数として処理するため、元の文字列を返す
        if not 1 <= month <= 12:
            return match.group(0)
        return f"{year}年{month}月"

    # 年/月パターンの変換
    text = __YEAR_MONTH_PATTERN.sub(convert_year_month, text)

    # 分数の処理
    def convert_fraction(match: re.Match[str]) -> str:
        try:
            numerator = int(match.group(1))
            denominator = int(match.group(2))
            return f"{num2words(denominator, lang='ja')}ぶんの{num2words(numerator, lang='ja')}"
        except ValueError:
            return match.group(0)

    # 分数パターンの変換
    text = __FRACTION_PATTERN.sub(convert_fraction, text)

    # 時刻の処理（漢字で書かれた時分秒）
    def convert_time(match: re.Match[str]) -> str:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3)) if match.group(3) else None

        # 時刻として処理
        result = f'{num2words(hours, lang="ja")}時'

        # 分の処理：0分で秒がない場合は省略、秒がある場合は零分を追加
        if minutes == 0:
            if seconds is not None:
                result += "零分"
        elif 0 <= minutes <= 59:
            result += f'{num2words(minutes, lang="ja")}分'
        else:
            result += f'{num2words(minutes, lang="ja")}'

        # 秒の処理
        if seconds is not None:
            if 0 <= seconds <= 59:
                result += f'{num2words(seconds, lang="ja")}秒'
            else:
                result += f'{num2words(seconds, lang="ja")}'
        return result

    # 時刻パターンの処理（漢字で書かれた時分秒）
    text = __TIME_PATTERN.sub(convert_time, text)

    # 時刻またはアスペクト比の処理
    # 時刻は 00:00:00 から 27:59:59 までの範囲であれば、漢数字に変換して「十四時五分三十秒」「二十四時」のように読み上げる
    # それ以外ならアスペクト比と判断し「十六タイ九」のように読み上げる (「対」にすると「つい」と読んでしまう場合がある)
    def convert_time_or_aspect(match: re.Match[str]) -> str:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3)) if match.group(3) else None

        # 時刻として処理する条件をチェック
        # 時刻らしさを判定：時が0-27の範囲で、分が2桁で表現されている
        looks_like_time = 0 <= hours <= 27 and len(match.group(2)) == 2

        if looks_like_time:
            # 時刻として処理
            result = f'{num2words(hours, lang="ja")}時'

            # 分の処理：0分で秒がない場合は省略、秒がある場合は零分を追加
            if minutes == 0:
                if seconds is not None:
                    result += "零分"
            elif 0 <= minutes <= 59:
                result += f'{num2words(minutes, lang="ja")}分'
            else:
                result += f'{num2words(minutes, lang="ja")}'

            # 秒の処理
            if seconds is not None:
                if 0 <= seconds <= 59:
                    result += f'{num2words(seconds, lang="ja")}秒'
                else:
                    result += f'{num2words(seconds, lang="ja")}'
            return result
        else:
            # アスペクト比として処理
            result = f'{num2words(match.group(1), lang="ja")}タイ{num2words(match.group(2), lang="ja")}'
            if seconds is not None:
                result += f'タイ{num2words(seconds, lang="ja")}'
            return result

    # 時刻またはアスペクト比パターンの処理（コロンで区切られた時分秒）
    text = __ASPECT_PATTERN.sub(convert_time_or_aspect, text)

    # 指数表記の処理
    ## 稀にランダムな英数字 ID にマッチしたことで OverflowError が発生するが、続行に支障はないため無視する
    try:
        text = __EXPONENT_PATTERN.sub(
            lambda m: f'{num2words(float(m.group(0)), lang="ja")}', text
        )
    except OverflowError:
        pass

    # 記号類を辞書で置換
    text = __SYMBOL_YOMI_PATTERN.sub(lambda x: __SYMBOL_YOMI_MAP[x.group()], text)

    # 数字の前のバックスラッシュを円記号に変換
    ## __convert_numbers_to_words() は「¥100」を「100円」と自動で読み替えるが、円記号としてバックスラッシュ (U+005C) が使われているとうまく動作しないため
    ## ref: https://ja.wikipedia.org/wiki/%E5%86%86%E8%A8%98%E5%8F%B7
    text = re.sub(r"\\(?=\d)", "¥", text)

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

    # 単位の変換（平方メートルなどの特殊な単位も含む）
    def convert_unit(match: re.Match[str]) -> str:
        number = match.group(1)
        unit = match.group(2)
        # 特殊な単位の処理
        if unit.endswith("2"):
            base_unit = unit[:-1]
            if base_unit in __UNIT_MAP:
                return f"{number}平方{__UNIT_MAP[base_unit]}"
        elif unit.endswith("3"):
            base_unit = unit[:-1]
            if base_unit in __UNIT_MAP:
                return f"{number}立方{__UNIT_MAP[base_unit]}"
        # 指数表記の場合も単位変換を適用
        if "e" in str(number).lower():
            try:
                num_str = num2words(float(number), lang="ja")
                unit_str = __UNIT_MAP.get(unit, unit)
                return f"{num_str}{unit_str}"
            except (ValueError, OverflowError):
                pass
        return f"{number}{__UNIT_MAP.get(unit, unit)}"

    res = __UNIT_PATTERN.sub(convert_unit, text)
    res = __NUMBER_WITH_SEPARATOR_PATTERN.sub(lambda m: m[0].replace(",", ""), res)
    res = __CURRENCY_PATTERN.sub(
        lambda m: (
            (m[2] + __CURRENCY_MAP.get(m[1], m[1]))
            if m[1]
            else (m[3] + __CURRENCY_MAP.get(m[4], m[4]))
        ),
        res,
    )

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

    def try_split_convert(word: str) -> str | None:
        """
        単語を2つに分割してカタカナ変換を試みる。
        中央から開始して左右に分割位置を移動しながら、両方の部分が辞書に存在する分割を探す。

        Args:
            word (str): 変換する単語

        Returns:
            str | None: 変換に成功した場合はカタカナ文字列、失敗した場合は None
        """

        # 単語を小文字に変換
        word = word.lower()
        n = len(word)

        # 分割位置の候補を生成（中央から左右に広がる順）
        center = n // 2
        # 中央から左右に移動する分割位置のリストを生成
        # 例: 長さ6の単語の場合、[3, 2, 4, 1, 5] の順で試す
        positions = []
        left = center
        right = center + 1
        while left > 0 or right < n:
            if left > 0:
                positions.append(left)
                left -= 1
            if right < n:
                positions.append(right)
                right += 1

        # 各分割位置で試行
        for pos in positions:
            part1 = word[:pos]
            part2 = word[pos:]

            # 両方の部分が辞書に存在するかチェック
            kata1 = KATAKANA_MAP.get(part1)
            if kata1 is None:
                continue

            kata2 = KATAKANA_MAP.get(part2)
            if kata2 is None:
                continue

            # 両方見つかった場合、カタカナを連結して返す
            logger.debug(
                f"Split conversion succeeded: {word} -> {part1}({kata1}) + {part2}({kata2})"
            )
            return kata1 + kata2

        return None

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
                # "-" の場合は、数値の区切りかどうかをチェック
                if separator in [".", "-"]:
                    parts = word.split(separator)
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
            # ハイフンで区切られた数字の場合はそのまま返す (例: 33-4)
            if "-" in word:
                parts = word.split("-")
                if all(part.isdigit() for part in parts):
                    return word

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

        # 9. 最終手段として、2単語への分割を試みる
        # 最低4文字以上の単語のみ対象とし、全て大文字の単語の場合はこの処理を実行しない
        if len(word) >= 4 and not word.isupper():
            split_result = try_split_convert(word)
            if split_result is not None:
                return split_result

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
        print("Usage: python -m style_bert_vits2.nlp.japanese.normalizer <text>")
        sys.exit(1)
    print(normalize_text(sys.argv[1]))
