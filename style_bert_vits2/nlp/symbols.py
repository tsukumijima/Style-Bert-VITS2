from __future__ import annotations


# Punctuations
PUNCTUATIONS = ["!", "?", "…", ",", ".", "'", "-"]

# Punctuations and special tokens
PUNCTUATION_SYMBOLS = PUNCTUATIONS + ["SP", "UNK"]

# Padding
PAD = "_"

# Nanairo 専用の絵文字モーラ
## すでに学習済みの non-JP-Extra / JP-Extra 系モデルとの語彙 ID の互換性を維持するため、
## 通常の SYMBOLS には追加せず、Nanairo 専用の追加モーラとして SYMBOLS の末尾に追加し、Nanairo 互換モデルでのみ用いる
## 含まれるのは日本語 BERT のトークナイザできちんと1文字として分割される（複数コードポイントに分割されない）絵文字のみ
## NOTE: 学習完了後は語彙 ID が確定するため、既存世代の定義順の変更や途中への挿入は禁止
## 万が一今後絵文字を追加する場合は、既存の語彙 ID に影響しないよう、新しい世代のリストを末尾に連結すること
## 各世代内は sorted() でコードポイント昇順を保証する
_NANAIRO_EMOJI_SYMBOLS_V1 = sorted(
    [
        "⏩",  # U+23E9
        "⏸",  # U+23F8
        "🌬",  # U+1F32C
        "🎵",  # U+1F3B5
        "🐢",  # U+1F422
        "👂",  # U+1F442
        "👅",  # U+1F445
        "👌",  # U+1F44C
        "💋",  # U+1F48B
        "💨",  # U+1F4A8
        "📞",  # U+1F4DE
        "📢",  # U+1F4E2
        "😆",  # U+1F606
        "😊",  # U+1F60A
        "😌",  # U+1F60C
        "😏",  # U+1F60F
        "😒",  # U+1F612
        "😖",  # U+1F616
        "😟",  # U+1F61F
        "😠",  # U+1F620
        "😪",  # U+1F62A
        "😭",  # U+1F62D
        "😮",  # U+1F62E
        "😰",  # U+1F630
        "😱",  # U+1F631
        "😲",  # U+1F632
        "🙄",  # U+1F644
        "🙏",  # U+1F64F
        "🤐",  # U+1F910
        "🤔",  # U+1F914
        "🤧",  # U+1F927
        "🤭",  # U+1F92D
        "🥤",  # U+1F964
        "🥱",  # U+1F971
        "🥴",  # U+1F974
        "🥵",  # U+1F975
        "🥺",  # U+1F97A
        "🫣",  # U+1FAE3
        "🫶",  # U+1FAF6
    ]
)
NANAIRO_EMOJI_SYMBOLS: list[str] = [
    *_NANAIRO_EMOJI_SYMBOLS_V1,
]

# Chinese symbols
ZH_SYMBOLS = [
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
    "AA",
    "EE",
    "OO",
]
NUM_ZH_TONES = 6

# Japanese
JP_SYMBOLS = [
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "dy",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "u:",
    "v",
    "w",
    "y",
    "z",
    "zy",
]
NUM_JP_TONES = 2

# English
EN_SYMBOLS = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "V",
    "w",
    "y",
    "z",
    "zh",
]
NUM_EN_TONES = 4

# Combine all symbols
NORMAL_SYMBOLS = sorted(set(ZH_SYMBOLS + JP_SYMBOLS + EN_SYMBOLS))
SYMBOLS = [PAD] + NORMAL_SYMBOLS + PUNCTUATION_SYMBOLS
NANAIRO_SYMBOLS = SYMBOLS + NANAIRO_EMOJI_SYMBOLS

# Transformer-based DP/SDP へ渡す duration 専用トークン種別
## 音素 ID は個々の記号を表し、こちらは休止・境界・絵文字制御の粗い性質だけを表す
## 値を変更すると学習済み duration_token_type_emb の意味が変わるため、既存値の並び替えは禁止
DURATION_TOKEN_TYPE_CONTENT = 0
DURATION_TOKEN_TYPE_BLANK = 1
DURATION_TOKEN_TYPE_COMMA = 2
DURATION_TOKEN_TYPE_PERIOD = 3
DURATION_TOKEN_TYPE_ELLIPSIS_DOT = 4
DURATION_TOKEN_TYPE_QUESTION = 5
DURATION_TOKEN_TYPE_EXCLAMATION = 6
DURATION_TOKEN_TYPE_QUOTE_BOUNDARY = 7
DURATION_TOKEN_TYPE_HYPHEN = 8
DURATION_TOKEN_TYPE_EVENT_EMOJI = 9
DURATION_TOKEN_TYPE_PROSODY_MARKER_EMOJI = 10
DURATION_TOKEN_TYPE_NAMES = [
    "content",
    "blank",
    "comma",
    "period",
    "ellipsis_dot",
    "question",
    "exclamation",
    "quote_boundary",
    "hyphen",
    "event_emoji",
    "prosody_marker_emoji",
]
DURATION_TOKEN_TYPE_COUNT = len(DURATION_TOKEN_TYPE_NAMES)

# その位置に非言語音声や発声イベントを置く意図が強い絵文字
## 個々の絵文字 ID は TextEncoder 側に入るため、duration 側では発声イベント性だけを粗く渡す
DURATION_EVENT_EMOJI_SYMBOLS: frozenset[str] = frozenset(
    {
        "🌬",
        "👅",
        "💋",
        "💨",
        "🤧",
        "🥤",
        "🥱",
    }
)

# 文全体や周辺音素の読み味を制御する絵文字
## 未分類の Nanairo 絵文字は局所イベントとして扱わず、短めの制御マーカーとして学習させる
DURATION_PROSODY_MARKER_EMOJI_SYMBOLS: frozenset[str] = frozenset(
    symbol
    for symbol in NANAIRO_EMOJI_SYMBOLS
    if symbol not in DURATION_EVENT_EMOJI_SYMBOLS
)

# Combine all tones
NUM_TONES = NUM_ZH_TONES + NUM_JP_TONES + NUM_EN_TONES

# Language maps
LANGUAGE_ID_MAP = {"ZH": 0, "JP": 1, "EN": 2}
NUM_LANGUAGES = len(LANGUAGE_ID_MAP.keys())

# Language tone start map
LANGUAGE_TONE_START_MAP = {
    "ZH": 0,
    "JP": NUM_ZH_TONES,
    "EN": NUM_ZH_TONES + NUM_JP_TONES,
}


if __name__ == "__main__":
    a = set(ZH_SYMBOLS)
    b = set(EN_SYMBOLS)
    print(sorted(a & b))
