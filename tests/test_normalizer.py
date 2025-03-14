from style_bert_vits2.nlp.japanese.normalizer import normalize_text


def test_normalize_text_basic():
    """基本的な正規化のテスト"""
    # 基本的な句読点の正規化
    assert normalize_text("こんにちは。さようなら。") == "こんにちは.さようなら."
    assert normalize_text("おはよう、こんばんは、") == "おはよう,こんばんは,"
    assert normalize_text("すごい！やばい！") == "すごい!やばい!"
    assert normalize_text("なに？どうして？") == "なに?どうして?"
    # 特殊な空白文字
    assert normalize_text("text\u200btext") == "テキストテキスト"  # ゼロ幅スペース
    assert normalize_text("text\u3000text") == "テキスト.テキスト"  # 全角スペース
    assert normalize_text("text\ttext") == "テキストテキスト"  # タブ
    # 制御文字
    assert normalize_text("text\ntext") == "テキスト.テキスト"  # 改行
    assert normalize_text("text\rtext") == "テキストテキスト"  # キャリッジリターン
    # 重複する記号
    assert normalize_text("！！！？？？。。。") == "!!!???..."
    assert normalize_text("。。。、、、") == "...,,,"


def test_normalize_text_units():
    """単位関連の正規化のテスト"""
    # 基本的な単位
    assert normalize_text("100m") == "100メートル"
    assert normalize_text("100cm") == "100センチメートル"
    assert normalize_text("1000.19mm") == "1000.19ミリメートル"
    assert normalize_text("1km") == "1キロメートル"
    assert normalize_text("500mL") == "500ミリリットル"
    assert normalize_text("1L") == "1リットル"
    assert normalize_text("1000.19kL") == "1000.19キロリットル"
    assert normalize_text("1000.19mg") == "1000.19ミリグラム"
    assert normalize_text("100g") == "100グラム"
    assert normalize_text("2kg") == "2キログラム"
    # スラッシュ付き単位（変換しない）
    assert normalize_text("100m/s") == "100m/s"
    assert normalize_text("100kL/m") == "100kL/m"
    assert normalize_text("100g/㎥") == "100g/m3"
    assert normalize_text("100km/h") == "100km/h"
    # データ容量
    assert normalize_text("1B") == "1バイト"
    assert normalize_text("1KB") == "1キロバイト"
    assert normalize_text("1MB") == "1メガバイト"
    assert normalize_text("1GB") == "1ギガバイト"
    assert normalize_text("1TB") == "1テラバイト"
    assert normalize_text("1000.11MiB") == "1000.11メビバイト"
    assert normalize_text("1000.11GiB") == "1000.11ギビバイト"
    assert normalize_text("1000.11TiB") == "1000.11テビバイト"
    # 面積・体積
    assert normalize_text("100m2") == "100平方メートル"
    assert normalize_text("1km2") == "1平方キロメートル"
    assert normalize_text("50m3") == "50立方メートル"
    # 単位付き指数
    assert normalize_text("1.23e-6") == "零点零零零零零一二三"
    assert normalize_text("1.23e+4") == "一万二千三百"
    assert normalize_text("1.23e-4") == "零点零零零一二三"
    assert normalize_text("1e6") == "百万"
    # 単位付きの範囲
    assert normalize_text("100m〜200m") == "100メートルから200メートル"
    assert normalize_text("1kg〜2kg") == "1キログラムから2キログラム"
    assert normalize_text("100dL〜200dL") == "100デシリットルから200デシリットル"
    # 追加のテストケース
    assert normalize_text("100tトラック") == "100トントラック"
    assert normalize_text("100.1919tトラック") == "100.1919トントラック"
    assert normalize_text("345.56t") == "345.56トン"
    assert normalize_text("345.56test") == "345.56テスト"
    assert normalize_text("345.56t") == "345.56トン"
    assert normalize_text("345.56ms") == "345.56ミリ秒"
    assert normalize_text("345ms") == "345ミリ秒"
    assert normalize_text("24hを") == "24時間を"
    assert normalize_text("24h営業") == "24時間営業"
    assert normalize_text("24ms営業") == "24ミリ秒営業"
    assert normalize_text("24s営業") == "24秒営業"
    assert normalize_text("500Kがある") == "500Kがある"
    assert normalize_text("50℃") == "50度"
    assert normalize_text("50ms") == "50ミリ秒"
    assert normalize_text("50s") == "50秒"
    assert normalize_text("50ns") == "50ナノ秒"
    assert normalize_text("50μs") == "50マイクロ秒"
    assert normalize_text("50ms") == "50ミリ秒"
    assert normalize_text("50s") == "50秒"
    assert normalize_text("50h") == "50時間"
    assert normalize_text("1h") == "1時間"
    assert normalize_text("1h3m5s") == "1時間3メートル5秒"
    assert normalize_text("1h5s") == "1時間5秒"
    assert normalize_text("300s") == "300秒"
    assert normalize_text("3hで") == "3時間で"
    assert normalize_text("3hzで") == "3ヘルツで"
    assert normalize_text("30d") == "30日"
    assert normalize_text("30dでなんとかした") == "30日でなんとかした"
    assert normalize_text("でも30dでなんとかした") == "でも30日でなんとかした"
    assert normalize_text("でも30daysでなんとかした") == "でも30デイズでなんとかした"
    assert normalize_text("でも30date") == "でも30デート"
    assert normalize_text("でも30dで") == "でも30日で"
    assert normalize_text("でも30Dで") == "でも30Dで"
    assert normalize_text("\\100") == "100円"
    assert normalize_text("$100で") == "100ドルで"
    assert normalize_text("€100で") == "100ユーロで"
    assert normalize_text("5㎞") == "5キロメートル"
    assert normalize_text("5㎡") == "5平方メートル"


def test_normalize_text_currency():
    """通貨関連の正規化のテスト"""
    # 各種通貨記号
    assert normalize_text("$100") == "100ドル"
    assert normalize_text("¥100") == "100円"
    assert normalize_text("€100") == "100ユーロ"
    assert normalize_text("£100") == "100ポンド"
    assert normalize_text("₩1000") == "1000ウォン"
    # 通貨記号の位置による違い
    assert normalize_text("100$") == "100ドル"
    assert normalize_text("100¥") == "100円"
    # 金額の桁区切り
    assert normalize_text("¥1,234,567") == "1234567円"
    assert normalize_text("$1,234.56") == "1234.56ドル"
    # 通貨の単位
    assert normalize_text("1億円") == "1億円"
    assert normalize_text("100万ドル") == "100万ドル"
    # 特殊な通貨
    assert normalize_text("₿1.5") == "1.5ビットコイン"
    assert normalize_text("₹100") == "100ルピー"
    assert normalize_text("₽50") == "50ルーブル"
    assert normalize_text("₺25") == "25リラ"
    assert normalize_text("฿1000") == "1000バーツ"
    assert normalize_text("₱100") == "100ペソ"
    assert normalize_text("₴50") == "50フリヴニャ"
    assert normalize_text("₫1000") == "1000ドン"
    assert normalize_text("₪100") == "100シェケル"
    assert normalize_text("₦500") == "500ナイラ"
    assert normalize_text("₡1000") == "1000コロン"


def test_normalize_text_dates():
    """日付関連の正規化のテスト"""
    # 様々な日付形式
    assert normalize_text("2024/01/01") == "2024年1月1日"
    assert normalize_text("2024-01-01") == "2024年1月1日"
    assert normalize_text("2024年01月01日") == "2024年1月1日"  # 0埋めを除去
    assert normalize_text("01/01") == "1月1日"
    assert normalize_text("1/1") == "1月1日"
    # 曜日付きの日付
    assert normalize_text("2024/01/01(月)") == "2024年1月1日月曜日"
    assert normalize_text("2024-01-01（火）") == "2024年1月1日火曜日"
    assert normalize_text("2024/01/01（月曜）") == "2024年1月1日'月曜'"
    assert normalize_text("2024/01/01（月曜日）") == "2024年1月1日'月曜日'"
    # 年月のみ
    assert normalize_text("2024/01") == "2024年1月"
    assert normalize_text("1930/9") == "1930年9月"
    assert normalize_text("1880/10") == "1880年10月"
    assert normalize_text("2081/12") == "2081年12月"
    assert normalize_text("2181/12") == "2181年12月"
    # 2桁年の自動補完 (50以上は1900年代、49以下は2000年代)
    assert normalize_text("98/01/01") == "1998年1月1日"
    assert normalize_text("24/01/01") == "2024年1月1日"
    # 和暦
    assert normalize_text("令和6年1月1日") == "令和6年1月1日"
    assert normalize_text("平成30年12月31日") == "平成30年12月31日"
    # 日付範囲
    assert normalize_text("1/1〜1/3") == "1月1日から1月3日"
    # 年月
    assert normalize_text("2024年1月") == "2024年1月"
    # 区切り文字のバリエーション
    assert normalize_text("2024.01.01") == "2024年1月1日"
    assert normalize_text("20240101") == "2024年1月1日"
    assert normalize_text("19640820") == "1964年8月20日"
    # 省略表記の和暦
    assert normalize_text("R6.1.1") == "令和6年1月1日"
    assert normalize_text("R6.01.01") == "令和6年1月1日"
    assert normalize_text("H31.4.30") == "平成31年4月30日"
    assert normalize_text("H31.04.30") == "平成31年4月30日"
    assert normalize_text("S64.1.7") == "昭和64年1月7日"
    assert normalize_text("S64.01.07") == "昭和64年1月7日"
    assert normalize_text("S47.12.31") == "昭和47年12月31日"
    # 零時チェック
    assert normalize_text("午前00時") == "午前零時"
    assert normalize_text("午後00時") == "午後零時"
    assert normalize_text("午前00時00分") == "午前零時"
    assert normalize_text("午後00時00分") == "午後零時"
    assert normalize_text("午前00時00分00秒") == "午前零時零分零秒"
    assert normalize_text("午後00時00分00秒") == "午後零時零分零秒"
    assert normalize_text("今日は0時に就寝します") == "今日は零時に就寝します"
    assert normalize_text("今日は00時に就寝します") == "今日は零時に就寝します"
    assert (
        normalize_text("今日は0時間勉強した") == "今日は0時間勉強した"
    )  # 変換されない
    assert normalize_text("1000時間勉強した") == "1000時間勉強した"  # 変換されない
    # 異常な日付
    assert (
        normalize_text("2024/13/01") == "十三ぶんの二千二十四/01"
    )  # 13月は異常値なので分数判定される
    assert (
        normalize_text("2024/01/32") == "2024年1月/32"
    )  # 32日は異常値なので年と月だけ変換される
    assert normalize_text("2024/02/30") == "2024年2月/30"  # 存在しない日付
    assert normalize_text("2024/00/00") == "零ぶんの二千二十四/00"  # ゼロの月日

    # 追加のテストケース
    assert normalize_text("2024年5月8日 （月）") == "2024年5月8日月曜日"
    assert normalize_text("2024年5月8日（月）") == "2024年5月8日月曜日"
    assert normalize_text("2024年5月8日　（月）") == "2024年5月8日月曜日"
    assert normalize_text("2024年5月8日　　（月）") == "2024年5月8日月曜日"
    assert normalize_text("2024年05月08日　　（月）") == "2024年5月8日月曜日"
    assert normalize_text("08日　　（月）") == "8日月曜日"
    assert normalize_text("05/31　　（月）") == "5月31日月曜日"
    assert normalize_text("05/30　　（月）") == "5月30日月曜日"
    assert normalize_text("05/20　　（月）") == "5月20日月曜日"
    assert normalize_text("02/21　　（月）") == "2月21日月曜日"
    assert normalize_text("12/21　　（月）") == "12月21日月曜日"
    assert normalize_text("24/12/21　　（月）") == "2024年12月21日月曜日"
    assert normalize_text("24/02/21　　（月）") == "2024年2月21日月曜日"
    assert normalize_text("24/02/1　　（月）") == "2024年2月1日月曜日"
    assert normalize_text("24/02/01　　（月）") == "2024年2月1日月曜日"
    assert normalize_text("24/02/01(月)") == "2024年2月1日月曜日"
    assert normalize_text("24/02/01 (月)") == "2024年2月1日月曜日"
    assert normalize_text("24/02/01 (火)") == "2024年2月1日火曜日"
    assert normalize_text("24/02/01 (水)") == "2024年2月1日水曜日"
    assert normalize_text("24/02/29 (木)") == "2024年2月29日木曜日"
    assert normalize_text("24/02/29 (金)") == "2024年2月29日金曜日"
    assert normalize_text("24/02/29 (土)") == "2024年2月29日土曜日"
    assert normalize_text("24/02/29 （日）") == "2024年2月29日日曜日"
    assert normalize_text("24/02/29") == "2024年2月29日"
    assert normalize_text("98/02/21（水）") == "1998年2月21日水曜日"
    assert normalize_text("98/02/21") == "1998年2月21日"
    assert normalize_text("98/02（水）") == "二ぶんの九十八水曜日"
    assert normalize_text("（水）") == "'水'"
    assert normalize_text("そうだ（水）") == "そうだ'水'"
    assert normalize_text("そうだ（水）に（）行こう") == "そうだ'水'に''行こう"
    assert normalize_text("そうだ（水）に行こう") == "そうだ'水'に行こう"
    assert normalize_text("01/01") == "1月1日"
    assert normalize_text("01月03") == "1月03"
    assert normalize_text("01月03日") == "1月3日"
    assert normalize_text("1/3") == "1月3日"
    assert normalize_text("01月03日") == "1月3日"
    assert normalize_text("95/01/03") == "1995年1月3日"
    assert normalize_text("01/03") == "1月3日"
    assert normalize_text("今年01/03にですね") == "今年1月3日にですね"
    assert normalize_text("今年12/3にですね") == "今年12月3日にですね"
    assert normalize_text("今年1/3にですね") == "今年1月3日にですね"
    assert normalize_text("今年9/13にですね") == "今年9月13日にですね"
    assert normalize_text("今年08-13にですね") == "今年08-13にですね"
    assert normalize_text("今年24-08-13にですね") == "今年2024年8月13日にですね"
    assert normalize_text("今年08/13にですね") == "今年8月13日にですね"
    assert normalize_text("今年24/12/03にですね") == "今年2024年12月3日にですね"
    assert normalize_text("今年12/03にですね") == "今年12月3日にですね"
    assert normalize_text("今年08/13にですね") == "今年8月13日にですね"
    assert normalize_text("20年には") == "20年には"
    assert normalize_text("05年には") == "05年には"
    assert normalize_text("85年には") == "85年には"
    assert normalize_text("05年01月には") == "05年1月には"
    assert normalize_text("09-01-03 24:34") == "2009年1月3日二十四時三十四分"
    assert normalize_text("87-01-03 24:34") == "1987年1月3日二十四時三十四分"
    assert normalize_text("明治45年07月30日") == "明治45年7月30日"
    assert normalize_text("大正15年12月25日") == "大正15年12月25日"
    assert normalize_text("昭和64年01月07日") == "昭和64年1月7日"
    assert normalize_text("平成31年04月30日") == "平成31年4月30日"
    assert normalize_text("令和05年12月31日") == "令和05年12月31日"
    assert normalize_text("西暦2024年1月1日") == "西暦2024年1月1日"
    assert normalize_text("AD2024") == "エーディー2024"
    assert normalize_text("BC356") == "ビーシー356"


def test_normalize_text_time():
    """時刻関連の正規化のテスト"""
    # 基本的な時刻表現
    assert normalize_text("9時30分") == "九時三十分"
    assert normalize_text("14時00分") == "十四時"
    assert normalize_text("7時45分30秒") == "七時四十五分三十秒"
    # コロン区切りの時刻
    assert normalize_text("09:30") == "九時三十分"
    assert normalize_text("14:00") == "十四時"
    assert normalize_text("07:45:30") == "七時四十五分三十秒"
    # アスペクト比（時刻として解釈されない数値の組み合わせ）
    assert normalize_text("16:9") == "十六タイ九"
    # 午前・午後
    assert normalize_text("午前9時30分") == "午前九時三十分"
    assert normalize_text("午後3時45分") == "午後三時四十五分"
    # 特殊な時刻
    assert normalize_text("0時0分") == "零時"
    assert normalize_text("24時00分") == "二十四時"
    assert normalize_text("25:00") == "二十五時"  # 25時までは許容
    assert normalize_text("30:00") == "三十タイ零"  # 30時はアスペクト比として解釈される
    # 秒以下の単位
    assert normalize_text("10時20分30.5秒") == "十時二十分30.5秒"
    # 異常な時刻
    assert normalize_text("24:60") == "二十四時六十"  # 存在しない分
    assert normalize_text("00:00:60") == "零時零分六十"  # 存在しない秒
    # 27時台までは許容
    assert normalize_text("27:59:00") == "二十七時五十九分零秒"
    assert (
        normalize_text("28:00:00") == "二十八タイ零タイ零"
    )  # 28時はアスペクト比として解釈される

    # 追加のテストケース
    assert normalize_text("03:34に") == "三時三十四分に"
    assert normalize_text("03:3:564に") == "三タイ三タイ五百六十四に"
    assert normalize_text("03:34:54に") == "三時三十四分五十四秒に"
    assert normalize_text("03:03:03に") == "三時三分三秒に"
    assert normalize_text("03:03:62に") == "三時三分六十二に"
    assert normalize_text("03:03:60に") == "三時三分六十に"
    assert normalize_text("03:03:59に") == "三時三分五十九秒に"
    assert normalize_text("03:03:01に") == "三時三分一秒に"
    assert normalize_text("03:03:5に") == "三時三分五秒に"
    assert normalize_text("04:03") == "四時三分"
    assert normalize_text("4:3") == "四タイ三"
    assert normalize_text("16:3") == "十六タイ三"
    assert normalize_text("04:3") == "四タイ三"
    assert normalize_text("4:30") == "四時三十分"
    assert normalize_text("04:30") == "四時三十分"
    assert normalize_text("04時30分") == "四時三十分"
    assert normalize_text("2024年05月08日 03時06分08秒") == "2024年5月8日三時六分八秒"
    assert normalize_text("2024年05月08日 00時00分00秒") == "2024年5月8日零時零分零秒"
    assert normalize_text("2024:05:08 00:00:00") == "二千二十四タイ五タイ八零時零分零秒"
    assert normalize_text("2024/05/08 00:03:00") == "2024年5月8日零時三分零秒"
    assert normalize_text("2024/05/08 00:03") == "2024年5月8日零時三分"
    assert normalize_text("2024/05/08 00") == "2024年5月8日00"
    assert normalize_text("2024/05/08 0:30") == "2024年5月8日零時三十分"
    assert normalize_text("2024年05月01日") == "2024年5月1日"
    assert normalize_text("2024/05/01") == "2024年5月1日"
    assert normalize_text("2024年05月01日") == "2024年5月1日"
    assert normalize_text("2024年05月01日 03時00分0秒") == "2024年5月1日三時零分零秒"
    assert normalize_text("2024年05月01日 03時0分0秒") == "2024年5月1日三時零分零秒"
    assert normalize_text("2024年05月08日 03時06分08秒") == "2024年5月8日三時六分八秒"
    assert normalize_text("2024年05月08日 00時00分30秒") == "2024年5月8日零時零分三十秒"
    assert normalize_text("2024年05月08日 00時00分０0秒") == "2024年5月8日零時零分零秒"
    assert normalize_text("2024年05月08日 00時00分00秒") == "2024年5月8日零時零分零秒"
    assert normalize_text("2024年05月08日 00時00分") == "2024年5月8日零時"
    assert normalize_text("2024年05月08日 03時00分") == "2024年5月8日三時"
    assert normalize_text("2024年05月08日 03時00分0秒") == "2024年5月8日三時零分零秒"
    assert normalize_text("2024年05月08日 03時00分") == "2024年5月8日三時"
    assert normalize_text("2024年05月08日 03時01分") == "2024年5月8日三時一分"
    assert normalize_text("2024年05月08日 03:01") == "2024年5月8日三時一分"
    assert normalize_text("2024/05/08 03:01") == "2024年5月8日三時一分"
    assert normalize_text("2024/05/08 03:01:00") == "2024年5月8日三時一分零秒"
    assert normalize_text("2024/05/08 3時1分00") == "2024年5月8日三時一分00"
    assert normalize_text("2024/05/08 3時1分0秒") == "2024年5月8日三時一分零秒"
    assert normalize_text("2024/05/08 3時1分60秒") == "2024年5月8日三時一分六十"
    assert normalize_text("2024/05/08 3時1分59秒") == "2024年5月8日三時一分五十九秒"
    assert normalize_text("27:59:00") == "二十七時五十九分零秒"
    assert normalize_text("27:59") == "二十七時五十九分"
    assert normalize_text("27:59:00") == "二十七時五十九分零秒"
    assert normalize_text("28:59:00") == "二十八タイ五十九タイ零"
    assert normalize_text("28:59") == "二十八タイ五十九"
    assert normalize_text("03:03:03") == "三時三分三秒"
    assert normalize_text("30:1:03") == "三十タイ一タイ三"
    assert normalize_text("30:1:03:5") == "三十タイ一タイ三,5"
    assert normalize_text("30:1:03:5:07") == "三十タイ一タイ三,五時七分"
    assert normalize_text("1:3:4") == "一タイ三タイ四"
    assert normalize_text("1:3:4:5") == "一タイ三タイ四,5"
    assert normalize_text("1:3:4:5:6") == "一タイ三タイ四,五タイ六"
    assert normalize_text("1:3:4:5:6:7") == "一タイ三タイ四,五タイ六タイ七"
    assert normalize_text("1:3:4:5:6:7:8:9") == "一タイ三タイ四,五タイ六タイ七,八タイ九"
    assert normalize_text("16:9") == "十六タイ九"
    assert normalize_text("4:3") == "四タイ三"
    assert normalize_text("3:2") == "三タイ二"
    assert normalize_text("03:02") == "三時二分"
    assert normalize_text("03:2") == "三タイ二"
    assert normalize_text("3:02") == "三時二分"
    assert normalize_text("03:02") == "三時二分"
    assert normalize_text("03:2") == "三タイ二"
    assert normalize_text("03:02") == "三時二分"
    assert normalize_text("03:02:00") == "三時二分零秒"
    assert normalize_text("03:00:00") == "三時零分零秒"
    assert normalize_text("24:00:00") == "二十四時零分零秒"
    assert normalize_text("24:00") == "二十四時"
    assert normalize_text("27:59:00") == "二十七時五十九分零秒"
    assert normalize_text("00:00:00.123") == "零時零分零秒.123"
    assert normalize_text("12:00 PM") == "十二時ピーエム"
    assert normalize_text("12:00 AM") == "十二時エーエム"
    assert normalize_text("深夜03時") == "深夜3時"
    assert normalize_text("未明04時") == "未明4時"
    assert normalize_text("早朝05時") == "早朝5時"
    assert normalize_text("夜09時") == "夜9時"
    assert normalize_text("正午") == "正午"
    assert normalize_text("正午12時") == "正午12時"
    assert normalize_text("0:00:00") == "零時零分零秒"


def test_normalize_text_fractions():
    """分数関連の正規化のテスト (明確に日付ではないパターンのみ分数として読まれる)"""
    assert normalize_text("123/456") == "四百五十六ぶんの百二十三"
    assert normalize_text("1/100") == "百ぶんの一"
    assert normalize_text("13/32") == "三十二ぶんの十三"
    # 分数を含む文
    assert normalize_text("材料の2/30を使用した。") == "材料の三十ぶんの二を使用した."
    assert normalize_text("残り時間は1/40です。") == "残り時間は四十ぶんの一です."

    # 追加のテストケース
    assert normalize_text("16/9") == "九ぶんの十六"
    assert normalize_text("1/2") == "1月2日"  # 日付として解釈される
    assert normalize_text("1/32") == "三十二ぶんの一"
    assert normalize_text("9/16") == "9月16日"  # 日付として解釈される


def test_normalize_text_symbols():
    """記号関連の正規化のテスト"""
    # 基本的な記号
    assert normalize_text("ABC+ABC") == "エービーシープラスエービーシー"
    assert normalize_text("ABC&ABC") == "エービーシーアンドエービーシー"
    # 数式
    assert normalize_text("1+1=2") == "1プラス1イコール2"
    assert normalize_text("5-3=2") == "5マイナス3イコール2"
    assert normalize_text("2×3=6") == "2かける3イコール6"
    assert normalize_text("6÷2=3") == "6わる2イコール3"
    # 比較演算子
    assert normalize_text("5>3") == "5大なり3"
    assert normalize_text("5≥3") == "5大なりイコール3"
    assert normalize_text("2<4") == "2小なり4"
    assert normalize_text("2≤4") == "2小なりイコール4"


def test_normalize_text_url_email():
    """URL・メールアドレス関連の正規化のテスト"""
    # URL
    assert (
        normalize_text("https://example.com")
        == "エイチティーティーピーエス,イグザンプルドットコム"
    )
    assert (
        normalize_text("http://test.jp")
        == "エイチティーティーピー,テストドットジェイピー"
    )
    # メールアドレス
    assert (
        normalize_text("test@example.com")
        == "テスト,アットマーク,イグザンプルドットコム"
    )
    assert (
        normalize_text("info@test.co.jp")
        == "インフォ,アットマーク,テストドットシーオードットジェイピー"
    )


def test_normalize_text_english():
    """英語関連の正規化のテスト"""
    # 基本的な英単語
    assert normalize_text("Hello") == "ハロー"
    assert normalize_text("Good Morning") == "グッドモーニング"
    assert normalize_text("Node.js") == "ノードジェイエス"
    # CamelCase
    assert normalize_text("JavaScript") == "ジャバスクリプト"
    assert normalize_text("TypeScript") == "タイプスクリプト"
    # 複合語
    assert normalize_text("e-mail") == "イーメール"
    assert normalize_text("YouTube") == "ユーチューブ"
    # 辞書にない単語の変換
    assert normalize_text("windsurfeditor") == "ウインドサーフエディター"
    assert (
        normalize_text("WINDSURFEDITOR") == "WINDSURFEDITOR"
    )  # 全て大文字の場合は変換しない
    assert normalize_text("DevinProgrammerAgents") == "デビンプログラマーエージェンツ"
    # 敬称の処理
    assert normalize_text("Mr. John") == "ミスタージョン"
    assert normalize_text("Mrs. Smith") == "ミセススミス"
    assert normalize_text("Ms. Jane") == "ミズジェーン"
    assert normalize_text("Dr. Brown") == "ドクターブラウン"
    assert normalize_text("John Smith Jr.") == "ジョンスミスジュニア"
    assert normalize_text("John Smith Sr.") == "ジョンスミスシニア"
    assert normalize_text("Mr Smith") == "ミスタースミス"  # ピリオドなし
    assert normalize_text("Dr Brown") == "ドクターブラウン"  # ピリオドなし
    assert normalize_text("Mr. and Mrs. Smith") == "ミスターアンドミセススミス"
    assert normalize_text("Dr. Smith Jr.") == "ドクタースミスジュニア"
    assert (
        normalize_text("Mr. John Smith Jr. PhD")
        == "ミスタージョンスミスジュニアピーエイチディー"
    )


def test_normalize_text_mathematical():
    """数学記号関連の正規化のテスト"""
    # 数学記号
    assert normalize_text("∞") == "無限大"
    assert normalize_text("π") == "パイ"
    assert normalize_text("√4") == "ルート4"
    assert normalize_text("∛8") == "立方根8"
    assert normalize_text("∜16") == "四乗根16"
    assert normalize_text("∑") == "シグマ"
    assert normalize_text("∫") == "インテグラル"
    assert normalize_text("∬") == "二重積分"
    assert normalize_text("∭") == "三重積分"
    assert normalize_text("∮") == "周回積分"
    assert normalize_text("∯") == "面積分"
    assert normalize_text("∰") == "体積分"
    assert normalize_text("∂") == "パーシャル"
    assert normalize_text("∇") == "ナブラ"
    assert normalize_text("∝") == "比例"
    # 集合記号
    assert normalize_text("∈") == "属する"
    assert normalize_text("∉") == "属さない"
    assert normalize_text("∋") == "含む"
    assert normalize_text("∌") == "含まない"
    assert normalize_text("∪") == "和集合"
    assert normalize_text("∩") == "共通部分"
    assert normalize_text("⊂") == "部分集合"
    assert normalize_text("⊃") == "上位集合"
    assert normalize_text("⊄") == "部分集合でない"
    assert normalize_text("⊅") == "上位集合でない"
    assert normalize_text("⊆") == "部分集合または等しい"
    assert normalize_text("⊇") == "上位集合または等しい"
    assert normalize_text("∅") == "空集合"
    assert normalize_text("∖") == "差集合"
    assert normalize_text("∆") == "対称差"
    # 幾何記号
    assert normalize_text("∥") == "平行"
    assert normalize_text("⊥") == "垂直"
    assert normalize_text("∠") == "角"
    assert normalize_text("∟") == "直角"
    assert normalize_text("∡") == "測定角"
    assert normalize_text("∢") == "球面角"


def test_normalize_text_ranges():
    """範囲表現の正規化のテスト"""
    # 数値範囲
    assert normalize_text("1〜10") == "1から10"
    assert normalize_text("1~10") == "1から10"
    assert normalize_text("1～10") == "1から10"
    # 文字を含む範囲
    assert normalize_text("AからZ") == "AからZ"
    assert normalize_text("1から100まで") == "1から100まで"
    # 単位付きの範囲
    assert normalize_text("100m〜200m") == "100メートルから200メートル"
    assert normalize_text("1kg〜2kg") == "1キログラムから2キログラム"


def test_normalize_text_enclosed_characters():
    """囲み文字の正規化のテスト"""
    # 丸付き数字
    assert normalize_text("①②③④⑤⑥⑦⑧⑨⑩") == "12345678910"
    assert normalize_text("⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳") == "11121314151617181920"
    # 囲み文字（漢字）
    assert normalize_text("㈱") == "株式会社"
    assert normalize_text("㈲") == "有限会社"
    assert normalize_text("㈳") == "社団法人"
    assert normalize_text("㈴") == "合名会社"
    assert normalize_text("㈵") == "特殊法人"
    assert normalize_text("㈶") == "財団法人"
    assert normalize_text("㈷") == "祝日"
    assert normalize_text("㈸") == "労働組合"
    assert normalize_text("㈹") == "代表電話"
    assert normalize_text("㈺") == "呼出し電話"
    assert normalize_text("㈻") == "学校法人"
    assert normalize_text("㈼") == "監査法人"
    assert normalize_text("㈽") == "企業組合"
    assert normalize_text("㈾") == "合資会社"
    assert normalize_text("㈿") == "協同組合"
    assert normalize_text("㊤㊥㊦") == "上中下"
    assert normalize_text("㊧㊨") == "左右"
    assert normalize_text("㊩") == "医療法人"
    assert normalize_text("㊪") == "宗教法人"
    assert normalize_text("㊫") == "学校法人"
    assert normalize_text("㊬") == "監査法人"
    assert normalize_text("㊭") == "企業組合"
    assert normalize_text("㊮") == "合資会社"
    assert normalize_text("㊯") == "協同組合"


def test_normalize_text_mixed_scripts():
    """文字種混在のテスト"""
    # 漢字・ひらがな・カタカナの混在
    assert (
        normalize_text("漢字とひらがなとカタカナの混在文")
        == "漢字とひらがなとカタカナの混在文"
    )
    # 英数字との混在
    assert normalize_text("123と漢字とABCの混在") == "123と漢字とエービーシーの混在"
    # 記号との混在
    assert normalize_text("漢字+カタカナ=混在!?") == "漢字プラスカタカナイコール混在!?"
    # 特殊文字との混在
    assert normalize_text("①漢字②ひらがな③カタカナ") == "1漢字2ひらがな3カタカナ"
    # 単位との混在
    assert (
        normalize_text("漢字100kg+カタカナ500m")
        == "漢字100キログラムプラスカタカナ500メートル"
    )


def test_normalize_text_edge_cases():
    """エッジケースの正規化のテスト"""
    # 空文字列
    assert normalize_text("") == ""
    # 記号のみ
    assert normalize_text("...") == "..."
    assert normalize_text("!!!") == "!!!"
    assert normalize_text("???") == "???"
    # 数字のみ
    assert normalize_text("12345") == "12345"

    # 結合文字の濁点・半濁点
    assert normalize_text("か゛") == "か"  # 結合文字の濁点は削除
    assert normalize_text("は゜") == "は"  # 結合文字の半濁点は削除

    # 極端に長い数値
    assert normalize_text("12345678901234567890") == "12345678901234567890"
    # 極端に長い英単語
    assert (
        normalize_text("supercalifragilisticexpialidocious")
        == "supercalifragilisticexpialidocious"
    )
    # 特殊な文字の組み合わせ
    assert normalize_text("㊊㊋㊌㊍㊎㊏㊐") == "月火水木金土日"  # 曜日の丸文字
    assert (
        normalize_text("㍉㌔㌢㍍㌘㌧㌃㌶㍑㍗")
        == "ミリキロセンチメートルグラムトンアールヘクタールリットルワット"
    )


def test_normalize_text_complex():
    """複合的なパターンの正規化のテスト"""
    # 日付・時刻・単位を含む文
    assert (
        normalize_text("2024/01/01(月)の14時30分に1.5kgの荷物を受け取った。")
        == "2024年1月1日月曜日の十四時三十分に1.5キログラムの荷物を受け取った."
    )
    assert (
        normalize_text("MacBookで1080p/60fpsの動画を2GB保存した。")
        == "マックブックで1080p/60fpsの動画を2ギガバイト保存した."
    )
    assert (
        normalize_text("¥1,000の商品を2個買うと、¥2,000です（1,000×2=2,000）。")
        == "1000円の商品を2個買うと,2000円です'1000かける2イコール2000'."
    )
    assert (
        normalize_text(
            "お問い合わせは、info@example.comまたはhttps://example.com/contactまで！"
        )
        == "お問い合わせは,インフォ,アットマーク,イグザンプルドットコムまたはエイチティーティーピーエス,イグザンプルドットコム,スラッシュ,コンタクトまで!"
    )
    assert (
        normalize_text("09:30に家を出発し、2km先のスーパーで500gのお肉を買った。")
        == "九時三十分に家を出発し,2キロメートル先のスーパーで500グラムのお肉を買った."
    )
    assert (
        normalize_text(
            "2024/05/01にWindowsのアップデート（2GB+500MB=2.5GB）を実施する。"
        )
        == "2024年5月1日にウィンドウズのアップデート'2ギガバイトプラス500メガバイトイコール2.5ギガバイト'を実施する."
    )
    assert (
        normalize_text(
            "CPU使用率が50%を超え、メモリ消費が2GBに達した時点で、Windows Serverは自動的に再起動します。"
        )
        == "シーピーユー使用率が50パーセントを超え,メモリ消費が2ギガバイトに達した時点で,ウィンドウズサーバーは自動的に再起動します."
    )
    assert (
        normalize_text(
            "新商品のiPhone 15 Pro Max (256GB)が¥158,000(税込)で発売！9/22(金)午前10時から予約受付開始。"
        )
        == "新商品のアイフォン15プロマックス'256ギガバイト'が158000円'税込'で発売!9月22日金曜日午前10時から予約受付開始."
    )
    assert (
        normalize_text(
            "株式会社Deeptest(担当：山田)様、10/1(月)15:00〜17:00にWeb会議(https://meet.example.com/test)を設定しました。"
        )
        == "株式会社ディープテスト'担当,山田'様,10月1日月曜日十五時から十七時にウェブ会議'エイチティーティーピーエス,ミートドット,イグザンプルドットコム,スラッシュ,テスト'を設定しました."
    )
    assert (
        normalize_text(
            "材料(4人分)：牛肉250g、玉ねぎ1個、水300mL、醤油大さじ2(30mL)、砂糖20g。"
        )
        == "材料'4人分',牛肉250グラム,玉ねぎ1個,水300ミリリットル,醤油大さじ2'30ミリリットル',砂糖20グラム."
    )
    assert (
        normalize_text(
            "2つの数 a, b があり、a:b = 2:3 で、a + b = 10 のとき、a = 4, b = 6 となります。"
        )
        == "2つの数a,bがあり,a,bイコール二タイ三で,aプラスbイコール10のとき,aイコール4,bイコール6となります."
    )
    assert (
        normalize_text(
            "JavaScriptでArray.prototype.map()を使用し、配列の要素を2倍にする処理を1/100秒で実行。"
        )
        == "ジャバスクリプトでアレイプロトタイプマップ''を使用し,配列の要素を2倍にする処理を百ぶんの一秒で実行."
    )
    assert (
        normalize_text(
            "今日01/03（月）にですね、16:9の映像を1/128の確率で表示するイベントをやっていて、85/09/30の08月01日(金)にお会いした人と久々に会うんです"
        )
        == "今日1月3日月曜日にですね,十六タイ九の映像を百二十八ぶんの一の確率で表示するイベントをやっていて,1985年9月30日の8月1日金曜日にお会いした人と久々に会うんです"
    )
