from pathlib import Path

from style_bert_vits2.utils.strenum import StrEnum


# Style-Bert-VITS2 のバージョン
VERSION = "2.7.0"

# Style-Bert-VITS2 のベースディレクトリ
BASE_DIR = Path(__file__).parent.parent

# デフォルトの学習用データセットのルートディレクトリ
## {model_folder_name} の学習データは {DATASET_ROOT}/{model_folder_name}/ に配置する
DEFAULT_DATASET_ROOT = BASE_DIR / "Data"

# デフォルトの推論用モデルアセットのルートディレクトリ
## 学習時は {ASSETS_ROOT}/{model_folder_name}/ にモデルが保存され、
## 推論時は {ASSETS_ROOT} 以下の全モデルを読み込む
DEFAULT_ASSETS_ROOT = BASE_DIR / "model_assets"

# デフォルトのパス設定ファイルのパス
DEFAULT_PATHS_CONFIG_PATH = BASE_DIR / "configs/paths.yml"
# デフォルトのパス設定ファイルのテンプレートのパス
DEFAULT_PATHS_TEMPLATE_PATH = BASE_DIR / "configs/default_paths.yml"


# 利用可能な言語
## JP-Extra モデル利用時は JP 以外の言語の音声合成はできない
class Languages(StrEnum):
    JP = "JP"
    EN = "EN"
    ZH = "ZH"


# 言語ごとのデフォルトの BERT モデルのパス
DEFAULT_BERT_MODEL_PATHS = {
    Languages.JP: BASE_DIR / "bert" / "deberta-v2-large-japanese-char-wwm",
    Languages.EN: BASE_DIR / "bert" / "deberta-v3-large",
    Languages.ZH: BASE_DIR / "bert" / "chinese-roberta-wwm-ext-large",
}

# 言語ごとのデフォルトの BERT モデル (ONNX 版) のパス
DEFAULT_ONNX_BERT_MODEL_PATHS = {
    Languages.JP: BASE_DIR / "bert" / "deberta-v2-large-japanese-char-wwm-onnx",
    Languages.EN: BASE_DIR / "bert" / "deberta-v3-large-onnx",
    Languages.ZH: BASE_DIR / "bert" / "chinese-roberta-wwm-ext-large-onnx",
}

# デフォルトのユーザー辞書ディレクトリ
## style_bert_vits2.nlp.japanese.user_dict モジュールのデフォルト値として利用される
## ライブラリとしての利用などで外部のユーザー辞書を指定したい場合は、user_dict 以下の各関数の実行時、引数に辞書データファイルのパスを指定する
DEFAULT_USER_DICT_DIR = BASE_DIR / "dict_data"

# デフォルトの推論パラメータ
DEFAULT_STYLE = "Neutral"
DEFAULT_STYLE_WEIGHT = 1.0
DEFAULT_SDP_RATIO = 0.2
DEFAULT_NOISE = 0.6
DEFAULT_NOISEW = 0.8
DEFAULT_LENGTH = 1.0
DEFAULT_LINE_SPLIT = True
DEFAULT_SPLIT_INTERVAL = 0.5
DEFAULT_ASSIST_TEXT_WEIGHT = 0.7

# 分散学習用環境変数
DEFAULT_TRAIN_ENV: dict[str, str] = {
    "MASTER_ADDR": "localhost",
    "MASTER_PORT": "10086",
    "WORLD_SIZE": "1",
    "LOCAL_RANK": "0",
    "RANK": "0",
}

# 学習済みモデルの聴き比べ・評価に使う固定の日本語テキスト
## speech_mos.py での MOS 評価のほか、外部ツールから未知文の合成確認に使うことを想定している
EVALUATION_TEXTS: list[str] = [
    # JVNVコーパスのテキスト
    # https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus
    # CC BY-SA 4.0
    "ああ？どうしてこんなに荒々しい態度をとるんだ？落ち着いて話を聞けばいいのに。",
    "いや、あんな醜い人間を見るのは本当に嫌だ。",
    "うわ、不景気の影響で失業してしまうかもしれない。どうしよう、心配で眠れない。",
    "今日の山登りは最高だった！山頂で見た景色は言葉に表せないほど美しかった！あはは、絶頂の喜びが胸に溢れるよ！",
    "あーあ、昨日の事故で大切な車が全損になっちゃった。もうどうしようもないよ。",
    "ああ、彼は本当に速い！ダッシュの速さは尋常じゃない！",
    # 以下 app.py の説明文章
    "音声合成は、機械学習を活用して、テキストから人の声を再現する技術です。この技術は、言語の構造を解析し、それに基づいて音声を生成します。",
    "この分野の最新の研究成果を使うと、より自然で表現豊かな音声の生成が可能である。深層学習の応用により、感情やアクセントを含む声質の微妙な変化も再現することが出来る。",
    # 追加テキスト
    "おはようございます！現在時刻は7時30分です。今日の東京の気温は18度で、天気は雨です。10時からミーティング、午後3時に歯医者の予約があります。今日も素敵な一日になりますように。",
    "やった〜！テストでようやく満点取れた〜！めちゃくちゃ嬉しい…。　そうそう、さっき読んでたこの漫画がめっちゃ面白くてさ〜！見てよこれ！",
    "ごめんね、今ちょっと風邪気味なんだよね…。それでもよければ会いたいけど、どう？　…………そっか…。コロナ流行ってるもんね。じゃまた今度にしようか。…元気になったらぜひご飯でも！",
    "イーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。またそのなかでいっしょになったたくさんのひとたち、ファゼーロとロザーロ、羊飼のミーロや、顔の赤いこどもたち、地主のテーモ、山猫博士のボーガント・デストゥパーゴなど、いまこの暗い巨きな家にはたったひとりがいません。",
    "小笠原近海で台風５号が発生しました。今後、北上し、関東から東北の太平洋側に沿って北上した後、北海道付近に到達する可能性が大きくなっています。",
    "もし関東へ上陸すれば６年ぶり、東北に上陸すれば２年連続、北海道へ上陸すれば９年ぶりとなります。この台風の進路の特徴とともに、詳しくみていきましょう。",
    "濁流は、メロスの叫びをせせら笑う如く、ますます激しく躍り狂う。浪は浪を呑み、捲き、煽り立て、そうして時は、刻一刻と消えて行く。今はメロスも覚悟した。泳ぎ切るより他に無い。ああ、神々も照覧あれ！",
    "万博協会が13日に発表した、12日（土）の大阪・関西万博の一般来場者数は速報値ベースで約16万4000人、パビリオンなどの関係者を含めた総来場者数は約18万2000人で、1日あたりの来場者数が“過去3番目”となりました。",
    "12日は、航空自衛隊の「ブルーインパルス」による展示飛行が行われ、多くの人が会場に詰めかけ、歓声を上げました。",
    "13日も午後2時40分ごろに関西空港を離陸後、大阪市の通天閣や吹田市の万博記念公園などの上空を通過した上で、午後3時ごろから15分程度、会場上空などで展示飛行が予定されています。",
    "サンプリングレート 44100Hz で出力されるクリアな音質は、IVR やカスタマーサポート、音声アシスタントなど、お客様と直接コミュニケーションをとる重要な場面でも、安心してご利用いただける品質です。",
    "感情表現の強さやテンポの調整も自由自在。例えば、重要なお知らせでは感情表現を抑えた信頼感のあるトーン、プロモーションでは明るく親しみやすいトーンといった具合に、お客様のブランドイメージに合わせた音声コミュニケーションを設計できます。",
    "血圧は 118/76、脈拍は 72 です。念のため、胸の痛みや息切れが続く場合は早めに受診してください。",
    "円相場は 1 ドル 150 円台まで円安が進みました。金利差の拡大を背景に、輸入コストの上昇が懸念されています。",
    "次の駅は新宿です。お降りの方は足元にご注意ください。",
    "ログの出力が増えたので、レベルを INFO から WARN に変更して挙動を確認します。",
    "この製品は 12V で動作し、消費電力は最大 18W です。連続使用は 30 分までにしてください。",
]

# Gradio のテーマ
## Built-in theme: "default", "base", "monochrome", "soft", "glass"
## See https://huggingface.co/spaces/gradio/theme-gallery for more themes
GRADIO_THEME = "NoCrypt/miku"
