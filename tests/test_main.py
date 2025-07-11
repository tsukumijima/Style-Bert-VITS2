import random
from collections.abc import Sequence
from typing import Any, Literal

import pytest
from scipy.io import wavfile

from style_bert_vits2.constants import BASE_DIR, Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.tts_model import TTSModelHolder


# 多様なサンプルテキストリスト (50個程度)
# キャッシュの影響を避け、より現実的なパフォーマンス測定を行うため、
# 固定テキストではなく、多様な長さと内容を持つリストからランダムに選択する
RANDOM_SAMPLE_TEXTS: list[str] = [
    "おはようございます。",
    "こんにちは。",
    "こんばんは。",
    "ありがとうございます。",
    "すみません。",
    "了解しました。",
    "これはペンです。",
    "今日の天気は晴れです。",
    "明日は雨が降るでしょう。",
    "週末の予定は決まりましたか？",
    "この映画はとても面白かったです。",
    "新しいカフェがオープンしました。",
    "駅まで歩いて 10 分です。",
    "電車が遅れています。",
    "会議の時間は午後 2 時です。",
    "レポートの締め切りは明日です。",
    "何かお手伝いできることはありますか？",
    "電話番号を教えていただけますか？",
    "メールアドレスはこちらです。",
    "ウェブサイトで詳細を確認できます。",
    "桜の花が満開で綺麗ですね。",
    "紅葉の季節が待ち遠しいです。",
    "夏は海に行って泳ぎたいです。",
    "冬はスキーやスノーボードを楽しみたいです。",
    "読書が好きで、特にミステリー小説をよく読みます。",
    "音楽を聴くのが趣味です。最近はクラシックにはまっています。",
    "料理をするのが好きで、週末は新しいレシピに挑戦しています。",
    "旅行が好きで、国内外問わず様々な場所を訪れています。",
    "ペットに犬を飼っています。名前はポチです。",
    "猫カフェに行って癒されたいです。",
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
    "祇園精舎の鐘の声、諸行無常の響きあり。沙羅双樹の花の色、盛者必衰の理をあらはす。",
    "走れメロス。メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。",
    "国境の長いトンネルを抜けると雪国であった。夜の底が白くなった。信号所に汽車が止まった。",
    "事実は小説よりも奇なり、とはよく言ったものです。まさかこんな結末を迎えるとは思いもしませんでした。",
    "AI 技術の発展は目覚ましく、私たちの生活のあらゆる側面に影響を与え始めています。",
    "再生可能エネルギーへの転換は、地球温暖化対策において避けて通れない道であり、世界各国で取り組みが進められています。",
    "経済格差の拡大は、社会全体の不安定化を招く可能性があり、持続可能な社会の実現に向けた課題の一つです。",
    "少子高齢化が急速に進む日本では、労働人口の減少や社会保障費の増大といった問題が深刻化しています。",
    "国際情勢は常に変化しており、地政学的なリスクや経済的な変動要因を的確に把握することが重要です。",
    "この度は、私たちのプロジェクトにご関心をお寄せいただき、誠にありがとうございます。詳細についてご説明させていただきます。",
    "ご不明な点がございましたら、いつでもお気軽に担当者までお問い合わせください。迅速に対応いたします。",
    "今後とも変わらぬご支援、ご鞭撻のほど、どうぞよろしくお願い申し上げます。",
    "皆様の益々のご発展とご健勝を心よりお祈り申し上げます。素晴らしい一日をお過ごしください。",
    "大変長らくお待たせいたしました。ご注文いただきました特製シーフードピザでございます。熱々ですのでお気をつけください。",
    "誠に申し訳ございませんが、ただいま満席となっております。空き次第、順番にご案内いたしますので、少々お待ちいただけますでしょうか。",
    "当店では、アレルギーをお持ちのお客様向けに、特別メニューもご用意しております。ご希望の際はお気軽にスタッフまでお申し付けください。",
    "食後のデザートに、当店パティシエ特製の季節のフルーツタルトはいかがでしょうか？ コーヒーまたは紅茶とセットでお得になります。",
    "お会計は、お席にて承ります。お呼び出しボタンを押していただくか、お近くのスタッフにお声がけください。",
    "本日はご来店いただき、誠にありがとうございました。またのお越しをスタッフ一同、心よりお待ちしております。",
]

# 推論に使用するモデル
TEST_MODELS: list[str] = [
    "koharune-ami",
    "amitaro",
    # "jvnv-F1-jp",
    # "jvnv-F2-jp",
    # "jvnv-M1-jp",
    # "jvnv-M2-jp",
]


def synthesize(
    inference_type: Literal["torch", "onnx"] = "torch",
    device: str = "cpu",
    onnx_providers: Sequence[tuple[str, dict[str, Any]]] = [
        ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
    ],
    use_random_texts: bool = False,
    use_fp16: bool = False,
):
    # 音声合成モデルが配置されていれば、音声合成を実行
    model_holder = TTSModelHolder(
        BASE_DIR / "model_assets",
        device=device,
        onnx_providers=onnx_providers,
        use_fp16=use_fp16,
    )
    if len(model_holder.models_info) > 0:
        for model_info in model_holder.models_info:
            if model_info.name in TEST_MODELS:
                # Safetensors 形式または ONNX 形式のモデルファイルに絞り込む
                if inference_type == "torch":
                    model_files = [
                        f
                        for f in model_info.files
                        if f.endswith(".safetensors") and not f.startswith(".")
                    ]
                else:
                    model_files = [
                        f
                        for f in model_info.files
                        if f.endswith(".onnx") and not f.startswith(".")
                    ]
                if len(model_files) == 0:
                    pytest.skip(
                        f'音声合成モデル "{model_info.name}" のモデルファイルが見つかりませんでした。'
                    )

                # モデルをロード
                model = model_holder.get_model(model_info.name, model_files[0])
                model.load()

                # ロードされた InferenceSession の ExecutionProvider が一致するか確認
                # 一致しない場合、指定された ExecutionProvider で推論できない状態
                if inference_type == "onnx":
                    assert model.onnx_session is not None
                    assert model.onnx_session.get_providers()[0] == onnx_providers[0][0]

                # すべてのスタイルに対して音声合成を実行
                for style in model_info.styles:
                    logger.info(f"Testing style: {style}")

                    # テストに使用するサンプルテキスト (use_random_texts=False の場合)
                    fixed_sample_texts = [
                        "こんにちは、初めまして。あなたの名前はなんていうの？",
                        "桜の樹の下には屍体が埋まっている！これは信じていいことなんだよ。",
                        "あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しい。",
                        "音声合成は、機械学習を活用して、テキストから人の声を再現する技術です。この技術は、言語の構造を解析し、それに基づいて音声を生成します。",
                    ]

                    # 各スタイルに対して 4 回音声合成を実行
                    for i in range(4):
                        # 使用するテキストを選択
                        if use_random_texts:
                            selected_text = random.choice(RANDOM_SAMPLE_TEXTS)
                        else:
                            selected_text = fixed_sample_texts[
                                i % len(fixed_sample_texts)
                            ]

                        # 音声合成を実行
                        sample_rate, audio_data = model.infer(
                            selected_text,
                            # 言語 (JP, EN, ZH / JP-Extra モデルの場合は JP のみ)
                            language=Languages.JP,
                            # 話者 ID (音声合成モデルに複数の話者が含まれる場合のみ必須、単一話者のみの場合は 0)
                            speaker_id=0,
                            # テンポの緩急 (0.0 〜 1.0)
                            sdp_ratio=0.4,
                            # スタイル (Neutral, Happy など)
                            style=style,
                            # スタイルの強さ (0.0 〜 100.0)
                            style_weight=2.0,
                        )

                        # 音声データを保存
                        output_dir = BASE_DIR / f"tests/wavs/{model_info.name}"
                        output_dir.mkdir(exist_ok=True, parents=True)
                        # ファイル名はスタイル名と実行インデックス (1~5) で一意にする
                        wav_file_path = output_dir / f"{style}_{i + 1:02d}.wav"
                        with open(wav_file_path, "wb") as f:
                            wavfile.write(f, sample_rate, audio_data)

                        # 音声データが保存されたことを確認
                        assert wav_file_path.exists()

                # モデルをアンロード
                model.unload()
    else:
        pytest.skip("音声合成モデルが見つかりませんでした。")


def test_synthesize_cpu():
    synthesize(inference_type="torch", device="cpu")


def test_synthesize_cpu_random_texts():
    synthesize(inference_type="torch", device="cpu", use_random_texts=True)


def test_synthesize_cuda():
    synthesize(inference_type="torch", device="cuda")


def test_synthesize_cuda_random_texts():
    synthesize(inference_type="torch", device="cuda", use_random_texts=True)


def test_synthesize_cuda_fp16():
    synthesize(inference_type="torch", device="cuda", use_fp16=True)


def test_synthesize_cuda_random_texts_fp16():
    synthesize(
        inference_type="torch", device="cuda", use_random_texts=True, use_fp16=True
    )


def test_synthesize_onnx_cpu():
    synthesize(
        inference_type="onnx",
        onnx_providers=[
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ],
    )


def test_synthesize_onnx_cpu_random_texts():
    synthesize(
        inference_type="onnx",
        onnx_providers=[
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ],
        use_random_texts=True,
    )


def test_synthesize_onnx_cuda():
    synthesize(
        inference_type="onnx",
        onnx_providers=[
            (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "DEFAULT",
                },
            ),
        ],
    )


def test_synthesize_onnx_cuda_random_texts():
    synthesize(
        inference_type="onnx",
        onnx_providers=[
            (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "DEFAULT",
                },
            ),
        ],
        use_random_texts=True,
    )


def test_synthesize_onnx_tensorrt():
    synthesize(
        inference_type="onnx",
        onnx_providers=[
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                },
            ),
        ],
    )


def test_synthesize_onnx_directml():
    synthesize(
        inference_type="onnx",
        onnx_providers=[
            ("DmlExecutionProvider", {"device_id": 0}),
        ],
    )


def test_synthesize_onnx_coreml():
    synthesize(
        inference_type="onnx",
        onnx_providers=[
            ("CoreMLExecutionProvider", {}),
        ],
    )
