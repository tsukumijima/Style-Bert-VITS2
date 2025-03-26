from collections.abc import Sequence
from typing import Any, Literal

import pytest
from scipy.io import wavfile

from style_bert_vits2.constants import BASE_DIR, Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.tts_model import TTSModelHolder


def synthesize(
    inference_type: Literal["torch", "onnx"] = "torch",
    device: str = "cpu",
    onnx_providers: Sequence[tuple[str, dict[str, Any]]] = [
        ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
    ],
):

    # 音声合成モデルが配置されていれば、音声合成を実行
    model_holder = TTSModelHolder(BASE_DIR / "model_assets", device, onnx_providers)
    if len(model_holder.models_info) > 0:

        # "koharune-ami" または "amitaro" モデルを探す
        for model_info in model_holder.models_info:
            if model_info.name == "koharune-ami" or model_info.name == "amitaro":

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

                    # テストに使用するサンプルテキスト
                    sample_texts = [
                        "こんにちは、初めまして。あなたの名前はなんていうの？",
                        "桜の樹の下には屍体が埋まっている！これは信じていいことなんだよ。",
                        "あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しい。",
                        "音声合成は、機械学習を活用して、テキストから人の声を再現する技術です。この技術は、言語の構造を解析し、それに基づいて音声を生成します。",
                    ]

                    # 各サンプルテキストに対して音声合成を実行
                    for i, text in enumerate(sample_texts):

                        # 音声合成を実行
                        sample_rate, audio_data = model.infer(
                            text,
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
                        (BASE_DIR / f"tests/wavs/{model_info.name}").mkdir(exist_ok=True, parents=True)  # fmt: skip
                        wav_file_path = BASE_DIR / f"tests/wavs/{model_info.name}/{style}_{i+1:02d}.wav"  # fmt: skip
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


def test_synthesize_cuda():
    synthesize(inference_type="torch", device="cuda")


def test_synthesize_onnx_cpu():
    synthesize(
        inference_type="onnx",
        onnx_providers=[
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ],
    )


def test_synthesize_onnx_cuda():
    synthesize(
        inference_type="onnx",
        onnx_providers=[
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "DEFAULT"}),  # fmt: skip
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
