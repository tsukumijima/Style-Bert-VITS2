#!/usr/bin/env python3

"""
実際の日本語テキストを使った BERT パディング動作の検証
"""

import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

import torch

from style_bert_vits2.nlp.japanese.bert_feature import extract_bert_feature
from style_bert_vits2.tts_model import TTSModel


# ログレベルを設定してパディング情報を表示
logging.basicConfig(level=logging.INFO)


def test_real_text_bert_padding():
    """実際の日本語テキストでの BERT パディング動作をテスト"""

    # 様々な長さの日本語テキストを準備
    test_texts = [
        "こんにちは、今日はいい天気ですね。",  # 短いテキスト
        "昨日は友達と一緒に映画を見に行きました。とても面白い映画でした。アクションシーンが迫力満点で、最後まで飽きることなく楽しめました。",  # 中程度
        "今年の夏休みには、家族と一緒に北海道旅行に行く予定です。札幌の時計台や小樽運河を見学し、美味しい海鮮料理を楽しみたいと思います。特に、新鮮なカニとウニを食べるのが楽しみです。北海道の大自然も満喫したいので、富良野のラベンダー畑や美瑛の青い池も訪れる予定です。きっと素晴らしい思い出になることでしょう。",  # 長いテキスト
        "この文章は約八十文字程度になるように調整されたテキストです。パディング動作を確認するために使用します。",  # 約80文字狙い
    ]

    print("=== 実際の日本語テキストでの BERT パディング検証 ===")

    # モデル設定を確認
    try:
        # モデルリストを取得して利用可能なモデルを確認
        from style_bert_vits2.constants import DEFAULT_MODEL_NAME

        model_name = DEFAULT_MODEL_NAME
        print(f"使用モデル: {model_name}")

        # TTSModelを初期化
        tts_model = TTSModel(model_path=f"model_assets/{model_name}")

    except Exception as e:
        print(f"モデル初期化エラー: {e}")
        print("extract_bert_feature を直接テストします")

        for i, text in enumerate(test_texts):
            print(f"\n--- テストケース {i + 1} ---")
            print(f"テキスト: {text}")
            print(f"文字数: {len(text)}")

            try:
                # パディングありでの BERT 特徴抽出
                bert_feature_padded = extract_bert_feature(
                    text, word2ph=None, device="cpu", enable_tensor_padding=True
                )
                print(f"パディングあり BERT特徴形状: {bert_feature_padded.shape}")

                # パディングなしでの BERT 特徴抽出
                bert_feature_normal = extract_bert_feature(
                    text, word2ph=None, device="cpu", enable_tensor_padding=False
                )
                print(f"パディングなし BERT特徴形状: {bert_feature_normal.shape}")

                # 形状の違いを確認
                if bert_feature_padded.shape != bert_feature_normal.shape:
                    print("⚠️  形状に違いあり - パディング効果を確認")
                    print(f"  オリジナル長: {bert_feature_normal.shape[-1]}")
                    print(f"  パディング長: {bert_feature_padded.shape[-1]}")
                    overhead = (
                        bert_feature_padded.shape[-1] / bert_feature_normal.shape[-1]
                    )
                    print(f"  オーバーヘッド: {overhead:.3f}")
                else:
                    print("✓ 形状は同じ（パディングなし）")

            except Exception as e:
                print(f"BERT特徴抽出エラー: {e}")

        return


if __name__ == "__main__":
    test_real_text_bert_padding()
