#!/usr/bin/env python3

"""
BERT パディングの 80トークン→96パディング問題をデバッグするスクリプト
"""

import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

from style_bert_vits2.models.tensor_padding import (
    BERT_TOKEN_PADDING_SIZES,
    get_optimal_padding_size,
    pad_bert_inputs,
)


def test_optimal_padding_size():
    """get_optimal_padding_size 関数の動作をテスト"""
    print("=== get_optimal_padding_size テスト ===")

    test_cases = [
        (79, 1.25),
        (80, 1.25),
        (81, 1.25),
        (95, 1.25),
        (96, 1.25),
        (97, 1.25),
        (80, 1.2),  # より厳格な max_overhead_ratio
        (80, 1.0),  # 完全一致のみ
    ]

    print(f"BERT_TOKEN_PADDING_SIZES: {BERT_TOKEN_PADDING_SIZES}")
    print()

    for actual_length, max_overhead_ratio in test_cases:
        result = get_optimal_padding_size(
            actual_length, BERT_TOKEN_PADDING_SIZES, max_overhead_ratio
        )
        overhead = result / actual_length if result != actual_length else 1.0
        print(
            f"長さ {actual_length:3}, max_ratio {max_overhead_ratio:4.2f} "
            f"→ パディング {result:3}, オーバーヘッド {overhead:4.2f}"
        )


def test_bert_padding():
    """実際の BERT inputs での pad_bert_inputs 動作をテスト"""
    print("\n=== pad_bert_inputs テスト ===")

    # 80トークンの BERT 入力を作成
    batch_size = 1
    seq_len = 80

    inputs = {
        "input_ids": torch.randint(0, 30000, (batch_size, seq_len)),
        "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        "token_type_ids": torch.zeros((batch_size, seq_len), dtype=torch.long),
    }

    print(f"元のトークン長: {seq_len}")
    print(f"input_ids shape: {inputs['input_ids'].shape}")

    # デフォルトの max_overhead_ratio でテスト
    padded_inputs, actual_length = pad_bert_inputs(inputs, use_pool=False)

    print(f"パディング後のトークン長: {padded_inputs['input_ids'].shape[1]}")
    print(f"実際の長さ: {actual_length}")

    # 異なる max_overhead_ratio でテスト
    for ratio in [1.0, 1.2, 1.25, 1.3]:
        print(f"\nmax_overhead_ratio = {ratio}:")
        padded_inputs, actual_length = pad_bert_inputs(
            inputs, max_overhead_ratio=ratio, use_pool=False
        )
        padded_length = padded_inputs["input_ids"].shape[1]
        overhead = padded_length / seq_len
        print(f"  パディング長: {padded_length}, オーバーヘッド: {overhead:4.2f}")


if __name__ == "__main__":
    test_optimal_padding_size()
    test_bert_padding()
