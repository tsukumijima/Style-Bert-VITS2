#!/usr/bin/env python3
"""
BERT バケツ化の数値精度検証テスト

このスクリプトは、BERTバケツ化前後で hidden_states が数値的に一致することを検証する。
レビュー指摘を受けて、「精度は絶対に劣化しない」を実証するためのテスト。

使用方法:
    .venv/bin/python test_bert_precision.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from style_bert_vits2.constants import Languages
from style_bert_vits2.models.memory_efficient import bucket_bert_inputs
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata


def test_bert_precision(
    device: str = "cuda",
    test_texts: list[str] | None = None,
    tolerance_atol: float = 1e-6,
    tolerance_rtol: float = 1e-6,
) -> dict[str, any]:
    """
    BERTバケツ化の数値精度検証を実行

    Args:
        device: 推論デバイス
        test_texts: テスト用テキストリスト
        tolerance_atol: 絶対許容誤差
        tolerance_rtol: 相対許容誤差

    Returns:
        検証結果の辞書
    """

    if test_texts is None:
        test_texts = [
            "こんにちは。",  # 8 → 8 (バケツ化なし)
            "今日はとてもいい天気です。",  # 15 → 15 (バケツ化なし)
            "短いテキストですがバケツ化テスト用です",  # 17トークン程度 → 32 (バケツ化発生)
            "人工知能の発展により、音声合成技術も大きく進歩しました。",  # 30 → 32 (バケツ化発生)
            "春の訪れとともに、桜の花が咲き始めました。淡いピンク色の花びらが風に舞い、街全体が華やかな雰囲気に包まれています。",
            "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。",
        ]

    print("=" * 80)
    print("BERT バケツ化数値精度検証テスト")
    print("=" * 80)
    print(f"デバイス: {device}")
    print(f"許容誤差: atol={tolerance_atol}, rtol={tolerance_rtol}")
    print(f"テストケース数: {len(test_texts)}")
    print("=" * 80)

    # BERT モデルをロード
    model = bert_models.load_model(Languages.JP, device_map=device, use_fp16=False)
    bert_models.transfer_model(Languages.JP, device)

    results = {
        "total_tests": len(test_texts),
        "passed_tests": 0,
        "failed_tests": 0,
        "test_details": [],
        "overall_passed": True,
    }

    with torch.inference_mode():
        tokenizer = bert_models.load_tokenizer(Languages.JP)

        for i, text in enumerate(test_texts):
            print(f"\nテスト {i + 1}/{len(test_texts)}: {text[:50]}...")

            # テキストを前処理
            sep_text = text_to_sep_kata(text, raise_yomi_error=False)[0]
            processed_text = "".join(sep_text)

            try:
                # オリジナル（バケツ化なし）
                inputs_original = tokenizer(processed_text, return_tensors="pt")
                for key in inputs_original:
                    inputs_original[key] = inputs_original[key].to(device)

                res_original = model(**inputs_original, output_hidden_states=True)
                hidden_states_original = torch.cat(
                    res_original["hidden_states"][-3:-2], -1
                )[0]

                # バケツ化あり（テスト用に詳細情報出力）
                inputs_bucketed, actual_length = bucket_bert_inputs(
                    {
                        k: v.clone() for k, v in inputs_original.items()
                    },  # deep copyで確実な分離
                    pool_type="bert_mid_lived",
                    use_pool=False,  # テスト用に新規作成
                )

                res_bucketed = model(**inputs_bucketed, output_hidden_states=True)
                hidden_states_bucketed = torch.cat(
                    res_bucketed["hidden_states"][-3:-2], -1
                )[0]

                # 実長部分のみ比較（パディング部分は除外）
                hidden_original_slice = hidden_states_original[:actual_length]
                hidden_bucketed_slice = hidden_states_bucketed[:actual_length]

                # 数値精度検証
                is_close = torch.allclose(
                    hidden_original_slice,
                    hidden_bucketed_slice,
                    atol=tolerance_atol,
                    rtol=tolerance_rtol,
                )

                # 詳細メトリクス計算
                abs_diff = torch.abs(hidden_original_slice - hidden_bucketed_slice)
                max_abs_diff = torch.max(abs_diff).item()
                mean_abs_diff = torch.mean(abs_diff).item()

                rel_diff = abs_diff / (torch.abs(hidden_original_slice) + 1e-8)
                max_rel_diff = torch.max(rel_diff).item()
                mean_rel_diff = torch.mean(rel_diff).item()

                test_detail = {
                    "text": text,
                    "token_length": actual_length,
                    "bucket_size": inputs_bucketed["input_ids"].shape[1],
                    "passed": is_close,
                    "max_abs_diff": max_abs_diff,
                    "mean_abs_diff": mean_abs_diff,
                    "max_rel_diff": max_rel_diff,
                    "mean_rel_diff": mean_rel_diff,
                }

                results["test_details"].append(test_detail)

                if is_close:
                    results["passed_tests"] += 1
                    print("  ✓ PASSED: 数値一致確認")
                else:
                    results["failed_tests"] += 1
                    results["overall_passed"] = False
                    print("  ✗ FAILED: 数値不一致検出")
                    print(f"    最大絶対誤差: {max_abs_diff:.2e}")
                    print(f"    最大相対誤差: {max_rel_diff:.2e}")

                    # デバッグ情報: 各テンソルの詳細確認
                    print(f"    デバッグ: 入力キー: {list(inputs_original.keys())}")
                    print(
                        f"    デバッグ: オリジナル attention_mask[:5]: {inputs_original['attention_mask'][0, :5]}"
                    )
                    print(
                        f"    デバッグ: バケツ化後 attention_mask[:5]: {inputs_bucketed['attention_mask'][0, :5]}"
                    )
                    print(
                        f"    デバッグ: パディング部分 attention_mask[-5:]: {inputs_bucketed['attention_mask'][0, -5:]}"
                    )
                    if "token_type_ids" in inputs_original:
                        print(
                            f"    デバッグ: オリジナル token_type_ids[:5]: {inputs_original['token_type_ids'][0, :5]}"
                        )
                        print(
                            f"    デバッグ: バケツ化後 token_type_ids[:5]: {inputs_bucketed['token_type_ids'][0, :5]}"
                        )
                        print(
                            f"    デバッグ: パディング部分 token_type_ids[-5:]: {inputs_bucketed['token_type_ids'][0, -5:]}"
                        )
                    print(
                        f"    デバッグ: input_ids形状 original vs bucketed: {inputs_original['input_ids'].shape} vs {inputs_bucketed['input_ids'].shape}"
                    )

                    # BERTトークナイザーのパディングトークンID確認
                    pad_token_id = (
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else 0
                    )
                    print(f"    デバッグ: パディングトークンID: {pad_token_id}")
                    print(
                        f"    デバッグ: オリジナル input_ids[:5]: {inputs_original['input_ids'][0, :5]}"
                    )
                    print(
                        f"    デバッグ: バケツ化後 input_ids[:5]: {inputs_bucketed['input_ids'][0, :5]}"
                    )
                    print(
                        f"    デバッグ: パディング部分 input_ids[-5:]: {inputs_bucketed['input_ids'][0, -5:]}"
                    )

                print(
                    f"  トークン長: {actual_length} → バケツサイズ: {inputs_bucketed['input_ids'].shape[1]}"
                )
                print(
                    f"  平均絶対誤差: {mean_abs_diff:.2e}, 最大絶対誤差: {max_abs_diff:.2e}"
                )
                print(
                    f"  平均相対誤差: {mean_rel_diff:.2e}, 最大相対誤差: {max_rel_diff:.2e}"
                )

            except Exception as e:
                print(f"  ✗ ERROR: テスト実行中にエラーが発生: {e}")
                results["failed_tests"] += 1
                results["overall_passed"] = False
                results["test_details"].append(
                    {
                        "text": text,
                        "passed": False,
                        "error": str(e),
                    }
                )

    # 結果サマリー
    print("\n" + "=" * 80)
    print("テスト結果サマリー")
    print("=" * 80)
    print(f"総テスト数: {results['total_tests']}")
    print(f"成功: {results['passed_tests']}")
    print(f"失敗: {results['failed_tests']}")

    if results["overall_passed"]:
        print("\n🎉 すべてのテストが合格しました！")
        print("   BERTバケツ化による数値精度への影響は検出されませんでした。")
    else:
        print("\n⚠️  一部のテストが失敗しました。")
        print("   実装の見直しが必要かもしれません。")

    # 統計情報
    if results["test_details"]:
        passed_details = [d for d in results["test_details"] if d.get("passed", False)]
        if passed_details:
            max_abs_diffs = [
                d["max_abs_diff"] for d in passed_details if "max_abs_diff" in d
            ]
            max_rel_diffs = [
                d["max_rel_diff"] for d in passed_details if "max_rel_diff" in d
            ]

            if max_abs_diffs and max_rel_diffs:
                print("\n成功テストの誤差範囲:")
                print(
                    f"  絶対誤差: 最大 {max(max_abs_diffs):.2e}, 平均 {np.mean(max_abs_diffs):.2e}"
                )
                print(
                    f"  相対誤差: 最大 {max(max_rel_diffs):.2e}, 平均 {np.mean(max_rel_diffs):.2e}"
                )

    return results


def main():
    parser = argparse.ArgumentParser(description="BERT バケツ化数値精度検証")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="推論デバイス (default: cuda)",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-6, help="絶対許容誤差 (default: 1e-6)"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-6, help="相対許容誤差 (default: 1e-6)"
    )

    args = parser.parse_args()

    try:
        results = test_bert_precision(
            device=args.device,
            tolerance_atol=args.atol,
            tolerance_rtol=args.rtol,
        )

        # 結果を終了コードで返す
        exit_code = 0 if results["overall_passed"] else 1
        exit(exit_code)

    except KeyboardInterrupt:
        print("\nテストが中断されました。")
        exit(1)
    except Exception as e:
        print(f"\nテスト実行中にエラーが発生しました: {e}")
        exit(1)


if __name__ == "__main__":
    main()
