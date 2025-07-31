#!/usr/bin/env python3
"""
実環境再現型メモリプロファイリングスクリプト

本番環境の使用パターンを模倣した断片化測定：
- ランダムなモデル選択
- ランダムなテキスト選択（多様な長さ）
- 累積的な断片化追跡
- 長時間運用をシミュレート
- テンソルパディング機能の検証

Usage: PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:True" .venv/bin/python -m tests.realistic_memory_profiling [--device cuda] [--iterations 50] [--interval 2] [--enable-padding]
"""

import os


# PYTORCH_CUDA_ALLOC_CONFの確認と警告
cuda_alloc_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "")
if "cudaMallocAsync" not in cuda_alloc_conf:
    print("WARNING: PYTORCH_CUDA_ALLOC_CONF is not set for optimal memory management!")
    print(
        'Please run with: PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:True"'
    )
    print(
        "This profiling may not accurately reflect production memory behavior without proper configuration."
    )
    print()
else:
    print(f"Using PYTORCH_CUDA_ALLOC_CONF: {cuda_alloc_conf}")
    print()

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from style_bert_vits2.constants import (
    BASE_DIR,
    DEFAULT_LENGTH,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models.infer import infer
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder


# 実環境を模倣した多様なテキストセット（100種類以上）
REALISTIC_TEXTS = [
    # ニュース風（10-50文字）
    "本日、東京で記者会見が開かれました。",
    "円相場は一時150円台まで上昇しました。",
    "新型コロナウイルスの感染者数が減少傾向にあります。",
    "気象庁は台風5号の接近に注意を呼びかけています。",
    "政府は新たな経済対策を発表しました。",
    "株価は前日比200円高で取引を終えました。",
    "オリンピックの開催まであと100日となりました。",
    "地震による被害状況が明らかになってきました。",
    "新しい法案が国会で可決されました。",
    "企業の決算発表が相次いでいます。",
    # 日常会話（5-30文字）
    "おはようございます。",
    "今日もお疲れ様でした。",
    "ありがとうございました。",
    "すみません、もう一度お願いします。",
    "はい、わかりました。",
    "それでは失礼します。",
    "お元気ですか？",
    "今日は何をしましたか？",
    "明日は晴れるといいですね。",
    "最近どうですか？",
    # 物語風（50-150文字）
    "昔々、ある山奥に一人の老人が住んでいました。老人は毎日山を歩き、薬草を集めて暮らしていました。",
    "春の訪れとともに、桜の花が咲き始めました。人々は花見を楽しみ、新しい季節の始まりを祝いました。",
    "深い森の中で、少女は不思議な光を見つけました。それは小さな妖精が放つ光でした。",
    "海辺の町では、毎年夏になると大きな祭りが開かれます。太鼓の音が響き、人々は踊り明かします。",
    "雨の日の午後、猫は窓辺で外を眺めていました。雨粒が窓を叩く音が、静かな部屋に響いていました。",
    # 説明文（100-200文字）
    "人工知能技術の発展により、私たちの生活は大きく変わりつつあります。音声認識、画像認識、自然言語処理など、様々な分野でAIが活用されています。今後もこの技術の進歩は続き、より便利で豊かな社会の実現が期待されています。",
    "地球温暖化は、現在人類が直面している最も深刻な環境問題の一つです。温室効果ガスの排出量を削減し、持続可能な社会を築くことが急務となっています。一人一人の行動が、地球の未来を決めることになるでしょう。",
    "日本の四季は、それぞれに独特の美しさがあります。春の桜、夏の花火、秋の紅葉、冬の雪景色。これらの季節の変化は、日本人の感性や文化に深く根ざしており、多くの芸術作品のインスピレーションとなってきました。",
    # ビジネス文書風（150-250文字）
    "本日は貴重なお時間をいただき、誠にありがとうございます。弊社の新サービスについてご説明させていただきます。このサービスは、お客様のニーズに合わせてカスタマイズ可能で、業務効率の大幅な向上が期待できます。導入事例では、平均して30%の時間短縮を実現しています。ぜひこの機会にご検討いただければ幸いです。",
    "プロジェクトの進捗についてご報告いたします。現在、全体の70%が完了しており、予定通り来月末には完成予定です。しかし、一部の機能については追加の要件が発生したため、スケジュールの見直しが必要となる可能性があります。詳細については、明日の会議でご相談させていただければと思います。",
    # 技術文書風（200-300文字）
    "機械学習モデルの性能を向上させるためには、適切なデータの前処理が不可欠です。データのクリーニング、正規化、特徴量エンジニアリングなど、様々な手法を組み合わせることで、モデルの精度を大幅に改善することができます。また、過学習を防ぐために、交差検証や正則化などの技術も重要です。これらの技術を適切に活用することで、実用的で信頼性の高いモデルを構築することが可能となります。",
    # 詩的な表現（20-80文字）
    "風が優しく頬を撫でる春の日。",
    "月明かりが水面に揺れている。静寂の中に美しさがある。",
    "朝露に濡れた花びらが、太陽の光を受けてきらめいている。",
    "遠くで汽笛が鳴る。旅立ちの時が来た。",
    # 指示・命令文（10-50文字）
    "右に曲がって、まっすぐ進んでください。",
    "この書類に記入してください。",
    "明日までに提出をお願いします。",
    "静かにしてください。",
    "ドアを閉めてください。",
    # 質問文（10-40文字）
    "今何時ですか？",
    "どこに行きたいですか？",
    "これはいくらですか？",
    "何か飲み物はいかがですか？",
    "手伝いましょうか？",
    # 感情表現（5-30文字）
    "すごい！素晴らしいですね！",
    "残念です。",
    "嬉しいです！",
    "心配しないでください。",
    "頑張ってください！",
    # 長文（300-500文字）
    "私たちが住むこの地球は、太陽系の第三惑星として知られています。地球の表面の約70%は海で覆われており、残りの30%が陸地となっています。地球には様々な気候帯があり、赤道付近の熱帯地域から、両極の極地まで、多様な環境が存在します。この多様性が、地球上の豊かな生態系を支えています。現在、地球上には約870万種の生物が存在すると推定されており、その中には私たち人類も含まれています。しかし、人類の活動により、多くの種が絶滅の危機に瀕しています。地球環境を守ることは、私たち自身の未来を守ることでもあるのです。",
    # 超長文（500文字以上）
    "日本の歴史は、縄文時代から始まり、現代に至るまで長い年月を経てきました。縄文時代の人々は、狩猟採集生活を営み、土器を作る技術を持っていました。その後、弥生時代になると、稲作が伝わり、農耕社会へと移行していきました。古墳時代には、大規模な墳墓が作られ、豪族による支配体制が確立されました。飛鳥時代から奈良時代にかけては、仏教が伝来し、中央集権的な律令国家が形成されました。平安時代には、独自の文化が花開き、源氏物語などの優れた文学作品が生まれました。鎌倉時代以降は、武士が台頭し、幕府による統治が続きました。明治維新により近代化が進み、第二次世界大戦を経て、現在の民主主義国家となりました。このような長い歴史の中で、日本は独自の文化を育み、世界に貢献する国へと成長してきたのです。",
]


class RealisticMemoryProfiler:
    """実環境を模倣したメモリプロファイラー"""

    def __init__(self, device: str):
        self.device = device
        self.iteration_results: list[dict[str, Any]] = []
        self.start_time = time.time()

    def get_memory_stats(self) -> dict[str, float]:
        """現在のメモリ統計を取得（累積値を保持）"""
        if self.device != "cuda":
            return {
                "allocated": 0.0,
                "reserved": 0.0,
                "free": 0.0,
                "fragmentation": 0.0,
                "fragmentation_ratio": 0.0,
            }

        # 重要: empty_cache()を呼ばない（実環境を再現）
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)

        # GPUの総メモリ量を取得
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = total_memory - reserved

        fragmentation = reserved - allocated
        fragmentation_ratio = fragmentation / reserved if reserved > 0 else 0.0

        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "fragmentation": fragmentation,
            "fragmentation_ratio": fragmentation_ratio,
        }

    def record_iteration(
        self,
        iteration: int,
        model_name: str,
        text: str,
        inference_time: float,
        memory_before: dict[str, float],
        memory_after: dict[str, float],
    ) -> None:
        """各イテレーションの結果を記録"""
        elapsed_time = time.time() - self.start_time

        result = {
            "iteration": iteration,
            "elapsed_time": elapsed_time,
            "model_name": model_name,
            "text_length": len(text),
            "text_preview": text[:30] + "..." if len(text) > 30 else text,
            "inference_time": inference_time,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_delta": {
                "allocated": memory_after["allocated"] - memory_before["allocated"],
                "reserved": memory_after["reserved"] - memory_before["reserved"],
                "fragmentation": memory_after["fragmentation"]
                - memory_before["fragmentation"],
            },
        }

        self.iteration_results.append(result)

        # リアルタイムログ出力
        logger.info(
            f"Iter {iteration}: {model_name} | "
            f"Text: {result['text_length']}chars | "
            f"Time: {inference_time:.2f}s | "
            f"Frag: {memory_after['fragmentation']:.3f}GB ({memory_after['fragmentation_ratio']:.1%}) | "
            f"Free: {memory_after['free']:.3f}GB"
        )


def simulate_production_usage(
    device: str,
    num_iterations: int,
    interval_seconds: float,
    use_fp16: bool,
    batch_simulation: int,
    use_padding: bool = False,
) -> dict[str, Any]:
    """本番環境の使用パターンをシミュレート"""

    logger.info("=== 実環境再現型メモリプロファイリング開始 ===")
    logger.info(f"デバイス: {device}")
    logger.info(f"イテレーション数: {num_iterations}")
    logger.info(f"インターバル: {interval_seconds}秒")
    logger.info(f"バッチシミュレーション: {batch_simulation}回/イテレーション")
    logger.info(f"テンソルパディング: {'有効' if use_padding else '無効'}")
    logger.info("=" * 80)

    profiler = RealisticMemoryProfiler(device)

    # 初期メモリ状態
    initial_memory = profiler.get_memory_stats()
    logger.info(
        f"初期状態: Allocated={initial_memory['allocated']:.3f}GB, "
        f"Reserved={initial_memory['reserved']:.3f}GB, "
        f"Free={initial_memory['free']:.3f}GB"
    )

    # BERTモデルをロード（実環境では常駐）
    logger.info("BERTモデルをロード中...")
    bert_models.load_model(Languages.JP, device_map=device, use_fp16=use_fp16)

    # TTSモデルホルダーを初期化
    model_holder = TTSModelHolder(
        BASE_DIR / "model_assets",
        device,
        onnx_providers=[],
        ignore_onnx=True,
        use_fp16=use_fp16,
    )

    if len(model_holder.models_info) == 0:
        logger.error("音声合成モデルが見つかりませんでした")
        return {}

    # 利用可能なモデルリストを作成
    available_models = []
    for info in model_holder.models_info:
        safetensors_files = [
            f
            for f in info.files
            if f.endswith(".safetensors") and not f.startswith(".")
        ]
        if safetensors_files:
            available_models.append((info.name, safetensors_files[0]))

    logger.info(f"利用可能なモデル数: {len(available_models)}")

    # 現在ロード中のモデルを追跡
    loaded_models: dict[str, TTSModel] = {}
    model_usage_count: dict[str, int] = {}

    # メインループ
    for iteration in range(num_iterations):
        # バッチシミュレーション（短時間に複数リクエスト）
        for batch_idx in range(batch_simulation):
            # ランダムにモデルとテキストを選択
            model_name, model_file = random.choice(available_models)
            text = random.choice(REALISTIC_TEXTS)

            # メモリ状態を記録
            memory_before = profiler.get_memory_stats()

            start_time = time.perf_counter()

            try:
                # モデルのロード管理（実環境を模倣）
                if model_name not in loaded_models:
                    # メモリ不足の場合、最も使用頻度の低いモデルをアンロード
                    if memory_before["free"] < 2.0 and len(loaded_models) > 0:
                        least_used = min(model_usage_count.items(), key=lambda x: x[1])[
                            0
                        ]
                        logger.info(f"メモリ不足のため {least_used} をアンロード")
                        loaded_models[least_used].unload()
                        del loaded_models[least_used]
                        del model_usage_count[least_used]
                        gc.collect()

                    # モデルをロード
                    model = model_holder.get_model(model_name, model_file)
                    model.load()
                    loaded_models[model_name] = model
                    model_usage_count[model_name] = 0

                model = loaded_models[model_name]
                model_usage_count[model_name] += 1

                # 推論実行
                assert model.net_g is not None
                style_vec = model.get_style_vector(0, 1.0)

                _ = infer(
                    text=text,
                    style_vec=style_vec,
                    sdp_ratio=DEFAULT_SDP_RATIO,
                    noise_scale=DEFAULT_NOISE,
                    noise_scale_w=DEFAULT_NOISEW,
                    length_scale=DEFAULT_LENGTH,
                    sid=0,  # 安全のため0固定
                    language=Languages.JP,
                    hps=model.hyper_parameters,
                    net_g=model.net_g,
                    device=device,
                    use_fp16=use_fp16,
                    clear_cuda_cache=False,  # 実環境を再現
                    enable_tensor_padding=use_padding,
                )

                inference_time = time.perf_counter() - start_time

                # メモリ状態を再度記録
                memory_after = profiler.get_memory_stats()

                # 結果を記録
                profiler.record_iteration(
                    iteration * batch_simulation + batch_idx,
                    model_name,
                    text,
                    inference_time,
                    memory_before,
                    memory_after,
                )

            except torch.cuda.OutOfMemoryError:
                logger.error(f"OOM エラー発生！イテレーション {iteration}")
                # OOM時の詳細情報を記録
                memory_after = profiler.get_memory_stats()
                profiler.record_iteration(
                    iteration * batch_simulation + batch_idx,
                    model_name,
                    text,
                    -1.0,  # エラーを示す
                    memory_before,
                    memory_after,
                )

                # 全モデルをアンロードして回復を試みる
                logger.info("全モデルをアンロードして回復を試みます")
                for m in loaded_models.values():
                    m.unload()
                loaded_models.clear()
                model_usage_count.clear()
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                logger.error(f"エラー発生: {e}")

        # インターバルを置く（実環境の時間経過を模倣）
        if iteration < num_iterations - 1:
            time.sleep(interval_seconds)

    # 最終分析
    return analyze_results(profiler, initial_memory)


def analyze_results(
    profiler: RealisticMemoryProfiler,
    initial_memory: dict[str, float],
) -> dict[str, Any]:
    """結果の詳細分析"""

    results = profiler.iteration_results

    if not results:
        return {}

    # 断片化の進行を分析
    fragmentation_progression = [
        {
            "iteration": r["iteration"],
            "elapsed_time": r["elapsed_time"],
            "fragmentation": r["memory_after"]["fragmentation"],
            "fragmentation_ratio": r["memory_after"]["fragmentation_ratio"],
            "free_memory": r["memory_after"]["free"],
        }
        for r in results
    ]

    # OOMエラーの発生を検出
    oom_occurrences = [r for r in results if r["inference_time"] < 0]

    # テキスト長別の統計
    text_length_stats = {}
    for r in results:
        if r["inference_time"] < 0:
            continue
        length_bucket = (r["text_length"] // 50) * 50  # 50文字ごとにバケット化
        if length_bucket not in text_length_stats:
            text_length_stats[length_bucket] = []
        text_length_stats[length_bucket].append(r["memory_delta"]["fragmentation"])

    # 最終メモリ状態
    final_memory = results[-1]["memory_after"] if results else initial_memory

    analysis = {
        "summary": {
            "total_iterations": len(results),
            "oom_count": len(oom_occurrences),
            "initial_fragmentation": initial_memory["fragmentation"],
            "final_fragmentation": final_memory["fragmentation"],
            "fragmentation_increase": final_memory["fragmentation"]
            - initial_memory["fragmentation"],
            "peak_fragmentation": max(
                r["memory_after"]["fragmentation"] for r in results
            ),
            "average_inference_time": np.mean(
                [r["inference_time"] for r in results if r["inference_time"] > 0]
            ),
        },
        "fragmentation_progression": fragmentation_progression,
        "text_length_impact": {
            f"{k}-{k + 49}chars": {
                "avg_fragmentation_delta": np.mean(v),
                "max_fragmentation_delta": max(v),
                "count": len(v),
            }
            for k, v in text_length_stats.items()
        },
        "oom_occurrences": oom_occurrences,
        "raw_results": results,
    }

    return analysis


def save_analysis(analysis: dict[str, Any], output_path: Path) -> None:
    """分析結果を保存"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info(f"分析結果を保存: {output_path}")


def print_analysis_summary(analysis: dict[str, Any]) -> None:
    """分析結果のサマリーを表示"""
    summary = analysis["summary"]

    print("\n" + "=" * 80)
    print("実環境再現型メモリプロファイリング結果")
    print("=" * 80)

    print(f"\n総イテレーション数: {summary['total_iterations']}")
    print(f"OOM発生回数: {summary['oom_count']}")

    print("\n断片化の進行:")
    print(f"  初期: {summary['initial_fragmentation']:.3f}GB")
    print(f"  最終: {summary['final_fragmentation']:.3f}GB")
    print(f"  増加量: {summary['fragmentation_increase']:.3f}GB")
    print(f"  ピーク: {summary['peak_fragmentation']:.3f}GB")

    print(f"\n平均推論時間: {summary['average_inference_time']:.3f}秒")

    print("\nテキスト長別の影響:")
    for length_range, stats in analysis["text_length_impact"].items():
        print(
            f"  {length_range}: 平均Δ{stats['avg_fragmentation_delta']:.4f}GB, "
            f"最大Δ{stats['max_fragmentation_delta']:.4f}GB ({stats['count']}回)"
        )

    # 断片化の進行をグラフ的に表示
    print("\n断片化の時系列推移:")
    progression = analysis["fragmentation_progression"]
    for i in range(0, len(progression), max(1, len(progression) // 20)):
        p = progression[i]
        bar = "█" * int(p["fragmentation_ratio"] * 50)
        print(
            f"  {p['iteration']:3d}: {bar} {p['fragmentation']:.3f}GB ({p['fragmentation_ratio']:.1%})"
        )


def main():
    parser = argparse.ArgumentParser(description="実環境再現型メモリプロファイリング")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="使用デバイス (default: cuda)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="イテレーション数 (default: 50)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="イテレーション間の待機時間（秒） (default: 2.0)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=3,
        help="各イテレーションでのバッチ数 (default: 3)",
    )
    parser.add_argument(
        "--fp16",
        dest="use_fp16",
        action="store_true",
        help="FP16推論を使用 (default)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="use_fp16",
        action="store_false",
        help="FP16推論を無効化",
    )
    parser.add_argument(
        "--enable-padding",
        dest="use_padding",
        action="store_true",
        help="テンソルパディング機能を有効化",
    )
    parser.set_defaults(use_fp16=True, use_padding=False)

    args = parser.parse_args()

    # ランダムシードは固定しない（実環境の多様性を再現）
    random.seed(None)
    np.random.seed(None)

    try:
        # プロファイリング実行
        analysis = simulate_production_usage(
            device=args.device,
            num_iterations=args.iterations,
            interval_seconds=args.interval,
            use_fp16=args.use_fp16,
            batch_simulation=args.batch,
            use_padding=args.use_padding,
        )

        if analysis:
            # 結果を保存
            output_dir = Path("tests/profiling_results")
            output_dir.mkdir(exist_ok=True)

            timestamp = int(time.time())
            output_path = output_dir / f"realistic_profiling_{timestamp}.json"
            save_analysis(analysis, output_path)

            # サマリーを表示
            print_analysis_summary(analysis)

    except KeyboardInterrupt:
        print("\nプロファイリングが中断されました")
    except Exception as e:
        logger.exception(f"プロファイリング中にエラーが発生: {e}")


if __name__ == "__main__":
    main()
