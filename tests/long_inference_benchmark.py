#!/usr/bin/env python3
"""
Style-Bert-VITS2 長文一括推論のパフォーマンス測定スクリプト

このスクリプトは infer.py に実装されている infer() 関数のパフォーマンスを測定し、
長いテキストでの処理時間とピークメモリ使用量をベンチマークする。
測定時の推論処理はすべて FP16 で行う。

測定項目:
- 総処理時間
- ピークメモリ使用量 (VRAM)
- 生成音声の長さごとの効果

使用方法:
    .venv/bin/python -m tests.long_inference_benchmark [--device cuda] [--model koharune-ami] [--runs 5]
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.io import wavfile

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


# 測定用サンプルテキスト (300文字以上)
BENCHMARK_TEXTS = [
    {
        # 初回ロード用（ダミー）
        "text": "あああ",
        "description": "Short (Dummy)",
    },
    {
        "text": "イーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。またそのなかでいっしょになったたくさんのひとたち、ファゼーロとロザーロ、羊飼のミーロや、顔の赤いこどもたち、地主のテーモ、山猫博士のボーガント・デストゥパーゴなど、いまこの暗い巨きな家にはたったひとりがいません。",
        "description": "Medium (ROUDOKU)",
    },
    {
        "text": "小笠原近海で台風５号が発生しました。今後、北上し、関東から東北の太平洋側に沿って北上した後、北海道付近に到達する可能性が大きくなっています。もし関東へ上陸すれば６年ぶり、東北に上陸すれば２年連続、北海道へ上陸すれば９年ぶりとなります。この台風の進路の特徴とともに、詳しくみていきましょう。",
        "description": "Medium (NEWS)",
    },
    {
        "text": "濁流は、メロスの叫びをせせら笑う如く、ますます激しく躍り狂う。浪は浪を呑み、捲き、煽り立て、そうして時は、刻一刻と消えて行く。今はメロスも覚悟した。泳ぎ切るより他に無い。ああ、神々も照覧あれ！　濁流にも負けぬ愛と誠の偉大な力を、いまこそ発揮して見せる。メロスは、ざんぶと流れに飛び込み、百匹の大蛇のようにのた打ち荒れ狂う浪を相手に、必死の闘争を開始した。満身の力を腕にこめて、押し寄せ渦巻き引きずる流れを、なんのこれしきと掻きわけ掻きわけ、めくらめっぽう獅子奮迅の人の子の姿には、神も哀れと思ったか、ついに憐愍を垂れてくれた。",
        "description": "Medium Long (ROUDOKU)",
    },
    {
        "text": "万博協会が13日に発表した、12日（土）の大阪・関西万博の一般来場者数は速報値ベースで約16万4000人、パビリオンなどの関係者を含めた総来場者数は約18万2000人で、1日あたりの来場者数が“過去3番目”となりました。12日は、航空自衛隊の「ブルーインパルス」による展示飛行が行われ、多くの人が会場に詰めかけ、歓声をを上げました。13日も午後2時40分ごろに関西空港を離陸後、大阪市の通天閣や吹田市の万博記念公園などの上空を通過した上で、午後3時ごろから15分程度、会場上空などで展示飛行が予定されています。また、「レジオネラ菌」の検出により中止されていた昼の噴水ショーと夜の水上ショーは11日から再開されたほか、部品落下トラブルにより中止されていた「空飛ぶクルマ」のデモ飛行も12日から再開されています。",
        "description": "Medium Long (NEWS)",
    },
    {
        "text": "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては、人一倍に敏感であった。きょう未明メロスは村を出発し、野を越え山越え、十里はなれた此のシラクスの市にやって来た。メロスには父も、母も無い。女房も無い。十六の、内気な妹と二人暮しだ。この妹は、村の或る律気な一牧人を、近々、花婿として迎える事になっていた。結婚式も間近かなのである。メロスは、それゆえ、花嫁の衣裳やら祝宴の御馳走やらを買いに、はるばる市にやって来たのだ。先ず、その品々を買い集め、それから都の大路をぶらぶら歩いた。メロスには竹馬の友があった。セリヌンティウスである。今は此のシラクスの市で、石工をしている。その友を、これから訪ねてみるつもりなのだ。久しく逢わなかったのだから、訪ねて行くのが楽しみである。歩いているうちにメロスは、まちの様子を怪しく思った。ひっそりしている。もう既に日も落ちて、まちの暗いのは当りまえだが、けれども、なんだか、夜のせいばかりでは無く、市全体が、やけに寂しい。",
        "description": "Long (ROUDOKU)",
    },
    {
        "text": "台風５号は、今後、発達しながら北上し、あす１４日は自動車並みの速度で関東から東北の太平洋側を北上し、あさって１５日には北海道付近に達する予想です。一般に台風に伴う風や雨は、進行方向に向かって、中心の右側ほど強く、左側ほど強くない傾向があります。これは台風の右側は、台風を流す風の向きと中心に吹き込む風の向きが一緒になり、風や雨の勢いが増すためで、左側は台風を流す風の向きと中心に吹き込む風の向きが逆になり、互いに打ち消し合って、右側ほど強くはならないためです。（右側は危険半円、左側は可航半円とも呼ばれます。）今回の台風５号は、関東以北の太平洋側を北上するため、おおむね陸地の上は、台風の進行方向の左側に入ることが予想されます。このため、右側に入るよりは、大荒れの度合いは小さいことになりそうですが、とはいえ、もちろん大雨や強風（暴風）、高波などには十分な警戒が必要です。今回の台風５号がもし関東へ上陸したら、６年前に千葉県に上陸し、甚大な暴風の被害をもたらした台風１５号以来となり、もし北海道へ上陸したら、９年前の台風１１号以来となります。一方、東北へは昨年も上陸していますので、２年連続となります。",
        "description": "Long (NEWS)",
    },
]


def save_audio_file(
    audio_data: NDArray[np.float32], sample_rate: int, text: str, output_type: str
) -> None:
    """音声データをWAVファイルとして保存する。"""
    try:
        output_dir = Path("tests/wavs/long_inference_benchmark")
        output_dir.mkdir(parents=True, exist_ok=True)

        # ファイル名に使えない文字を置換
        safe_filename = (
            text.replace("/", "_")
            .replace("\\", "_")
            .replace(":", "")
            .replace("*", "")
            .replace("?", "")
            .replace("<", "")
            .replace(">", "")
            .replace("|", "")
            .replace('"', "")
            .replace("!", "")
        )

        # ファイル名が長すぎる場合は切り詰める（拡張子と output_type を考慮して安全な長さに）
        max_filename_length = 70  # 安全なファイル名長制限
        if len(safe_filename) > max_filename_length:
            safe_filename = safe_filename[:max_filename_length] + "..."

        output_path = output_dir / f"{safe_filename}_{output_type}.wav"

        wavfile.write(str(output_path), sample_rate, audio_data)

    except ImportError:
        logger.warning("scipy is required to save audio files, skipping audio save")
    except Exception as ex:
        logger.error(f"Failed to save audio file: {ex}")


def measure_infer_performance(
    model: TTSModel,
    text: str,
    device: str,
    bert_memory_usage: float,
    **infer_kwargs: Any,
) -> tuple[float, float, float, float, NDArray[np.float32]]:
    """
    infer() 関数のパフォーマンスを測定する。

    Returns:
        tuple: (総処理時間, ピークメモリ使用量(MB), BERT除外メモリ使用量(MB), 生成音声長(秒), 音声データ)
    """
    # メモリ統計をリセット
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        print("CUDA キャッシュをクリアしました")

    start_time = time.perf_counter()

    net_g = model.net_g
    assert net_g is not None

    style_vec = model.get_style_vector(0, 1.0)  # デフォルトスタイル

    # 推論を実行
    with torch.inference_mode():
        audio_data = infer(
            text=text,
            style_vec=style_vec,
            sdp_ratio=DEFAULT_SDP_RATIO,
            noise_scale=DEFAULT_NOISE,
            noise_scale_w=DEFAULT_NOISEW,
            length_scale=DEFAULT_LENGTH,
            sid=0,
            language=Languages.JP,
            hps=model.hyper_parameters,
            net_g=net_g,
            device=model.device,
            **infer_kwargs,
        )

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # ピークメモリ使用量を取得
        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            # BERT の重み分を除いたメモリ使用量
            peak_memory_without_bert = max(0.0, peak_memory - bert_memory_usage)
        else:
            peak_memory = 0.0  # CPUでは測定しない
            peak_memory_without_bert = 0.0

        # 生成音声の長さを計算
        audio_duration = len(audio_data) / model.hyper_parameters.data.sampling_rate

        return (
            total_time,
            peak_memory,
            peak_memory_without_bert,
            audio_duration,
            audio_data,
        )


def run_benchmark(
    device: str = "cpu",
    model_name: str = "koharune-ami",
    num_runs: int = 5,
    use_fp16: bool = True,
) -> None:
    """
    ベンチマークを実行する。
    """
    print("=" * 80)
    print("Style-Bert-VITS2 長文一括推論パフォーマンス測定")
    print("=" * 80)
    print(f"デバイス: {device}")
    print(f"モデル: {model_name}")
    print(f"測定回数: {num_runs}")
    print(f"FP16: {use_fp16}")
    print("=" * 80)

    # BERT メモリ使用量を測定
    bert_memory_usage = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # BERT ロード前のメモリ使用量
        memory_before_bert = torch.cuda.memory_allocated() / (1024 * 1024)

        # BERT モデルをロード
        logger.info("Loading BERT model for memory measurement...")
        bert_models.load_model(Languages.JP, device_map=device, use_fp16=use_fp16)

        # BERT ロード後のメモリ使用量
        memory_after_bert = torch.cuda.memory_allocated() / (1024 * 1024)
        bert_memory_usage = memory_after_bert - memory_before_bert

        print(f"BERT メモリ使用量: {bert_memory_usage:.2f}MB")
        print("=" * 80)

    # モデルホルダーを初期化
    model_holder = TTSModelHolder(
        BASE_DIR / "model_assets",
        device,
        onnx_providers=[],
        ignore_onnx=True,
        use_fp16=use_fp16,
    )
    if len(model_holder.models_info) == 0:
        print("エラー: 音声合成モデルが見つかりませんでした。")
        return

    # 指定されたモデルを検索
    model_info = None
    for info in model_holder.models_info:
        if info.name == model_name:
            model_info = info
            break

    if model_info is None:
        print(f'エラー: モデル "{model_name}" が見つかりませんでした。')
        print("利用可能なモデル:")
        for info in model_holder.models_info:
            print(f"  - {info.name}")
        return

    # Safetensors 形式のモデルファイルを検索
    model_files = [
        f
        for f in model_info.files
        if f.endswith(".safetensors") and not f.startswith(".")
    ]
    if len(model_files) == 0:
        print(
            f'エラー: モデル "{model_name}" の .safetensors ファイルが見つかりませんでした。'
        )
        return

    model_file = model_files[0]
    print(f"使用するモデルファイル: {model_file}")
    print()

    # モデルをロード
    model = model_holder.get_model(model_name, model_file)
    model.load()

    # 結果を保存するリスト
    results = []

    # 各テキストでベンチマークを実行
    for i, test_case in enumerate(BENCHMARK_TEXTS):
        text = test_case["text"]
        description = test_case["description"]
        text_length = len(text)

        print(f"測定中: {description}")
        print(f"テキスト: {text[:50]}... (全{text_length}文字)")

        # 複数回実行して平均を取る
        infer_times = []
        peak_memories = []
        peak_memories_without_bert = []
        infer_durations = []
        last_audio = None
        last_sample_rate = None

        for run in range(num_runs):
            try:
                # 推論を測定
                (
                    infer_time,
                    peak_memory,
                    peak_memory_without_bert,
                    infer_duration,
                    audio_data,
                ) = measure_infer_performance(
                    model,
                    text,
                    device,
                    bert_memory_usage,
                    use_fp16=use_fp16,
                )
                infer_times.append(infer_time)
                peak_memories.append(peak_memory)
                peak_memories_without_bert.append(peak_memory_without_bert)
                infer_durations.append(infer_duration)
                if run == num_runs - 1:
                    last_audio = audio_data
                    last_sample_rate = model.hyper_parameters.data.sampling_rate

                print(
                    f"  実行{run + 1}: 時間 {infer_time:.3f}秒, ピークメモリ {peak_memory:.2f}MB, BERT除外 {peak_memory_without_bert:.2f}MB"
                )

            except Exception as ex:
                logger.exception(f"測定中にエラーが発生しました: {ex}")
                continue

        if not infer_times:
            print("  測定に失敗しました。")
            continue

        # 初回はロードが入るため捨てる
        if i == 0:
            continue

        # 音声ファイルを保存（初回のダミーは除く）
        if last_audio is not None and last_sample_rate is not None:
            save_audio_file(last_audio, last_sample_rate, text, "normal")

        # 平均値を計算
        avg_infer_time = np.mean(infer_times)
        avg_peak_memory = np.mean(peak_memories)
        avg_peak_memory_without_bert = np.mean(peak_memories_without_bert)
        avg_infer_duration = np.mean(infer_durations)

        # 結果を保存
        result = {
            "text": text,
            "description": description,
            "text_length": text_length,
            "actual_duration": avg_infer_duration,
            "infer_time": avg_infer_time,
            "peak_memory": avg_peak_memory,
            "peak_memory_without_bert": avg_peak_memory_without_bert,
        }
        results.append(result)

        # 個別結果を表示
        print(f"  文字数: {text_length}文字")
        print(f"  平均音声長: {avg_infer_duration:.2f}秒")
        print(f"  平均推論時間: {avg_infer_time:.3f}秒")
        print(f"  平均ピークメモリ: {avg_peak_memory:.2f}MB")
        print(f"  平均ピークメモリ(BERT除外): {avg_peak_memory_without_bert:.2f}MB")
        print("=" * 80)

    # モデルをアンロード
    model.unload()

    # 総合結果を表示
    print("総合結果")
    print("=" * 90)
    print(
        f"{'説明':<21} {'文字数':>4} {'音声長(秒)':>8} {'推論時間(秒)':>8} {'ピークメモリ(MB)':>8} {'BERT除外(MB)':>8}"
    )
    print("-" * 90)

    for result in results:
        print(
            f"{result['description']:<25} "
            f"{result['text_length']:>6} "
            f"{result['actual_duration']:>12.2f} "
            f"{result['infer_time']:>12.3f} "
            f"{result['peak_memory']:>12.2f} "
            f"{result['peak_memory_without_bert']:>12.2f}"
        )

    print("=" * 90)
    print("分析:")
    avg_time = np.mean([r["infer_time"] for r in results])
    avg_memory = np.mean([r["peak_memory"] for r in results])
    avg_memory_without_bert = np.mean([r["peak_memory_without_bert"] for r in results])
    print(f"- 全体平均推論時間: {avg_time:.3f}秒")
    print(f"- 全体平均ピークメモリ: {avg_memory:.2f}MB")
    print(f"- 全体平均ピークメモリ(BERT除外): {avg_memory_without_bert:.2f}MB")
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Style-Bert-VITS2 長文一括推論パフォーマンス測定"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="推論に使用するデバイス (default: cuda)",
    )
    parser.add_argument(
        "--model",
        default="koharune-ami",
        help="使用するモデル名 (default: koharune-ami)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="各テストケースの実行回数 (default: 5)",
    )
    parser.add_argument(
        "--fp16",
        dest="use_fp16",
        action="store_true",
        help="FP16 で推論を行う (default)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="use_fp16",
        action="store_false",
        help="FP16 を無効化する",
    )
    parser.set_defaults(use_fp16=True)

    args = parser.parse_args()

    try:
        run_benchmark(
            device=args.device,
            model_name=args.model,
            num_runs=args.runs,
            use_fp16=args.use_fp16,
        )
    except KeyboardInterrupt:
        print("\nベンチマークが中断されました。")
    except Exception as ex:
        logger.exception(f"ベンチマーク実行中にエラーが発生しました: {ex}")


if __name__ == "__main__":
    main()
