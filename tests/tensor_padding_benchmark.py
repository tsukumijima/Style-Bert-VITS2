#!/usr/bin/env python3
"""
Usage: PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:True" .venv/bin/python -m tests.tensor_padding_benchmark [--device cuda] [--model koharune-ami] [--iterations 30]

メモリ効率化テンソルパディングのベンチマークスクリプト
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

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
from style_bert_vits2.models.tensor_padding import (
    clear_memory_pools,
    get_memory_pool_stats,
)
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder

from .utils import save_benchmark_audio, set_random_seeds


# テストテキスト（様々な長さ）
TEST_TEXTS = [
    # ニュース風
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
    # 日常会話
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
    # 物語風
    "昔々、ある山奥に一人の老人が住んでいました。老人は毎日山を歩き、薬草を集めて暮らしていました。",
    "春の訪れとともに、桜の花が咲き始めました。人々は花見を楽しみ、新しい季節の始まりを祝いました。",
    "深い森の中で、少女は不思議な光を見つけました。それは小さな妖精が放つ光でした。",
    "海辺の町では、毎年夏になると大きな祭りが開かれます。太鼓の音が響き、人々は踊り明かします。",
    "雨の日の午後、猫は窓辺で外を眺めていました。雨粒が窓を叩く音が、静かな部屋に響いていました。",
    # 説明文
    "人工知能技術の発展により、私たちの生活は大きく変わりつつあります。音声認識、画像認識、自然言語処理など、様々な分野でAIが活用されています。今後もこの技術の進歩は続き、より便利で豊かな社会の実現が期待されています。",
    "地球温暖化は、現在人類が直面している最も深刻な環境問題の一つです。温室効果ガスの排出量を削減し、持続可能な社会を築くことが急務となっています。一人一人の行動が、地球の未来を決めることになるでしょう。",
    "日本の四季は、それぞれに独特の美しさがあります。春の桜、夏の花火、秋の紅葉、冬の雪景色。これらの季節の変化は、日本人の感性や文化に深く根ざしており、多くの芸術作品のインスピレーションとなってきました。",
    # ビジネス文書風
    "本日は貴重なお時間をいただき、誠にありがとうございます。弊社の新サービスについてご説明させていただきます。このサービスは、お客様のニーズに合わせてカスタマイズ可能で、業務効率の大幅な向上が期待できます。導入事例では、平均して30%の時間短縮を実現しています。ぜひこの機会にご検討いただければ幸いです。",
    "プロジェクトの進捗についてご報告いたします。現在、全体の70%が完了しており、予定通り来月末には完成予定です。しかし、一部の機能については追加の要件が発生したため、スケジュールの見直しが必要となる可能性があります。詳細については、明日の会議でご相談させていただければと思います。",
    # 技術文書風
    "機械学習モデルの性能を向上させるためには、適切なデータの前処理が不可欠です。データのクリーニング、正規化、特徴量エンジニアリングなど、様々な手法を組み合わせることで、モデルの精度を大幅に改善することができます。また、過学習を防ぐために、交差検証や正則化などの技術も重要です。これらの技術を適切に活用することで、実用的で信頼性の高いモデルを構築することが可能となります。",
    # 詩的な表現
    "風が優しく頬を撫でる春の日。",
    "月明かりが水面に揺れている。静寂の中に美しさがある。",
    "朝露に濡れた花びらが、太陽の光を受けてきらめいている。",
    "遠くで汽笛が鳴る。旅立ちの時が来た。",
    # 指示・命令文
    "右に曲がって、まっすぐ進んでください。",
    "この書類に記入してください。",
    "明日までに提出をお願いします。",
    "静かにしてください。",
    "ドアを閉めてください。",
    # 質問文
    "今何時ですか？",
    "どこに行きたいですか？",
    "これはいくらですか？",
    "何か飲み物はいかがですか？",
    "手伝いましょうか？",
    # 感情表現
    "すごい！素晴らしいですね！",
    "残念です。",
    "嬉しいです！",
    "心配しないでください。",
    "頑張ってください！",
    # 長文
    "私たちが住むこの地球は、太陽系の第三惑星として知られています。地球の表面の約70%は海で覆われており、残りの30%が陸地となっています。地球には様々な気候帯があり、赤道付近の熱帯地域から、両極の極地まで、多様な環境が存在します。この多様性が、地球上の豊かな生態系を支えています。現在、地球上には約870万種の生物が存在すると推定されており、その中には私たち人類も含まれています。しかし、人類の活動により、多くの種が絶滅の危機に瀕しています。地球環境を守ることは、私たち自身の未来を守ることでもあるのです。",
    # 超長文
    "日本の歴史は、縄文時代から始まり、現代に至るまで長い年月を経てきました。縄文時代の人々は、狩猟採集生活を営み、土器を作る技術を持っていました。その後、弥生時代になると、稲作が伝わり、農耕社会へと移行していきました。古墳時代には、大規模な墳墓が作られ、豪族による支配体制が確立されました。飛鳥時代から奈良時代にかけては、仏教が伝来し、中央集権的な律令国家が形成されました。平安時代には、独自の文化が花開き、源氏物語などの優れた文学作品が生まれました。鎌倉時代以降は、武士が台頭し、幕府による統治が続きました。明治維新により近代化が進み、第二次世界大戦を経て、現在の民主主義国家となりました。このような長い歴史の中で、日本は独自の文化を育み、世界に貢献する国へと成長してきたのです。",
    "濁流は、メロスの叫びをせせら笑う如く、ますます激しく躍り狂う。浪は浪を呑み、捲き、煽り立て、そうして時は、刻一刻と消えて行く。今はメロスも覚悟した。泳ぎ切るより他に無い。ああ、神々も照覧あれ！　濁流にも負けぬ愛と誠の偉大な力を、いまこそ発揮して見せる。メロスは、ざんぶと流れに飛び込み、百匹の大蛇のようにのた打ち荒れ狂う浪を相手に、必死の闘争を開始した。満身の力を腕にこめて、押し寄せ渦巻き引きずる流れを、なんのこれしきと掻きわけ掻きわけ、めくらめっぽう獅子奮迅の人の子の姿には、神も哀れと思ったか、ついに憐愍を垂れてくれた。",
    "万博協会が13日に発表した、12日（土）の大阪・関西万博の一般来場者数は速報値ベースで約16万4000人、パビリオンなどの関係者を含めた総来場者数は約18万2000人で、1日あたりの来場者数が“過去3番目”となりました。12日は、航空自衛隊の「ブルーインパルス」による展示飛行が行われ、多くの人が会場に詰めかけ、歓声をを上げました。13日も午後2時40分ごろに関西空港を離陸後、大阪市の通天閣や吹田市の万博記念公園などの上空を通過した上で、午後3時ごろから15分程度、会場上空などで展示飛行が予定されています。また、「レジオネラ菌」の検出により中止されていた昼の噴水ショーと夜の水上ショーは11日から再開されたほか、部品落下トラブルにより中止されていた「空飛ぶクルマ」のデモ飛行も12日から再開されています。",
    "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては、人一倍に敏感であった。きょう未明メロスは村を出発し、野を越え山越え、十里はなれた此のシラクスの市にやって来た。メロスには父も、母も無い。女房も無い。十六の、内気な妹と二人暮しだ。この妹は、村の或る律気な一牧人を、近々、花婿として迎える事になっていた。結婚式も間近かなのである。メロスは、それゆえ、花嫁の衣裳やら祝宴の御馳走やらを買いに、はるばる市にやって来たのだ。先ず、その品々を買い集め、それから都の大路をぶらぶら歩いた。メロスには竹馬の友があった。セリヌンティウスである。今は此のシラクスの市で、石工をしている。その友を、これから訪ねてみるつもりなのだ。久しく逢わなかったのだから、訪ねて行くのが楽しみである。歩いているうちにメロスは、まちの様子を怪しく思った。ひっそりしている。もう既に日も落ちて、まちの暗いのは当りまえだが、けれども、なんだか、夜のせいばかりでは無く、市全体が、やけに寂しい。",
    "台風５号は、今後、発達しながら北上し、あす１４日は自動車並みの速度で関東から東北の太平洋側を北上し、あさって１５日には北海道付近に達する予想です。一般に台風に伴う風や雨は、進行方向に向かって、中心の右側ほど強く、左側ほど強くない傾向があります。これは台風の右側は、台風を流す風の向きと中心に吹き込む風の向きが一緒になり、風や雨の勢いが増すためで、左側は台風を流す風の向きと中心に吹き込む風の向きが逆になり、互いに打ち消し合って、右側ほど強くはならないためです。（右側は危険半円、左側は可航半円とも呼ばれます。）今回の台風５号は、関東以北の太平洋側を北上するため、おおむね陸地の上は、台風の進行方向の左側に入ることが予想されます。このため、右側に入るよりは、大荒れの度合いは小さいことになりそうですが、とはいえ、もちろん大雨や強風（暴風）、高波などには十分な警戒が必要です。今回の台風５号がもし関東へ上陸したら、６年前に千葉県に上陸し、甚大な暴風の被害をもたらした台風１５号以来となり、もし北海道へ上陸したら、９年前の台風１１号以来となります。一方、東北へは昨年も上陸していますので、２年連続となります。",
]


class MemoryTracker:
    """GPU メモリ使用量を追跡するクラス"""

    def __init__(self, device: str):
        self.device = device
        self.snapshots: list[dict[str, float]] = []

    def snapshot(self, label: str):
        """現在のメモリ使用量をスナップショット"""
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            fragmentation = reserved - allocated
        else:
            allocated = reserved = fragmentation = 0.0

        self.snapshots.append(
            {
                "label": label,  # type: ignore
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "fragmentation_gb": fragmentation,
            }
        )

    def print_report(self):
        """メモリ使用状況のレポートを出力"""
        print("\n=== Memory Usage Report ===")
        print(
            f"{'Label':<30} {'Allocated (GB)':>15} {'Reserved (GB)':>15} {'Fragmentation (GB)':>20}"
        )
        print("-" * 82)

        for snap in self.snapshots:
            print(
                f"{snap['label']:<30} {snap['allocated_gb']:>15.3f} "
                f"{snap['reserved_gb']:>15.3f} {snap['fragmentation_gb']:>20.3f}"
            )


def measure_inference_performance(
    model: TTSModel,
    text: str,
    device: str,
    use_padding: bool,
    use_fp16: bool,
    save_audio: bool = False,
) -> tuple[float, float, NDArray[np.float32]]:
    """
    推論のパフォーマンスを測定する

    Args:
        save_audio: 音声ファイルを保存するかどうか

    Returns:
        tuple: (推論時間, ピークメモリ使用量MB, 音声データ)
    """
    # メモリプール統計をクリア（新しい測定のため）
    if use_padding:
        clear_memory_pools()

    # テキスト情報をログ出力
    logger.info(
        f"Processing text (length: {len(text)}): '{text[:50]}{'...' if len(text) > 50 else ''}'"
    )

    # メモリ統計をリセット
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

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
            device=device,
            use_fp16=use_fp16,
            clear_cuda_cache=False,  # ベンチマーク中はキャッシュクリアを無効化
            enable_tensor_padding=use_padding,
        )

    end_time = time.perf_counter()
    inference_time = end_time - start_time

    # パディング統計情報を出力
    if use_padding:
        pool_stats = get_memory_pool_stats()
        logger.info(f"  Padding enabled - Pool stats: {pool_stats}")

    # ピークメモリ使用量を取得
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    else:
        peak_memory = 0.0

    # 音声ファイル保存（オプション）
    if save_audio:
        output_type = "with_padding" if use_padding else "without_padding"
        save_benchmark_audio(
            audio_data,
            model.hyper_parameters.data.sampling_rate,
            text,
            "tensor_padding_benchmark",
            output_type,
        )

    return inference_time, peak_memory, audio_data


def run_benchmark(
    device: str = "cuda",
    model_name: str = "koharune-ami",
    num_iterations: int = 30,
    use_fp16: bool = True,
    fix_seed: bool = False,
) -> None:
    """ベンチマークを実行する"""

    # ランダムシード固定
    if fix_seed:
        set_random_seeds()

    print("=" * 80)
    print("Style-Bert-VITS2 メモリ効率化テンソルパディングベンチマーク")
    print("=" * 80)
    print(f"デバイス: {device}")
    print(f"モデル: {model_name}")
    print(f"反復回数: {num_iterations}")
    print(f"FP16: {use_fp16}")
    print(f"ランダムシード固定: {'有効' if fix_seed else '無効'}")
    print("=" * 80)

    # BERT メモリ使用量を測定
    bert_memory_usage = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        memory_before_bert = torch.cuda.memory_allocated() / (1024 * 1024)
        logger.info("Loading BERT model for memory measurement...")
        bert_models.load_model(Languages.JP, device_map=device, use_fp16=use_fp16)
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

    # パディングなしでベンチマーク
    print("\n" + "=" * 60)
    print("パディングなしでベンチマーク実行中...")
    print("=" * 60)

    tracker_without = MemoryTracker(device)
    tracker_without.snapshot("Initial")

    # ウォームアップ
    for i in range(2):
        _, _, _ = measure_inference_performance(
            model, TEST_TEXTS[0], device, use_padding=False, use_fp16=use_fp16
        )

    torch.cuda.empty_cache()
    gc.collect()
    tracker_without.snapshot("After warmup")

    # 本測定
    inference_times_without = []
    for i in range(num_iterations):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        # 最後の数回のみ音声を保存（主観評価用）
        save_audio = i >= max(0, num_iterations - min(num_iterations, len(TEST_TEXTS)))
        inference_time, _, _ = measure_inference_performance(
            model,
            text,
            device,
            use_padding=False,
            use_fp16=use_fp16,
            save_audio=save_audio,
        )
        inference_times_without.append(inference_time)

        if (i + 1) % 10 == 0:
            tracker_without.snapshot(f"After {i + 1} iterations")
            print(f"  反復 {i + 1}/{num_iterations} 完了")

    tracker_without.snapshot("Final (before cleanup)")
    torch.cuda.empty_cache()
    gc.collect()
    tracker_without.snapshot("After cleanup")

    # パディングありでベンチマーク
    print("\n" + "=" * 60)
    print("パディングありでベンチマーク実行中...")
    print("=" * 60)

    tracker_with = MemoryTracker(device)
    tracker_with.snapshot("Initial")

    # ウォームアップ
    for i in range(2):
        _, _, _ = measure_inference_performance(
            model, TEST_TEXTS[0], device, use_padding=True, use_fp16=use_fp16
        )

    torch.cuda.empty_cache()
    gc.collect()
    tracker_with.snapshot("After warmup")

    # 本測定
    inference_times_with = []
    for i in range(num_iterations):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        # 最後の数回のみ音声を保存（主観評価用）
        save_audio = i >= max(0, num_iterations - len(TEST_TEXTS))
        inference_time, _, _ = measure_inference_performance(
            model,
            text,
            device,
            use_padding=True,
            use_fp16=use_fp16,
            save_audio=save_audio,
        )
        inference_times_with.append(inference_time)

        if (i + 1) % 10 == 0:
            tracker_with.snapshot(f"After {i + 1} iterations")
            print(f"  反復 {i + 1}/{num_iterations} 完了")

    tracker_with.snapshot("Final (before cleanup)")

    # クリーンアップ
    clear_memory_pools()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    tracker_with.snapshot("After cleanup")

    # モデルをアンロード
    model.unload()

    # 結果比較
    print("\n" + "=" * 60)
    print("ベンチマーク結果比較")
    print("=" * 60)

    # 推論時間比較
    avg_without = np.mean(inference_times_without)
    avg_with = np.mean(inference_times_with)
    overhead = (avg_with / avg_without - 1) * 100

    print("\n推論時間:")
    print(
        f"  パディングなし: {avg_without:.3f} s (標準偏差: {np.std(inference_times_without):.3f})"
    )
    print(
        f"  パディングあり: {avg_with:.3f} s (標準偏差: {np.std(inference_times_with):.3f})"
    )
    print(f"  オーバーヘッド: {overhead:+.1f}%")

    # メモリ断片化比較
    final_without = next(
        s for s in tracker_without.snapshots if s["label"] == "Final (before cleanup)"
    )
    final_with = next(
        s for s in tracker_with.snapshots if s["label"] == "Final (before cleanup)"
    )

    print("\nメモリ断片化:")
    print(f"  パディングなし: {final_without['fragmentation_gb']:.3f} GB")
    print(f"  パディングあり: {final_with['fragmentation_gb']:.3f} GB")
    print(
        f"  削減量: {final_without['fragmentation_gb'] - final_with['fragmentation_gb']:.3f} GB"
    )

    # メモリプール統計
    pool_stats = get_memory_pool_stats()
    if pool_stats:
        print("\nメモリプール統計:")
        for pool_name, stats in pool_stats.items():
            print(f"  {pool_name}: {stats}")

    # 詳細メモリレポート
    print("\n=== パディングなし - メモリ使用状況 ===")
    tracker_without.print_report()

    print("\n=== パディングあり - メモリ使用状況 ===")
    tracker_with.print_report()

    # 判定
    print("\n=== 総合評価 ===")
    if overhead < 5:
        print("✓ オーバーヘッドは許容範囲内です (< 5%)")
    elif overhead < 15:
        print("⚠ 軽微なオーバーヘッドがあります (5-15%)")
    else:
        print("✗ 大きなオーバーヘッドがあります (> 15%)")

    fragmentation_reduction = (
        final_without["fragmentation_gb"] - final_with["fragmentation_gb"]
    )
    if fragmentation_reduction > 0.1:
        print("✓ メモリ断片化が大幅に改善されました")
    elif fragmentation_reduction > 0.05:
        print("✓ メモリ断片化が改善されました")
    else:
        print("⚠ メモリ断片化の改善は限定的です")

    # 音声ファイル保存の通知
    audio_dir = Path("tests/wavs/tensor_padding_benchmark")
    if audio_dir.exists() and any(audio_dir.iterdir()):
        print(f"\n🎧 音声ファイルが保存されました: {audio_dir}")
        print("   - *_without_padding.wav: パディングなしの音声")
        print("   - *_with_padding.wav: パディングありの音声")
        print("   主観評価で音質の比較をしてください。")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient tensor padding benchmark"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="koharune-ami",
        help="Model name to use (default: koharune-ami)",
    )
    parser.add_argument(
        "--iterations", type=int, default=60, help="Number of iterations (default: 60)"
    )
    parser.add_argument(
        "--fp16",
        dest="use_fp16",
        action="store_true",
        help="Use FP16 inference (default)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="use_fp16",
        action="store_false",
        help="Disable FP16 inference",
    )
    parser.add_argument(
        "--fix-seed",
        action="store_true",
        help="Fix random seed for reproducibility",
    )
    parser.set_defaults(use_fp16=True)

    args = parser.parse_args()

    try:
        run_benchmark(
            device=args.device,
            model_name=args.model,
            num_iterations=args.iterations,
            use_fp16=args.use_fp16,
            fix_seed=args.fix_seed,
        )
    except KeyboardInterrupt:
        print("\nベンチマークが中断されました。")
    except Exception as ex:
        logger.exception(
            f"テンソルパディングベンチマーク実行中にエラーが発生しました: {ex}"
        )


if __name__ == "__main__":
    main()
