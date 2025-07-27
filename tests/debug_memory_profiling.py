#!/usr/bin/env python3
"""
Usage: .venv/bin/python -m tests.debug_memory_profiling

各モジュールでのメモリ使用量と処理時間を測定して、
実際のボトルネックがどこにあるかを特定する。
"""

import gc
import time

import torch

from style_bert_vits2.constants import (
    BASE_DIR,
    DEFAULT_LENGTH,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    Languages,
)
from style_bert_vits2.models import commons
from style_bert_vits2.models.infer import prepare_inference_data
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.nlp import (
    bert_models,
    clean_text_with_given_phone_tone,
    extract_bert_feature,
)
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder


USE_FP16 = True


def get_memory_usage() -> float:
    """現在のGPUメモリ使用量をMBで返す"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def reset_memory() -> None:
    """メモリをリセット"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def profile_memory_usage(
    model: TTSModel,
    text: str,
    device: str,
) -> dict[str, float]:
    """
    推論処理の各段階でのメモリ使用量と処理時間を測定する。

    Returns:
        dict: 各段階でのメモリ使用量（MB）と処理時間（秒）
    """
    results = {}
    time_results = {}

    reset_memory()
    baseline_memory = get_memory_usage()
    results["00_baseline"] = baseline_memory
    print(f"ベースライン: {baseline_memory:.2f} MB")

    style_vec = model.get_style_vector(0, 1.0)

    with torch.inference_mode():
        # === 1. BERT特徴量抽出とデータ前処理 ===
        reset_memory()
        start_memory = get_memory_usage()
        start_time = time.perf_counter()

        (
            x_tst,
            x_tst_lengths,
            sid_tensor,
            tones,
            lang_ids,
            zh_bert,
            ja_bert,
            en_bert,
            style_vec_tensor,
        ) = prepare_inference_data(
            text,
            style_vec=style_vec,
            sid=0,
            language=Languages.JP,
            hps=model.hyper_parameters,
            device=device,
        )

        bert_time = time.perf_counter() - start_time
        bert_memory = get_memory_usage()
        results["01_bert_preprocessing"] = bert_memory - start_memory
        time_results["01_bert_preprocessing"] = bert_time
        print(
            f"BERT前処理後: +{bert_memory - start_memory:.2f} MB, {bert_time:.3f}秒 (累計: {bert_memory:.2f} MB)"
        )

        # === 2. TextEncoder（BERT処理含む） ===
        net_g = model.net_g
        assert net_g is not None

        # JP-Extraモデルかどうかを判定
        is_jp_extra = model.hyper_parameters.version.endswith("JP-Extra")

        if is_jp_extra:
            # SynthesizerTrnJPExtraの場合
            net_g_jp = net_g
            assert isinstance(net_g_jp, SynthesizerTrnJPExtra)

            # Speaker embedding
            start_memory = get_memory_usage()
            start_time = time.perf_counter()
            g = net_g_jp.emb_g(sid_tensor).unsqueeze(-1)
            spk_emb_time = time.perf_counter() - start_time
            spk_emb_memory = get_memory_usage()
            results["02_speaker_embedding"] = spk_emb_memory - start_memory
            time_results["02_speaker_embedding"] = spk_emb_time
            print(
                f"Speaker embedding後: +{spk_emb_memory - start_memory:.2f} MB, {spk_emb_time:.3f}秒"
            )

            # TextEncoder
            start_memory = get_memory_usage()
            start_time = time.perf_counter()

            # BERT特徴量を明示的にFP32に変換（FP16使用時のエラー対策）
            ja_bert_float = ja_bert.float()

            x, m_p, logs_p, x_mask = net_g_jp.enc_p(
                x_tst,
                x_tst_lengths,
                tones,
                lang_ids,
                ja_bert_float,
                style_vec_tensor,
                g=g,
                use_fp16=USE_FP16,
            )
            text_encoder_time = time.perf_counter() - start_time
            text_encoder_memory = get_memory_usage()
            results["03_text_encoder"] = text_encoder_memory - start_memory
            time_results["03_text_encoder"] = text_encoder_time
            print(
                f"TextEncoder後: +{text_encoder_memory - start_memory:.2f} MB, {text_encoder_time:.3f}秒"
            )

            # === 3. Duration Predictor (SDP/DP) ===
            start_memory = get_memory_usage()
            start_time = time.perf_counter()

            # SDP + DP
            logw = net_g_jp.sdp(
                x, x_mask, g=g, reverse=True, noise_scale=DEFAULT_NOISEW
            ) * (DEFAULT_SDP_RATIO) + net_g_jp.dp(x, x_mask, g=g) * (
                1 - DEFAULT_SDP_RATIO
            )

            w = torch.exp(logw) * x_mask * DEFAULT_LENGTH
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(
                torch.ones(y_lengths.size(0), y_lengths.max().item(), device=device),  # type: ignore
                1,
            ).to(x_mask.dtype)

            # Attention mask and path generation
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = commons.generate_path(w_ceil, attn_mask)

            # Expand prior
            m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
                1, 2
            )

            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * DEFAULT_NOISE

            duration_time = time.perf_counter() - start_time
            duration_memory = get_memory_usage()
            results["04_duration_predictor"] = duration_memory - start_memory
            time_results["04_duration_predictor"] = duration_time
            print(
                f"Duration Predictor後: +{duration_memory - start_memory:.2f} MB, {duration_time:.3f}秒"
            )

            # === 4. Flow（TransformerCouplingBlock） ===
            start_memory = get_memory_usage()
            start_time = time.perf_counter()

            z = net_g_jp.flow(z_p, y_mask, g=g, reverse=True)

            flow_time = time.perf_counter() - start_time
            flow_memory = get_memory_usage()
            results["05_flow"] = flow_memory - start_memory
            time_results["05_flow"] = flow_time
            print(f"Flow後: +{flow_memory - start_memory:.2f} MB, {flow_time:.3f}秒")

            # === 5. Generator (Decoder) ===
            start_memory = get_memory_usage()
            start_time = time.perf_counter()

            z_input = z * y_mask
            # Generator (Decoder) はFP16なので、入力をFP16に変換
            if USE_FP16:
                output = net_g_jp.dec(z_input.half(), g=g.half())
            else:
                output = net_g_jp.dec(z_input, g=g)

            generator_time = time.perf_counter() - start_time
            generator_memory = get_memory_usage()
            results["06_generator"] = generator_memory - start_memory
            time_results["06_generator"] = generator_time
            print(
                f"Generator後: +{generator_memory - start_memory:.2f} MB, {generator_time:.3f}秒"
            )

        else:
            # 通常モデルの場合（省略）
            results["02_speaker_embedding"] = 0
            results["03_text_encoder"] = 0
            results["04_duration_predictor"] = 0
            results["05_flow"] = 0
            results["06_generator"] = 0
            time_results["02_speaker_embedding"] = 0.0
            time_results["03_text_encoder"] = 0.0
            time_results["04_duration_predictor"] = 0.0
            time_results["05_flow"] = 0.0
            time_results["06_generator"] = 0.0

        # === 総メモリ使用量と総処理時間 ===
        final_memory = get_memory_usage()
        total_time = sum(time_results.values())
        results["07_total"] = final_memory - baseline_memory
        time_results["07_total"] = total_time
        print(
            f"最終メモリ使用量: {final_memory:.2f} MB (総増加: {final_memory - baseline_memory:.2f} MB)"
        )
        print(f"総処理時間: {total_time:.3f}秒")

        # === ピークメモリ ===
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            results["08_peak"] = peak_memory
            print(f"ピークメモリ: {peak_memory:.2f} MB")

        # メモリと時間の結果を合体
        for key, value in time_results.items():
            results[f"{key}_time"] = value

        return results


def profile_bert_separately(device: str, texts: list[str]) -> dict[str, float]:
    """BERTのメモリ使用量を個別に測定"""
    print("\n=== BERT単体のメモリプロファイリング ===")

    results = {}

    # BERTモデルロード前
    reset_memory()
    before_bert = get_memory_usage()
    print(f"BERT読込前: {before_bert:.2f} MB")

    # BERTモデルロード
    bert_models.load_model(Languages.JP, device_map=device, use_fp16=USE_FP16)
    after_bert_load = get_memory_usage()
    bert_load_memory = after_bert_load - before_bert
    results["bert_model_load"] = bert_load_memory
    print(f"BERT読込後: +{bert_load_memory:.2f} MB")

    # 各テキストでBERT特徴量抽出のメモリ測定
    for i, text in enumerate(texts):
        reset_memory()
        start_memory = get_memory_usage()

        # BERT特徴量の抽出のみ

        # テキスト前処理の測定
        nlp_start_time = time.perf_counter()
        norm_text, phone, tone, word2ph, sep_text, _, _ = (
            clean_text_with_given_phone_tone(
                text,
                Languages.JP,
                use_jp_extra=True,
                raise_yomi_error=False,
            )
        )
        nlp_time = time.perf_counter() - nlp_start_time

        # BERT特徴量抽出の測定
        bert_extract_start_time = time.perf_counter()
        bert_features = extract_bert_feature(
            norm_text,
            word2ph,
            Languages.JP,
            device,
            None,  # assist_text
            0.7,  # assist_text_weight
            sep_text,
        )
        bert_extract_time = time.perf_counter() - bert_extract_start_time

        bert_extraction_memory = get_memory_usage() - start_memory
        results[f"bert_extraction_text_{i}"] = bert_extraction_memory
        results[f"nlp_time_text_{i}"] = nlp_time
        results[f"bert_extract_time_text_{i}"] = bert_extract_time
        print(
            f"テキスト{i}（{len(text)}文字）BERT抽出: +{bert_extraction_memory:.2f} MB"
        )
        print(
            f"  NLP処理時間: {nlp_time:.3f}秒, BERT抽出時間: {bert_extract_time:.3f}秒"
        )

        del bert_features, norm_text, phone, tone, word2ph, sep_text

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "koharune-ami"

    # テスト用テキスト
    texts = [
        "あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市",  # 中文
        "あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。またそのなかでいっしょになったたくさんのひとたち、ファゼーロとロザーロ、羊飼のミーロや、顔の赤いこどもたち、地主のテーモ、山猫博士のボーガント・デストゥパーゴなど、いまこの暗い巨きな家にはたったひとりがいません。夏の夕暮れどきに白い月がのぼり、草の波から出る霧がそのへんをいつそう白くしたころ、たくさんのおじいさんやおばあさんが、木の机の前にすわって、青い蝋燭をたてて、なにやら熱心に読書をしています。そこにはまた、青い服を着たこどもたちが、たくさん集まって、なにやら楽しげに遊んでいます。",  # 長文
    ]

    print("=" * 80)
    print("VITS2推論詳細メモリプロファイリング")
    print("=" * 80)
    print(f"デバイス: {device}")
    print(f"モデル: {model_name}")
    print("=" * 80)

    # BERT単体の測定
    bert_results = profile_bert_separately(device, texts)

    # モデルホルダー初期化
    model_holder = TTSModelHolder(
        BASE_DIR / "model_assets",
        device,
        onnx_providers=[],
        ignore_onnx=True,
        use_fp16=USE_FP16,
    )

    if len(model_holder.models_info) == 0:
        print("エラー: 音声合成モデルが見つかりませんでした。")
        return

    # モデル検索
    model_info = None
    for info in model_holder.models_info:
        if info.name == model_name:
            model_info = info
            break

    if model_info is None:
        print(f'エラー: モデル "{model_name}" が見つかりませんでした。')
        return

    # モデルファイル検索
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

    # モデルロード
    model = model_holder.get_model(model_name, model_file)
    model.load()

    # 各テキストでプロファイリング実行
    all_results = {}

    for i, text in enumerate(texts):
        print(f"\n{'=' * 60}")
        print(f"テキスト{i}の測定（{len(text)}文字）")
        print(f"{'=' * 60}")
        print(f"テキスト: {text[:50]}...")

        try:
            results = profile_memory_usage(model, text, device)
            all_results[f"text_{i}"] = results

        except Exception as ex:
            print(f"エラー: {ex}")
            import traceback

            traceback.print_exc()

    # 結果まとめ
    print(f"\n{'=' * 80}")
    print("結果まとめ")
    print(f"{'=' * 80}")

    # BERT結果
    print("BERT単体のメモリ使用量:")
    for key, value in bert_results.items():
        print(f"  {key}: {value:.2f} MB")

    # 各テキストの結果
    for text_key, results in all_results.items():
        print(f"\n{text_key}:")
        for stage, value in results.items():
            if stage.endswith("_time"):
                print(f"  {stage}: {value:.3f}秒")
            else:
                print(f"  {stage}: {value:.2f} MB")

    # ボトルネック分析
    print(f"\n{'=' * 80}")
    print("ボトルネック分析")
    print(f"{'=' * 80}")

    if all_results:
        # 長文の結果を使用
        long_text_results = all_results.get("text_1", all_results["text_0"])

        stages = [
            "01_bert_preprocessing",
            "03_text_encoder",
            "04_duration_predictor",
            "05_flow",
            "06_generator",
        ]
        stage_names = [
            "BERT前処理",
            "TextEncoder",
            "Duration Predictor",
            "Flow",
            "Generator",
        ]

        print("各段階のメモリ使用量:")
        for stage, name in zip(stages, stage_names):
            if stage in long_text_results:
                memory = long_text_results[stage]
                percentage = (
                    memory / long_text_results["07_total"] * 100
                    if long_text_results["07_total"] > 0
                    else 0
                )
                print(f"  {name}: {memory:.2f} MB ({percentage:.1f}%)")

        print("\n各段階の処理時間:")
        time_stages = [f"{stage}_time" for stage in stages]
        for stage, name in zip(time_stages, stage_names):
            if stage in long_text_results:
                time_val = long_text_results[stage]
                percentage = (
                    time_val / long_text_results["07_total_time"] * 100
                    if long_text_results["07_total_time"] > 0
                    else 0
                )
                print(f"  {name}: {time_val:.3f}秒 ({percentage:.1f}%)")

    # モデルアンロード
    model.unload()


if __name__ == "__main__":
    main()
