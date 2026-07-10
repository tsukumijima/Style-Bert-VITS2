"""
# 既知の問題点・バグ (対応済み)
---------------------------------------------------------------------------
## WavLM Discriminator の Generator 側損失が使用されない問題 (対応済み)
オリジナル Bert-VITS2 の JP-Extra 版から存在していたバグ。
WavLM Discriminator の損失（loss_lm, loss_lm_gen）を計算しても、Generator の学習に反映されなかった。
Duration Discriminator を意図的に無効化した際に、WavLM 損失の条件分岐が連動して無効になることが見落とされていたと推測される。
現在の事前学習モデルのベースである Bert-VITS2 時代からこの状態で学習されていたと思われる。

原因は、WavLM 損失を loss_gen_all に加算する処理が、
Duration Discriminator の条件分岐 `if net_dur_disc is not None:` の内部にネストされていたこと。
JP-Extra のデフォルト設定では `use_duration_discriminator: false` のため、
この条件分岐に入らず、WavLM 損失が計算されても使用されなかった。

これにより、WavLM Discriminator を有効にしていても Discriminator 側だけが鍛えられるだけで、
Generator は WavLM 損失を一切受け取れず、WavLM は学習に寄与しない状態だった。
にもかかわらず WavLM モデルのロード・forward pass・損失計算で数百 MB のメモリと計算時間を消費していた。
また TensorBoard には loss_lm, loss_lm_gen が記録されるが、実際には Generator の学習に使われていなかった。

### このコードでの対応
1. **ネストバグの修正**: WavLM 損失の加算を Duration Discriminator の条件分岐から独立させ、
   Duration Discriminator / WavLM Discriminator を個別に有効/無効化できるようにした。
2. **デフォルト設定の変更**: `use_wavlm_discriminator: false` に変更し、WavLM 関連のリソース消費を削減した。
   事前学習モデルの Generator は WavLM 損失を使用せずに学習されていたため、
   Generator の学習結果は従来と完全に等価である。

### WavLM を無効のままとする理由
事前学習モデルは WavLM なしで学習されたにもかかわらず、十分に高品質な音声を生成できている。
WavLM を有効化して学習に組み込むには JP-Extra ベースモデルの再学習が必要となる大変更になるため、
現時点では WavLM は存在しなかったものとして扱い、config.json で無効化している。
将来的に WavLM を導入する可能性は完全には排除しないが、積極的に予定はしていない。
"""

from __future__ import annotations

import argparse
import datetime
import gc
import os
import platform
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import HfApi
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler

from style_bert_vits2.constants import (
    DEFAULT_TRAIN_ENV,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons, utils
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models_nanairo import (
    DurationDiscriminator,
    MultiPeriodDiscriminator,
    SynthesizerTrn,
    WavLMDiscriminator,
)
from style_bert_vits2.nlp.symbols import NANAIRO_SYMBOLS
from style_bert_vits2.utils.paths import (
    TrainingModelPaths,
    add_model_argument,
    get_paths_config,
)
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT

# logging.getLogger("numba").setLevel(logging.WARNING)
from training import default_style
from training.data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from training.duration_observability import (
    build_generator_optimizer_parameter_groups,
    clip_generator_gradients,
    collect_duration_residual_tail_metrics,
    collect_duration_symbol_type_metrics,
)
from training.losses import (
    WavLMLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from training.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from training.runtime import (
    EMAModel,
    GradientMonitor,
    TrainRuntimeConfig,
)
from training.utils import check_git_hash, get_steps, is_resuming, summarize


if TYPE_CHECKING:
    from loguru import Logger as LoguruLogger


# PyTorch 最適化設定 (torch >= 2.1 前提)
# TF32: Ampere 以降の GPU で FP32 演算を高速化
torch.backends.cuda.matmul.allow_tf32 = True
# If encountered training problem, please try to disable TF32.
torch.backends.cudnn.allow_tf32 = True
# 行列演算精度: "medium" は TF32 を使用し、速度と精度のバランスを取る
torch.set_float32_matmul_precision("medium")
# cuDNN ベンチマーク: 固定入力サイズの学習で畳み込みアルゴリズムを自動選択し高速化
torch.backends.cudnn.benchmark = True
# Scaled Dot-Product Attention (SDPA) バックエンド設定
# Flash Attention: 最も高速だがハードウェア要件あり (Ampere 以降)
torch.backends.cuda.enable_flash_sdp(True)
# Memory Efficient Attention: Flash が使えない場合のフォールバック
torch.backends.cuda.enable_mem_efficient_sdp(True)
# Math Attention: 上記が使えない場合の最終フォールバック
torch.backends.cuda.enable_math_sdp(True)

global_step = 0

api = HfApi()


# グローバル勾配モニターインスタンス (分散学習との互換性のため run() 内で初期化)
gradient_monitor: GradientMonitor | None = None
# グローバル EMA インスタンス
ema_model: EMAModel | None = None

# 固定評価サンプル: 初回 evaluate() 時にキャッシュし、以降の評価ステップで再利用する
# 常に同一入力から合成することで、学習進行に伴う品質変化を追跡できる
_fixed_eval_batch: tuple[Any, ...] | None = None

# style_vec の話者情報を遮断する検証では、学習と評価で同じ共有平均を使う
_style_vec_shared_mean: torch.Tensor | None = None


def run():
    paths_config = get_paths_config()
    parser = argparse.ArgumentParser(
        description="Train Nanairo model.",
    )
    add_model_argument(parser)
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default=None,
        help="Directory that contains G_0.safetensors / D_0.safetensors for initialization. "
        "If omitted, model_dir is used.",
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        help="Root directory of model assets needed for inference.",
        default=str(paths_config.assets_root),
    )
    parser.add_argument(
        "--keep_ckpts",
        type=int,
        default=1,
        help="Number of checkpoints to keep. Set to 0 to keep all. Default: 1",
    )
    parser.add_argument(
        "--skip_default_style",
        action="store_true",
        help="Skip saving default style config and mean vector.",
    )
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Do not show the progress bar while training.",
    )
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="Speed up training by disabling logging and evaluation.",
    )
    parser.add_argument(
        "--repo_id",
        help="Huggingface model repo id to backup the model.",
        default=None,
    )
    parser.add_argument(
        "--not_use_custom_batch_sampler",
        help="Don't use custom batch sampler for training, which was used in the version < 2.5",
        action="store_true",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use Exponential Moving Average (EMA) for model weights. "
        "EMA weights are smoother and often produce more stable inference results.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay rate. Higher values (e.g., 0.9999) produce smoother weights. Default: 0.999",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating weights. "
        "Effective batch size = batch_size * gradient_accumulation_steps. Default: 1 (no accumulation)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Stop training after this global step and save the final checkpoint. "
        "Default: run until the configured epoch count finishes.",
    )
    args = parser.parse_args()

    # TrainingModelPaths を使ってパスを解決
    model_folder_name: str = args.model
    paths = TrainingModelPaths(model_folder_name=model_folder_name)
    assets_root = Path(args.assets_root)

    # チェックポイント保存ディレクトリ (Data/{model_folder_name}/models/)
    model_dir = str(paths.models_dir)
    pretrained_model_dir = args.pretrained_model_dir or model_dir
    if args.pretrained_model_dir is not None and not os.path.isdir(
        pretrained_model_dir
    ):
        logger.warning(
            f"Pretrained model dir not found: {pretrained_model_dir}. Falling back to model_dir."
        )
        pretrained_model_dir = model_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add(str(paths.dataset_dir / f"train_{timestamp}.log"))

    # 分散学習用の環境変数をセット
    # 未設定の変数のみ DEFAULT_TRAIN_ENV から設定される
    for env_name, env_value in DEFAULT_TRAIN_ENV.items():
        if env_name not in os.environ.keys():
            logger.info(f"Setting default environment variable: {env_name}={env_value}")
            os.environ[env_name] = str(env_value)
    logger.info(
        "Loading environment variables \nMASTER_ADDR: {},\nMASTER_PORT: {},\nWORLD_SIZE: {},\nRANK: {},\nLOCAL_RANK: {}".format(
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
            os.environ["WORLD_SIZE"],
            os.environ["RANK"],
            os.environ["LOCAL_RANK"],
        )
    )

    backend = "nccl"
    if platform.system() == "Windows":
        backend = "gloo"  # If Windows,switch to gloo backend.
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=300),
    )  # Use torchrun instead of mp.spawn
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    n_gpus = dist.get_world_size()

    hps = HyperParameters.load_from_json(paths.config_path)

    # 共有平均を要求した実験は入力ファイルの欠落を黙って別条件へ切り替えず、起動時に停止する
    global _style_vec_shared_mean
    if hps.train.style_vec_shared_mean is True:
        style_vec_mean_path = paths.dataset_dir / "style_vec_mean.npy"
        if not style_vec_mean_path.exists():
            raise FileNotFoundError(
                "Shared style vector mean not found. "
                f"Run prepare_nanairo_zeroshot_stats.py first: {style_vec_mean_path}"
            )
        style_vec_mean_array = np.load(style_vec_mean_path)
        if style_vec_mean_array.shape != (256,):
            raise ValueError(
                "Shared style vector mean shape must be (256,). "
                f"actual: {style_vec_mean_array.shape}"
            )
        _style_vec_shared_mean = torch.from_numpy(
            style_vec_mean_array.astype(np.float32, copy=False)
        )
        logger.info(f"Loaded shared style vector mean: {style_vec_mean_path}")
    else:
        _style_vec_shared_mean = None
    runtime_config = TrainRuntimeConfig(
        model_name=hps.model_name,
        model_dir=model_dir,
        out_dir=str(assets_root / model_folder_name),
        dataset_path=str(paths.dataset_dir),
        keep_ckpts=args.keep_ckpts,
        repo_id=args.repo_id,
        speedup=args.speedup,
        # spec キャッシュは音声ごとに約 10MB が追加され、数万ファイルに及ぶ多話者学習では数百 GB 級に膨らんで
        # ストレージを枯渇させるため無効化する (spec は DataLoader worker が毎回計算する)
        spec_cache=False,
    )

    """
    パス定数について:
    - args.model: 学習データのフォルダ名 (Data/ 以下)
    - runtime_config.model_name: config.json の model_name フィールド（ファイル名に使用）
    - runtime_config.model_dir: チェックポイント保存先 (Data/{model}/models/)
    - runtime_config.out_dir: 推論用モデルの出力先 (model_assets/{model}/)
    - runtime_config.dataset_path: データセットパス (Data/{model}/)
    """

    if args.repo_id is not None:
        # First try to upload config.json to check if the repo exists
        assert runtime_config.dataset_path is not None
        try:
            api.upload_file(
                path_or_fileobj=str(paths.config_path),
                path_in_repo=f"Data/{runtime_config.model_name}/config.json",
                repo_id=args.repo_id,
            )
        except Exception as ex:
            logger.error(
                f"Failed to upload files to the repo {args.repo_id}. "
                "Please check if the repo exists and you have logged in using `huggingface-cli login`.",
                exc_info=ex,
            )
            raise ex
        # Upload Data dir for resuming training
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=runtime_config.dataset_path,
            path_in_repo=f"Data/{runtime_config.model_name}",
            delete_patterns="*.pth",  # Only keep the latest checkpoint
            ignore_patterns="raw/**",  # Ignore raw data
            run_as_future=True,
        )

    assert runtime_config.out_dir is not None
    os.makedirs(runtime_config.out_dir, exist_ok=True)

    if not args.skip_default_style:
        default_style.save_styles_by_dirs(
            str(paths.wavs_dir),
            runtime_config.out_dir,
            config_path=str(paths.config_path),
            config_output_path=os.path.join(runtime_config.out_dir, "config.json"),
        )

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(local_rank)
    if _style_vec_shared_mean is not None:
        _style_vec_shared_mean = _style_vec_shared_mean.cuda(local_rank)

    global global_step
    writer = None
    writer_eval = None
    if rank == 0 and not args.speedup:
        # logger = get_logger(runtime_config.model_dir)
        # logger.info(hps)
        check_git_hash(model_dir)
        writer = SummaryWriter(log_dir=model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(model_dir, "eval"))
    train_dataset = TextAudioSpeakerLoader(
        hps.data.training_files,
        hps.data,
        wavs_dir=paths.wavs_dir,
        spec_cache=runtime_config.spec_cache,
        return_duration_symbol_type_ids=hps.model.use_duration_symbol_type_embedding,
    )
    collate_fn = TextAudioSpeakerCollate(
        use_jp_extra=True,
        use_speaker_embedding=hps.data.use_speaker_embedding,
        return_duration_symbol_type_ids=hps.model.use_duration_symbol_type_embedding,
    )
    if not args.not_use_custom_batch_sampler:
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size,
            # バケット境界（スペクトログラムのフレーム数単位）
            # 44100Hz / hop_length=512 の場合: 1 frame ≈ 0.0116 秒
            # 上限 1724 frames ≈ 20.01 秒（20 秒の音声を確実にカバーするため 1724 に設定）
            # 1000 frames 以降は長尺音声の絶対数が少ないため 200 frames 刻み (≈2.3 秒) に拡大
            [32, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1724],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            # メモリ消費量を減らそうとnum_workersを1にしてみる
            # num_workers=min(config.train_ms_config.num_workers, os.cpu_count() // 2),
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            # batch_size=hps.train.batch_size,
            persistent_workers=True,
            # これもメモリ消費量を減らそうとしてコメントアウト
            # prefetch_factor=6,
        )
    else:
        train_sampler = DistributedLengthGroupedSampler(
            dataset=train_dataset,
            batch_size=hps.train.batch_size,
            num_replicas=n_gpus,
            rank=rank,
            lengths=train_dataset.lengths,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            # メモリ消費量を減らそうとnum_workersを1にしてみる
            # num_workers=min(config.train_ms_config.num_workers, os.cpu_count() // 2),
            num_workers=1,
            # shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
            batch_size=hps.train.batch_size,
            persistent_workers=True,
            # これもメモリ消費量を減らそうとしてコメントアウト
            # prefetch_factor=6,
        )
        logger.info("Using DistributedLengthGroupedSampler for training.")
        logger.debug(f"len(train_dataset): {len(train_dataset)}")
        logger.debug(f"len(train_loader): {len(train_loader)}")

    eval_dataset = None
    eval_loader = None
    if rank == 0 and not args.speedup:
        eval_dataset = TextAudioSpeakerLoader(
            hps.data.validation_files,
            hps.data,
            wavs_dir=paths.wavs_dir,
            spec_cache=runtime_config.spec_cache,
            return_duration_symbol_type_ids=hps.model.use_duration_symbol_type_embedding,
        )
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    if hps.model.use_noise_scaled_mas is True:
        logger.info("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        logger.info("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0
    if hps.model.use_duration_discriminator is True:
        logger.info("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(local_rank)
    else:
        net_dur_disc = None
    if hps.model.use_wavlm_discriminator is True:
        net_wd = WavLMDiscriminator(
            hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel
        ).cuda(local_rank)
    else:
        net_wd = None
    if hps.model.use_spk_conditioned_encoder is True:
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        logger.info("Using normal encoder for VITS1")

    # Nanairo 専用の学習スクリプトなので、config 側の use_nanairo フラグが有効でない場合にはエラーとする
    # さもなければ、NLP パイプラインが通常モード（絵文字モーラを削除する）で動作し、語彙数と不整合を起こして学習が破綻しうる
    if hps.is_nanairo_like_model() is not True:
        raise RuntimeError(
            "train_ms_nanairo.py requires data.use_nanairo=true in config. "
            "Check your config.json or use train_ms_jp_extra.py for non-Nanairo models."
        )

    net_g = SynthesizerTrn(
        len(NANAIRO_SYMBOLS),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        # hps.model 以下のすべての値を引数に渡す
        use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
        use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
        use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
        use_duration_discriminator=hps.model.use_duration_discriminator,
        use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        p_dropout=hps.model.p_dropout,
        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        n_layers_q=hps.model.n_layers_q,
        use_spectral_norm=hps.model.use_spectral_norm,
        gin_channels=hps.model.gin_channels,
        slm=hps.model.slm,
        use_transformer_duration_predictor=hps.model.use_transformer_duration_predictor,
        duration_filter_channels=hps.model.duration_filter_channels,
        use_duration_symbol_type_embedding=hps.model.use_duration_symbol_type_embedding,
        duration_symbol_type_count=hps.model.duration_symbol_type_count,
        use_speaker_adapter=hps.model.use_speaker_adapter,
        speaker_adapter_input_dim=hps.model.speaker_adapter_input_dim,
        speaker_adapter_bottleneck_dim=hps.model.speaker_adapter_bottleneck_dim,
        use_fixed_isometric_speaker_projection=hps.model.use_fixed_isometric_speaker_projection,
        speaker_projection_scale_init=hps.model.speaker_projection_scale_init,
        # 主成分別分散モニタの PC 数
        pc_variance_monitor_k=hps.train.pc_variance_monitor_k,
    ).cuda(local_rank)

    if getattr(hps.train, "freeze_JP_bert", False):
        logger.info("Freezing (JP) bert encoder !!!")
        for param in net_g.enc_p.bert_proj.parameters():
            param.requires_grad = False

    if getattr(hps.train, "freeze_style", False):
        logger.info("Freezing style encoder !!!")
        for param in net_g.enc_p.style_proj.parameters():
            param.requires_grad = False

    if getattr(hps.train, "freeze_decoder", False):
        logger.info("Freezing decoder !!!")
        for param in net_g.dec.parameters():
            param.requires_grad = False

    # ゼロショット型の full 学習では lookup table を更新せず、全発話を SpeakerAdapter 経由で条件付けする
    if hps.train.freeze_emb_g is True:
        logger.info("Freezing speaker lookup embedding.")
        for param in net_g.emb_g.parameters():
            param.requires_grad = False

    # 固定写像モードでは既存 Adapter を checkpoint 互換のため保持しつつ学習対象から外す
    ## DDP が forward で使われないパラメータの勾配を待たないよう、ラップ前に凍結する
    if hps.model.use_fixed_isometric_speaker_projection is True:
        logger.info("Freezing speaker adapter modules for fixed projection mode.")
        if net_g.speaker_control_encoder is not None:
            for param in net_g.speaker_control_encoder.parameters():
                param.requires_grad = False
        if net_g.speaker_adapter is not None:
            for param in net_g.speaker_adapter.parameters():
                param.requires_grad = False

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(local_rank)
    # Transformer-based DP/SDP では duration 系だけ weight decay の有無を明示的に分ける
    ## 旧構成では従来通り単一パラメータ群にして既存 FT の学習条件を変えない
    generator_parameters: Any
    if hps.model.use_transformer_duration_predictor is True:
        generator_parameters = build_generator_optimizer_parameter_groups(
            net_g,
            hps.train.learning_rate,
        )
    else:
        generator_parameters = filter(lambda p: p.requires_grad, net_g.parameters())
    optim_g = torch.optim.AdamW(
        generator_parameters,
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None
    if net_wd is not None:
        optim_wd = torch.optim.AdamW(
            net_wd.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_wd = None
    net_g = DDP(
        net_g,
        device_ids=[local_rank],
        # bucket_cap_mb=512
    )
    net_d = DDP(
        net_d,
        device_ids=[local_rank],
        # bucket_cap_mb=512
    )
    if net_dur_disc is not None:
        net_dur_disc = DDP(
            net_dur_disc,
            device_ids=[local_rank],
            # bucket_cap_mb=512,
        )
    if net_wd is not None:
        net_wd = DDP(
            net_wd,
            device_ids=[local_rank],
            #  bucket_cap_mb=512
        )

    is_resuming_training = is_resuming(model_dir)
    if is_resuming_training:
        if net_dur_disc is not None:
            # チェックポイントが見つからない場合のデフォルト学習率
            dur_resume_lr = hps.train.learning_rate
            try:
                _, _, dur_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                    utils.checkpoints.get_latest_checkpoint_path(
                        model_dir, "DUR_*.pth"
                    ),
                    net_dur_disc,
                    optim_dur_disc,
                    skip_optimizer=hps.train.skip_optimizer,
                )
                assert optim_dur_disc is not None
                if not optim_dur_disc.param_groups[0].get("initial_lr"):
                    optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
            except Exception as ex:
                # チェックポイントのロードに失敗した場合、デフォルト学習率で初期化
                logger.warning(f"Failed to load DUR checkpoint: {ex}")
                assert optim_dur_disc is not None
                if not optim_dur_disc.param_groups[0].get("initial_lr"):
                    optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
                logger.info("Initialize dur_disc with default learning rate")
        if net_wd is not None:
            # チェックポイントが見つからない場合のデフォルト学習率
            wd_resume_lr = hps.train.learning_rate
            try:
                _, optim_wd, wd_resume_lr, epoch_str = (
                    utils.checkpoints.load_checkpoint(
                        utils.checkpoints.get_latest_checkpoint_path(
                            model_dir, "WD_*.pth"
                        ),
                        net_wd,
                        optim_wd,
                        skip_optimizer=hps.train.skip_optimizer,
                    )
                )
                assert optim_wd is not None
                if not optim_wd.param_groups[0].get("initial_lr"):
                    optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr
            except Exception as ex:
                # チェックポイントのロードに失敗した場合、デフォルト学習率で初期化
                logger.warning(f"Failed to load WD checkpoint: {ex}")
                assert optim_wd is not None
                if not optim_wd.param_groups[0].get("initial_lr"):
                    optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr
                logger.info("Initialize wavlm with default learning rate")

        try:
            generator_checkpoint_path = utils.checkpoints.get_latest_checkpoint_path(
                model_dir, "G_*.pth"
            )
            try:
                _, optim_g, g_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                    generator_checkpoint_path,
                    net_g,
                    optim_g,
                    skip_optimizer=hps.train.skip_optimizer,
                )
            except ValueError as ex:
                if "different number of parameter groups" not in str(ex):
                    raise
                logger.warning(
                    f"Skip loading generator optimizer state because parameter groups changed: {ex}"
                )
                _, optim_g, g_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                    generator_checkpoint_path,
                    net_g,
                    optim_g,
                    skip_optimizer=True,
                )
            _, optim_d, d_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "D_*.pth"),
                net_d,
                optim_d,
                skip_optimizer=hps.train.skip_optimizer,
            )
            if not optim_g.param_groups[0].get("initial_lr"):
                optim_g.param_groups[0]["initial_lr"] = g_resume_lr
            if not optim_d.param_groups[0].get("initial_lr"):
                optim_d.param_groups[0]["initial_lr"] = d_resume_lr

            epoch_str = max(epoch_str, 1)
            # global_step = (epoch_str - 1) * len(train_loader)
            steps = get_steps(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth")
            )
            if steps is None:
                raise ValueError("Failed to parse global step from checkpoint path.")
            global_step = steps
            logger.info(
                f"******************Found the model. Current epoch is {epoch_str}, global step is {global_step}*********************"
            )
        except Exception as e:
            logger.warning(e)
            logger.warning(
                "It seems that you are not using the pretrained models, so we will train from scratch."
            )
            epoch_str = 1
            global_step = 0
    else:
        try:
            _ = utils.safetensors.load_safetensors(
                os.path.join(pretrained_model_dir, "G_0.safetensors"),
                net_g,
                allow_partial_load_embedding_keys=("enc_p.emb.weight",),
            )
            _ = utils.safetensors.load_safetensors(
                os.path.join(pretrained_model_dir, "D_0.safetensors"), net_d
            )
            if net_dur_disc is not None:
                _ = utils.safetensors.load_safetensors(
                    os.path.join(pretrained_model_dir, "DUR_0.safetensors"),
                    net_dur_disc,
                )
            if net_wd is not None:
                _ = utils.safetensors.load_safetensors(
                    os.path.join(pretrained_model_dir, "WD_0.safetensors"), net_wd
                )
            logger.info("Loaded the pretrained models.")
        except Exception as e:
            logger.warning(e)
            logger.warning(
                "It seems that you are not using the pretrained models, so we will train from scratch."
            )
        finally:
            epoch_str = 1
            global_step = 0

    # 話者アダプタ経路の基準 g は、新規学習開始時のみ emb_g の平均から初期化する
    # 学習再開時は checkpoint に保存された g_neutral を保持し、学習状態の連続性を壊さない
    if is_resuming_training is False and (
        hasattr(net_g.module, "set_g_neutral")
        and hasattr(net_g.module, "emb_g")
        and (
            getattr(net_g.module, "use_speaker_adapter", False) is True
            or getattr(
                net_g.module,
                "use_fixed_isometric_speaker_projection",
                False,
            )
            is True
        )
    ):
        with torch.no_grad():
            # ゲート付き delta 経路用に、ロード済み emb_g の平均から g_neutral を初期化する
            g_neutral = net_g.module.emb_g.weight.detach().mean(dim=0, keepdim=True)
            net_g.module.set_g_neutral(g_neutral)

    # 固定写像の中心は学習対象にせず、全発話から事前計算した平均を checkpoint buffer へ保存する
    if hps.model.use_fixed_isometric_speaker_projection is True:
        speaker_embedding_mean_path = paths.dataset_dir / "spk_emb_mean.npy"
        if not speaker_embedding_mean_path.exists():
            raise FileNotFoundError(
                "Speaker embedding mean not found. "
                f"Run prepare_nanairo_zeroshot_stats.py first: {speaker_embedding_mean_path}"
            )
        speaker_embedding_mean_array = np.load(speaker_embedding_mean_path)
        expected_shape = (hps.model.speaker_adapter_input_dim,)
        if speaker_embedding_mean_array.shape != expected_shape:
            raise ValueError(
                "Speaker embedding mean shape mismatch. "
                f"expected: {expected_shape}, actual: {speaker_embedding_mean_array.shape}"
            )
        net_g.module.set_speaker_embedding_stats(
            torch.from_numpy(
                speaker_embedding_mean_array.astype(np.float32, copy=False)
            ).cuda(local_rank)
        )
        if rank == 0:
            logger.info(f"Loaded speaker embedding mean: {speaker_embedding_mean_path}")

    # 事前学習 emb_g の分布統計 (平均・分散・PCA) を初期化する
    ## 学習再開時も統計が checkpoint に残るが、emb_g 自体は学習中ほぼ更新されないので再計算で問題ない
    if hasattr(net_g.module, "set_emb_g_statistics") and hasattr(net_g.module, "emb_g"):
        net_g.module.set_emb_g_statistics()
        if rank == 0:
            logger.info(
                "Initialized emb_g distribution statistics for Adapter monitoring."
            )

    def lr_lambda(epoch: int) -> float:
        """
        Learning rate scheduler for warmup and exponential decay.
        - During the warmup period, the learning rate increases linearly.
        - After the warmup period, the learning rate decreases exponentially.
        """
        if epoch < hps.train.warmup_epochs:
            return float(epoch) / float(max(1, hps.train.warmup_epochs))
        else:
            return float(hps.train.lr_decay) ** (epoch - hps.train.warmup_epochs)

    scheduler_last_epoch = epoch_str - 2
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optim_g, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optim_d, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    if net_dur_disc is not None:
        assert optim_dur_disc is not None
        scheduler_dur_disc = torch.optim.lr_scheduler.LambdaLR(
            optim_dur_disc,
            lr_lambda=lr_lambda,
            last_epoch=scheduler_last_epoch,
        )
    else:
        scheduler_dur_disc = None
    if net_wd is not None:
        assert optim_wd is not None
        scheduler_wd = torch.optim.lr_scheduler.LambdaLR(
            optim_wd, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
        )
        wl = WavLMLoss(
            hps.model.slm.model,
            net_wd,
            hps.data.sampling_rate,
            hps.model.slm.sr,
        ).to(local_rank)
    else:
        scheduler_wd = None
        wl = None

    # NOTE: GradScaler は本来 FP16 (AMP) 用であり、BF16 では不要
    ## BF16 は FP32 と同じ動的レンジを持つため、勾配スケーリングは必要ない
    ## ただし、enabled=False で初期化すれば害はないため、現状維持としている
    scaler = GradScaler(device="cuda", enabled=not hps.train.bf16_run)
    logger.info("Start training.")

    diff = abs(
        epoch_str * len(train_loader) - (hps.train.epochs + 1) * len(train_loader)
    )
    pbar: tqdm[Any] | None = None
    if not args.no_progress_bar:
        pbar = tqdm(
            total=global_step + diff,
            initial=global_step,
            smoothing=0.05,
            file=SAFE_STDOUT,
            dynamic_ncols=True,
        )
    initial_step = global_step

    # 自動学習率調整用の勾配モニターを初期化
    # 分散学習時は各 GPU 間での学習率同期が複雑になるため無効化
    global gradient_monitor
    if n_gpus > 1:
        gradient_monitor = None
        logger.info(
            "[GradientMonitor] Disabled (distributed training with multiple GPUs detected)."
        )
    else:
        gradient_monitor = GradientMonitor()
        logger.info("[GradientMonitor] Automatic learning rate adjustment enabled.")

    # EMA (Exponential Moving Average) の初期化
    # 推論時により安定した出力を得るために、モデル重みの移動平均を保持
    global ema_model
    if args.use_ema:
        ema_model = EMAModel(net_g, decay=args.ema_decay)
        logger.info(
            f"[EMA] Enabled with decay: {args.ema_decay}. EMA weights will be saved for inference."
        )
    else:
        ema_model = None

    if ema_model is not None:
        try:
            ema_path = utils.checkpoints.get_latest_checkpoint_path(
                model_dir, "EMA_*.pth"
            )
            try:
                ema_state = torch.load(
                    ema_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                ema_state = torch.load(ema_path, map_location="cpu")
            ema_model.load_state_dict(ema_state.get("state_dict", {}))
            if "decay" in ema_state:
                ema_model.decay = float(ema_state["decay"])
            logger.info(f"[EMA] Loaded state from {ema_path}")
        except Exception as ex:
            logger.warning(ex)
            logger.warning("[EMA] Checkpoint not found. EMA will start fresh.")

    # 勾配累積のログ出力
    if args.gradient_accumulation_steps > 1:
        effective_batch_size = hps.train.batch_size * args.gradient_accumulation_steps
        logger.info(
            f"[GradientAccumulation] Enabled with {args.gradient_accumulation_steps} steps. "
            f"Effective batch size: {hps.train.batch_size} x {args.gradient_accumulation_steps} = {effective_batch_size}"
        )

    writers = (
        (writer, writer_eval)
        if writer is not None and writer_eval is not None
        else None
    )
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            is_max_steps_reached = train_and_evaluate(
                rank,
                local_rank,
                epoch,
                hps,
                runtime_config,
                (net_g, net_d, net_dur_disc, net_wd, wl),
                (optim_g, optim_d, optim_dur_disc, optim_wd),
                (scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd),
                scaler,
                (train_loader, eval_loader),
                logger,
                writers,
                pbar,
                initial_step,
                args.gradient_accumulation_steps,
                args.max_steps,
            )
        else:
            is_max_steps_reached = train_and_evaluate(
                rank,
                local_rank,
                epoch,
                hps,
                runtime_config,
                (net_g, net_d, net_dur_disc, net_wd, wl),
                (optim_g, optim_d, optim_dur_disc, optim_wd),
                (scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd),
                scaler,
                (train_loader, None),
                None,
                None,
                pbar,
                initial_step,
                args.gradient_accumulation_steps,
                args.max_steps,
            )
        if is_max_steps_reached is True:
            break
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            assert scheduler_dur_disc is not None
            scheduler_dur_disc.step()
        if net_wd is not None:
            assert scheduler_wd is not None
            scheduler_wd.step()

        if epoch == hps.train.epochs:
            # Save the final models
            assert optim_g is not None
            utils.checkpoints.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(model_dir, f"G_{global_step}.pth"),
            )
            assert optim_d is not None
            utils.checkpoints.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(model_dir, f"D_{global_step}.pth"),
            )
            if net_dur_disc is not None:
                assert optim_dur_disc is not None
                utils.checkpoints.save_checkpoint(
                    net_dur_disc,
                    optim_dur_disc,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(model_dir, f"DUR_{global_step}.pth"),
                )
            if net_wd is not None:
                assert optim_wd is not None
                utils.checkpoints.save_checkpoint(
                    net_wd,
                    optim_wd,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(model_dir, f"WD_{global_step}.pth"),
                )
            # EMA が有効な場合は EMA 重みを保存
            model_to_save = (
                ema_model.get_ema_model() if ema_model is not None else net_g
            )
            utils.safetensors.save_safetensors(
                model_to_save,
                epoch,
                os.path.join(
                    runtime_config.out_dir,
                    f"{runtime_config.model_name}_e{epoch}_s{global_step}.safetensors",
                ),
                for_infer=True,
            )
            if runtime_config.repo_id is not None:
                future1 = api.upload_folder(
                    repo_id=runtime_config.repo_id,
                    folder_path=runtime_config.dataset_path,
                    path_in_repo=f"Data/{runtime_config.model_name}",
                    delete_patterns="*.pth",  # Only keep the latest checkpoint
                    ignore_patterns="raw/**",  # Ignore raw data
                    run_as_future=True,
                )
                future2 = api.upload_folder(
                    repo_id=runtime_config.repo_id,
                    folder_path=runtime_config.out_dir,
                    path_in_repo=f"model_assets/{runtime_config.model_name}",
                    run_as_future=True,
                )
                try:
                    future1.result()
                    future2.result()
                except Exception as ex:
                    logger.error("Failed to upload to HuggingFace", exc_info=ex)

    if pbar is not None:
        pbar.close()


def train_and_evaluate(
    rank: int,
    local_rank: int,
    epoch: int,
    hps: HyperParameters,
    runtime_config: TrainRuntimeConfig,
    nets: tuple[DDP, DDP, DDP | None, DDP | None, Any | None],
    optims: tuple[
        torch.optim.Optimizer,
        torch.optim.Optimizer,
        torch.optim.Optimizer | None,
        torch.optim.Optimizer | None,
    ],
    schedulers: tuple[
        torch.optim.lr_scheduler.LambdaLR,
        torch.optim.lr_scheduler.LambdaLR,
        torch.optim.lr_scheduler.LambdaLR | None,
        torch.optim.lr_scheduler.LambdaLR | None,
    ],
    scaler: GradScaler,
    loaders: tuple[DataLoader[Any], DataLoader[Any] | None],
    logger: LoguruLogger | None,
    writers: tuple[SummaryWriter, SummaryWriter] | None,
    pbar: tqdm[Any] | None,
    initial_step: int,
    gradient_accumulation_steps: int = 1,
    max_steps: int | None = None,
) -> bool:
    net_g, net_d, net_dur_disc, net_wd, wl = nets
    optim_g, optim_d, optim_dur_disc, optim_wd = optims
    scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd = schedulers
    train_loader, eval_loader = loaders
    writer: SummaryWriter | None = None
    writer_eval: SummaryWriter | None = None
    if writers is not None:
        writer, writer_eval = writers

    # マルチ GPU 学習は基本行わないため、DistributedBucketSampler でのシャッフルを固定して再現性を優先する
    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    if net_wd is not None:
        net_wd.train()

    use_speaker_embedding = getattr(hps.data, "use_speaker_embedding", False)
    use_duration_symbol_type_ids = hps.model.use_duration_symbol_type_embedding
    should_use_duration_clip = (
        hps.model.use_transformer_duration_predictor is True
        or use_duration_symbol_type_ids is True
    )
    disable_discriminators = False  # 現在未使用
    if disable_discriminators:
        net_d.eval()
        if net_dur_disc is not None:
            net_dur_disc.eval()
        if net_wd is not None:
            net_wd.eval()

    # 勾配累積用の変数
    is_accumulating = gradient_accumulation_steps > 1

    for batch_idx, batch in enumerate(train_loader):
        if use_speaker_embedding and use_duration_symbol_type_ids:
            (
                x,
                x_lengths,
                spec,
                spec_lengths,
                y,
                y_lengths,
                speakers,
                tone,
                language,
                bert,
                style_vec,
                duration_symbol_type_ids,
                speaker_embedding,
            ) = batch
        elif use_speaker_embedding:
            (
                x,
                x_lengths,
                spec,
                spec_lengths,
                y,
                y_lengths,
                speakers,
                tone,
                language,
                bert,
                style_vec,
                speaker_embedding,
            ) = batch
            duration_symbol_type_ids = None
        elif use_duration_symbol_type_ids:
            (
                x,
                x_lengths,
                spec,
                spec_lengths,
                y,
                y_lengths,
                speakers,
                tone,
                language,
                bert,
                style_vec,
                duration_symbol_type_ids,
            ) = batch
            speaker_embedding = None
        else:
            (
                x,
                x_lengths,
                spec,
                spec_lengths,
                y,
                y_lengths,
                speakers,
                tone,
                language,
                bert,
                style_vec,
            ) = batch
            speaker_embedding = None
            duration_symbol_type_ids = None
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        x, x_lengths = (
            x.cuda(local_rank, non_blocking=True),
            x_lengths.cuda(local_rank, non_blocking=True),
        )
        spec, spec_lengths = (
            spec.cuda(local_rank, non_blocking=True),
            spec_lengths.cuda(local_rank, non_blocking=True),
        )
        y, y_lengths = (
            y.cuda(local_rank, non_blocking=True),
            y_lengths.cuda(local_rank, non_blocking=True),
        )
        speakers = speakers.cuda(local_rank, non_blocking=True)
        tone = tone.cuda(local_rank, non_blocking=True)
        language = language.cuda(local_rank, non_blocking=True)
        bert = bert.cuda(local_rank, non_blocking=True)
        style_vec = style_vec.cuda(local_rank, non_blocking=True)
        if duration_symbol_type_ids is not None:
            duration_symbol_type_ids = duration_symbol_type_ids.cuda(
                local_rank, non_blocking=True
            )
        if speaker_embedding is not None:
            speaker_embedding = speaker_embedding.cuda(local_rank, non_blocking=True)
        # 共有平均モードでは dropout 設定にかかわらず、学習の全行を同じ style_vec に固定する
        ## 対象音声由来の話者情報を遮断し、prior 側の話者条件を speaker_embedding に限定する
        if hps.train.style_vec_shared_mean is True:
            if _style_vec_shared_mean is None:
                raise RuntimeError("Shared style vector mean is not initialized")
            style_vec = (
                _style_vec_shared_mean.to(dtype=style_vec.dtype)
                .unsqueeze(0)
                .expand(style_vec.shape[0], -1)
            )

        with autocast(
            device_type="cuda",
            enabled=hps.train.bf16_run,
            dtype=torch.bfloat16,
        ):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),  # , logw_sdp),
                g,
            ) = net_g(
                x,
                x_lengths,
                spec,
                spec_lengths,
                speakers,
                tone,
                language,
                bert,
                style_vec,
                speaker_embedding,
                duration_symbol_type_ids=duration_symbol_type_ids,
            )
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

        # 勾配累積を使用する場合は、累積ステップの最初でのみ zero_grad() を呼ぶ
        is_first_accumulation_step = (batch_idx % gradient_accumulation_steps) == 0
        is_last_accumulation_step = ((batch_idx + 1) % gradient_accumulation_steps) == 0
        should_step = is_last_accumulation_step or not is_accumulating

        loss_dur_disc_all: torch.Tensor | None = None
        losses_dur_disc_r: list[float] | None = None
        losses_dur_disc_g: list[float] | None = None
        loss_slm: torch.Tensor | None = None
        loss_lm: torch.Tensor | None = None
        loss_lm_gen: torch.Tensor | None = None
        loss_dur_gen: torch.Tensor | None = None
        losses_dur_gen: list[torch.Tensor] | None = None
        generator_grad_metrics: dict[str, float] = {}
        # 分布診断メトリクス
        adapter_dist_metrics: dict[str, torch.Tensor] | None = None

        # Discriminator
        if disable_discriminators:
            loss_disc = torch.tensor(0.0, device=y.device)
            loss_disc_all = torch.tensor(0.0, device=y.device)
            losses_disc_r = []
            losses_disc_g = []
            grad_norm_d = 0.0
            grad_norm_dur = 0.0
            grad_norm_wd = 0.0
        else:
            if is_first_accumulation_step:
                optim_d.zero_grad()
                if net_dur_disc is not None:
                    assert optim_dur_disc is not None
                    optim_dur_disc.zero_grad()
                if net_wd is not None:
                    assert optim_wd is not None
                    optim_wd.zero_grad()
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(
                device_type="cuda",
                enabled=hps.train.bf16_run,
                dtype=torch.bfloat16,
            ):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw_.detach(),
                    logw.detach(),
                    g.detach(),
                )
                with autocast(
                    device_type="cuda",
                    enabled=hps.train.bf16_run,
                    dtype=torch.bfloat16,
                ):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
            if net_wd is not None:
                # logger.debug(f"y.shape: {y.shape}, y_hat.shape: {y_hat.shape}")
                # shape: (batch, 1, time)
                with autocast(
                    device_type="cuda",
                    enabled=hps.train.bf16_run,
                    dtype=torch.bfloat16,
                ):
                    assert wl is not None
                    loss_slm = wl.discriminator(
                        y.detach().squeeze(1), y_hat.detach().squeeze(1)
                    ).mean()

            loss_disc_scaled = (
                loss_disc_all / gradient_accumulation_steps
                if is_accumulating
                else loss_disc_all
            )
            net_d_no_sync = net_d.no_sync if hasattr(net_d, "no_sync") else nullcontext
            with (
                net_d_no_sync()
                if is_accumulating and not is_last_accumulation_step
                else nullcontext()
            ):
                scaler.scale(loss_disc_scaled).backward()

            if net_dur_disc is not None:
                assert loss_dur_disc_all is not None
                loss_dur_disc_scaled = (
                    loss_dur_disc_all / gradient_accumulation_steps
                    if is_accumulating
                    else loss_dur_disc_all
                )
                net_dur_no_sync = (
                    net_dur_disc.no_sync
                    if hasattr(net_dur_disc, "no_sync")
                    else nullcontext
                )
                with (
                    net_dur_no_sync()
                    if is_accumulating and not is_last_accumulation_step
                    else nullcontext()
                ):
                    scaler.scale(loss_dur_disc_scaled).backward()

            if net_wd is not None:
                assert loss_slm is not None
                loss_slm_scaled = (
                    loss_slm / gradient_accumulation_steps
                    if is_accumulating
                    else loss_slm
                )
                net_wd_no_sync = (
                    net_wd.no_sync if hasattr(net_wd, "no_sync") else nullcontext
                )
                with (
                    net_wd_no_sync()
                    if is_accumulating and not is_last_accumulation_step
                    else nullcontext()
                ):
                    scaler.scale(loss_slm_scaled).backward()

            grad_norm_d = 0.0
            grad_norm_dur = 0.0
            grad_norm_wd = 0.0
            if should_step:
                scaler.unscale_(optim_d)
                if getattr(hps.train, "bf16_run", False):
                    torch.nn.utils.clip_grad_norm_(
                        parameters=net_d.parameters(), max_norm=200
                    )
                grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
                scaler.step(optim_d)

                if net_dur_disc is not None:
                    assert optim_dur_disc is not None
                    scaler.unscale_(optim_dur_disc)
                    # torch.nn.utils.clip_grad_norm_(
                    # parameters=net_dur_disc.parameters(), max_norm=5
                    # )
                    grad_norm_dur = commons.clip_grad_value_(
                        net_dur_disc.parameters(), None
                    )
                    scaler.step(optim_dur_disc)

                if net_wd is not None:
                    assert optim_wd is not None
                    scaler.unscale_(optim_wd)
                    grad_norm_wd = commons.clip_grad_value_(net_wd.parameters(), None)
                    scaler.step(optim_wd)

        with autocast(
            device_type="cuda",
            enabled=hps.train.bf16_run,
            dtype=torch.bfloat16,
        ):
            # Generator
            y_dur_hat_g_gen: list[torch.Tensor] | None = None
            if disable_discriminators:
                loss_fm = torch.tensor(0.0, device=y.device)
                loss_gen = torch.tensor(0.0, device=y.device)
                losses_gen = []
            else:
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                if net_dur_disc is not None:
                    _, y_dur_hat_g_gen = net_dur_disc(hidden_x, x_mask, logw_, logw, g)
                if net_wd is not None:
                    assert wl is not None
                    loss_lm = wl(y.detach().squeeze(1), y_hat.squeeze(1)).mean()
                    loss_lm_gen = wl.generator(y_hat.squeeze(1))
                with autocast(
                    device_type="cuda",
                    enabled=hps.train.bf16_run,
                    dtype=torch.bfloat16,
                ):
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
            with autocast(
                device_type="cuda",
                enabled=hps.train.bf16_run,
                dtype=torch.bfloat16,
            ):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                # NOTE: loss_commit は JP-Extra 版では使用されておらず、config.json の "c_commit": 100 は参照されない
                ## オリジナル Bert-VITS2 では VQ (Vector Quantization) のコミットメント損失として
                ## 使用されていたが、JP-Extra 版では VQ が削除されたため不要になった
                # loss_commit = loss_commit * hps.train.c_commit

                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

                # Duration Discriminator の損失を追加
                if net_dur_disc is not None and not disable_discriminators:
                    assert y_dur_hat_g_gen is not None
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g_gen)
                    loss_gen_all += loss_dur_gen

                # ======================================================================
                ## オリジナルの Bert-VITS2 JP-Extra では、WavLM 損失 (loss_lm, loss_lm_gen) の
                ## 加算が Duration Discriminator の条件分岐の内部にネストされていた。
                ## 一方 JP-Extra 版のデフォルトハイパラは use_duration_discriminator: false のため、
                ## WavLM 損失が計算されても loss_gen_all には加算されない状態になっていた。
                ## このコードでは、WavLM 損失と Duration Discriminator 損失を独立して処理するため、
                ## Duration Discriminator / WavLM Discriminator を個別に有効/無効化できるようになっている。
                ## なお、事前学習モデルの Generator は WavLM 損失を使用せずに学習されてしまっていたと考えられるため、
                ## 現在のデフォルトハイパラでは use_wavlm_discriminator: false としている。そのため、この修正は実質的に影響しない。
                ## 将来的に WavLM 損失を有効化したモデルを学習する場合に、初めてこの修正が機能するようになる。
                # ======================================================================
                # WavLM Discriminator の損失を追加 (Duration Discriminator の有無に関係なく)
                if net_wd is not None and not disable_discriminators:
                    assert loss_lm is not None
                    assert loss_lm_gen is not None
                    loss_gen_all += loss_lm + loss_lm_gen

        if is_first_accumulation_step:
            optim_g.zero_grad()

        # 勾配累積時は損失を累積ステップ数で割って平均化
        if is_accumulating:
            loss_gen_all = loss_gen_all / gradient_accumulation_steps

        net_g_no_sync = net_g.no_sync if hasattr(net_g, "no_sync") else nullcontext
        with (
            net_g_no_sync()
            if is_accumulating and not is_last_accumulation_step
            else nullcontext()
        ):
            scaler.scale(loss_gen_all).backward()

        # 累積ステップの最後でのみオプティマイザを更新
        grad_norm_g = 0.0
        if should_step:
            scaler.unscale_(optim_g)
            # 固定写像の初回 backward で、等長性と学習対象が設計どおりかを同じ実行ログへ残す
            if (
                rank == 0
                and global_step == 0
                and hps.model.use_fixed_isometric_speaker_projection is True
            ):
                if logger is None:
                    raise RuntimeError("Training logger is not initialized")
                fixed_projection = net_g.module.get_buffer(
                    "speaker_isometric_projection"
                )
                if fixed_projection is None:
                    raise RuntimeError("Fixed speaker projection buffer is missing")
                identity = torch.eye(
                    fixed_projection.shape[1],
                    device=fixed_projection.device,
                    dtype=fixed_projection.dtype,
                )
                orthogonality_max_error = float(
                    (fixed_projection.T @ fixed_projection - identity)
                    .abs()
                    .max()
                    .item()
                )
                projection_scale = net_g.module.get_parameter(
                    "speaker_projection_scale"
                )
                if projection_scale is None:
                    raise RuntimeError("Speaker projection scale is missing")
                projection_scale_grad = projection_scale.grad
                style_vec_max_deviation = 0.0
                if _style_vec_shared_mean is not None:
                    shared_style_vec = _style_vec_shared_mean.to(
                        device=style_vec.device,
                        dtype=style_vec.dtype,
                    )
                    style_vec_max_deviation = float(
                        (style_vec - shared_style_vec.unsqueeze(0)).abs().max().item()
                    )
                g_norm_mean = float(g.detach().squeeze(-1).norm(dim=-1).mean().item())
                emb_g_norm_median = float(
                    net_g.module.emb_g.weight.detach().norm(dim=-1).median().item()
                )
                logger.info(
                    "Fixed isometric projection dry-run probe. "
                    f"orthogonality_max_error: {orthogonality_max_error:.8e}, "
                    f"projection_scale_grad: {None if projection_scale_grad is None else float(projection_scale_grad.item())}, "
                    f"projection_buffer_grad_is_none: {fixed_projection.grad is None}, "
                    f"style_vec_max_deviation: {style_vec_max_deviation:.8e}, "
                    f"emb_g_grad_is_none: {net_g.module.emb_g.weight.grad is None}, "
                    f"g_norm_mean: {g_norm_mean:.6f}, "
                    f"emb_g_norm_median: {emb_g_norm_median:.6f}"
                )
            # Transformer-based DP/SDP は duration 系だけ強めに抑えて初期 flow の暴れを監視する
            grad_norm_g, generator_grad_metrics = clip_generator_gradients(
                net_g,
                should_use_duration_clip=should_use_duration_clip,
            )
            scaler.step(optim_g)
            scaler.update()

            # EMA の更新（オプティマイザステップ後に実行）
            if ema_model is not None:
                ema_model.update(net_g)

            # 勾配ノルムに基づく自動学習率調整
            # 分散学習での競合を避けるため、rank 0 のみがモニタリングと調整を行う
            if rank == 0 and gradient_monitor is not None:
                gradient_monitor.update(grad_norm_g, optim_g, scheduler_g, global_step)

        if rank == 0:
            if (
                should_step
                and global_step % hps.train.log_interval == 0
                and not runtime_config.speedup
            ):
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                # logger.info(
                #     "Train Epoch: {} [{:.0f}%]".format(
                #         epoch, 100.0 * batch_idx / len(train_loader)
                #     )
                # )
                # logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )
                scalar_dict.update(generator_grad_metrics)
                if use_duration_symbol_type_ids is True:
                    scalar_dict.update(
                        collect_duration_symbol_type_metrics(
                            logw=logw,
                            logw_target=logw_,
                            x_mask=x_mask,
                            duration_symbol_type_ids=duration_symbol_type_ids,
                        )
                    )
                    scalar_dict.update(
                        collect_duration_residual_tail_metrics(
                            net_g=net_g,
                            hidden_x=hidden_x,
                            x_mask=x_mask,
                            g=g,
                            logw=logw,
                            duration_symbol_type_ids=duration_symbol_type_ids,
                        )
                    )
                scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
                scalar_dict.update(
                    {f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)}
                )

                if net_dur_disc is not None and not disable_discriminators:
                    assert loss_dur_disc_all is not None
                    assert losses_dur_disc_g is not None
                    assert losses_dur_disc_r is not None
                    assert loss_dur_gen is not None
                    assert losses_dur_gen is not None

                    scalar_dict.update({"loss/dur_disc/total": loss_dur_disc_all})

                    scalar_dict.update(
                        {
                            f"loss/dur_disc_g/{i}": v
                            for i, v in enumerate(losses_dur_disc_g)
                        }
                    )
                    scalar_dict.update(
                        {
                            f"loss/dur_disc_r/{i}": v
                            for i, v in enumerate(losses_dur_disc_r)
                        }
                    )

                    scalar_dict.update({"loss/g/dur_gen": loss_dur_gen})
                    scalar_dict.update(
                        {f"loss/g/dur_gen_{i}": v for i, v in enumerate(losses_dur_gen)}
                    )

                # NOTE: 現在のコードでは、ここでログに記録される損失は実際に loss_gen_all に加算されて学習に使用されるようになっている
                ## 修正前は loss_lm, loss_lm_gen がログには表示されるが学習には使われない状態だった
                if net_wd is not None and not disable_discriminators:
                    assert loss_slm is not None
                    assert loss_lm is not None
                    assert loss_lm_gen is not None
                    scalar_dict.update(
                        {
                            "loss/wd/total": loss_slm,
                            "grad_norm_wd": grad_norm_wd,
                            "loss/g/lm": loss_lm,
                            "loss/g/lm_gen": loss_lm_gen,
                        }
                    )

                # g ベクトルの統計をログに記録
                # Adapter が出力する g 空間の挙動を監視するためのメトリクス
                if g is not None:
                    g_norms = g.detach().squeeze(-1).norm(dim=-1)
                    scalar_dict["g/norm_mean"] = g_norms.mean()
                    scalar_dict["g/norm_std"] = g_norms.std()

                    # speaker embedding 使用時、ゲート付き delta 経路の統計を TensorBoard 用スカラーに記録する
                    if (
                        use_speaker_embedding
                        and speaker_embedding is not None
                        and hasattr(net_g.module, "speaker_control_encoder")
                        and net_g.module.speaker_control_encoder is not None
                        and hasattr(net_g.module, "speaker_adapter")
                        and net_g.module.speaker_adapter is not None
                    ):
                        with torch.no_grad():
                            if speaker_embedding.dim() == 1:
                                speaker_embedding = speaker_embedding.unsqueeze(0)
                            ctrl = net_g.module.speaker_control_encoder(
                                speaker_embedding
                            )
                            gated_delta, delta, gate_val = (
                                net_g.module.speaker_adapter.forward_with_intermediates(
                                    ctrl
                                )
                            )
                            scalar_dict["g/adapter_delta_norm_mean"] = delta.norm(
                                dim=-1
                            ).mean()
                            scalar_dict["g/adapter_gated_delta_norm_mean"] = (
                                gated_delta.norm(dim=-1).mean()
                            )
                            scalar_dict["g/adapter_gate"] = gate_val

                # 分布診断メトリクス
                ## 診断専用なので勾配を持たせず、TensorBoard と警告ログにだけ利用する
                assert g is not None
                if speaker_embedding is not None and g.shape[0] >= 4:
                    if adapter_dist_metrics is None:
                        with torch.no_grad():
                            adapter_dist_metrics = (
                                net_g.module.compute_adapter_distribution_metrics(
                                    g.detach().float(),
                                )
                            )
                    assert adapter_dist_metrics is not None
                    if adapter_dist_metrics["statistics_initialized"].item():
                        scalar_dict["adapter_dist/var_ratio_mean"] = (
                            adapter_dist_metrics["var_ratio_mean"]
                        )
                        scalar_dict["adapter_dist/var_ratio_min"] = (
                            adapter_dist_metrics["var_ratio_min"]
                        )
                        scalar_dict["adapter_dist/mean_shift"] = adapter_dist_metrics[
                            "mean_shift"
                        ]
                        scalar_dict["adapter_dist/pc_var_ratio_mean"] = (
                            adapter_dist_metrics["pc_var_ratio_mean"]
                        )
                        scalar_dict["adapter_dist/pc_var_ratio_min"] = (
                            adapter_dist_metrics["pc_var_ratio_min"]
                        )
                        # 警告閾値: 分散比が著しく低いときに警告ログを出す (Early stopping のシグナル)
                        var_ratio_min_val = float(
                            adapter_dist_metrics["var_ratio_min"].item()
                        )
                        warning_threshold = float(
                            hps.train.adapter_min_variance_ratio_warning
                        )
                        if var_ratio_min_val < warning_threshold and logger is not None:
                            logger.warning(
                                f"Adapter output distribution shrinkage detected: "
                                f"var_ratio_min: {var_ratio_min_val:.3f} "
                                f"(threshold: {warning_threshold:.3f}, step: {global_step}). "
                                f"Review SpeakerAdapter design or training data diversity."
                            )

                # 以降のログは計算が重い気がするし誰も見てない気がするのでコメントアウト
                # image_dict = {
                #     "slice/mel_org": utils.plot_spectrogram_to_numpy(
                #         y_mel[0].data.cpu().numpy()
                #     ),
                #     "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                #         y_hat_mel[0].data.cpu().numpy()
                #     ),
                #     "all/mel": utils.plot_spectrogram_to_numpy(
                #         mel[0].data.cpu().numpy()
                #     ),
                #     "all/attn": utils.plot_alignment_to_numpy(
                #         attn[0, 0].data.cpu().numpy()
                #     ),
                # }
                assert writer is not None
                summarize(
                    writer=writer,
                    global_step=global_step,
                    # images=image_dict,
                    scalars=scalar_dict,
                )

            if (
                global_step % hps.train.eval_interval == 0
                and global_step != 0
                and initial_step != global_step
            ):
                if not runtime_config.speedup:
                    assert eval_loader is not None
                    assert writer_eval is not None
                    evaluate(hps, net_g, eval_loader, writer_eval)
                assert runtime_config.model_dir is not None
                utils.checkpoints.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(runtime_config.model_dir, f"G_{global_step}.pth"),
                )
                utils.checkpoints.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(runtime_config.model_dir, f"D_{global_step}.pth"),
                )
                if net_dur_disc is not None:
                    assert optim_dur_disc is not None
                    utils.checkpoints.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(
                            runtime_config.model_dir, f"DUR_{global_step}.pth"
                        ),
                    )
                if net_wd is not None:
                    assert optim_wd is not None
                    utils.checkpoints.save_checkpoint(
                        net_wd,
                        optim_wd,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(runtime_config.model_dir, f"WD_{global_step}.pth"),
                    )
                if ema_model is not None:
                    ema_state = {
                        "state_dict": ema_model.state_dict(),
                        "decay": ema_model.decay,
                    }
                    torch.save(
                        ema_state,
                        os.path.join(
                            runtime_config.model_dir, f"EMA_{global_step}.pth"
                        ),
                    )
                if runtime_config.keep_ckpts > 0:
                    utils.checkpoints.clean_checkpoints(
                        model_dir_path=runtime_config.model_dir,
                        n_ckpts_to_keep=runtime_config.keep_ckpts,
                        sort_by_time=True,
                    )
                # Save safetensors (for inference) to `model_assets/{model_name}`
                # EMA が有効な場合は EMA 重みを保存（より安定した推論結果が期待できる）
                model_to_save = (
                    ema_model.get_ema_model() if ema_model is not None else net_g
                )
                utils.safetensors.save_safetensors(
                    model_to_save,
                    epoch,
                    os.path.join(
                        runtime_config.out_dir,
                        f"{runtime_config.model_name}_e{epoch}_s{global_step}.safetensors",
                    ),
                    for_infer=True,
                )
                if runtime_config.repo_id is not None:
                    api.upload_folder(
                        repo_id=runtime_config.repo_id,
                        folder_path=runtime_config.dataset_path,
                        path_in_repo=f"Data/{runtime_config.model_name}",
                        delete_patterns="*.pth",  # Only keep the latest checkpoint
                        ignore_patterns="raw/**",  # Ignore raw data
                        run_as_future=True,
                    )
                    api.upload_folder(
                        repo_id=runtime_config.repo_id,
                        folder_path=runtime_config.out_dir,
                        path_in_repo=f"model_assets/{runtime_config.model_name}",
                        run_as_future=True,
                    )

        global_step += 1
        if max_steps is not None and global_step >= max_steps:
            if rank == 0:
                assert runtime_config.model_dir is not None
                assert runtime_config.out_dir is not None

                # step 上限で止める smoke でも後段測定に使う checkpoint を必ず残す
                utils.checkpoints.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(runtime_config.model_dir, f"G_{global_step}.pth"),
                )
                utils.checkpoints.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(runtime_config.model_dir, f"D_{global_step}.pth"),
                )
                if net_dur_disc is not None:
                    assert optim_dur_disc is not None
                    utils.checkpoints.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(
                            runtime_config.model_dir, f"DUR_{global_step}.pth"
                        ),
                    )
                if net_wd is not None:
                    assert optim_wd is not None
                    utils.checkpoints.save_checkpoint(
                        net_wd,
                        optim_wd,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(runtime_config.model_dir, f"WD_{global_step}.pth"),
                    )
                if ema_model is not None:
                    ema_state = {
                        "state_dict": ema_model.state_dict(),
                        "decay": ema_model.decay,
                    }
                    torch.save(
                        ema_state,
                        os.path.join(
                            runtime_config.model_dir, f"EMA_{global_step}.pth"
                        ),
                    )
                if runtime_config.keep_ckpts > 0:
                    utils.checkpoints.clean_checkpoints(
                        model_dir_path=runtime_config.model_dir,
                        n_ckpts_to_keep=runtime_config.keep_ckpts,
                        sort_by_time=True,
                    )

                # 推論用の safetensors も同じ step 名で出力し、測定対象を明確にする
                model_to_save = (
                    ema_model.get_ema_model() if ema_model is not None else net_g
                )
                utils.safetensors.save_safetensors(
                    model_to_save,
                    epoch,
                    os.path.join(
                        runtime_config.out_dir,
                        f"{runtime_config.model_name}_e{epoch}_s{global_step}.safetensors",
                    ),
                    for_infer=True,
                )
                if logger is not None:
                    logger.info(f"Reached max_steps: {global_step}. Stop training.")
            return True
        if pbar is not None:
            pbar.set_description(
                f"Epoch {epoch}({100.0 * batch_idx / len(train_loader):.0f}%)/{hps.train.epochs}"
            )
            pbar.update()
    # 本家ではこれをスピードアップのために消すと書かれていたので、一応消してみる
    # と思ったけどメモリ使用量が減るかもしれないのでつけてみる
    gc.collect()
    torch.cuda.empty_cache()
    if pbar is None and rank == 0:
        assert logger is not None
        logger.info(f"====> Epoch: {epoch}, step: {global_step}")
    return False


def evaluate(
    hps: HyperParameters,
    generator: DDP,
    eval_loader: DataLoader[Any],
    writer_eval: SummaryWriter,
) -> None:
    global _fixed_eval_batch
    generator.eval()
    image_dict = {}
    audio_dict = {}
    scalar_dict: dict[str, float | torch.Tensor] = {}
    print()
    logger.info("Evaluating ...")
    use_speaker_embedding = getattr(hps.data, "use_speaker_embedding", False)
    use_duration_symbol_type_ids = hps.model.use_duration_symbol_type_embedding

    # Adapter 経由 g の分布診断: eval_loader 全体で g を集約し、分布縮小を Aggregate 統計で検出する
    ## ステップ単位ログ (バッチ内分散) はノイズが大きいため、より信頼性の高い検査を eval 時に行う
    aggregated_g: list[torch.Tensor] = []
    is_nanairo_adapter_active = use_speaker_embedding is True and (
        hasattr(generator.module, "compute_adapter_distribution_metrics")
        and getattr(generator.module, "use_speaker_adapter", False) is True
        and getattr(
            generator.module,
            "use_fixed_isometric_speaker_projection",
            False,
        )
        is not True
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            # 初回の evaluate() 時に最初のバッチを固定サンプルとしてキャッシュ
            # 以降の評価ステップで常に同一入力から合成し、学習進行に伴う品質変化を追跡する
            if _fixed_eval_batch is None and batch_idx == 0:
                _fixed_eval_batch = tuple(
                    t.clone() if isinstance(t, torch.Tensor) else t for t in batch
                )

            if use_speaker_embedding and use_duration_symbol_type_ids:
                (
                    x,
                    x_lengths,
                    spec,
                    spec_lengths,
                    y,
                    y_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    style_vec,
                    duration_symbol_type_ids,
                    speaker_embedding,
                ) = batch
            elif use_speaker_embedding:
                (
                    x,
                    x_lengths,
                    spec,
                    spec_lengths,
                    y,
                    y_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    style_vec,
                    speaker_embedding,
                ) = batch
                duration_symbol_type_ids = None
            elif use_duration_symbol_type_ids:
                (
                    x,
                    x_lengths,
                    spec,
                    spec_lengths,
                    y,
                    y_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    style_vec,
                    duration_symbol_type_ids,
                ) = batch
                speaker_embedding = None
            else:
                (
                    x,
                    x_lengths,
                    spec,
                    spec_lengths,
                    y,
                    y_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    style_vec,
                ) = batch
                speaker_embedding = None
                duration_symbol_type_ids = None
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            style_vec = style_vec.cuda()
            # 評価でも学習と同じ共有平均を使い、対象音声の style_vec から話者情報が戻らないようにする
            if hps.train.style_vec_shared_mean is True:
                if _style_vec_shared_mean is None:
                    raise RuntimeError("Shared style vector mean is not initialized")
                style_vec = (
                    _style_vec_shared_mean.to(
                        device=style_vec.device,
                        dtype=style_vec.dtype,
                    )
                    .unsqueeze(0)
                    .expand(style_vec.shape[0], -1)
                )
            if duration_symbol_type_ids is not None:
                duration_symbol_type_ids = duration_symbol_type_ids.cuda()
            if speaker_embedding is not None:
                speaker_embedding = speaker_embedding.cuda()

            # Adapter 出力 g をバッチごとに収集 (eval 全体で aggregate 分布診断するため)
            if is_nanairo_adapter_active and speaker_embedding is not None:
                if speaker_embedding.dim() == 1:
                    speaker_embedding_for_g = speaker_embedding.unsqueeze(0)
                else:
                    speaker_embedding_for_g = speaker_embedding
                ctrl = generator.module.speaker_control_encoder(speaker_embedding_for_g)
                gated_delta = generator.module.speaker_adapter(ctrl)
                g_adapter = (
                    generator.module.get_buffer("g_neutral") + gated_delta
                ).float()
                aggregated_g.append(g_adapter.detach())

            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.module.infer(
                    x,
                    x_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    style_vec,
                    y=spec,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                    speaker_embedding=speaker_embedding,
                    duration_symbol_type_ids=duration_symbol_type_ids,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
                # 以降のログは計算が重い気がするし誰も見てない気がするのでコメントアウト
                # mel = spec_to_mel_torch(
                #     spec,
                #     hps.data.filter_length,
                #     hps.data.n_mel_channels,
                #     hps.data.sampling_rate,
                #     hps.data.mel_fmin,
                #     hps.data.mel_fmax,
                # )
                # y_hat_mel = mel_spectrogram_torch(
                #     y_hat.squeeze(1).float(),
                #     hps.data.filter_length,
                #     hps.data.n_mel_channels,
                #     hps.data.sampling_rate,
                #     hps.data.hop_length,
                #     hps.data.win_length,
                #     hps.data.mel_fmin,
                #     hps.data.mel_fmax,
                # )
                # image_dict.update(
                #     {
                #         f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                #             y_hat_mel[0].cpu().numpy()
                #         )
                #     }
                # )
                # image_dict.update(
                #     {
                #         f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                #             mel[0].cpu().numpy()
                #         )
                #     }
                # )
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

        # 固定サンプルによる定期合成 (学習進行に伴う品質変化の追跡用)
        # 常に同一入力から合成することで、ステップ間の品質推移を TensorBoard で聴き比べ可能にする
        if _fixed_eval_batch is not None:
            fixed_eval_batch = _fixed_eval_batch
            if use_speaker_embedding and use_duration_symbol_type_ids:
                (
                    fixed_x,
                    fixed_x_lengths,
                    fixed_spec,
                    fixed_spec_lengths,
                    fixed_y,
                    fixed_y_lengths,
                    fixed_speakers,
                    fixed_tone,
                    fixed_language,
                    fixed_bert,
                    fixed_style_vec,
                    fixed_duration_symbol_type_ids,
                    fixed_speaker_embedding,
                ) = fixed_eval_batch
            elif use_speaker_embedding:
                (
                    fixed_x,
                    fixed_x_lengths,
                    fixed_spec,
                    fixed_spec_lengths,
                    fixed_y,
                    fixed_y_lengths,
                    fixed_speakers,
                    fixed_tone,
                    fixed_language,
                    fixed_bert,
                    fixed_style_vec,
                    fixed_speaker_embedding,
                ) = fixed_eval_batch
                fixed_duration_symbol_type_ids = None
            elif use_duration_symbol_type_ids:
                (
                    fixed_x,
                    fixed_x_lengths,
                    fixed_spec,
                    fixed_spec_lengths,
                    fixed_y,
                    fixed_y_lengths,
                    fixed_speakers,
                    fixed_tone,
                    fixed_language,
                    fixed_bert,
                    fixed_style_vec,
                    fixed_duration_symbol_type_ids,
                ) = fixed_eval_batch
                fixed_speaker_embedding = None
            else:
                (
                    fixed_x,
                    fixed_x_lengths,
                    fixed_spec,
                    fixed_spec_lengths,
                    fixed_y,
                    fixed_y_lengths,
                    fixed_speakers,
                    fixed_tone,
                    fixed_language,
                    fixed_bert,
                    fixed_style_vec,
                ) = fixed_eval_batch
                fixed_speaker_embedding = None
                fixed_duration_symbol_type_ids = None
            fixed_x = fixed_x.cuda()
            fixed_x_lengths = fixed_x_lengths.cuda()
            fixed_spec = fixed_spec.cuda()
            fixed_spec_lengths = fixed_spec_lengths.cuda()
            fixed_y = fixed_y.cuda()
            fixed_y_lengths = fixed_y_lengths.cuda()
            fixed_speakers = fixed_speakers.cuda()
            fixed_bert = fixed_bert.cuda()
            fixed_tone = fixed_tone.cuda()
            fixed_language = fixed_language.cuda()
            fixed_style_vec = fixed_style_vec.cuda()
            # 固定評価サンプルも共有平均へ置換し、step 間で同じ条件を比較する
            if hps.train.style_vec_shared_mean is True:
                if _style_vec_shared_mean is None:
                    raise RuntimeError("Shared style vector mean is not initialized")
                fixed_style_vec = (
                    _style_vec_shared_mean.to(
                        device=fixed_style_vec.device,
                        dtype=fixed_style_vec.dtype,
                    )
                    .unsqueeze(0)
                    .expand(fixed_style_vec.shape[0], -1)
                )
            if fixed_duration_symbol_type_ids is not None:
                fixed_duration_symbol_type_ids = fixed_duration_symbol_type_ids.cuda()
            if fixed_speaker_embedding is not None:
                fixed_speaker_embedding = fixed_speaker_embedding.cuda()
            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.module.infer(
                    fixed_x,
                    fixed_x_lengths,
                    fixed_speakers,
                    fixed_tone,
                    fixed_language,
                    fixed_bert,
                    fixed_style_vec,
                    y=fixed_spec,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                    speaker_embedding=fixed_speaker_embedding,
                    duration_symbol_type_ids=fixed_duration_symbol_type_ids,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
                audio_dict[f"fixed/audio_{use_sdp}"] = y_hat[0, :, : y_hat_lengths[0]]
            audio_dict["fixed/gt_audio"] = fixed_y[0, :, : fixed_y_lengths[0]]

    # Aggregate 分布診断: eval セット全体を通じて収集した Adapter 出力 g に対し
    # 信頼度の高い分布縮小検査を行い、TensorBoard に記録する
    if is_nanairo_adapter_active and len(aggregated_g) > 0:
        with torch.no_grad():
            g_concat = torch.cat(aggregated_g, dim=0).float()  # [N, gin_channels]
            if g_concat.shape[0] >= 4:
                eval_metrics = generator.module.compute_adapter_distribution_metrics(
                    g_concat,
                )
                if eval_metrics["statistics_initialized"].item():
                    scalar_dict["eval/adapter_dist/var_ratio_mean"] = float(
                        eval_metrics["var_ratio_mean"].item()
                    )
                    scalar_dict["eval/adapter_dist/var_ratio_min"] = float(
                        eval_metrics["var_ratio_min"].item()
                    )
                    scalar_dict["eval/adapter_dist/mean_shift"] = float(
                        eval_metrics["mean_shift"].item()
                    )
                    scalar_dict["eval/adapter_dist/pc_var_ratio_mean"] = float(
                        eval_metrics["pc_var_ratio_mean"].item()
                    )
                    scalar_dict["eval/adapter_dist/pc_var_ratio_min"] = float(
                        eval_metrics["pc_var_ratio_min"].item()
                    )
                    scalar_dict["eval/adapter_dist/sample_count"] = float(
                        g_concat.shape[0]
                    )
                    # Aggregate でも分散縮小が見られたら警告 (Early stopping の主シグナル)
                    eval_var_ratio_min = float(eval_metrics["var_ratio_min"].item())
                    warning_threshold = float(
                        hps.train.adapter_min_variance_ratio_warning
                    )
                    if eval_var_ratio_min < warning_threshold:
                        logger.warning(
                            f"[EVAL] Adapter output distribution shrinkage confirmed: "
                            f"var_ratio_min: {eval_var_ratio_min:.3f} "
                            f"(threshold: {warning_threshold:.3f}, eval N: {g_concat.shape[0]}). "
                            f"Aggregate measurement is more reliable than step-wise warnings; "
                            f"consider stopping training and revisiting Adapter setup "
                            f"(bottleneck_dim / data diversity)."
                        )

    summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        scalars=scalar_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


if __name__ == "__main__":
    run()
