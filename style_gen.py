"""
スタイルベクトル生成スクリプト。

train.list / val.list から読み込んだ音声ファイルに対応する
スタイルベクトルを生成し、.npy ファイルとして保存する。
"""

import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


# pyannote.audio のインポート前に TF32 警告を抑制
# TF32 は再現性のために意図的に無効化されており、この警告自体は無害
# ref: https://github.com/pyannote/pyannote-audio/issues/1370
warnings.filterwarnings("ignore", message="TensorFloat-32.*")

from pyannote.audio import Inference, Model
from tqdm import tqdm

from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.utils.paths import (
    TrainingModelPaths,
    add_dataset_root_argument,
    add_model_argument,
)
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


class NaNValueError(ValueError):
    """カスタム例外クラス。NaN 値が見つかった場合に使用されます。"""


# グローバル変数として推論モデルを保持
# main で初期化され、process_line から参照される
_inference: Inference | None = None


def initialize_inference(device: str) -> Inference:
    """
    推論モデルを初期化する。

    Args:
        device (str): 使用するデバイス（"cuda" または "cpu"）

    Returns:
        Inference: 初期化された Inference インスタンス
    """

    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    assert model is not None
    inference = Inference(model, window="whole")
    inference.to(torch.device(device))
    return inference


def get_style_vector(wav_path: str) -> NDArray[Any]:
    """
    音声ファイルからスタイルベクトルを抽出する。

    Args:
        wav_path (str): 音声ファイルのパス

    Returns:
        NDArray[Any]: スタイルベクトル（numpy 配列）
    """

    assert _inference is not None, "Inference model is not initialized"
    return _inference(wav_path)  # type: ignore


def save_style_vector(wav_path: str) -> None:
    """
    スタイルベクトルを生成して保存する。

    Args:
        wav_path (str): 音声ファイルのパス

    Raises:
        NaNValueError: スタイルベクトルに NaN 値が含まれている場合
    """

    try:
        style_vec = get_style_vector(wav_path)
    except Exception as ex:
        print("\n")
        logger.error(f"Error occurred with file: {wav_path}", exc_info=ex)
        raise

    # 値に NaN が含まれていると悪影響なのでチェックする
    if np.isnan(style_vec).any():
        print("\n")
        logger.warning(f"NaN value found in style vector: {wav_path}")
        raise NaNValueError(f"NaN value found in style vector: {wav_path}")

    np.save(f"{wav_path}.npy", style_vec)  # `test.wav` -> `test.wav.npy`


def process_line(x: tuple[str, Path]) -> tuple[str, str | None]:
    """
    1行のデータを処理し、スタイルベクトルを生成する。

    Args:
        x (tuple[str, Path]): (行データ, wavs_dir) のタプル

    Returns:
        tuple[str, str | None]: (行データ, エラー種別) のタプル。エラーがなければエラー種別は None。
    """

    line, wavs_dir = x
    # wav_path は wavs_dir からの相対パスなので、フルパスを構築
    wav_relative_path = line.split("|")[0]
    wav_path = str(wavs_dir / wav_relative_path)
    try:
        save_style_vector(wav_path)
        return line, None
    except NaNValueError:
        return line, "nan_error"
    except Exception as ex:
        logger.error(f"Failed to generate style vector: {wav_path}", exc_info=ex)
        return line, "error"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate style vectors for audio files.",
    )
    add_model_argument(parser)
    add_dataset_root_argument(parser)
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of parallel processes (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for style vector extraction (default: cuda if available)",
    )
    args = parser.parse_args()

    # TrainingModelPaths を使ってパスを解決
    model_folder_name: str = args.model
    paths = TrainingModelPaths(model_folder_name, dataset_root=Path(args.dataset_root))
    config_path = paths.config_path
    wavs_dir = paths.wavs_dir

    num_processes: int = args.num_processes
    device: str = args.device

    # 推論モデルを初期化
    _inference = initialize_inference(device)

    hps = HyperParameters.load_from_json(config_path)

    train_files_path = Path(hps.data.training_files)
    val_files_path = Path(hps.data.validation_files)

    training_lines: list[str] = []
    with open(train_files_path, encoding="utf-8") as f:
        training_lines.extend(f.readlines())

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        training_results = list(
            tqdm(
                executor.map(
                    process_line,
                    [(line, wavs_dir) for line in training_lines],
                ),
                total=len(training_lines),
                file=SAFE_STDOUT,
                dynamic_ncols=True,
            )
        )

    ok_training_lines = [line for line, error in training_results if error is None]
    nan_training_lines = [
        line for line, error in training_results if error == "nan_error"
    ]
    error_training_lines = [
        line for line, error in training_results if error == "error"
    ]
    if nan_training_lines:
        nan_files = [line.split("|")[0] for line in nan_training_lines]
        logger.warning(
            f"Found NaN value in {len(nan_training_lines)} files: {nan_files}, so they will be deleted from training data."
        )
    if error_training_lines:
        error_files = [line.split("|")[0] for line in error_training_lines]
        logger.warning(
            f"Failed to generate style vectors for {len(error_training_lines)} files: {error_files}, so they will be deleted from training data."
        )

    val_lines: list[str] = []
    with open(val_files_path, encoding="utf-8") as f:
        val_lines.extend(f.readlines())

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        val_results = list(
            tqdm(
                executor.map(
                    process_line,
                    [(line, wavs_dir) for line in val_lines],
                ),
                total=len(val_lines),
                file=SAFE_STDOUT,
                dynamic_ncols=True,
            )
        )

    ok_val_lines = [line for line, error in val_results if error is None]
    nan_val_lines = [line for line, error in val_results if error == "nan_error"]
    error_val_lines = [line for line, error in val_results if error == "error"]
    if nan_val_lines:
        nan_files = [line.split("|")[0] for line in nan_val_lines]
        logger.warning(
            f"Found NaN value in {len(nan_val_lines)} files: {nan_files}, so they will be deleted from validation data."
        )
    if error_val_lines:
        error_files = [line.split("|")[0] for line in error_val_lines]
        logger.warning(
            f"Failed to generate style vectors for {len(error_val_lines)} files: {error_files}, so they will be deleted from validation data."
        )

    with open(train_files_path, "w", encoding="utf-8") as f:
        f.writelines(ok_training_lines)

    with open(val_files_path, "w", encoding="utf-8") as f:
        f.writelines(ok_val_lines)

    ok_num = len(ok_training_lines) + len(ok_val_lines)

    logger.info(f"Finished generating style vectors! total: {ok_num} npy files.")
