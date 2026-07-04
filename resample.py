"""
音声ファイルのリサンプリングスクリプト。

raw/ ディレクトリ内の音声ファイルを指定したサンプリングレートに変換し、
wavs/ ディレクトリに出力する。
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import librosa
import pyloudnorm as pyln
import soundfile
from numpy.typing import NDArray
from tqdm import tqdm

from style_bert_vits2.logging import logger
from style_bert_vits2.utils.paths import TrainingModelPaths, add_model_argument
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


DEFAULT_BLOCK_SIZE: float = 0.400  # seconds


class BlockSizeException(Exception):
    pass


def normalize_audio(data: NDArray[Any], sr: int) -> NDArray[Any]:
    """
    音声データをラウドネス正規化する。

    ITU-R BS.1770 規格に基づいたラウドネス測定を行い、
    -23 LUFS に正規化する。

    Args:
        data (NDArray[Any]): 音声データ（numpy 配列）
        sr (int): サンプリングレート

    Returns:
        NDArray[Any]: 正規化された音声データ

    Raises:
        BlockSizeException: 音声が短すぎてラウドネス測定ができない場合
    """

    meter = pyln.Meter(sr, block_size=DEFAULT_BLOCK_SIZE)  # create BS.1770 meter
    try:
        loudness = meter.integrated_loudness(data)
    except ValueError as ex:
        raise BlockSizeException(ex) from ex

    data = pyln.normalize.loudness(data, loudness, -23.0)
    return data


def resample(
    file: Path,
    input_dir: Path,
    output_dir: Path,
    target_sr: int,
    normalize: bool,
    trim: bool,
) -> None:
    """
    音声ファイルを読み込み、指定したサンプリングレートで保存する。

    input_dir からの相対パスを保持したまま output_dir に保存する。
    出力ファイルの拡張子は常に .wav になる。

    Args:
        file (Path): 入力ファイルのパス
        input_dir (Path): 入力ディレクトリのルートパス
        output_dir (Path): 出力ディレクトリのルートパス
        target_sr (int): 目標サンプリングレート
        normalize (bool): ラウドネス正規化を行うかどうか
        trim (bool): 無音部分をトリムするかどうか
    """

    try:
        # librosa が読めるファイルかチェック
        # wav 以外にも mp3 や ogg や flac なども読める
        wav, sr = librosa.load(file, sr=target_sr)
        if normalize:
            try:
                wav = normalize_audio(wav, int(sr))
            except BlockSizeException:
                print("")
                logger.info(
                    f"Skip normalize due to less than {DEFAULT_BLOCK_SIZE} second audio: {file}"
                )
        if trim:
            wav, _ = librosa.effects.trim(wav, top_db=30)
        relative_path = file.relative_to(input_dir)
        # ここで拡張子が .wav 以外でも .wav に置き換えられる
        output_path = output_dir / relative_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(output_path, wav, int(sr))
    except Exception as ex:
        logger.warning(f"Cannot load file, so skipping: {file}", exc_info=ex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample audio files in raw/ directory to wavs/ directory.",
    )
    add_model_argument(parser)
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Target sampling rate (default: 44100)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of parallel processes (0 = auto)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Apply loudness normalization",
    )
    parser.add_argument(
        "--trim",
        action="store_true",
        default=False,
        help="Trim silence at start and end",
    )
    args = parser.parse_args()

    # TrainingModelPaths を使ってパスを解決
    model_folder_name: str = args.model
    paths = TrainingModelPaths(model_folder_name)

    if args.num_processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes: int = args.num_processes

    input_dir = paths.raw_dir
    output_dir = paths.wavs_dir
    logger.info(f"Resampling {input_dir} to {output_dir}")
    sr: int = int(args.sr)
    normalize_flag: bool = args.normalize
    trim_flag: bool = args.trim

    # librosa / soundfile がサポートする音声ファイルの拡張子でフィルタリング
    # ref: https://librosa.org/doc/0.11.0/troubleshooting.html
    ## .DS_Store などの非音声ファイルを除外しないと librosa が audioread にフォールバックして警告が出る
    supported_extensions = {
        ".wav",
        ".flac",
        ".ogg",
        ".mp3",
        ".m4a",
        ".aac",
        ".wma",
        ".opus",
    }
    original_files = [
        f
        for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if len(original_files) == 0:
        logger.error(f"No files found in {input_dir}")
        raise ValueError(f"No files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=processes) as executor:
        futures = [
            executor.submit(
                resample, file, input_dir, output_dir, sr, normalize_flag, trim_flag
            )
            for file in original_files
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(original_files),
            file=SAFE_STDOUT,
            dynamic_ncols=True,
        ):
            pass

    logger.info("Resampling Done!")
