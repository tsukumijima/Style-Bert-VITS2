"""
モデル関連のパス管理ユーティリティ。

学習・前処理スクリプトで使用する共通のパス解決機能を提供する。
"""

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from style_bert_vits2.constants import (
    DEFAULT_ASSETS_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_PATHS_CONFIG_PATH,
    DEFAULT_PATHS_TEMPLATE_PATH,
)
from style_bert_vits2.logging import logger


@dataclass(frozen=True)
class PathsConfig:
    """
    パス設定を管理するデータクラス。

    Args:
        dataset_root: 学習データセットのルートディレクトリ
        assets_root: 推論用モデルアセットのルートディレクトリ
    """

    dataset_root: Path
    assets_root: Path


_CACHED_PATHS_CONFIG: PathsConfig | None = None


def get_paths_config(config_path: Path | None = None) -> PathsConfig:
    """
    paths.yml からパス設定を読み込み、PathsConfig を返す。

    Args:
        config_path: 読み込む設定ファイルパス。未指定時は configs/paths.yml。

    Returns:
        パス設定を格納した PathsConfig
    """

    def _normalize_paths_config_value(value: str | Path | None, fallback: Path) -> Path:
        """
        paths.yml の値を正規化して Path として返す。

        Args:
            value: 取得した値
            fallback: 値が空の場合に使うフォールバックパス

        Returns:
            正規化済みのパス
        """

        if value is None or value == "":
            return fallback
        return Path(value)

    global _CACHED_PATHS_CONFIG

    if config_path is None and _CACHED_PATHS_CONFIG is not None:
        return _CACHED_PATHS_CONFIG

    target_config_path = config_path or DEFAULT_PATHS_CONFIG_PATH
    if not target_config_path.exists():
        if DEFAULT_PATHS_TEMPLATE_PATH.exists():
            shutil.copy(DEFAULT_PATHS_TEMPLATE_PATH, target_config_path)
            logger.info(
                f"A configuration file {target_config_path} has been generated based on the default configuration file {DEFAULT_PATHS_TEMPLATE_PATH}."
            )
            logger.info(
                f"Please do not modify {DEFAULT_PATHS_TEMPLATE_PATH}. Instead, modify {target_config_path}."
            )
        else:
            logger.warning(
                f"Paths configuration file not found: {target_config_path}. Falling back to defaults."
            )
            paths_config = PathsConfig(
                dataset_root=DEFAULT_DATASET_ROOT,
                assets_root=DEFAULT_ASSETS_ROOT,
            )
            if config_path is None:
                _CACHED_PATHS_CONFIG = paths_config
            return paths_config

    with open(target_config_path, encoding="utf-8") as file:
        yaml_config: dict[str, Any] = yaml.safe_load(file.read()) or {}

    dataset_root = _normalize_paths_config_value(
        yaml_config.get("dataset_root"),
        DEFAULT_DATASET_ROOT,
    )
    assets_root = _normalize_paths_config_value(
        yaml_config.get("assets_root"),
        DEFAULT_ASSETS_ROOT,
    )
    paths_config = PathsConfig(
        dataset_root=dataset_root,
        assets_root=assets_root,
    )
    if config_path is None:
        _CACHED_PATHS_CONFIG = paths_config
    return paths_config


def add_model_argument(
    parser: argparse.ArgumentParser,
    required: bool = True,
) -> None:
    """
    共通の --model 引数を ArgumentParser に追加するヘルパー関数。

    Args:
        parser (argparse.ArgumentParser): ArgumentParser インスタンス
        required (bool): 必須引数かどうか（デフォルト: True）
    """

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=required,
        help="Model folder name (folder name under Data/)",
    )


def add_dataset_root_argument(parser: argparse.ArgumentParser) -> None:
    """
    共通の --dataset_root 引数を ArgumentParser に追加するヘルパー関数。
    未指定時は paths.yml の dataset_root がそのまま使われるため、従来の挙動は変わらない。

    Args:
        parser (argparse.ArgumentParser): ArgumentParser インスタンス
    """

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(get_paths_config().dataset_root),
        help="Root directory of training datasets (the parent directory of the model folder). "
        "Defaults to dataset_root in configs/paths.yml.",
    )


@dataclass
class TrainingModelPaths:
    """
    学習時のモデル関連のパスを管理するデータクラス。

    Data/ 以下のフォルダ名（model_folder_name）を元に、
    学習・前処理で使用する各種パスを自動的に導出する。
    """

    # Data/ 以下のフォルダ名
    model_folder_name: str
    # データセットのルートディレクトリ
    dataset_root: Path = field(default_factory=lambda: get_paths_config().dataset_root)
    # 推論用モデルアセットのルートディレクトリ
    assets_root: Path = field(default_factory=lambda: get_paths_config().assets_root)

    @property
    def dataset_dir(self) -> Path:
        """データセットディレクトリ: Data/{model_folder_name}/"""
        return self.dataset_root / self.model_folder_name

    @property
    def config_path(self) -> Path:
        """ハイパーパラメータ設定ファイル: Data/{model_folder_name}/config.json"""
        return self.dataset_dir / "config.json"

    @property
    def esd_list_path(self) -> Path:
        """書き起こし・アノテーションデータ: Data/{model_folder_name}/esd.list"""
        return self.dataset_dir / "esd.list"

    @property
    def esd_list_cleaned_path(self) -> Path:
        """前処理済み書き起こし・アノテーションデータ: Data/{model_folder_name}/esd.list.cleaned"""
        return self.dataset_dir / "esd.list.cleaned"

    @property
    def train_list_path(self) -> Path:
        """学習用アノテーションデータ: Data/{model_folder_name}/train.list"""
        return self.dataset_dir / "train.list"

    @property
    def val_list_path(self) -> Path:
        """検証用アノテーションデータ: Data/{model_folder_name}/val.list"""
        return self.dataset_dir / "val.list"

    @property
    def raw_dir(self) -> Path:
        """元音声データディレクトリ: Data/{model_folder_name}/raw/"""
        return self.dataset_dir / "raw"

    @property
    def wavs_dir(self) -> Path:
        """前処理済み音声データディレクトリ: Data/{model_folder_name}/wavs/"""
        return self.dataset_dir / "wavs"

    @property
    def models_dir(self) -> Path:
        """チェックポイント保存ディレクトリ: Data/{model_folder_name}/models/"""
        return self.dataset_dir / "models"
