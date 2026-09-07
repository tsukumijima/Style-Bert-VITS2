import copy
from dataclasses import dataclass
from typing import Any, cast

import torch


@dataclass(frozen=True)
class TrainRuntimeConfig:
    """
    学習スクリプトでのみ使用されるランタイム設定。

    Args:
        model_name: config.json 由来のモデル名
        model_dir: チェックポイントの保存先ディレクトリ
        out_dir: 推論用モデルの出力先ディレクトリ
        dataset_path: データセットのルートディレクトリ
        keep_ckpts: 保持するチェックポイント数
        repo_id: Hugging Face へのバックアップ用リポジトリ ID
        speedup: 速度優先モードの有効化フラグ
        spec_cache: Spectrogram キャッシュの有効化フラグ
    """

    model_name: str
    model_dir: str
    out_dir: str
    dataset_path: str
    keep_ckpts: int
    repo_id: str | None
    speedup: bool
    spec_cache: bool = True


class EMAModel:
    """
    モデル重みの指数移動平均 (Exponential Moving Average) を管理するクラス。

    学習中のモデル重みは勾配更新により振動するが、EMA はその平滑化されたバージョンを保持する。
    推論時に EMA 重みを使用することで、より安定した出力が得られる傾向がある。

    使い方:
        1. 学習開始時に EMAModel を初期化
        2. 各オプティマイザステップ後に update() を呼び出し
        3. チェックポイント保存時に get_ema_model() で EMA モデルを取得して保存

    Args:
        model: EMA を適用する対象モデル (通常は Generator)
        decay: EMA の減衰率 (0.999 が一般的)。
            値が大きいほど過去の重みを重視し、より滑らかになる。
        device: EMA モデルを配置するデバイス
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,  # 減衰率: 0.999 が一般的な値
        device: torch.device | None = None,
    ):
        self.decay = decay
        self.device = device

        # モデルの深いコピーを作成して EMA 用の重みを保持
        # DDP の場合は .module を使用して内部モデルを取得
        if hasattr(model, "module"):
            self.ema_model = cast(torch.nn.Module, copy.deepcopy(model.module))
        else:
            self.ema_model = copy.deepcopy(model)

        # EMA モデルは学習しないので勾配計算を無効化
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        if device is not None:
            self.ema_model.to(device)

    @torch.no_grad()  # type: ignore[misc]
    def update(self, model: torch.nn.Module) -> None:
        """
        現在のモデル重みで EMA を更新する。

        各オプティマイザステップ後に呼び出すこと。
        EMA 更新式: ema_weight = decay * ema_weight + (1 - decay) * current_weight

        Args:
            model: 現在の学習中モデル
        """

        # DDP の場合は .module を使用
        source_model = cast(
            torch.nn.Module, model.module if hasattr(model, "module") else model
        )

        for ema_param, param in zip(
            self.ema_model.parameters(),
            source_model.parameters(),
        ):
            # EMA 更新: ema = decay * ema + (1 - decay) * current
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def get_ema_model(self) -> torch.nn.Module:
        """
        EMA モデルを取得する。

        チェックポイント保存時や推論時に使用。

        Returns:
            EMA 重みを持つモデル
        """

        return self.ema_model

    def state_dict(self) -> dict[str, Any]:
        """
        EMA モデルの状態辞書を取得する。

        Returns:
            EMA モデルの状態辞書
        """

        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        EMA モデルの状態を復元する。

        Args:
            state_dict: 以前のチェックポイントからの状態変数を含む辞書。
        """

        self.ema_model.load_state_dict(state_dict)
