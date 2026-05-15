"""
Nanairo duration 学習の監視と勾配制御 helper を提供する
"""

from __future__ import annotations

from typing import Any

import torch

from style_bert_vits2.models import commons
from style_bert_vits2.models.models_nanairo import SynthesizerTrn
from style_bert_vits2.nlp.symbols import DURATION_TOKEN_TYPE_NAMES


GENERATOR_DURATION_PARAM_PREFIXES = (
    "dp.",
    "sdp.",
    "duration_token_type_emb.",
)
GENERATOR_NO_WEIGHT_DECAY_MARKERS = (
    "bias",
    "norm",
    "norm_layers_1",
    "norm_layers_2",
    "norms_1",
    "norms_2",
    "emb_rel_k",
    "emb_rel_v",
)
WEIGHT_DECAY_DEFAULT = 1e-2
WEIGHT_DECAY_ZERO = 0.0
GRAD_CLIP_MAX_NORM_FALLBACK = 500
GRAD_CLIP_MAX_NORM_OTHER = 200
GRAD_CLIP_MAX_NORM_DURATION = 50
SDP_SAMPLING_NOISE_SCALE = 1.0


def is_duration_generator_parameter(parameter_name: str) -> bool:
    """
    Generator の duration 系パラメータかどうかを判定する

    Args:
        parameter_name (str): `SynthesizerTrn` 内のパラメータ名

    Returns:
        bool: DP / SDP / duration type embedding のパラメータかどうか
    """

    return parameter_name.startswith(GENERATOR_DURATION_PARAM_PREFIXES)


def should_disable_weight_decay(parameter_name: str) -> bool:
    """
    AdamW の weight decay から外すパラメータかどうかを判定する

    Args:
        parameter_name (str): `SynthesizerTrn` 内のパラメータ名

    Returns:
        bool: bias / 正規化層 / 相対位置埋め込みに該当するかどうか
    """

    return any(
        marker in parameter_name.split(".")
        for marker in GENERATOR_NO_WEIGHT_DECAY_MARKERS
    )


def unwrap_generator(net_g: Any) -> SynthesizerTrn:
    """
    DDP 有無に関係なく実体の Generator を取得する

    Args:
        net_g (Any): Generator または DDP でラップされた Generator

    Returns:
        SynthesizerTrn: 実体の Generator
    """

    return getattr(net_g, "module", net_g)


def build_generator_optimizer_parameter_groups(
    net_g: Any,
    learning_rate: float,
) -> list[dict[str, Any]]:
    """
    Transformer-based DP/SDP 向けに Generator の AdamW パラメータ群を分ける

    Args:
        net_g (SynthesizerTrn): Generator
        learning_rate (float): 各パラメータ群へ設定する学習率

    Returns:
        list[dict[str, Any]]: AdamW に渡すパラメータ群
    """

    duration_weight_params: list[torch.nn.Parameter] = []
    duration_no_decay_params: list[torch.nn.Parameter] = []
    other_weight_params: list[torch.nn.Parameter] = []
    other_no_decay_params: list[torch.nn.Parameter] = []

    model = unwrap_generator(net_g)
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad is False:
            continue

        if is_duration_generator_parameter(parameter_name) is False:
            if should_disable_weight_decay(parameter_name) is True:
                other_no_decay_params.append(parameter)
            else:
                other_weight_params.append(parameter)
            continue

        if should_disable_weight_decay(parameter_name) is True:
            duration_no_decay_params.append(parameter)
        else:
            duration_weight_params.append(parameter)

    parameter_groups: list[dict[str, Any]] = []
    if len(duration_weight_params) > 0:
        parameter_groups.append(
            {
                "params": duration_weight_params,
                "lr": learning_rate,
                "weight_decay": WEIGHT_DECAY_DEFAULT,
                "name": "duration_weight",
            }
        )
    if len(duration_no_decay_params) > 0:
        parameter_groups.append(
            {
                "params": duration_no_decay_params,
                "lr": learning_rate,
                "weight_decay": WEIGHT_DECAY_ZERO,
                "name": "duration_no_decay",
            }
        )
    if len(other_weight_params) > 0:
        parameter_groups.append(
            {
                "params": other_weight_params,
                "lr": learning_rate,
                "weight_decay": WEIGHT_DECAY_DEFAULT,
                "name": "other_weight",
            }
        )
    if len(other_no_decay_params) > 0:
        parameter_groups.append(
            {
                "params": other_no_decay_params,
                "lr": learning_rate,
                "weight_decay": WEIGHT_DECAY_ZERO,
                "name": "other_no_decay",
            }
        )

    return parameter_groups


def collect_duration_type_metrics(
    logw: torch.Tensor,
    logw_target: torch.Tensor,
    x_mask: torch.Tensor,
    token_types: torch.Tensor | None,
) -> dict[str, float]:
    """
    DP の type 別 MSE と token 数を TensorBoard 用に集計する

    Args:
        logw (torch.Tensor): DP が予測した log duration
        logw_target (torch.Tensor): MAS 由来の教師 log duration
        x_mask (torch.Tensor): 有効 token のマスク
        token_types (torch.Tensor | None): duration token type ID

    Returns:
        dict[str, float]: TensorBoard に出力するスカラー
    """

    if token_types is None:
        return {}

    metrics: dict[str, float] = {}
    valid_mask = x_mask.detach().squeeze(1) > 0.5
    squared_error = ((logw.detach() - logw_target.detach()).squeeze(1) ** 2).float()
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return metrics

    for type_id, type_name in enumerate(DURATION_TOKEN_TYPE_NAMES):
        type_mask = (token_types == type_id) & valid_mask
        type_count = int(type_mask.sum().item())
        metrics[f"count/dur_types/{type_name}"] = float(type_count)
        metrics[f"fraction/dur_types/{type_name}"] = type_count / valid_count
        if type_count > 0:
            metrics[f"loss/dur_dp/{type_name}"] = float(
                squared_error[type_mask].mean().item()
            )

    return metrics


def collect_duration_residual_tail_metrics(
    net_g: Any,
    hidden_x: torch.Tensor,
    x_mask: torch.Tensor,
    g: torch.Tensor,
    logw: torch.Tensor,
    token_types: torch.Tensor | None,
) -> dict[str, float]:
    """
    SDP sampled residual の tail を観測専用メトリクスとして集計する

    Args:
        net_g (Any): DDP でラップされた Generator
        hidden_x (torch.Tensor): TextEncoder の出力
        x_mask (torch.Tensor): 有効 token のマスク
        g (torch.Tensor): 話者条件ベクトル
        logw (torch.Tensor): DP が予測した log duration
        token_types (torch.Tensor | None): duration token type ID

    Returns:
        dict[str, float]: TensorBoard に出力するスカラー
    """

    if token_types is None:
        return {}

    metrics: dict[str, float] = {}
    with torch.no_grad():
        model = unwrap_generator(net_g)
        duration_x = model.apply_duration_token_types(hidden_x, token_types)
        sampled_logw = model.sdp(
            duration_x,
            x_mask,
            g=g,
            reverse=True,
            noise_scale=SDP_SAMPLING_NOISE_SCALE,
        )
        residual_abs = (sampled_logw - logw.detach()).abs().squeeze(1).float()
        valid_mask = x_mask.detach().squeeze(1) > 0.5

        for type_id, type_name in enumerate(DURATION_TOKEN_TYPE_NAMES):
            type_values = residual_abs[(token_types == type_id) & valid_mask]
            if type_values.numel() == 0:
                continue
            metrics[f"duration_residual/{type_name}/p50"] = float(
                torch.quantile(type_values, 0.50).item()
            )
            metrics[f"duration_residual/{type_name}/p95"] = float(
                torch.quantile(type_values, 0.95).item()
            )
            metrics[f"duration_residual/{type_name}/p99"] = float(
                torch.quantile(type_values, 0.99).item()
            )
            metrics[f"duration_residual/{type_name}/max"] = float(
                type_values.max().item()
            )

    return metrics


def clip_generator_gradients(
    net_g: Any,
    should_use_duration_clip: bool,
) -> tuple[float, dict[str, float]]:
    """
    Generator の勾配を duration 系とそれ以外で分けてクリップする

    Args:
        net_g (Any): DDP でラップされた Generator
        should_use_duration_clip (bool): duration 専用クリップを使うかどうか

    Returns:
        tuple[float, dict[str, float]]: 全体の勾配ノルムと監視用スカラー
    """

    grad_metrics: dict[str, float] = {}
    model = unwrap_generator(net_g)
    if should_use_duration_clip is False:
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=GRAD_CLIP_MAX_NORM_FALLBACK,
        )
        grad_norm_g = commons.clip_grad_value_(model.parameters(), None)
        return grad_norm_g, grad_metrics

    duration_parameters: list[torch.nn.Parameter] = []
    other_parameters: list[torch.nn.Parameter] = []
    duration_prefix_parameters: dict[str, list[torch.nn.Parameter]] = {
        "dp": [],
        "sdp_prior": [],
        "sdp_posterior": [],
        "sdp_flow": [],
        "sdp_other": [],
        "token_type_emb": [],
    }

    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if is_duration_generator_parameter(parameter_name) is True:
            duration_parameters.append(parameter)
            if parameter_name.startswith("dp."):
                duration_prefix_parameters["dp"].append(parameter)
            elif parameter_name.startswith(
                ("sdp.prior_flow_layers.", "sdp.posterior_flow_layers.")
            ):
                duration_prefix_parameters["sdp_flow"].append(parameter)
            elif parameter_name.startswith("sdp.prior_"):
                duration_prefix_parameters["sdp_prior"].append(parameter)
            elif parameter_name.startswith("sdp.posterior_"):
                duration_prefix_parameters["sdp_posterior"].append(parameter)
            elif parameter_name.startswith("sdp."):
                duration_prefix_parameters["sdp_other"].append(parameter)
            elif parameter_name.startswith("duration_token_type_emb."):
                duration_prefix_parameters["token_type_emb"].append(parameter)
        else:
            other_parameters.append(parameter)

    if len(other_parameters) > 0:
        other_norm = torch.nn.utils.clip_grad_norm_(
            parameters=other_parameters,
            max_norm=GRAD_CLIP_MAX_NORM_OTHER,
        )
        grad_metrics["grad_norm/g_other_preclip"] = float(other_norm.item())
    if len(duration_parameters) > 0:
        duration_norm = torch.nn.utils.clip_grad_norm_(
            parameters=duration_parameters,
            max_norm=GRAD_CLIP_MAX_NORM_DURATION,
        )
        grad_metrics["grad_norm/g_duration_preclip"] = float(duration_norm.item())

    for prefix_name, parameters in duration_prefix_parameters.items():
        if len(parameters) == 0:
            continue
        grad_metrics[f"grad_norm/duration/{prefix_name}"] = commons.clip_grad_value_(
            parameters,
            None,
        )

    grad_norm_g = commons.clip_grad_value_(model.parameters(), None)
    return grad_norm_g, grad_metrics
