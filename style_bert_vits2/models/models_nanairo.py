import math
from typing import Any

import torch
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from style_bert_vits2.models import attentions, commons, modules, monotonic_alignment
from style_bert_vits2.nlp.symbols import (
    DURATION_SYMBOL_TYPE_COUNT,
    NANAIRO_SYMBOLS,
    NUM_LANGUAGES,
    NUM_TONES,
)


class DurationDiscriminator(nn.Module):  # vits2
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.LSTM = nn.LSTM(
            2 * filter_channels, filter_channels, batch_first=True, bidirectional=True
        )

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(
            nn.Linear(2 * filter_channels, 1), nn.Sigmoid()
        )

    def forward_probability(self, x: torch.Tensor, dur: torch.Tensor) -> torch.Tensor:
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = x.transpose(1, 2)
        x, _ = self.LSTM(x)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        dur_r: torch.Tensor,
        dur_hat: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        x = torch.detach(x)
        if g is not None and hasattr(self, "cond"):
            g = torch.detach(g)
            x = commons.auto_inplace_add(x, self.cond(g))
        x = self.conv_1(x * x_mask)
        x = commons.auto_inplace_relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = commons.auto_inplace_relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, dur)
            output_probs.append(output_prob)

        return output_probs


class TransformerCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
        share_parameter: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = (
            # attentions.FFT(
            #     hidden_channels,
            #     filter_channels,
            #     n_heads,
            #     n_layers,
            #     kernel_size,
            #     p_dropout,
            #     isflow=True,
            #     gin_channels=self.gin_channels,
            # )
            None if share_parameter else None
        )

        for i in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                    gin_channels=self.gin_channels,
                )
            )
            self.flows.append(modules.Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        # メモリ効率化のため、長い系列ではチャンク処理を適用
        seq_len = x.size(2)
        CHUNK_SIZE = 1024
        if seq_len > CHUNK_SIZE:
            return self._chunked_forward(x, x_mask, g, reverse, CHUNK_SIZE)
        else:
            return self._standard_forward(x, x_mask, g, reverse)

    def _standard_forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None,
        reverse: bool,
    ) -> torch.Tensor:
        """従来の Flow 処理の実装"""
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def _chunked_forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None,
        reverse: bool,
        chunk_size: int,
    ) -> torch.Tensor:
        """チャンクごとに処理しメモリ消費の肥大化を抑えた Flow 処理の実装"""
        batch_size, channels, seq_len = x.shape
        overlap_size = 16

        outputs = []

        for start in range(0, seq_len, chunk_size - overlap_size):
            end = min(start + chunk_size, seq_len)
            # チャンク抽出
            chunk_x = x[:, :, start:end]
            chunk_mask = x_mask[:, :, start:end]

            # チャンクごとの Flow 処理
            chunk_output = self._standard_forward(chunk_x, chunk_mask, g, reverse)

            # オーバーラップ処理
            if start == 0:
                # 最初のチャンク
                if end < seq_len:
                    outputs.append(chunk_output[:, :, : -overlap_size // 2])
                else:
                    outputs.append(chunk_output)
            elif end >= seq_len:
                # 最後のチャンク
                outputs.append(chunk_output[:, :, overlap_size // 2 :])
            else:
                # 中間チャンク
                outputs.append(
                    chunk_output[:, :, overlap_size // 2 : -overlap_size // 2]
                )

            # メモリ解放
            del chunk_x, chunk_mask, chunk_output

        # 結合
        result = torch.cat(outputs, dim=2)

        # 長さ調整
        if result.size(2) != seq_len:
            if result.size(2) < seq_len:
                # パディング
                pad_size = seq_len - result.size(2)
                result = torch.nn.functional.pad(result, (0, pad_size))
            else:
                # トリミング
                result = result[:, :, :seq_len]

        return result


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None and hasattr(self, "cond"):
            g = torch.detach(g)
            x = commons.auto_inplace_add(x, self.cond(g))
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = torch.detach(x)
        if g is not None and hasattr(self, "cond"):
            g = torch.detach(g)
            x = commons.auto_inplace_add(x, self.cond(g))
        x = self.conv_1(x * x_mask)
        x = commons.auto_inplace_relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = commons.auto_inplace_relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TransformerDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        window_size: int = 4,
        gin_channels: int = 0,
        cond_layer_idx: int = 2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.gin_channels = gin_channels

        self.transformer_encoder = attentions.Encoder(
            in_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
            gin_channels=gin_channels,
            cond_layer_idx=cond_layer_idx,
        )
        self.output_proj = nn.Conv1d(in_channels, 1, 1)
        self.output_proj.weight.data.zero_()
        assert self.output_proj.bias is not None
        self.output_proj.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if g is not None:
            g = torch.detach(g)
        x = self.transformer_encoder(x, x_mask, g=g)
        x = self.output_proj(x * x_mask)
        return x * x_mask


class TransformerStochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        transformer_filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        window_size: int = 4,
        gin_channels: int = 0,
        cond_layer_idx: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.transformer_filter_channels = transformer_filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.window_size = window_size
        self.gin_channels = gin_channels

        self.duration_log_flow = modules.Log()

        self.prior_input_proj = nn.Conv1d(in_channels, filter_channels, 1)
        self.prior_cond_encoder = attentions.Encoder(
            filter_channels,
            transformer_filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
            gin_channels=gin_channels,
            cond_layer_idx=cond_layer_idx,
        )
        self.prior_output_proj = nn.Conv1d(filter_channels, filter_channels, 1)

        self.posterior_input_proj = nn.Conv1d(1, filter_channels, 1)
        self.posterior_cond_encoder = attentions.Encoder(
            filter_channels,
            transformer_filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
        )
        self.posterior_output_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.posterior_output_proj.weight.data.zero_()
        assert self.posterior_output_proj.bias is not None
        self.posterior_output_proj.bias.data.zero_()

        self.prior_flow_layers = nn.ModuleList()
        self.prior_flow_layers.append(modules.ElementwiseAffine(2))
        for _ in range(n_flows):
            self.prior_flow_layers.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.prior_flow_layers.append(modules.Flip())

        self.posterior_flow_layers = nn.ModuleList()
        self.posterior_flow_layers.append(modules.ElementwiseAffine(2))
        for _ in range(n_flows):
            self.posterior_flow_layers.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.posterior_flow_layers.append(modules.Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        if g is not None:
            g = torch.detach(g)
        x = self.prior_input_proj(x)
        x = self.prior_cond_encoder(x, x_mask, g=g)
        x = self.prior_output_proj(x) * x_mask

        if not reverse:
            assert w is not None

            logdet_tot_q = 0
            h_w = self.posterior_input_proj(w)
            h_w = self.posterior_cond_encoder(h_w, x_mask)
            h_w = self.posterior_output_proj(h_w) * x_mask
            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            for flow in self.posterior_flow_layers:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.duration_log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in self.prior_flow_layers:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq

        flows = list(reversed(self.prior_flow_layers))
        flows = flows[:-2] + [flows[-1]]
        z = (
            torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
            * noise_scale
        )
        for flow in flows:
            z = flow(z, x_mask, g=x, reverse=reverse)
        z0, _ = torch.split(z, [1, 1], 1)
        return z0


class Bottleneck(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        super().__init__(c_fc1, c_fc2)


class Block(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(self.norm(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class SpeakerControlEncoder(nn.Module):
    """
    anime-speaker-embedding から、話者制御用の低次元表現（制御部分空間）を抽出する。
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpeakerAdapter(nn.Module):
    """
    制御部分空間から g 空間への差分を、ゲート付き残差として生成する。
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, ctrl: torch.Tensor) -> torch.Tensor:
        delta = self.net(ctrl)
        gate = torch.sigmoid(self.gate_logit)
        return gate * delta

    def forward_with_intermediates(
        self, ctrl: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        中間値を返す forward 。学習中の監視・診断用。

        Args:
            ctrl (torch.Tensor): control subspace のベクトル

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                gated_delta, delta (gate 前), gate (scalar)
        """

        delta = self.net(ctrl)
        gate = torch.sigmoid(self.gate_logit)
        return gate * delta, delta, gate


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # Keep TextEncoder vocabulary size consistent with the caller contract.
        # Nanairo extends the token inventory, so detect mismatches against NANAIRO_SYMBOLS early.
        assert n_vocab == len(NANAIRO_SYMBOLS), (
            f"Nanairo TextEncoder expected n_vocab: {len(NANAIRO_SYMBOLS)}, got: {n_vocab}"
        )
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.tone_emb = nn.Embedding(NUM_TONES, hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)
        self.language_emb = nn.Embedding(NUM_LANGUAGES, hidden_channels)
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)

        # Remove emo_vq since it's not working well.
        self.style_proj = nn.Linear(256, hidden_channels)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        g: torch.Tensor | None = None,
        use_fp16: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bert_emb = self.bert_proj(bert).transpose(1, 2)
        style_emb = self.style_proj(style_vec.unsqueeze(1))
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + bert_emb
            + style_emb
        ) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )

        x = self.encoder(x * x_mask, x_mask, g=g, use_fp16=use_fp16)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock_str: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock_str == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        ch = None
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))  # type: ignore

        assert ch is not None
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(commons.init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None and hasattr(self, "cond"):
            x = commons.auto_inplace_add(x, self.cond(g))

        for i in range(self.num_upsamples):
            x = commons.auto_inplace_leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            assert xs is not None
            x = xs / self.num_kernels
        x = commons.auto_inplace_leaky_relu(x)
        x = self.conv_post(x)
        x = commons.auto_inplace_tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        # print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()  # type: ignore


class DiscriminatorP(torch.nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = commons.auto_inplace_leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = commons.auto_inplace_leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class WavLMDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(
        self,
        slm_hidden: int = 768,
        slm_layers: int = 13,
        initial_channel: int = 64,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.pre = norm_f(
            Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0)
        )

        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv1d(
                        initial_channel, initial_channel * 2, kernel_size=5, padding=2
                    )
                ),
                norm_f(
                    nn.Conv1d(
                        initial_channel * 2,
                        initial_channel * 4,
                        kernel_size=5,
                        padding=2,
                    )
                ),
                norm_f(
                    nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)
                ),
            ]
        )

        self.conv_post = norm_f(Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)

        fmap = []
        for l in self.convs:
            x = l(x)
            x = commons.auto_inplace_leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels: int, gin_channels: int = 0) -> None:
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = commons.auto_inplace_relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(
        self, L: int, kernel_size: int, stride: int, pad: int, n_convs: int
    ) -> int:
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        n_speakers: int = 256,
        gin_channels: int = 256,
        use_sdp: bool = True,
        n_flow_layer: int = 4,
        n_layers_trans_flow: int = 6,
        flow_share_parameter: bool = False,
        use_transformer_flow: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_layers_trans_flow = n_layers_trans_flow
        self.use_spk_conditioned_encoder = kwargs.get(
            "use_spk_conditioned_encoder", True
        )
        self.use_speaker_adapter = kwargs.get("use_speaker_adapter", False)
        self.speaker_adapter_input_dim = kwargs.get(
            "speaker_adapter_input_dim", gin_channels
        )
        self.speaker_adapter_bottleneck_dim = kwargs.get(
            "speaker_adapter_bottleneck_dim", 96
        )
        self.use_sdp = use_sdp
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        self.use_transformer_duration_predictor = kwargs.get(
            "use_transformer_duration_predictor", False
        )
        self.duration_filter_channels = int(
            kwargs.get("duration_filter_channels", hidden_channels)
        )
        self.use_duration_symbol_type_embedding = kwargs.get(
            "use_duration_symbol_type_embedding", False
        )
        self.duration_symbol_type_count = int(
            kwargs.get("duration_symbol_type_count", DURATION_SYMBOL_TYPE_COUNT)
        )
        self.enc_gin_channels = 0
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.enc_gin_channels,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers_trans_flow,
                5,
                p_dropout,
                n_flow_layer,
                gin_channels=gin_channels,
                share_parameter=flow_share_parameter,
            )
        else:
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                n_flow_layer,
                gin_channels=gin_channels,
            )
        if self.use_transformer_duration_predictor is True:
            self.sdp = TransformerStochasticDurationPredictor(
                hidden_channels,
                hidden_channels,
                self.duration_filter_channels,
                n_heads,
                4,
                kernel_size,
                p_dropout,
                4,
                window_size=4,
                gin_channels=gin_channels,
                cond_layer_idx=2,
            )
            self.dp = TransformerDurationPredictor(
                hidden_channels,
                self.duration_filter_channels,
                n_heads,
                6,
                kernel_size,
                p_dropout,
                window_size=4,
                gin_channels=gin_channels,
                cond_layer_idx=2,
            )
        else:
            self.sdp = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
            )
            self.dp = DurationPredictor(
                hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
            )
        if self.use_duration_symbol_type_embedding is True:
            self.duration_symbol_type_emb = nn.Embedding(
                self.duration_symbol_type_count,
                hidden_channels,
            )
            # 未学習の symbol type の埋め込みが初期ランダム値のまま残らないようにする
            nn.init.zeros_(self.duration_symbol_type_emb.weight)
        else:
            self.duration_symbol_type_emb = None

        if n_speakers >= 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        else:
            self.ref_enc = ReferenceEncoder(spec_channels, gin_channels)
        if self.use_speaker_adapter is True:
            self.speaker_control_encoder = SpeakerControlEncoder(
                self.speaker_adapter_input_dim,
                self.speaker_adapter_bottleneck_dim,
            )
            self.speaker_adapter = SpeakerAdapter(
                self.speaker_adapter_bottleneck_dim,
                gin_channels,
            )
            self.register_buffer("g_neutral", torch.zeros(1, gin_channels))
        else:
            self.speaker_control_encoder = None
            self.speaker_adapter = None
            # Adapter 無効時でも g_neutral を登録し、チェックポイントの state_dict 互換性を維持する
            self.register_buffer("g_neutral", torch.zeros(1, gin_channels))

        # 事前学習済み emb_g の分布統計を保持する
        ## Adapter 経路の出力分布が事前学習 emb_g 分布から逸脱していないかの監視と、分散保持 loss に使う
        ## Adapter 無効時でも buffer を登録しチェックポイント互換性を維持する
        ## 全次元の分散 (per channel)、平均ベクトル、上位 k 個の主成分とその分散を保持する
        pc_k = max(1, int(kwargs.get("pc_variance_monitor_k", 8)))
        self._pc_variance_monitor_k = pc_k
        self.register_buffer("emb_g_mean", torch.zeros(1, gin_channels))
        self.register_buffer("emb_g_var", torch.ones(1, gin_channels))
        self.register_buffer("emb_g_pca_components", torch.zeros(pc_k, gin_channels))
        self.register_buffer("emb_g_pca_variances", torch.ones(pc_k))
        # 実際に SVD で計算できた PC 数 (= min(pc_k, n_speakers - 1, gin_channels))
        # 少数話者モデルで存在しない PC がメトリクスに混ざらないよう、有効範囲を保持する
        self.register_buffer("emb_g_pca_k_actual", torch.tensor(pc_k, dtype=torch.long))
        # emb_g_statistics が初期化済みかどうかのフラグ (学習開始時に set_emb_g_statistics() を呼ぶ)
        self.register_buffer("emb_g_statistics_initialized", torch.zeros(1))

    def set_g_neutral(self, neutral_g: torch.Tensor) -> None:
        """
        g_neutral を設定する。このメソッドは学習開始前に一度だけ呼び出す。

        Args:
            neutral_g (torch.Tensor): 基準となる g ベクトル
        """

        if neutral_g.dim() == 1:
            neutral_g = neutral_g.unsqueeze(0)
        # register_buffer 由来のテンソル更新（属性経由だと型チェッカーに誤解釈される場合がある）
        self.get_buffer("g_neutral").copy_(neutral_g)

    def set_emb_g_statistics(self) -> None:
        """
        事前学習済み emb_g の分布統計 (平均・分散・主成分) を計算して buffer に保存する。
        学習開始時に一度だけ呼び出すことを想定している。

        Notes:
            - n_speakers <= 0 の場合 (ref_enc 利用時) は何もしない
            - 主成分は PCA via SVD で計算する。計算量は O(N*D^2) で N=話者数、D=gin_channels
              典型的な多話者事前学習モデル (N~数百〜数千、D=512) では一瞬で終わる
            - PC 数は __init__ で確定した self._pc_variance_monitor_k に従う
        """

        if not hasattr(self, "emb_g") or self.n_speakers <= 0:
            return

        with torch.no_grad():
            weights = self.emb_g.weight.detach().float()  # [n_speakers, gin_channels]
            n_speakers, _ = weights.shape

            # 平均と分散 (per channel)
            mean = weights.mean(dim=0, keepdim=True)
            var = weights.var(dim=0, unbiased=False, keepdim=True)
            self.get_buffer("emb_g_mean").copy_(mean)
            self.get_buffer("emb_g_var").copy_(var)

            # 主成分分析 (SVD ベース)
            pc_k_target = self._pc_variance_monitor_k
            k_actual = min(pc_k_target, max(1, n_speakers - 1))
            weights_centered = weights - mean
            try:
                # full_matrices=False で thin SVD、Vh は [min(N,D), D]
                _, S, Vh = torch.linalg.svd(weights_centered, full_matrices=False)
                components = Vh[:k_actual]  # [k_actual, gin_channels]
                pc_variances = (S[:k_actual] ** 2) / max(1, n_speakers - 1)
            except Exception:
                # SVD 収束失敗時は zero 埋めで継続 (warning は呼び出し側で出す)
                components = torch.zeros(
                    k_actual, weights.shape[1], device=weights.device
                )
                pc_variances = torch.zeros(k_actual, device=weights.device)

            # buffer 形状にパディング (k_actual < pc_k_target の場合)
            pc_components_buf = self.get_buffer("emb_g_pca_components")
            pc_variances_buf = self.get_buffer("emb_g_pca_variances")
            pc_components_buf.zero_()
            pc_variances_buf.zero_()
            pc_components_buf[:k_actual].copy_(components)
            pc_variances_buf[:k_actual].copy_(pc_variances)

            # 有効 PC 数を buffer に保存 (compute_adapter_distribution_metrics で使う)
            self.get_buffer("emb_g_pca_k_actual").fill_(k_actual)

            # 初期化フラグを立てる
            self.get_buffer("emb_g_statistics_initialized").fill_(1.0)

    def compute_adapter_distribution_metrics(
        self,
        g_batch: torch.Tensor,
        target_variance_ratio: float = 0.8,
    ) -> dict[str, torch.Tensor]:
        """
        Adapter 経由で生成された g バッチに対し、事前学習 emb_g 分布からの逸脱を測定する。

        Args:
            g_batch (torch.Tensor): Adapter 出力の g バッチ。形状 [B, gin_channels] または
                [B, gin_channels, 1] を許容する
            target_variance_ratio (float): 分散保持 hinge loss の閾値。
                channel 別の `batch_var / emb_g_var` がこの値を下回った量を罰則とする

        Returns:
            dict[str, torch.Tensor]: 以下のキーを含む辞書
                - loss_variance_preserve: 分散保持 hinge loss (スカラー)。
                  分散が十分なら 0。ミニバッチサイズが小さいと信頼度は下がる
                - var_ratio_mean: 全 channel 平均の分散比 (1.0 が理想)
                - var_ratio_min: 最小の channel 分散比
                - mean_shift: バッチ平均と emb_g 平均の L2 距離
                - pc_var_ratio_mean: 主成分 k 個の平均分散比
                - pc_var_ratio_min: 主成分 k 個の最小分散比
                - statistics_initialized: emb_g 統計が set_emb_g_statistics() で初期化済みかの真偽
        """

        if g_batch.dim() == 3:
            g_batch = g_batch.squeeze(-1)

        eps = 1e-8
        device = g_batch.device
        statistics_initialized = self.get_buffer("emb_g_statistics_initialized") > 0.5

        # 統計未初期化の場合はゼロ loss を返す (set_emb_g_statistics() 未呼び出し時の安全装置)
        if not statistics_initialized.item():
            return {
                "loss_variance_preserve": torch.tensor(0.0, device=device),
                "var_ratio_mean": torch.tensor(1.0, device=device),
                "var_ratio_min": torch.tensor(1.0, device=device),
                "mean_shift": torch.tensor(0.0, device=device),
                "pc_var_ratio_mean": torch.tensor(1.0, device=device),
                "pc_var_ratio_min": torch.tensor(1.0, device=device),
                "statistics_initialized": statistics_initialized,
            }

        # チャネル別の分散比
        batch_var = g_batch.var(dim=0, unbiased=False)  # [gin_channels]
        target_var = self.get_buffer("emb_g_var").squeeze(0)  # [gin_channels]
        var_ratio_per_dim = batch_var / (target_var + eps)

        # Hinge: 分散比が target を下回った量だけ罰則 (上回るのは自由)
        loss_variance_preserve = torch.clamp(
            target_variance_ratio - var_ratio_per_dim, min=0.0
        ).mean()

        # 平均シフト (中央集中していなくても平均がずれているかを別軸で監視)
        batch_mean = g_batch.mean(dim=0)
        emb_g_mean = self.get_buffer("emb_g_mean").squeeze(0)
        mean_shift = (batch_mean - emb_g_mean).norm()

        # 主成分方向の分散比 (主要方向で縮小していないかをチェック)
        ## 有効 PC 数 (k_actual) でスライス、少数話者でゼロ埋めされた存在しない PC を除外する
        k_actual = int(self.get_buffer("emb_g_pca_k_actual").item())
        pc_components = self.get_buffer("emb_g_pca_components")[
            :k_actual
        ]  # [k_actual, gin_channels]
        pc_target_var = self.get_buffer("emb_g_pca_variances")[:k_actual]  # [k_actual]
        g_centered = g_batch - batch_mean.unsqueeze(0)
        # [B, gin_channels] @ [gin_channels, k_actual] -> [B, k_actual]
        g_projected = g_centered @ pc_components.t()
        batch_pc_var = g_projected.var(dim=0, unbiased=False)  # [k_actual]
        pc_var_ratio = batch_pc_var / (pc_target_var + eps)

        return {
            "loss_variance_preserve": loss_variance_preserve,
            "var_ratio_mean": var_ratio_per_dim.mean(),
            "var_ratio_min": var_ratio_per_dim.min(),
            "mean_shift": mean_shift,
            "pc_var_ratio_mean": pc_var_ratio.mean(),
            "pc_var_ratio_min": pc_var_ratio.min(),
            "statistics_initialized": statistics_initialized,
        }

    def apply_duration_symbol_type_ids(
        self,
        x: torch.Tensor,
        duration_symbol_type_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        # duration loss が TextEncoder へ逆流しない性質を保ちつつ、
        # duration 専用種別埋め込みだけは DP/SDP 側の学習対象にする
        x = torch.detach(x)
        if self.duration_symbol_type_emb is None:
            return x
        if duration_symbol_type_ids is None:
            # 種別 ID が未指定の場合は全位置を content 相当として扱う
            duration_symbol_type_ids = torch.zeros(
                x.size(0),
                x.size(2),
                dtype=torch.long,
                device=x.device,
            )
        symbol_type_emb = self.duration_symbol_type_emb(
            duration_symbol_type_ids
        ).transpose(1, 2)
        return x + symbol_type_emb

    def _resolve_g(
        self,
        sid: torch.Tensor,
        y: torch.Tensor | None,
        speaker_embedding: torch.Tensor | None,
        g_adjust: torch.Tensor | None,
    ) -> torch.Tensor:
        if speaker_embedding is not None:
            if self.speaker_control_encoder is None or self.speaker_adapter is None:
                raise ValueError("Speaker control encoder and adapter are required")
            if speaker_embedding.dim() == 1:
                speaker_embedding = speaker_embedding.unsqueeze(0)
            # g_neutral にゲート付き SpeakerAdapter の出力を加算して g を求める
            ctrl = self.speaker_control_encoder(speaker_embedding)
            gated_delta = self.speaker_adapter(ctrl)
            g = self.g_neutral + gated_delta
            g = g.unsqueeze(-1)
        else:
            if self.n_speakers > 0:
                g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
            else:
                if y is None:
                    raise ValueError("y must be provided when n_speakers <= 0")
                g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)

        if g_adjust is not None:
            if g_adjust.dim() == 1:
                g_adjust = g_adjust.unsqueeze(0)
            if g_adjust.dim() == 2:
                g_adjust = g_adjust.unsqueeze(-1)
            g = g + g_adjust

        return g

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        sid: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        speaker_embedding: torch.Tensor | None = None,
        g_adjust: torch.Tensor | None = None,
        duration_symbol_type_ids: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        g = self._resolve_g(sid, y, speaker_embedding, g_adjust)
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, style_vec, g=g
        )
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            if self.use_noise_scaled_mas:
                epsilon = (
                    torch.std(neg_cent)
                    * torch.randn_like(neg_cent)
                    * self.current_mas_noise_scale
                )
                neg_cent = neg_cent + epsilon

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_alignment.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)
        duration_x = self.apply_duration_symbol_type_ids(x, duration_symbol_type_ids)

        l_length_sdp = self.sdp(duration_x, x_mask, w, g=g)
        l_length_sdp = l_length_sdp / torch.sum(x_mask)

        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(duration_x, x_mask, g=g)
        # logw_sdp = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=1.0)
        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )  # for averaging
        # l_length_sdp += torch.sum((logw_sdp - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

        l_length = l_length_dp + l_length_sdp

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),  # type: ignore
            (x, logw, logw_),  # , logw_sdp),
            g,
        )

    def infer_input_feature(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        sdp_ratio: float = 0.0,
        y: torch.Tensor | None = None,
        use_fp16: bool = False,
        durations_frames_override: torch.Tensor | None = None,
        durations_frames_override_mask: torch.Tensor | None = None,
        speaker_embedding: torch.Tensor | None = None,
        g_adjust: torch.Tensor | None = None,
        duration_symbol_type_ids: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Generator への入力特徴量（潜在変数）を生成する。通常推論・ストリーミング推論の両方で共通。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                z (latent), y_mask, g (global conditioning), attn (attention), z_p, m_p, logs_p
        """
        # x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert)
        # g = self.gst(y)
        g = self._resolve_g(sid, y, speaker_embedding, g_adjust)

        # BERT モデルが FP16 でロードされている場合は特徴量も FP16 になるので、明示的に FP32 に変換することでエラーを防ぐ
        if use_fp16 is True:
            bert = bert.float()

        # Encoder は基本 FP32 で実行し、相対位置エンコーディング部分のみ FP16 化
        x, m_p, logs_p, x_mask = self.enc_p(
            x,
            x_lengths,
            tone,
            language,
            bert,
            style_vec,
            g=g,
            use_fp16=use_fp16,
        )

        # 精度クリティカルな部分 (SDP/DP, Flow) は常に FP32 で実行する

        # SDP (Stochastic Duration Predictor) / DP (Duration Predictor)
        duration_x = self.apply_duration_symbol_type_ids(x, duration_symbol_type_ids)
        logw = self.sdp(
            duration_x,
            x_mask,
            g=g,
            reverse=True,
            noise_scale=noise_scale_w,
        ) * (sdp_ratio) + self.dp(duration_x, x_mask, g=g) * (1 - sdp_ratio)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)

        # durations_frames_override が指定された場合、指定されたトークンのみ duration を上書きする
        if durations_frames_override is not None:
            override = durations_frames_override
            original_is_floating_point = override.is_floating_point()
            if override.dim() == 1:
                override = override.unsqueeze(0).unsqueeze(0)
            elif override.dim() == 2:
                override = override.unsqueeze(1)
            elif override.dim() != 3:
                raise ValueError(
                    "durations_frames_override must be 1D, 2D, or 3D tensor"
                )

            override = override.to(device=w_ceil.device, dtype=w_ceil.dtype)
            if override.shape != w_ceil.shape:
                raise ValueError(
                    "durations_frames_override shape mismatch. "
                    f"expected: {tuple(w_ceil.shape)}, actual: {tuple(override.shape)}"
                )

            if durations_frames_override_mask is not None:
                mask = durations_frames_override_mask
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                elif mask.dim() == 2:
                    mask = mask.unsqueeze(1)
                elif mask.dim() != 3:
                    raise ValueError(
                        "durations_frames_override_mask must be 1D, 2D, or 3D tensor"
                    )
                mask = mask.to(device=w_ceil.device, dtype=torch.bool)
                if mask.shape != w_ceil.shape:
                    raise ValueError(
                        "durations_frames_override_mask shape mismatch. "
                        f"expected: {tuple(w_ceil.shape)}, actual: {tuple(mask.shape)}"
                    )
            else:
                # mask が省略された場合、NaN 以外を上書き対象とみなす (float のみ)
                # それ以外の型では、0 より大きい値のみを上書き対象とみなす
                if original_is_floating_point is True:
                    mask = ~torch.isnan(override)
                else:
                    mask = override > 0

            if original_is_floating_point is True:
                invalid_value_mask = ~torch.isfinite(override)
                if torch.any(invalid_value_mask & mask):
                    raise ValueError("durations_frames_override must be finite")

            w_ceil = torch.where(mask, override, w_ceil)
            # 物理的に不正な値の混入を防ぐ
            w_ceil = torch.clamp_min(w_ceil, 0)

        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # Flow
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        return z, y_mask, g, attn, z_p, m_p, logs_p

    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        max_len: int | None = None,
        sdp_ratio: float = 0.0,
        y: torch.Tensor | None = None,
        use_fp16: bool = False,
        durations_frames_override: torch.Tensor | None = None,
        durations_frames_override_mask: torch.Tensor | None = None,
        speaker_embedding: torch.Tensor | None = None,
        g_adjust: torch.Tensor | None = None,
        duration_symbol_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        # Generator 実行前の共通処理
        z, y_mask, g, attn, z_p, m_p, logs_p = self.infer_input_feature(
            x,
            x_lengths,
            sid,
            tone,
            language,
            bert,
            style_vec,
            noise_scale,
            length_scale,
            noise_scale_w,
            sdp_ratio,
            y,
            use_fp16,
            durations_frames_override,
            durations_frames_override_mask,
            speaker_embedding,
            g_adjust,
            duration_symbol_type_ids,
        )

        # Generator (Decoder) のみ全体を FP16 / AMP (Automatic Mixed Precision) で実行
        if use_fp16 is True:
            # z テンソルを Generator の入力用に FP16 に変換
            z_input = (z * y_mask)[:, :, :max_len]
            # デバイスタイプを動的に取得
            device_type = z_input.device.type
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=True,
            ):
                # Generator への入力を FP16 に変換
                o = self.dec(z_input.half(), g=g.half())
        else:
            # FP16 を使わない場合は通常通り実行
            o = self.dec((z * y_mask)[:, :, :max_len], g=g)

        return (o, attn, y_mask, (z, z_p, m_p, logs_p))
