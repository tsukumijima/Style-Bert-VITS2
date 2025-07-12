import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from style_bert_vits2.models import commons


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


@torch.jit.script  # type: ignore
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: list[int]
) -> torch.Tensor:
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
        isflow: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        # if isflow:
        #  cond_layer = torch.nn.Conv1d(256, 2*hidden_channels*n_layers, 1)
        #  self.cond_pre = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, 1)
        #  self.cond_layer = weight_norm(cond_layer, name='weight')
        #  self.gin_channels = 256
        self.cond_layer_idx = self.n_layers
        if "gin_channels" in kwargs:
            self.gin_channels = kwargs["gin_channels"]
            if self.gin_channels != 0:
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)
                # vits2 says 3rd block, so idx is 2 by default
                self.cond_layer_idx = (
                    kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2
                )
                # logger.debug(self.gin_channels, self.cond_layer_idx)
                assert self.cond_layer_idx < self.n_layers, (
                    "cond_layer_idx should be less than n_layers"
                )
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        use_fp16: bool = False,
    ) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            if i == self.cond_layer_idx and g is not None:
                g = self.spk_emb_linear(g.transpose(1, 2))
                assert g is not None
                g = g.transpose(1, 2)
                x = x + g
                x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask, use_fp16=use_fp16)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        proximal_bias: bool = False,
        proximal_init: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.encdec_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        h: torch.Tensor,
        h_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: decoder input
        h: encoder output
        """
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(
            device=x.device, dtype=x.dtype
        )
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int | None = None,
        heads_share: bool = True,
        block_length: int | None = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ) -> None:
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                assert self.conv_k.bias is not None
                assert self.conv_q.bias is not None
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_fp16: bool = False,
    ) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask, use_fp16=use_fp16)

        x = self.conv_o(x)
        return x

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_fp16: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention with optional SDPA acceleration.

        Args:
            query, key, value: Conv1d で投影済みのテンソル ([b, d, t])
            mask: パディング等のマスク。0 の位置は‐inf にマッピングされる。
            use_fp16: True のとき FP16 + SDPA を使用（対応 GPU のみ）

        Returns:
            out: [b, d, t_t]
            attn: アテンション重み（デバッグ用）。SDPA パスでは空 Tensor が返る。
        """
        # 入力テンソル形状
        # key/val の長さ=t_s, query の長さ=t_t
        b, d, t_s, t_t = (*key.size(), query.size(2))
        d_k = d // self.n_heads

        # FP16 推論時のみ、SDPA を使いメモリ使用量の削減を図る
        if use_fp16 is True:
            # 入力は既に Conv1d 済みなので、ここでは reshape のみ行う
            q = query.view(b, self.n_heads, d_k, t_t).transpose(2, 3)  # [b,h,t_t,d_k]
            k = key.view(b, self.n_heads, d_k, t_s).transpose(2, 3)  # [b,h,t_s,d_k]
            v = value.view(b, self.n_heads, d_k, t_s).transpose(2, 3)  # [b,h,t_s,d_k]

            attn_bias: torch.Tensor | None = None  # SDPA に渡すバイアス/マスク

            # Self-attention 固有のバイアス類 (t_s == t_t の場合のみ適用)
            if t_s == t_t:
                if self.window_size is not None:
                    device_type = query.device.type  # デバイスタイプを動的に取得
                    # メモリ削減のため、相対位置計算のみ FP16 で実行
                    # 全体を FP16 化すると後続のモジュールの精度が保てない
                    with torch.autocast(device_type, dtype=torch.float16, enabled=True):
                        rel_k = self._get_relative_embeddings(
                            self.emb_rel_k, t_s
                        )  # [h|1, 2l-1, d_k]
                        rel_score = self._matmul_with_relative_keys(q, rel_k)
                        rel_score = self._relative_position_to_absolute_position(
                            rel_score
                        )
                        attn_bias = rel_score

                if self.proximal_bias:
                    bias = self._attention_bias_proximal(t_t).to(
                        device=q.device, dtype=q.dtype
                    )
                    bias = (
                        bias.unsqueeze(0).unsqueeze(0).expand(b, self.n_heads, t_t, t_t)
                    )
                    attn_bias = bias if attn_bias is None else attn_bias + bias

                if self.block_length is not None:
                    blk = self._make_local_attention_block_mask(t_t, self.block_length)
                    blk = (
                        blk.to(device=q.device, dtype=q.dtype)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand(b, self.n_heads, t_t, t_t)
                    )
                    attn_bias = blk if attn_bias is None else attn_bias + blk

            # Pad / causal マスク
            if mask is not None:
                # 任意の多次元 [b, 1, 1, t_t, t_s] などを [b, t_t, t_s] に正規化
                while mask.dim() > 3:
                    mask = mask.squeeze(1)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(2) * mask.unsqueeze(1)
                mask = mask.unsqueeze(1).expand(b, self.n_heads, t_t, t_s)
                mask = mask.masked_fill(mask == 0, float("-inf"))
                attn_bias = mask if attn_bias is None else attn_bias + mask

            # relative value を後段で加える必要があるか
            need_rel_v = self.window_size is not None and t_s == t_t

            # SDPA 関数を実行
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.p_dropout if self.training else 0.0,
            )  # [b, h, t_t, d_k]

            # 相対位置 value の反映（必要なら追加計算）
            if need_rel_v:
                # 1. attention scores を再計算（softmax 事前値）
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                    d_k
                )  # [b, h, l, l]
                if attn_bias is not None:
                    scores = scores + attn_bias
                p_attn = F.softmax(scores, dim=-1)

                # 2. relative value を加算
                rel_w = self._absolute_position_to_relative_position(p_attn)
                rel_v = self._get_relative_embeddings(self.emb_rel_v, t_s)
                out = out + self._matmul_with_relative_values(rel_w, rel_v)

            out = out.transpose(2, 3).contiguous().view(b, d, t_t)
            out = out.to(query.dtype)  # Restore original dtype to match conv_o bias
            return self.drop(out), torch.empty(0, device=out.device, dtype=out.dtype)

        # 明示的に FP16 推論が指定されている時以外は、実績のある従来の実装を使用する
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert t_s == t_t, (
                "Relative attention is only available for self-attention."
            )
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, (
                    "Local attention is only available for self-attention."
                )
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = (
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]

        return output, p_attn

    def _matmul_with_relative_values(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(
        self, relative_embeddings: torch.Tensor, length: int
    ) -> torch.Tensor:
        assert self.window_size is not None
        2 * self.window_size + 1  # type: ignore
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # # Concat columns of pad to shift from relative to absolute indexing.
        # x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # # Concat extra elements so to add up to shape (len+1, 2*len-1).
        # x_flat = x.view([batch, heads, length * 2 * length])
        # x_flat = F.pad(
        #     x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        # )

        # # Reshape and slice out the padded elements.
        # x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
        #     :, :, :length, length - 1 :
        # ]

        # --- new implementation (for memory efficiency) ---

        # Generate row and column indices using broadcasting
        row_indices = (
            torch.arange(length, device=x.device).unsqueeze(1).expand(length, length)
        )  # [length, length]
        col_indices = row_indices + (
            length - 1
        )  # Shift to match relative positioning: [length, length]

        # Flatten the indices for gathering
        flat_indices = col_indices.view(-1)  # [length * length]

        # Flatten x for gathering: [batch, heads, length * (2 * length - 1)]
        x_flat = x.view(batch, heads, -1)

        # Gather the required elements directly without padding or large intermediates
        x_gathered = torch.gather(
            x_flat, dim=2, index=flat_indices.expand(batch, heads, -1)
        )

        # Reshape back to [batch, heads, length, length]
        x_final = x_gathered.view(batch, heads, length, length)

        return x_final

    def _absolute_position_to_relative_position(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # pad along column
        x = F.pad(
            x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length: int) -> torch.Tensor:
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)

    def _make_local_attention_block_mask(
        self, sz: int, right: int, left: int = -1
    ) -> torch.Tensor:
        """Prevent local attention from paying attention further than `sz` tokens away.
        Args:
            sz: the full sequence length
            right: number of visible tokens to the left
            left: number of visible tokens to the right, -1 means unlimited
        Returns:
            a [sz, sz] mask with -inf where attention is not allowed and 0 where it is
        """
        mask = torch.ones((sz, sz))
        mask = torch.triu(mask, diagonal=1 + left)
        mask = torch.tril(mask, diagonal=-1 - right)
        # NaN を避けるため log ではなく -inf を直接埋め込む
        mask = mask.masked_fill(mask == 0, float("-inf"))
        return mask


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: str | None = None,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x
