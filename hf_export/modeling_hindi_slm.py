import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from .configuration_hindi_slm import HindiSLMConfig


def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]

    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos

    return torch.cat((y1, y2), dim=-1)


class Linear(nn.Linear):
    def forward(self, x):
        # Match nanochat's explicit mixed-precision behavior:
        # FP32 master weights are cast to the activation dtype for matmul.
        return F.linear(x, self.weight.to(dtype=x.dtype))


class HindiSLMAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        if (
            self.n_kv_head > self.n_head
            or self.n_head % self.n_kv_head != 0
        ):
            raise ValueError("Invalid GQA configuration")

        self.c_q = Linear(
            self.n_embd,
            self.n_head * self.head_dim,
            bias=False,
        )

        self.c_k = Linear(
            self.n_embd,
            self.n_kv_head * self.head_dim,
            bias=False,
        )

        self.c_v = Linear(
            self.n_embd,
            self.n_kv_head * self.head_dim,
            bias=False,
        )

        self.c_proj = Linear(
            self.n_embd,
            self.n_embd,
            bias=False,
        )

        self.ve_gate_channels = 12

        self.ve_gate = (
            Linear(
                self.ve_gate_channels,
                self.n_kv_head,
                bias=False,
            )
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(self, x, ve, cos_sin):
        B, T, _ = x.shape

        q = self.c_q(x).view(
            B, T, self.n_head, self.head_dim
        )

        k = self.c_k(x).view(
            B, T, self.n_kv_head, self.head_dim
        )

        v = self.c_v(x).view(
            B, T, self.n_kv_head, self.head_dim
        )

        if ve is not None:
            ve = ve.view(
                B, T, self.n_kv_head, self.head_dim
            )

            gate = 3 * torch.sigmoid(
                self.ve_gate(
                    x[..., :self.ve_gate_channels]
                )
            )

            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = rms_norm(q)
        k = rms_norm(k)

        # Match native nanochat sharpening.
        q = q * 1.2
        k = k * 1.2

        # Native tensors are [B, T, H, D].
        # torch SDPA expects [B, H, T, D].
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Current locked checkpoint uses n_head == n_kv_head == 18.
        # Keep GQA support for completeness.
        if self.n_kv_head != self.n_head:
            repeats = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, self.n_embd)

        return self.c_proj(y)


class HindiSLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=False,
        )

        self.c_proj = Linear(
            4 * config.n_embd,
            config.n_embd,
            bias=False,
        )

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class HindiSLMBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        self.attn = HindiSLMAttention(
            config,
            layer_idx,
        )

        self.mlp = HindiSLMMLP(config)

    def forward(self, x, ve, cos_sin):
        x = x + self.attn(
            rms_norm(x),
            ve,
            cos_sin,
        )

        x = x + self.mlp(
            rms_norm(x)
        )

        return x


class HindiSLMPreTrainedModel(PreTrainedModel):
    config_class = HindiSLMConfig
    base_model_prefix = "hindi_slm"

    supports_gradient_checkpointing = False
    _supports_cache_class = False
    _no_split_modules = ["HindiSLMBlock"]


class HindiSLMForCausalLM(
    HindiSLMPreTrainedModel,
    GenerationMixin,
):
    def __init__(self, config):
        super().__init__(config)

        padded_vocab_size = (
            (config.vocab_size + 63) // 64
        ) * 64

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(
                padded_vocab_size,
                config.n_embd,
                dtype=torch.bfloat16,
            ),
            "h": nn.ModuleList([
                HindiSLMBlock(config, i)
                for i in range(config.n_layer)
            ]),
        })

        self.lm_head = Linear(
            config.n_embd,
            padded_vocab_size,
            bias=False,
        )

        self.resid_lambdas = nn.Parameter(
            torch.ones(config.n_layer)
        )

        self.x0_lambdas = nn.Parameter(
            torch.zeros(config.n_layer)
        )

        self.smear_gate = Linear(
            24,
            1,
            bias=False,
        )

        self.smear_lambda = nn.Parameter(
            torch.zeros(1)
        )

        self.backout_lambda = nn.Parameter(
            torch.zeros(1)
        )

        kv_dim = (
            config.n_kv_head
            * (config.n_embd // config.n_head)
        )

        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(
                padded_vocab_size,
                kv_dim,
                dtype=torch.bfloat16,
            )
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        })

        self.rotary_seq_len = (
            config.sequence_len * 10
        )

        self.post_init()

    def _compute_dtype(self):
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        return mapping[self.config.compute_dtype]

    def _rotary(self, T, device):
        head_dim = (
            self.config.n_embd
            // self.config.n_head
        )

        channel_range = torch.arange(
            0,
            head_dim,
            2,
            dtype=torch.float32,
            device=device,
        )

        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (channel_range / head_dim)
        )

        positions = torch.arange(
            T,
            dtype=torch.float32,
            device=device,
        )

        freqs = torch.outer(
            positions,
            inv_freq,
        )

        dtype = self._compute_dtype()

        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)

        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        return cos, sin

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_inputs_for_generation(
        self,
        input_ids,
        **kwargs,
    ):
        # Faithful simple implementation: no KV cache.
        return {
            "input_ids": input_ids,
            "use_cache": False,
        }

    def forward(
        self,
        input_ids=None,
        labels=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required")

        B, T = input_ids.shape

        if T > self.rotary_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds rotary cache "
                f"limit {self.rotary_seq_len}"
            )

        dtype = self._compute_dtype()

        x = self.transformer.wte(input_ids)
        x = x.to(dtype)
        x = rms_norm(x)

        # Native smear operation.
        if T > 1:
            gate = (
                self.smear_lambda.to(x.dtype)
                * torch.sigmoid(
                    self.smear_gate(
                        x[:, 1:, :24]
                    )
                )
            )

            x = torch.cat(
                [
                    x[:, :1],
                    x[:, 1:]
                    + gate * x[:, :-1],
                ],
                dim=1,
            )

        x0 = x

        cos_sin = self._rotary(
            T,
            input_ids.device,
        )

        backout_layer = (
            self.config.n_layer // 2
        )

        x_backout = None

        for i, block in enumerate(
            self.transformer.h
        ):
            x = (
                self.resid_lambdas[i].to(x.dtype)
                * x
                + self.x0_lambdas[i].to(x.dtype)
                * x0
            )

            if str(i) in self.value_embeds:
                ve = self.value_embeds[
                    str(i)
                ](input_ids).to(x.dtype)
            else:
                ve = None

            x = block(
                x,
                ve,
                cos_sin,
            )

            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = (
                x
                - self.backout_lambda.to(x.dtype)
                * x_backout
            )

        x = rms_norm(x)

        logits = self.lm_head(x)
        logits = logits[
            ...,
            :self.config.vocab_size,
        ]

        logits = logits.float()

        softcap = self.config.logit_softcap

        logits = (
            softcap
            * torch.tanh(
                logits / softcap
            )
        )

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                labels.reshape(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,)
            return (
                ((loss,) + output)
                if loss is not None
                else output
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )
