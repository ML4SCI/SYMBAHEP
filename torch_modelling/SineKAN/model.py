import torch
import torch.nn.functional as F
import math
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import List, Union


def forward_step(i_n, grid_size, A, K, C):
    ratio = A * grid_size**(-K) + C
    i_n1 = ratio * i_n
    return i_n1


class SineKANLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device: Union[str, int] = 'cuda', grid_size=5, is_first=False, add_bias=True, norm_freq=True):
        super(SineKANLayer, self).__init__()
        self.grid_size = grid_size
        self.device = device
        self.is_first = is_first
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A, self.K, self.C = 0.9724108095811765, 0.9884401790754128, 0.999449553483052

        self.grid_norm_factor = (torch.arange(grid_size) + 1)
        self.grid_norm_factor = self.grid_norm_factor.reshape(1, 1, grid_size)

        if is_first:
            self.amplitudes = torch.nn.Parameter(torch.empty(
                output_dim, input_dim, 1).normal_(0, .4) / output_dim / self.grid_norm_factor)
        else:
            self.amplitudes = torch.nn.Parameter(torch.empty(
                output_dim, input_dim, 1).uniform_(-1, 1) / output_dim / self.grid_norm_factor)

        grid_phase = torch.arange(
            1, grid_size + 1).reshape(1, 1, 1, grid_size) / (grid_size + 1)
        self.input_phase = torch.linspace(
            0, math.pi, input_dim).reshape(1, 1, input_dim, 1).to(device)
        phase = grid_phase.to(device) + self.input_phase

        if norm_freq:
            self.freq = torch.nn.Parameter(torch.arange(
                1, grid_size + 1).float().reshape(1, 1, 1, grid_size) / (grid_size + 1)**(1 - is_first))
        else:
            self.freq = torch.nn.Parameter(torch.arange(
                1, grid_size + 1).float().reshape(1, 1, 1, grid_size))

        for i in range(1, self.grid_size):
            phase = forward_step(phase, i, self.A, self.K, self.C)
        # self.phase = torch.nn.Parameter(phase)
        self.register_buffer('phase', phase)

        if self.add_bias:
            self.bias = torch.nn.Parameter(
                torch.ones(1, output_dim) / output_dim)

    def forward(self, x):
        x_shape = x.shape
        output_shape = x_shape[0:-1] + (self.output_dim,)
        x = torch.reshape(x, (-1, self.input_dim))
        x_reshaped = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        s = torch.sin(x_reshaped * self.freq + self.phase)
        y = torch.einsum('ijkl,jkl->ij', s, self.amplitudes)
        if self.add_bias:
            y += self.bias
        y = torch.reshape(y, output_shape)
        return y


def apply_rotary_pos_emb(x, pos_emb):
    x_cos, x_sin = torch.split(pos_emb, x.shape[-1] // 2, dim=-1)
    x1_rot = (x[..., ::2] * x_cos) + (rotate_half(x[..., 1::2]) * x_sin)
    x2_rot = (x[..., 1::2] * x_cos) + (rotate_half(x[..., ::2]) * x_sin)
    x_rot = torch.cat([x1_rot, x2_rot], dim=-1)
    return x_rot


def rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class FeedForwardBlock(nn.Module):

    def __init__(self, features: int, ff_dims: List[int], grid_size: int = 8, device: Union[str, int] = 'cuda') -> None:
        super().__init__()
        self.ffn = torch.nn.ModuleList()
        in_size = features
        local_ff_dims = ff_dims.copy()
        local_ff_dims += [features]
        for i, d in enumerate(local_ff_dims):
            self.ffn.append(SineKANLayer(
                in_size, d, grid_size=grid_size, device=device, is_first=(i == 0)))
            in_size = d

    def forward(self, x):
        for f in self.ffn:
            x = f(x)
        return x


class MultiheadKANAttention(nn.Module):
    def __init__(self, d_model, num_heads, is_cross, rope, device: Union[str, int] = 'cuda'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope
        self.is_cross_attention = is_cross

        # Linear transformation depending on the type of attention
        if self.is_cross_attention:
            self.qkv_enc_linear = SineKANLayer(
                d_model * 2, d_model * 2, grid_size=8, device=device, is_first=True)
            self.qkv_dec_linear = SineKANLayer(
                d_model, d_model, grid_size=8, device=device, is_first=True)
        else:
            self.qkv_linear = SineKANLayer(
                d_model, d_model * 3, grid_size=8, device=device, is_first=True)

        self.out = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.out.weight)

    def _attention(self, queries, keys, mask):

        scores = torch.matmul(queries, keys.transpose(2, 3))

        scores = scores / (self.head_dim ** 0.5)
        scores = scores.to(torch.float32)

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)

        return attention

    def forward(self, q, mask=None, k=None, v=None):
        if self.is_cross_attention:
            assert k is not None and v is not None, "For cross-attention, k and v cannot be None."
            kv = torch.cat([k, v], dim=-1)
            batch_size, seq_length = kv.size()[:2]
            kv = self.qkv_enc_linear(kv)
            kv = kv.reshape(
                batch_size, seq_length, self.num_heads, 2 * self.head_dim)
            kv = kv.transpose(1, 2)
            keys, values = kv.chunk(2, dim=-1)

            batch_size, seq_length = q.size()[:2]
            q = self.qkv_dec_linear(q)
            q = q.reshape(
                batch_size, seq_length, self.num_heads, self.head_dim)
            queries = q.transpose(1, 2)

        else:
            x = q
            batch_size, seq_length = x.size()[:2]
            qkv = self.qkv_linear(x)
            qkv = qkv.reshape(
                batch_size, seq_length, self.num_heads, 3 * self.head_dim)
            qkv = qkv.transpose(1, 2)

            queries, keys, values = qkv.chunk(3, dim=-1)

        # Perform linear transformation
        q_rotation_matrix = self.rope(
            queries.size()[2]).to(queries.device)
        queries = apply_rotary_pos_emb(
            queries, q_rotation_matrix)
        k_rotation_matrix = self.rope(
            keys.size()[2]).to(keys.device)
        keys = apply_rotary_pos_emb(keys, k_rotation_matrix)

        attention_score = self._attention(queries, keys, mask=mask)

        context = torch.matmul(attention_score, values)
        context = context.transpose(1, 2)

        context = context.reshape(batch_size, seq_length, self.d_model)
        output = self.out(context)

        return output


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (100 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.register_buffer(
            'pos_enc', self._generate_positional_encoding(max_seq_len))

    def _generate_positional_encoding(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device,
                         dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        pos_enc = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return pos_enc

    def forward(self, seq_len):
        return self.pos_enc[:seq_len, :]


class TokenEmbedding(nn.Module):
    """
    Token embedding module for transformer architectures.

    Args:
        vocab_size (int): Size of the vocabulary.
        emb_size (int): The embedding size.
    """

    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        nn.init.normal_(self.embedding.weight, mean=0, std=1)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class ResidualConnection(nn.Module):

    def __init__(self, features: int) -> None:
        super().__init__()
        self.norm = RMSNorm(features)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiheadKANAttention, feed_forward_block: FeedForwardBlock) -> None:

        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features) for _ in range(1)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, src_mask))
        x = self.feed_forward_block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiheadKANAttention, cross_attention_block: MultiheadKANAttention,
                 feed_forward_block: FeedForwardBlock) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features) for _ in range(2)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(
            x, src_mask, encoder_output, encoder_output))
        x = self.feed_forward_block(x)
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: TokenEmbedding, tgt_embed: TokenEmbedding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_kanformer(src_vocab_size: int, tgt_vocab_size: int, max_seq_len: int, device: Union[str, int] = 'cuda',
                      d_model: int = 512, num_layers: int = 6, num_heads: int = 8, ff_dims: List[int] = [256]) -> Transformer:

    # Create the embedding layers
    src_embed = TokenEmbedding(src_vocab_size, d_model)
    tgt_embed = TokenEmbedding(tgt_vocab_size, d_model)

    assert d_model % num_heads == 0, "d_model is not divisible by number of heads"

    head_dim = d_model // num_heads
    assert head_dim % 2 == 0, "head_dim must be even for ROPE"

    rope = RotaryPositionalEmbedding(head_dim, max_seq_len)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(num_layers):
        encoder_self_attention_block = MultiheadKANAttention(
            d_model, num_heads, False, rope, device=device)
        feed_forward_block = FeedForwardBlock(d_model, ff_dims)
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, feed_forward_block)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(num_layers):
        decoder_self_attention_block = MultiheadKANAttention(
            d_model, num_heads, False, rope, device=device)
        decoder_cross_attention_block = MultiheadKANAttention(
            d_model, num_heads, True, rope, device=device)
        feed_forward_block = FeedForwardBlock(d_model, ff_dims)
        decoder_block = DecoderBlock(
            d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, projection_layer)

    return transformer
