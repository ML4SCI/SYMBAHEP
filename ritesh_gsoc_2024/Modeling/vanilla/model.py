import torch
import torch.nn as nn
from torch.nn import Transformer
from torch import Tensor
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for transformer architectures.

    Args:
        emb_size (int): The embedding size.
        dropout (float): Dropout rate.
        maxlen (int, optional): Maximum sequence length. Defaults to 5000.
    """

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2)
                        * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


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
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Model(nn.Module):
    """
    Transformer-based model for sequence-to-sequence tasks.

    Args:
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        emb_size (int): Size of the embedding.
        nhead (int): Number of attention heads.
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        dim_feedforward (int, optional): Dimension of the feedforward network. Defaults to 512.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """

    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Model, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=False,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        """
        Forward pass of the model.

        Args:
            src (Tensor): Source input.
            trg (Tensor): Target input.
            src_mask (Tensor): Mask for source input.
            tgt_mask (Tensor): Mask for target input.
            src_padding_mask (Tensor): Padding mask for source input.
            tgt_padding_mask (Tensor): Padding mask for target input.
            memory_key_padding_mask (Tensor): Padding mask for memory.

        Returns:
            Tensor: Output tensor.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, memory_key_padding_mask
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor, src_pad_mask: Tensor):
        """
        Encode the source input.

        Args:
            src (Tensor): Source input.
            src_mask (Tensor): Mask for source input.

        Returns:
            Tensor: Encoded tensor.
        """
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask, src_pad_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, memory_mask: Tensor, tgt_pad_mask: Tensor, memory_pad_mask: Tensor):
        """
        Decode the target input.

        Args:
            tgt (Tensor): Target input.
            memory (Tensor): Memory tensor.
            tgt_mask (Tensor): Mask for target input.

        Returns:
            Tensor: Decoded tensor.
        """
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask, memory_mask, tgt_pad_mask, memory_pad_mask)

