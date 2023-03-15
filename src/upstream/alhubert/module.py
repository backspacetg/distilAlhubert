"""
    Distiller Modules
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..wav2vec2.wav2vec2_model import (ConvFeatureExtractionModel,
                                                    GradMultiply,
                                                    MultiheadAttention,
                                                    SamePad, get_activation_fn, compute_mask_indices, LayerNorm)
from .config import AlhubertConfig


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)

        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        input_after_layer_norm = None
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            input_after_layer_norm = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False if self.layer_norm_first else need_weights,
            attn_mask=self_attn_mask,
        )
        x = self.dropout1(x)
        x = residual + x
        if not self.layer_norm_first:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.layer_norm_first:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = residual + x
        if not self.layer_norm_first:
            x =  self.final_layer_norm(x)

        return x, attn, input_after_layer_norm


class TransformerEncoder(nn.Module):
    def __init__(self, args:AlhubertConfig):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
                for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop
        self.repeat_time = args.repeat_time
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, attn_mask=None, get_hidden=False):
        x, layer_results = self.extract_features(
            x, padding_mask, attn_mask, get_hidden=get_hidden
        )

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, attn_mask=None, get_hidden=False):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        
        for i in range(self.repeat_time):
            for j, layer in enumerate(self.layers):
                dropout_probability = np.random.random()
                if not self.training or (dropout_probability > self.layerdrop):
                    x, z, input_after_layer_norm = layer(
                        x,
                        self_attn_padding_mask=padding_mask,
                        need_weights=False, # change it to True to get weights
                        self_attn_mask=attn_mask,
                    )
                    if get_hidden:
                        layer_results.append((x.transpose(0, 1), input_after_layer_norm.transpose(0, 1) if input_after_layer_norm is not None else None))
                # print(i, j, self.layerdrop)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results
