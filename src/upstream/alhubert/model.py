import logging
import torch
from torch import nn

from typing import Any, List, Optional, Tuple
from .module import (ConvFeatureExtractionModel, GradMultiply,
                     TransformerEncoder, compute_mask_indices, LayerNorm)
from .config import AlhubertConfig

class AlhubertModel(nn.Module):
    """
    Distiller Model
    """

    def __init__(self, config: AlhubertConfig, dictionaries: Optional[List[Any]]=[]):
        super().__init__()

        self.config = config
        self.dictionaries = dictionaries

        self.conv_layers = eval(config.extractor_conv_feature_layers)
        feat_emb_dim = self.conv_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            self.conv_layers,
            dropout=0.0,
            mode=config.extractor_mode,
            conv_bias=config.conv_bias,
        )
        self.feature_grad_mult = config.feature_grad_mult

        self.post_extract_proj = (
            nn.Linear(feat_emb_dim, config.encoder_embed_dim)
            if feat_emb_dim != config.encoder_embed_dim
            else None
        )

        if config.encoder_layers > 0:
            self.encoder = TransformerEncoder(config)
        else:
            self.encoder = nn.GELU()
        self.layer_norm = LayerNorm(feat_emb_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(config.encoder_embed_dim).uniform_()
        )

        if len(self.dictionaries) == 0:
            logging.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            final_dim = config.final_dim if config.final_dim > 0 else config.encoder_embed_dim
            self.final_proj = nn.Linear(config.encoder_embed_dim, final_dim)
            # print(self.dictionaries[0].symbols)
            self.num_classes = [len(d) for d in self.dictionaries]
            
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat) #TODO 初始化

    def apply_mask(self, x: torch.Tensor, padding_mask: torch.Tensor, target_list: list, generated_mask: torch.Tensor=None):
        B, T, C = x.shape
        if self.config.mask_prob > 0:
            if generated_mask is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.config.mask_prob,
                    self.config.mask_length,
                    self.config.mask_selection,
                    self.config.mask_other,
                    min_masks=2,
                    no_overlap=self.config.no_mask_overlap,
                    min_space=self.config.mask_min_space,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            else:
                mask_indices = generated_mask
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.config.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.config.mask_channel_prob,
                self.config.mask_channel_length,
                self.config.mask_channel_selection,
                self.config.mask_channel_other,
                no_overlap=self.config.no_mask_channel_overlap,
                min_space=self.config.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices
    
    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(1)
        targ_tsz = min([t.size(1) for t in target_list])
        # print(feat_tsz, targ_tsz)
        # 对特征进行截断
        if feat_tsz > targ_tsz:
            features = features[:, :feat_tsz, :]
        # 对标签也进行截断
        target_inds = torch.arange(feat_tsz).long()
        target_list = [t[:, target_inds] for t in target_list]
        return features, target_list

    def forward_feature(self, wave: torch.Tensor, pad_mask: torch.Tensor):
        """Forward feature extractor"""

        if self.feature_grad_mult > 0:
            feat = self.feature_extractor(wave)
            if self.feature_grad_mult != 1.0:
                feat = GradMultiply.apply(feat, self.feature_grad_mult)
        else:
            with torch.no_grad():
                feat = self.feature_extractor(wave)

        feat = feat.transpose(1, 2)  # B x T x D
        pad_mask = self.cal_pad_mask(pad_mask, feat.shape[1])

        return feat, pad_mask

    def forward(
        self, 
        wave: torch.Tensor, 
        target_list: Optional[List[torch.Tensor]] = None,
        pad_mask: Optional[torch.Tensor] = None, 
        mask: bool=False,
        generated_mask: torch.Tensor = None,
        features_only: bool = False
        ):
        # 注意这里和Hubert模型不一样 pad_mask 1是保留 0是Mask
        # feat的形状也不一样 是BTD而不是BDT
        feat, pad_mask = self.forward_feature(wave, pad_mask)
        # print(feat.shape)
        feat_pen = feat.float().pow(2).mean()
        feat = self.layer_norm(feat)
        
        if target_list is not None:
            feat, target_list = self.forward_targets(feat, target_list)

        if self.post_extract_proj is not None:
            feat = self.post_extract_proj(feat)
        # feat: B x T x D
        
        if mask:
            feat, mask_indices = self.apply_mask(feat, ~pad_mask.bool(), target_list, generated_mask=generated_mask)
        else:
            mask_indices = None

        layer_hiddens = []
        if self.config.encoder_layers > 0:
            x, layer_hiddens = self.encoder(
                feat, ~pad_mask.bool(), get_hidden=True
            )
        else:
            x = self.encoder(feat)
        
        ret_dict = {
                "feat_pen": feat_pen,
                "last_hidden": x, 
                "layer_hiddens": layer_hiddens,
                "pad_mask": pad_mask
            }

        if features_only:
            return ret_dict
        
        # label_embs_list：随机初始化的一个embedding [Num_classes, 256]
        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
        # 分别计算mask和非mask部分的损失
        if not self.config.skip_masked: 
            masked_indices = torch.logical_and(pad_mask, mask_indices) # 没有被padding掉 并且被Mask了的
            # print(target_list[0].shape)
            # 两个Mask都是False表示留
            # 布尔索引：在索引维度会被拉平
            # proj_x_m: [all the unmasked frames, 256]
            proj_x_m = self.final_proj(x[masked_indices])
            proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                self.compute_pred(x_m, target[masked_indices], label_embs_list[i])
                for i, (x_m, target) in enumerate(zip(proj_x_m_list, target_list))
            ]
        else:
            logit_m_list = []

        if not self.config.skip_nomask:
            nomask_indices = torch.logical_and(pad_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            proj_x_u_list = [proj_x_u for _ in range(len(target_list))]
            logit_u_list = [
                self.compute_pred(x_u, target[nomask_indices], label_embs_list[i])
                for i, (x_u, target) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = []

        target_m_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logit_m_list]
        target_u_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logit_u_list]
        
        ret_dict["logit_m_list"] = logit_m_list
        ret_dict["logit_u_list"] = logit_u_list
        ret_dict["target_m_list"] = target_m_list
        ret_dict["target_u_list"] = target_u_list
        # ret_dict["mask_indices"] = mask_indices
        
        return ret_dict

    def cal_pad_mask(self, pad_mask, max_len):
        """Calculates pad mask after conv."""
        pad_len = (pad_mask > 0).sum(1).long()
        for _, k_size, s_size in self.conv_layers:
            pad_len = torch.div((pad_len - k_size), s_size, rounding_mode="trunc") + 1

        new_pad_mask = torch.ones(
            (pad_mask.shape[0], max_len), dtype=pad_mask.dtype, device=pad_mask.device
        )

        for idx in range(pad_len.shape[0]):
            new_pad_mask[idx, pad_len[idx] :] = 0

        return new_pad_mask
    
    def compute_pred(self, proj_x, target, label_embs):
        # compute logits for the i-th label set
        # y：正类的embedding [num_sample, 256]
        # negs: [Num_classes, num_sample, 256]
        y = torch.index_select(label_embs, 0, target.long())
        # expand: 传入-1 表示不修改
        negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
        # proj_x: (S, D)
        # y: (S, D)
        # negs: (Neg, S, D)
        return self.compute_nce(proj_x, y, negs)
    
    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1) # always true somewhere in this code
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0) # 正类放在第一位

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x) # 计算余弦相似度
        logits /= self.config.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits.float() # from hubert_model.py/get_logits