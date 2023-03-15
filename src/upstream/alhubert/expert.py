"""
    Upstream expert for Distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""
import sys
import logging

import numpy as np
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence

from ... import optimizers

from ..interfaces import UpstreamBase
from .model import AlhubertModel
from .config import AlhubertConfig

class UpstreamExpert(UpstreamBase):
    """
    The Distiller wrapper
    """

    def __init__(self, ckpt, model_config=None, **kwargs):
        super().__init__(**kwargs)
        if model_config is not None:
            self.config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
        else:
            # Since some old checkpoints contained pickled scheduler which needs 'optimizers'
            # module which is now moved into s3prl package.
            original_optimizer = sys.modules.get("optimizers")
            sys.modules["optimizers"] = optimizers

            all_states = torch.load(ckpt, map_location="cpu")
            self.config = all_states["Config"]

            del sys.modules["optimizers"]
            if original_optimizer is not None:
                sys.modules["optimizers"] = original_optimizer
            
            pretrained_modules = ["label_embs_concat", "final_proj.weight", "final_proj.bias"]
            new_dict = {}
            for key, value in all_states["alhubert"].items():
                if key not in pretrained_modules:
                    new_dict[key] = value
                else:
                    logging.info(f"skip {key}")
        
        self.model_config = AlhubertConfig(self.config["alhubert"])
        self.model = AlhubertModel(self.model_config)
        self.model.load_state_dict(new_dict)
        del all_states

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        wave_len = [len(wave) for wave in wavs]
        wave_len = torch.LongTensor(wave_len).to(device)
        wave_inputs = pad_sequence(wavs, batch_first=True)
        pad_mask = torch.ones(wave_inputs.shape).to(device)
        for idx in range(wave_inputs.shape[0]):
            pad_mask[idx, wave_len[idx] :] = 0

        ret_dict = self.model(wave_inputs, pad_mask=pad_mask, mask=False, features_only=True)
        hidden_states = [l[0] for l in ret_dict["layer_hiddens"]]

        states = {
            "last_hidden_state": ret_dict["last_hidden"],
            "hidden_states": hidden_states,
            "pad_mask": ret_dict["pad_mask"],
            "paper": ret_dict["last_hidden"],
        }
        return states
