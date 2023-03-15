import tempfile
from pathlib import Path
from typing import List

import torch

from .hubert_model import (
    HubertConfig,
    HubertModel,
    HubertPretrainingConfig,
)
from ..utils import merge_with_parent

def load_converted_model(ckpt: str):
    ckpt_state = torch.load(ckpt, map_location="cpu")

    for required_key in [
        "task_cfg",
        "model_cfg",
        "model_weight",
        "dictionaries_symbols",
    ]:
        if required_key not in ckpt_state:
            raise ValueError(
                f"{ckpt} is not a valid checkpoint since the required key: {required_key} is missing"
            )

    task_cfg = merge_with_parent(HubertPretrainingConfig, ckpt_state["task_cfg"])
    model_cfg = merge_with_parent(HubertConfig, ckpt_state["model_cfg"])
    model = HubertModel(model_cfg, task_cfg, ckpt_state["dictionaries_symbols"])

    model.load_state_dict(ckpt_state["model_weight"])
    return model, task_cfg
