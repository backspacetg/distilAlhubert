from pathlib import Path

import torch

from ..utils import merge_with_parent
from .wav2vec2_model import (
    AudioPretrainingConfig,
    Wav2Vec2Config,
    Wav2Vec2Model,
)

def load_converted_model(ckpt: str):
    ckpt_state = torch.load(ckpt, map_location="cpu")

    for required_key in ["task_cfg", "model_cfg", "model_weight"]:
        if required_key not in ckpt_state:
            raise ValueError(
                f"{ckpt} is not a valid checkpoint since the required key: {required_key} is missing"
            )

    task_cfg = merge_with_parent(AudioPretrainingConfig, ckpt_state["task_cfg"])
    model_cfg = merge_with_parent(Wav2Vec2Config, ckpt_state["model_cfg"])
    # print(model_cfg)
    model = Wav2Vec2Model(model_cfg)
    model.load_state_dict(ckpt_state["model_weight"])
    return model, task_cfg
