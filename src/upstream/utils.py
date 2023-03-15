import argparse
import logging
from copy import deepcopy
from dataclasses import dataclass, is_dataclass

import torch

from ..util.pseudo_data import get_pseudo_wavs

logger = logging.getLogger(__name__)

def merge_with_parent(dc: dataclass, cfg: dict):

    assert is_dataclass(dc)
    assert type(cfg) == dict
    cfg = deepcopy(cfg)

    def fix_cfg(cfg):
        target_keys = set(dc.__dataclass_fields__.keys())
        for k in list(cfg.keys()):
            if k not in target_keys:
                del cfg[k]

    fix_cfg(cfg)
    assert len(cfg) > 0
    return dc(**cfg)


def extract_hidden_states(model):
    model.eval()
    with torch.no_grad():
        return model(get_pseudo_wavs())["hidden_states"]


def are_same_models(model1, model2):
    hs1 = extract_hidden_states(model1)
    hs2 = extract_hidden_states(model2)
    for h1, h2 in zip(hs1, hs2):
        assert torch.allclose(h1, h2)


def models_all_close(*models):
    assert len(models) > 1
    for model in models[1:]:
        are_same_models(models[0], model)
