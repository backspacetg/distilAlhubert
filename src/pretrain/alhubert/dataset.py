"""
    Dataset for distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import os
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import layer_norm
import torchaudio
from ..bucket_dataset import WaveDataset


class OnlineWaveDataset(WaveDataset):
    """Online waveform dataset"""

    def __init__(
        self,
        task_config,
        bucket_size,
        file_path,
        sets,
        normalize, 
        max_timestep=0,
        libri_root=None,
        target_level=-25,
        **kwargs
    ):
        super().__init__(
            task_config,
            bucket_size,
            file_path,
            sets,
            max_timestep,
            libri_root,
            **kwargs
        )
        self.target_level = target_level
        self.normalize = normalize
        print(f"[Dataset] - normalize: {self.normalize}")

    def _load_feat(self, feat_path):
        if self.libri_root is None:
            return torch.FloatTensor(np.load(os.path.join(self.root, feat_path)))
        wav, _ = torchaudio.load(os.path.join(self.libri_root, feat_path))
        return wav.squeeze()  # (seq_len)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch_unnorm = [self._sample(self._load_feat(x_file)) for x_file in self.X[index]]
        if self.normalize:
            x_batch = [layer_norm(x, x.shape) for x in x_batch_unnorm]
        else:
            x_batch = x_batch_unnorm
        x_lens = [len(x) for x in x_batch]
        x_lens = torch.LongTensor(x_lens)
        x_pad_batch = pad_sequence(x_batch, batch_first=True)

        pad_mask = torch.ones(x_pad_batch.shape)  # (batch_size, seq_len)
        # zero vectors for padding dimension
        for idx in range(x_pad_batch.shape[0]):
            pad_mask[idx, x_lens[idx] :] = 0

        return [x_pad_batch, x_lens, pad_mask]
    
    def collate_fn(self, items):
        items = items[0]  # hack bucketing
        assert (
            len(items) == 3
        ), "__getitem__ should return (wave_input, wave_orig, wave_len, pad_mask, labels)"
        return items
