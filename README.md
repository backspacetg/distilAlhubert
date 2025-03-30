# DistilALHuBERT

This is an implementation of our paper *DistilALHuBERT: A Distilled Parameter Sharing Audio Representation Model*. This repository contains the pre-trained models and the codes. To use our model, you can either add our code to [S3PRL](https://github.com/s3prl/s3prl) as an additional "Upstream model" or use it separately. 

## Adding to S3PRL (Recommended)

Our model is implemented by S3RPL, a flexible toolbox for pre-trained speech models. S3PRL support adding customized pre-trained models (called Upstream models) and customized pre-training methods. 

To add our code into S3PRL, you should: 

* Make sure your S3PRL is installed in development mode. 

* copy `src/upstream/alhubert` to `s3prl/upstream/alhubert`.

* copy `src/upstream/hubert` to `s3prl/upstream/hubert` since we have changed some of the HuBERT implementation code in S3PRL to support multi-GPU training. Please remember to keep a copy of those codes!

* copy `src/pretrain/alhubert` to `s3prl/pretrain/alhubert`.

* add `from .upstream.alhubert.hubconf import *` to `s3prl/hub.py`. You will find many similar import statements there, so just append this one to the last one.

* copy `config/alhubert/*.yaml` to anywhere you like. e.g., `s3prl/pretrain/alhubert`.

We recommend adding our code to your S3PRL installation to evaluate our pre-trained models on all the downstream tasks.

## Using the code separately

To use our code without S3PRL, you should

* run `pip install -r requirement.txt` to install all the dependencies. 

## Extracting features

You can extract features from the pre-trained model by

```python
import torch
from src.upstream.alhubert.expert import UpstreamExpert
# when using in s3prl, you can use 
# from s3prl.upstream.alhubert.expert import UpstreamExpert
model_ckpt_path = "small.ckpt"
model = UpstreamExpert(model_ckpt_path)
data = [torch.randn(10000) for _ in range(2)] # 16KHz
states = model(data)
print(states["last_hidden_state"].shape) # torch.Tensor: hidden state of the last layer
print(len(states["hidden_states"])) # list[torch.Tensor] hidden states of each layer
# please note that if layer_norm_first=False (default), "hidden_states" will be the outputs of transformer layer 0,1,...11
# layer_norm_first=True (for HuBERT Large teachers), "hidden_states" will be the outputs of the CNN feature estractor and transformer layer 0,1,...10.
# in that case, the output of transformer layer 11 is in states["last_hidden_state"].
# This is because that the feature after layer norm is better for distillation.
```

## Pre-trained models

The pre-trained models can be downloaded at: 

| Model  | Link                                                                                               |
| ------ | -------------------------------------------------------------------------------------------------- |
| small  | [Google Drive](https://drive.google.com/file/d/1agjmHWhbAE_ZBGOHI7sy_9UP-XZk4V11/view?usp=sharing) |
| middle | [Google Drive](https://drive.google.com/file/d/1ummRt6_BxbCtJaqi-PnJ88UNaVglhYaQ/view?usp=sharing) |
| large  | [Google Drive](https://drive.google.com/file/d/1ZQY3I44qAZ59ZXicqx3aNX0ytSoYf24N/view?usp=sharing) |

## Pre-training

Take the small model (2*6=12 layers) for an example. 

### Step 1

Prepare the data according to [the instructions](https://github.com/s3prl/s3prl/blob/master/s3prl/pretrain/README.md#pre-training--upstream-models) in S3PRL. 

### Step 2

Edit  `s3prl/pretrain/alhubert/config_runner.yaml` to add the dataset path. 

```yaml
    libri_root: '/mnt/data/LibriSpeech/'
    # path to the librispeech dataset
    # contains folders like ‘train-clean-100’ or 'train-other-500'
    file_path: '/mnt/exp/len_for_bucket'
    # path to the audio length files generated in step 1. 


```

Edit `s3prl/pretrain/alhubert/config_model_l2.yaml` to add the path of the teacher model.  

```yaml
teacher:
  model: hubert_local
  path: "/path/to/teacher/model.pt"
```

We use the [pre-trained Hubert Base model](https://huggingface.co/s3prl/converted_ckpts/resolve/main/hubert_base_ls960.pt) provided by [hugging face](https://huggingface.co/). 

### Step 3:

Perform distillation. 

```bash
#!/bin/bash
export OMP_NUM_THREADS=1

set -ue

name=l2
expdir=exp

python run_pretrain.py \
    --upstream alhubert \
    --upstream_config "pretrain/alhubert/config_model_l2.yaml" \
    --config "pretrain/alhubert/config_runner.yaml" \
    --expname $name \
    --expdir $expdir/$name
```

## Fine-tuning

If you have added our code to S3PRL, you can follow [the official instructions](https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/docs/superb.md) to evaluate our model in all the downstream tasks. Our model is registered as `alhubert_local`. E.g., You can perform ASR fine-tuning by

```bash
#!/bin/bash
set -ue

export OMP_NUM_THREADS=1

name="asr"

python3 run_downstream.py \
    --config downstream/asr/config.yaml \
    --upstream alhubert_local \
    --upstream_feature_selection hidden_states \
    --downstream asr \
    --expname $name \
    --mode train \
    --upstream_ckpt "small.ckpt" \
    --expdir /mnt/exp/$name
```

We also add ASR fine-tuning code to this repository and you can use similar code to evaluate the pre-trained model on the ASR task without S3PRL. For other tasks, we still recommend using S3PRL's official implementations. 

## Reference Repositories

Most of the source code is based on [s3prl](https://github.com/s3prl/s3prl/) and [DistilHuBERT](https://github.com/s3prl/s3prl/tree/master/s3prl/upstream/distiller).


