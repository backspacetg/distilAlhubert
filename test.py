import torch
from src.upstream.alhubert.expert import UpstreamExpert
# when using in s3prl, you can use 
# from s3prl.upstream.alhubert.expert import UpstreamExpert
model_ckpt_path = "/mnt/c1/wanghaoyu/exp/alhubert/pretrain/small/states-epoch-4.ckpt"
model = UpstreamExpert(model_ckpt_path)
data = [torch.randn(10000) for _ in range(2)] # 16KHz
states = model(data)
print(states["last_hidden_state"].shape) # torch.Tensor: hidden state of the last layer
print(len(states["hidden_states"])) # list[torch.Tensor] hidden states of each layer
# please note that if layer_norm_first=False (default), "hidden_states" will be the outputs of transformer layer 0,1,...11
# layer_norm_first=True (for HuBERT Large teachers), "hidden_states" will be the outputs of the CNN feature estractor and transformer layer 0,1,...10.
# in that case, the output of transformer layer 11 is in states["last_hidden_state"].
# This is because that the feature after layer norm is better for distillation.
