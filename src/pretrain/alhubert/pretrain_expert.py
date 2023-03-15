from easydict import EasyDict as edict
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import OnlineWaveDataset
from ...upstream.alhubert.model import AlhubertModel
from ...upstream.alhubert.config import AlhubertConfig
from ...upstream.hubert.expert import UpstreamExpert as teacher_expert


def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


class UpstreamPretrainExpert(nn.Module):
    def __init__(
        self, datarc, upstream_config, device="cuda", multi_gpu=False, **kwargs
    ):
        super().__init__()

        self.datarc = datarc
        self.device = device
        self.multi_gpu = multi_gpu

        if type(upstream_config) == str:
            self.upstream_config = yaml.load(
                open(upstream_config, "r"), Loader=yaml.FullLoader
            )
            print(
                "[UpstreamPretrainExpert] - Using upstream config from:",
                upstream_config,
            )
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            print(
                "[UpstreamPretrainExpert] - Using upstream config from the previous experiment."
            )
        else:
            raise ValueError

        print("[UpstreamPretrainExpert] - Initializing model...")
        model_config = AlhubertConfig(self.upstream_config["alhubert"])
        if datarc.get("normalize", None) is None:
            print(f"[Warning] - set dataset normalize to {model_config.normalize}")
            datarc["normalize"] = model_config.normalize
        else:
            assert datarc["normalize"] == model_config.normalize

        self._get_train_dataloader()
        self.model = AlhubertForPretrain(
            config = model_config, 
            teacher_config = edict(self.upstream_config["teacher"]), 
            dictionaries = self.dictionaries
        )

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print(
                "[UpstreamPretrainExpert] - Multi-GPU training Enabled: "
                + str(torch.cuda.device_count())
            )
        print(
            "[UpstreamPretrainExpert] - Number of parameters: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        )

    def _get_train_dataloader(self):

        dataset = OnlineWaveDataset(
            self.upstream_config["task"],
            self.datarc["train_batch_size"],
            target_level=self.upstream_config["audio"]["target_level"],
            **self.datarc,
        )
        self.dictionaries = []

        self.dataloader = DataLoader(
            dataset,
            batch_size=1,  # for bucketing
            shuffle=True,
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def load_model(self, all_states):
        if self.multi_gpu:
            self.model.module.student.load_state_dict(all_states["alhubert"])
        else:
            self.model.student.load_state_dict(all_states["alhubert"])

    # Interface
    def add_state_to_save(self, all_states):
        all_states["alhubert"] = (
            self.model.float().student.state_dict()
            if not self.multi_gpu
            else self.model.float().module.student.state_dict()
        )
        all_states["Config"] = self.upstream_config
        all_states["dictionaries"] = [d.symbols for d in self.dictionaries]
        return all_states

    # Interface
    def get_train_dataloader(self):
        return self.dataloader

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs):
        """
        Args:
            data:
                [wave_input, pad_mask]

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss
        """

        wave_input, wave_len, pad_mask = data
        target = None
        
        wave_input = wave_input.to(self.device)
        wave_len = wave_len.to(self.device)
        pad_mask = pad_mask.type(wave_input.dtype).to(self.device)
        target = target.to(self.device) if target is not None else None

        loss, other_res = self.model(
            wave_input,
            wave_len,
            pad_mask,
            global_step=global_step,
            return_other=global_step % log_step == 0,
            target=target
        )

        if global_step % log_step == 0:
            for key, value in other_res.items():
                if isinstance(value, torch.Tensor):
                    value = float(value.mean().cpu().item())
                records[key] = value

        return loss, records

    # interface
    def on_before_zero_grad(self):
        pass

    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        for key, values in records.items():
            if isinstance(values, torch.Tensor) and len(values.shape) > 1:
                logger.add_image(f"{prefix}{key}", values, global_step=global_step)
            elif isinstance(values, float):
                logger.add_scalar(f"{prefix}{key}", values, global_step=global_step)
            elif isinstance(values, torch.Tensor) and values.numel() == 1:
                logger.add_scalar(f"{prefix}{key}", values.detach().cpu().item(), global_step=global_step)


class AlhubertForPretrain(nn.Module):
    def __init__(self, config: AlhubertConfig, teacher_config: edict, dictionaries: list=[]):
        super().__init__()
        self.config = config
        print(self.config.__dict__)
        self.student = AlhubertModel(config, dictionaries=dictionaries)

        self.teacher_config = teacher_config
        teacher = teacher_expert(
            ckpt = teacher_config.path,
            feature_selection=config.teacher_feature_selection
        )
        if (
            teacher_config.model.find("hubert") >= 0
            or teacher_config.model.find("wav2vec2") >= 0
        ):
            teacher.model.encoder.layerdrop = 0
            print("[AlhubertForPretrain] - Disabled teacher's encoder layerdrop")

        self.teacher = teacher
        freeze_model(self.teacher)

        assert teacher.task_cfg.normalize == self.config.normalize
        assert teacher.model.encoder.layer_norm_first == self.config.layer_norm_first
        if teacher.model.encoder.layer_norm_first:
            # 设置hook 获取经过layer norm之后的输出
            print("[AlhubertForPretrain] - Using hooking to get output features")
            assert self.config.teacher_feature_selection == "no_feat" # 禁用原有expert的hook
            self.teacher_hiddens_after_norm = []
            def layer_hook(module, input, output):
                self.teacher_hiddens_after_norm.append(output.transpose(0, 1))
            for i in range(len(self.teacher.model.encoder.layers)):
                self.teacher.model.encoder.layers[i].self_attn_layer_norm.register_forward_hook(hook=layer_hook)

        if config.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="none")
        elif config.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError(config.loss_type)

        self.student_mid_layers = eval(self.config.student_mid_layers)
        self.teacher_mid_layers = eval(self.config.teacher_mid_layers)
        assert len(self.student_mid_layers) == len(self.teacher_mid_layers)
        print(f"[AlhubertForPretrain] - use mid layers {self.teacher_mid_layers} -> {self.student_mid_layers}")

        self.cosine_loss = config.cosine_loss
        if self.cosine_loss > 0:
            print("[AlhubertForPretrain] - Enabled cosine similarity loss.")

        if config.init_teacher_conv_layers:
            print(
                "[AlhubertForPretrain] - "
                "Initializing feature extractor from teacher"
            )
            self.student.feature_extractor.load_state_dict(
                self.teacher.model.feature_extractor.state_dict()
            )
            if self.student.post_extract_proj is not None:
                self.student.post_extract_proj.load_state_dict(
                    self.teacher.model.post_extract_proj.state_dict()
                )

        if config.init_teacher_encoder_layers:
            print("[AlhubertForPretrain] - " "Initializing encoder from teacher")
            self.student.encoder.pos_conv.load_state_dict(
                self.teacher.model.encoder.pos_conv.state_dict()
            )
            for l in range(config.encoder_layers):
                self.student.encoder.layers[l].load_state_dict(
                    self.teacher.model.encoder.layers[l].state_dict()
                )
        

    def forward(
        self,
        wave_input: torch.Tensor,
        wave_len: torch.Tensor,
        pad_mask: torch.Tensor,
        global_step: int,
        return_other: bool = False,
        target: torch.Tensor = None,
    ):
        """
        Forward function.
        Input:
            wave_input: FloatTensor (B x T_wave)
            wave_orig: List of FloatTensor
            wave_len: LongTensor (B)
            pad_mask: FloatTensor (B x T)
            return_other: Bool (returns other information for logging)
        """

        with torch.no_grad():
            # wave_orig = [wave.to(wave_input.device) for wave in wave_orig]
            with torch.cuda.amp.autocast(False):
                # print(self.config.teacher_layer_selection)
                teacher_res = self.teacher.model(
                    source = wave_input,
                    target_list = None,
                    padding_mask = ~pad_mask.bool(),
                    mask=False,
                    features_only=True
                )
                teacher_hiddens = []
                for li in self.teacher_mid_layers:
                    if self.config.layer_norm_first:
                        teacher_hiddens.append(self.teacher_hiddens_after_norm[li+1])
                    else:
                        teacher_hiddens.append(teacher_res["layer_results"][li][0].transpose(0, 1)) 
                teacher_hiddens.append(teacher_res["x"])
                # ["hidden_states"][self.config.teacher_layer_selection]

        # Forward model
        student_ret = self.student(
            wave_input, 
            target_list = [target] if target is not None else None,
            pad_mask=pad_mask,
            mask=False,
            features_only=True,
            generated_mask=None
        )
        student_hiddens = []
        for li in self.student_mid_layers:
            if self.config.layer_norm_first:
                student_hiddens.append(student_ret["layer_hiddens"][li+1][1])
            else:
                student_hiddens.append(student_ret["layer_hiddens"][li][0])
        # print(len(self.teacher_hiddens_after_norm))
        student_hiddens.append(student_ret["last_hidden"])
        feat_pen = student_ret["feat_pen"]

        # 计算损失
        total_loss = 0 # 初始值
        # MSE 损失
        rec_loss, sim_loss = self.compute_loss(
            student_hiddens, 
            teacher_hiddens
            )
        if self.cosine_loss > 0:
            total_loss = total_loss + rec_loss + sim_loss*self.cosine_loss
        else:
            total_loss = total_loss + rec_loss
        # 正则化系数
        total_loss += feat_pen*self.config.feat_pen_loss

        if return_other:
            with torch.no_grad():
                other_res = {"feat_pen": feat_pen}
                other_res["rec_loss"] = rec_loss
                other_res["sim_loss"] = sim_loss
        else:
            other_res = None

        if self.config.layer_norm_first:
            self.teacher_hiddens_after_norm = []
        # del teacher_res, student_ret
        return total_loss, other_res

    def compute_loss(self, student_hiddens, teacher_hiddens):
        """
        Computes loss.
        Inputs:
            feat_pen: tensor
            pred: B x T x D
            target: B x T x D
        """
        # assert len(student_hiddens) == len(teacher_hiddens)
        # Reconstruction loss
        rec_loss = 0
        sim_loss = 0
        for student_hidden, teacher_hidden in zip(student_hiddens, teacher_hiddens):
            diff = student_hidden.shape[1] - teacher_hidden.shape[1]
            if diff > 0:
                student_hidden = student_hidden[:,:-diff,:]
            elif diff < 0:
                teacher_hidden = teacher_hidden[:,:diff,:]
            rec_loss_layer = self.loss_func(student_hidden, teacher_hidden)  # B x T x D
            rec_loss += rec_loss_layer.mean()
            # Cosine similarity loss
            if self.cosine_loss > 0:
                sim_loss_layer = -F.logsigmoid(F.cosine_similarity(student_hidden, teacher_hidden, dim=-1))
                # B x N x T
                sim_loss += sim_loss_layer.mean()
            # print(rec_loss_layer.mean(), sim_loss_layer.mean())
        # print(total_loss, rec_loss, feat_pen, sim_loss)
        return rec_loss, sim_loss
    