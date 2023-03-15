class AlhubertConfig:
    """
    Configuration class
    """

    def __init__(self, config: dict):
        # Feature extractor
        self.extractor_mode = str(config.get("extractor_mode", "default"))
        self.extractor_conv_feature_layers = str(
            config.get(
                "extractor_conv_feature_layers",
                "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
            )
        )
        self.feature_grad_mult = float(config.get("feature_grad_mult", 1.0))
        self.normalize = bool(config.get("normalize", False))

        # Convolutional relative positional encoding
        self.conv_pos = int(config.get("conv_pos", 128))
        self.conv_bias = bool(config.get("conv_bias", False))
        self.conv_pos_groups = int(config.get("conv_pos_groups", 16))

        # Transformer encoder
        self.encoder_layers = int(config.get("encoder_layers", 1))
        self.encoder_embed_dim = int(config.get("encoder_embed_dim", 768))
        self.encoder_ffn_embed_dim = int(config.get("encoder_ffn_embed_dim", 3072))
        self.encoder_attention_heads = int(config.get("encoder_attention_heads", 12))
        self.activation_fn = str(config.get("activation_fn", "gelu"))
        self.layer_norm_first = bool(config.get("layer_norm_first", False))

        # Dropout
        self.dropout = float(config.get("dropout", 0.1))
        self.attention_dropout = float(config.get("attention_dropout", 0.1))
        self.activation_dropout = float(config.get("activation_dropout", 0.1))
        self.encoder_layerdrop = float(config.get("encoder_layerdrop", 0.0))

        # mask
        self.mask_length = int(config.get("mask_length", 10))
        self.mask_prob = float(config.get("mask_prob", 0.65))
        self.mask_selection = str(config.get("mask_selection", "static")) # "static", "uniform", "normal", "poisson"
        self.mask_other = int(config.get("mask_other", 0))
        self.no_mask_overlap = bool(config.get("no_mask_overlap", False))
        self.mask_min_space = int(config.get("mask_min_space", 1))

        self.mask_channel_length = int(config.get("mask_channel_length", 10))
        self.mask_channel_prob = float(config.get("mask_channel_prob", 0.0))
        self.mask_channel_selection = str(config.get("mask_channel_selection", "static")) # "static", "uniform", "normal", "poisson"
        self.mask_channel_other = int(config.get("mask_channel_other", 0))
        self.no_mask_channel_overlap = bool(config.get("no_mask_channel_overlap", False))
        self.mask_channel_min_space = int(config.get("mask_channel_min_space", 1))

        # Task & loss
        self.loss_type = str(config.get("loss_type", "l1"))
        self.feat_pen_loss = float(config.get("feat_pen_loss", 0.0))
        self.cosine_loss = float(config.get("cosine_loss", 0.0))
        self.student_mid_layers = str(config.get("student_mid_layers", "[]"))
        self.teacher_mid_layers = str(config.get("teacher_mid_layers", "[]"))

        # Initialization
        self.init_teacher_conv_layers = bool(config.get("init_teacher_conv_layers", False))
        self.init_teacher_encoder_layers = bool(config.get("init_teacher_encoder_layers", False))

        self.repeat_time = int(config.get("repeat_time", 6))
        self.teacher_feature_selection = str(config.get("teacher_feature_selection", "original"))