from msg_transformer_mindspore import MSGTransformer

def build_model(config):
    model_type = config.MODEL.TYPE
    if 'msg' in model_type:
        model = MSGTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.MSG.PATCH_SIZE,
                                in_chans=config.MODEL.MSG.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.MSG.EMBED_DIM,
                                depths=config.MODEL.MSG.DEPTHS,
                                num_heads=config.MODEL.MSG.NUM_HEADS,
                                window_size=config.MODEL.MSG.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.MSG.MLP_RATIO,
                                qkv_bias=config.MODEL.MSG.QKV_BIAS,
                                qk_scale=config.MODEL.MSG.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.MSG.APE,
                                patch_norm=config.MODEL.MSG.PATCH_NORM,
                                shuffle_size=config.MODEL.MSG.SHUF_SIZE,
                                manip_type=config.MODEL.MSG.MANIP_TYPE,)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
