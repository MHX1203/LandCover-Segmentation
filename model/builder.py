from model import attention_unet, rec_attention_unet, unet_1, unet, nested_unet, r2u_net

def build_model(in_channels, num_classes, model_name='unet'):

    if model_name == 'unet':
        return unet_1.UNet(in_channels, num_classes)
    elif model_name == 'r2u':
        return r2u_net.R2U_Net(in_channels, num_classes)
    elif model_name == 'nested_unet':
        return nested_unet.NestedUNet(in_channels, num_classes)
    elif model_name == 'rec_attention_unet':
        return rec_attention_unet.R2AttU_Net(in_channels, num_classes)
    elif model_name == 'attention_unet':
        return attention_unet.AttU_Net(in_channels, num_classes)
    else:
        raise NotImplementedError(f'Unknown model_name {model_name}')
    pass