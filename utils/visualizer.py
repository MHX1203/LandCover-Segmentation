import torch as t
import numpy as np
import cv2

def convert_color(color_dicts):
    dicts = {}
    for k, color in color_dicts.items():
        # print(color[:2])
        # print(color[2:4])
        # print(color[4:])
        dicts[k] = [int(str(color[:2]), 16), int(str(color[2:4]), 16), int(str(color[4:]), 16)]

    return dicts

def vis_mask(mask, dicts, return_type='tensor'):

    # if isinstance(mask, np.ndarray):
    #     mask = t.from_numpy(mask)

    dicts = convert_color(dicts)
    
    if len(mask.shape) == 2:
        mask = mask[None]

    mask = mask.type(t.uint8)
    ch1 = mask.clone()
    ch2 = mask.clone()
    ch3 = mask.clone()

    for k, color in dicts.items():
        ch1[ch1 == k] = color[0]
        ch2[ch2 == k] = color[1]
        ch3[ch3 == k] = color[2]

    mask = t.stack([ch1, ch2, ch3], dim=1)
        # mask[mask == k][:, 0, ...] = color[0]
        # mask[mask == k][..., 1, ...] = color[1]
        # mask[mask == k][..., 2, ...] = color[2]

    # print(mask.shape, mask.dtype)
    if return_type == 'tensor':
        return mask

    elif return_type == 'array':
        return mask.numpy()
    else:
        raise NotImplementedError
    