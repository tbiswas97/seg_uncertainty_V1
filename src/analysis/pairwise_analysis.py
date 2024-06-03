import toolbox as tb
import pandas as pd

def get_info_at_layer(
        df,
        SM,
        layer=0,
        image_idx=0,
        im_shape=(256,256)
):

    pmap = SM.pmaps[model][n_components][layer]

    if pmap.shape[1:] != im_shape:
        pmap = tb.crop(pmap,size=im_shape)

    df_labels = [elem for elem in df.columns if "np_coord" in elem]

    pass 

    return None