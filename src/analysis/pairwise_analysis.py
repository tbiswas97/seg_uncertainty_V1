import toolbox as tb
import pandas as pd
import numpy as np
from natsort import natsorted as ns
from analysis import single_neuron_analysis as sna


def get_info_at_layer(df, SM, layer_idx=0, image_idx=0, im_shape=(256, 256)):
    model = SM.model
    n_components = SM.n_components

    pmap = SM.pmaps[model][n_components][layer_idx]

    if pmap.shape[1:] != im_shape:
        # output of crop function is a list of slices that should be converted to an array
        pmap = np.asarray(tb.crop(pmap, size=im_shape))

    df_labels = [elem for elem in df.columns if "np_coord" in elem]

    segment_labels = [
        elem.split("_")[0] + "_p{}".format(i)
        for i in range(n_components)
        for elem in df_labels
    ]

    argmax_labels = [elem.split("_")[0] + "_segment_argmax" for elem in df_labels]

    labels = segment_labels + argmax_labels

    labels = ns(labels)

    coords = df.loc[df.img_idx == SM.iid_idx, df_labels].reset_index(drop=True)

    str_to_int = lambda x: np.asarray(
        [
            int(item)
            for item in x.replace("[", "").replace("]", "").split(" ")
            if len(item) > 0
        ]
    )

    coords = coords.applymap(str_to_int)

    neuron1_df = sna.get_pmap_infolist(coords.iloc[:,0],pmap,labels[:len(labels)//2])

    neuron2_df = sna.get_pmap_infolist(coords.iloc[:,1],pmap,labels[len(labels)//2:len(labels)])

    out = pd.concat([neuron1_df,neuron2_df],axis=1,ignore_index=True)
    out["layer"] = layer_idx

    to_join = df.loc[df.img_idx==SM.iid_idx].reset_index(drop=True)
    names = list(to_join.columns) + list(out.columns)

    joined = pd.concat(
        [to_join,out],
        axis=1,
        ignore_index=True
    ).rename(
        {
            k:v for k,v in zip(range(len(names)),names)
        }, axis=1
    )

    return joined

def get_info_at_im(
    df,
    SM,
    layers_of_interest,
    im_shape=(256, 256),
    spatial_average=(20, 20)
):

    info_all_layers = pd.concat(
        [
            get_info_at_layer(
                df,
                SM,
                layer_idx=k,
                im_shape=im_shape,
                spatial_average=spatial_average,
            )
            for k in range(len(layers_of_interest))
        ],
        axis=0,
        ignore_index=True,
    )

    return info_all_layers


def get_info(
    df,
    SMs,
    layers_of_interest,
    im_shape=(256, 256),
    spatial_average=(20, 20)
):
    all_ims = pd.concat(
        [
            get_info_at_im(
                df,
                SM,
                layers_of_interest,
                im_shape=im_shape,
                spatial_average=spatial_average,
            )
            for i, SM in enumerate(SMs)
        ],
        axis=0,
        ignore_index=True,
    )

    return all_ims


def stitch_info(
    df,
    SMs,
    layers_of_interest,
    im_shape=(256, 256),
    bounding_box = 180,
    spatial_average=(20, 20)
):

    df = df.loc[
        np.asarray(
            [
                (df.neuron_r[i] < bounding_box)
                for i in df.neuron_np_coord.index
            ]
        )
    ]

    info = get_info(
        df,
        SMs,
        layers_of_interest,
        im_shape=im_shape,
        spatial_average=spatial_average,
    )

    return info
