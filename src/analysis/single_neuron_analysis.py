import toolbox as tb
import pandas as pd
import numpy as np
from scipy.stats import entropy


def _get_pmap_infolist(coord, pmap, labels, spatial_average=(20, 20)) -> dict:
    if spatial_average is not None:
        #TODO: spatial average over a circular area (rather than rectuangular)
        dy, dx = spatial_average
        p = pmap[:, coord[0] - dy : coord[0] + dy, coord[1] - dx : coord[1] + dx]
        p = np.mean(p, axis=(1, 2))
        p = list(p)
    else:
        p = list(pmap[:, coord[0], coord[1]])

    p.append(pmap[:, coord[0], coord[1]].argmax(0))
    d = {k: v for k, v in zip(labels, p)}

    return d


# function for list of coordinates
def get_pmap_infolist(coords, pmap, labels, spatial_average=(20, 20)) -> pd.DataFrame:
    # helper function

    ld = [_get_pmap_infolist(coord, pmap, labels, spatial_average) for coord in coords]
    dl = {k: [d[k] for d in ld] for k in ld[0]}

    return pd.DataFrame.from_dict(dl)


def get_info_at_layer(
    df,
    SM,
    layer_idx=0,
    im_shape=(256, 256),
    spatial_average=(20, 20),
    calculate_global_entropy=True
):
    """
    Extracts uncertinaty information and segment assignment for a given coordinate
    """
    assert hasattr(SM, "pmaps"), "SM does not have pmap attribute"

    model = SM.model
    n_components = SM.n_components

    pmap = SM.pmaps[model][n_components][layer_idx]

    if pmap.shape[1:] != im_shape:
        # output of crop function is a list of slices that should be converted to an array
        pmap = np.asarray(tb.crop(pmap, size=im_shape))

    # There is only one label for single-neuron analysis
    df_labels = [elem for elem in df.columns if "np_coord" in elem][0]

    segment_labels = [
        elem.split("_")[0] + "_p{}".format(i)
        for i in range(n_components)
        for elem in [df_labels]
    ]

    segment_labels.append("neuron_segment_argmax")

    
    coords = df.loc[df.img_idx == SM.iid_idx, df_labels].reset_index(drop=True)

    out = get_pmap_infolist(coords, pmap, segment_labels, spatial_average)
    out["layer"] = layer_idx

    if calculate_global_entropy:
        ge = np.sum(entropy(pmap,axis=0))
    
    out["global_entropy"] = ge

    to_join = df.loc[df.img_idx == SM.iid_idx].reset_index(drop=True)

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
    spatial_average=(20, 20),
    calculate_global_entropy=True
):

    info_all_layers = pd.concat(
        [
            get_info_at_layer(
                df,
                SM,
                layer_idx=k,
                im_shape=im_shape,
                spatial_average=spatial_average,
                calculate_global_entropy=calculate_global_entropy
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
    spatial_average=(20, 20),
    calculate_global_entropy=True
):
    all_ims = pd.concat(
        [
            get_info_at_im(
                df,
                SM,
                layers_of_interest,
                im_shape=im_shape,
                spatial_average=spatial_average,
                calculate_global_entropy=calculate_global_entropy
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
    spatial_average=(20, 20),
    calculate_global_entropy=True
):
    df = df.loc[df.presentation == "large"]

    df = df.loc[
        np.asarray(
            [
                (df.neuron_r[i] < bounding_box)
                for i in df.neuron_np_coord.index
            ]
        )
    ]

    #use only those images that have not been excluded:
    SMs_arr = np.asarray(SMs)
    #included images: 
    incl = df.img_idx.unique().astype(int)
    
    SMs = list(SMs_arr[incl])

    info = get_info(
        df,
        SMs,
        layers_of_interest,
        im_shape=im_shape,
        spatial_average=spatial_average,
        calculate_global_entropy=calculate_global_entropy
    )

    return info
