from seg.models_deep_seg import *
from seg.pyramid import *

from scipy.stats import entropy
import pandas as pd

import datetime
import os
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from joblib import Parallel, delayed, Memory, parallel_backend
from joblib import dump, load

# select device
device = torch.device("cpu")
# load model and send it to device for evaluation only
#TODO:the parameter "pretrained" is deprecrtaed and may be removed, please used "weights" instead 
#from torchvision.models import VGG_19.Weights
#weights = VGG_19.Weights.DEFAULT
#or 
#weights = VGG_19.Weights.IMAGENET1K_V1
#or use strings
#weights = "DEFAULT"
#weights = "IMAGENET1K_V1"
pretrained = True
deepnet = models.vgg19(pretrained=pretrained).features.to(device).eval()
# number of layers (max 16)
L = 16


def make_image_stack(dat, n_im=None):
    """
    Input: dat - a 4D array (n_im, height, width, channels)
    Output: im_all - input is output as images normalized with a standardized rotation
    """
    if n_im is not None:
        n_im = n_im
    else:
        n_im = dat.shape[0]
    ny, nx = dat[0, 0].shape[0:2]
    N_list = np.array(
        [
            (ny, nx),
            (ny, nx),
            (ny // 2, nx // 2),
            (ny // 2, nx // 2),
            (ny // 4, nx // 4),
            (ny // 4, nx // 4),
            (ny // 4, nx // 4),
            (ny // 4, nx // 4),
            (ny // 8, nx // 8),
            (ny // 8, nx // 8),
            (ny // 8, nx // 8),
            (ny // 8, nx // 8),
            (ny // 16, nx // 16),
            (ny // 16, nx // 16),
            (ny // 16, nx // 16),
            (ny // 16, nx // 16),
        ]
    )
    im_all = np.zeros((n_im, ny, nx, 3))
    for i in range(n_im):
        m = dat[i][0].min()
        M = dat[i][0].max()
        if dat[i][0].shape[0] == ny:  # checks rotation is all is one direction
            im_all[i] = (dat[i][0] - m) / (M - m)
        else:
            im_all[i] = (np.swapaxes(dat[i][0], 0, 1) - m) / (M - m)

    return im_all, N_list  


# MODEL SELECTION
def _fit_model(
        dat,
        model_type="ref",
        n_components=np.array([3]),
        layer=None,
        keep=False,
        init=None,
        init_eps=None
    ):
    """
    Use to fit segmentation map to input image
    Parameters:
    ------------
        dat: np.ndarray
            ND array of n_im,height,weight,channels
        model_type: str
            'ref': indeprendent layers / no smoothing
            'a': independent layers
            'b': single prior probability map
            'c': smoothed prior probability maps
        n_components: np.array
            number of components for segmentation map (each image will have the same number) as array
        layer : int 
            number of layers of VGG neural network to extract features from 
        keep : bool
            Set to True to keep the segmentation maps from every iteration of the EM algorithm
        init : np.array 
            Array of shape(image height, image width), this is the initial guess during segmentation fitting 
        init_eps: float
            This is the amount of uncertainty injected with the initial guess, if None 0.0001 is used as default 

    Output:
        array of n_im model results (singleton if n_im=1)
    """
    #im_all = dat
    im = dat
    K_list = n_components
    ny, nx = im.shape[0:2]
    N_list = np.array(
        [
            (ny, nx),
            (ny, nx),
            (ny // 2, nx // 2),
            (ny // 2, nx // 2),
            (ny // 4, nx // 4),
            (ny // 4, nx // 4),
            (ny // 4, nx // 4),
            (ny // 4, nx // 4),
            (ny // 8, nx // 8),
            (ny // 8, nx // 8),
            (ny // 8, nx // 8),
            (ny // 8, nx // 8),
            (ny // 16, nx // 16),
            (ny // 16, nx // 16),
            (ny // 16, nx // 16),
            (ny // 16, nx // 16),
        ]
    )

    neigh_size_list = 1.0 * np.array(
        [17, 17, 13, 13, 9, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3]
    )  # -1

    d_list = np.array(
        [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
    )

    # FIXME: for ppca = True, model b and c don't work
    ppca = False
    light = True
    n_pca = 10

    # models are defined in models_deep_seg.py
    # FIXME: ref model output is not the correct size?
    # FIXME: add init option to model a 
    if model_type == "ref":
        _fit = lambda x: model_ref(
            deepnet,
            x,
            K_list=K_list,
            L=16,
            d_list=d_list,
            N_list=N_list,
            n_iter=25,
            params="",
            ppca=ppca,
            gmm=False,
            light=light,
            verbose=False,
        )
    elif model_type == "a":
        _fit = lambda x: model_a(
            deepnet,
            x,
            K_list=K_list,
            neigh_size_list=neigh_size_list,
            L=layer,
            d_list=d_list,
            N_list=N_list,
            n_iter=25,
            params="",
            ppca=ppca,
            gmm=False,
            light=light,
            n_pca=n_pca,
            verbose=False,
        )
    elif model_type == "b":
        _fit = lambda x: model_b(
            deepnet,
            x,
            K_list=K_list,
            neigh_size_list=neigh_size_list,
            L=layer,
            d_list=d_list,
            N_list=N_list,
            n_iter=25,
            params="",
            ppca=ppca,
            gmm=False,
            light=light,
            n_pca=n_pca,
            verbose=False,
        )
    elif model_type == "c":
        _fit = lambda x: model_c(
            deepnet,
            x,
            gt=init,
            gt_eps=init_eps,
            K_list=K_list,
            neigh_size_list=neigh_size_list[:layer],
            L=layer,
            d_list=d_list[:layer],
            N_list=N_list[:layer],
            n_iter=25,
            params="",
            ppca=ppca,
            gmm=False,
            light=light,
            n_pca=n_pca,
            verbose=False,
            keep=keep
        )

        if keep:
            res_arr,proba_maps = _fit(im)
        else:
            res_arr = _fit(im)
    #res_arr = np.asarray([_fit(im) for im in im_all])
    if not keep:
        res_arr = _fit(im)

    if len(res_arr) == 1:
        res_arr = res_arr[0]

    if keep:
        return res_arr, proba_maps
    else: 
        return res_arr

def _gen_seg_map_from_weights(weights, size):
    Ny,Nx = size
    ny,nx = weights.shape
    smap = weights.argmax(1).reshape((size))
    if ny != Ny:
        assert Ny//ny == Nx//nx
        multiplier = Ny // ny
        m = multiplier
        smap = smap.repeat(m,0).repeat(m,1)

    return smap

def _gen_seg_map(res, N_list, standard_size=True):
    """
    Parameters:
        res = output of _model function for one image, all layers
    Outputs:
        segmentation map (16 x ny x nx) based on model output
    """
    Ny, Nx = N_list[0]
    out = []

    for l in range(len(res)):
        layer = res[l, 0, 2, 0]
        ny, nx = N_list[l]
        weights = np.float32(layer.weights_)
        smap = weights.argmax(1).reshape((ny, nx))
        if standard_size:
            assert Ny // ny == Nx // nx
            multiplier = Ny // ny
            m = multiplier
            smap = smap.repeat(m, 0).repeat(m, 1)

        out.append(smap)

    return np.asarray(out)

def _reshape_model_weights(SM,layers_of_interest=None):
    d = SM.model_res
    SM.layers_of_interest = layers_of_interest
    SM.model = list(SM.seg_maps.keys())[0]
    SM.n_components = list(SM.seg_maps[SM.model].keys())[0]
    SM.pmaps = {}
    for key in d.keys():
        #self.seg_maps[key] = {}
        SM.pmaps[key] = {}
        n_n_components = d[key].shape[1]

        # different values of i will have different n_components
        for i in range(n_n_components):
            # index 2 below is the index for the smm object
            #layer 0
            smm = d[key][0, i, 2, 0]
            #layer 16
            smm_last = d[key][-1, i, 2, 0]
            Ny, Nx = smm.im_shape
            _Ny, _Nx = smm_last.im_shape
            SM.pmaps[key][smm.n_components] = []
            if layers_of_interest is not None:
                layers = d[key][
                    layers_of_interest, i, 2, 0
                ]  # generates seg map from every 4th layer
            else:
                layers = d[key][
                    LAYER_START:LAYER_STOP:LAYER_STEP, i, 2, 0
                ]  # generates seg map from every 4th layer

            for layer in layers:
                ny, nx = layer.im_shape
                pmap = layer.weights_.reshape((ny,nx,layer.n_components))
                #reshape so that the component probabilities are the first dimension
                pmap = np.moveaxis(pmap,-1,0)

                if ny != Ny:
                    assert Ny // ny == Nx // nx
                    multiplier = Ny // ny
                    m = multiplier
                    pmap = pmap.repeat(m, 1).repeat(m, 2)

                SM.pmaps[key][smm.n_components].append(pmap)

    return None

def __get_global_entropy_at_layer(SM,layer,bounding_box=(256,256)):
    if hasattr(SM, 'pmaps'):
        pass
    else:
        _reshape_model_weights(SM)

    assert layer in SM.layers_of_interest

    layer_idx = SM.layers_of_interest.index(layer)

    pmap = SM.pmaps[SM.model][SM.n_components][layer_idx]
    pmap_resize = np.asarray(tb.crop(pmap,size=bounding_box))
    entropies = np.nansum(entropy(pmap_resize))

    return entropies


def _get_global_entropy(SM,bounding_box=(256,256)):
    if hasattr(SM, 'pmaps'):
        pass
    else:
        _reshape_model_weights(SM)
    
    pmaps  = [pmap for pmap in SM.pmaps[SM.model][SM.n_components]]

    pmaps_resize = [np.asarray(tb.crop(pmap,size=bounding_box)) for pmap in pmaps]

    entropies = [np.sum(entropy(pmap)) for pmap in pmaps_resize]

    d = {"global_entropy":entropies}

    df = pd.DataFrame.from_dict(d)

    df.insert(0,"layer",SM.layers_of_interest)
    df.insert(0,"img_idx",[SM.iid_idx]*len(SM.layers_of_interest))

    return df

def main(model_type="a"):
    """
    Output: array of segmentation maps and input images

    if n input images are h x w then output is n x 16 x h x w

    """
    im_all, N_list = make_image_stack(dat, n_im=2)

    res_arr = _fit_model(im_all, model_type=model_type)

    seg_maps = []

    for im_res in res_arr:
        im_seg_map = _gen_seg_map(im_res, N_list)
        seg_maps.append(im_seg_map)

    out = np.asarray(seg_maps)

    return out


if __name__ == "__main__":
    # INPUT
    cd = os.getcwd()
    rel_path = "../data/bsd500-im-color-full.npz"
    filepath = os.path.abspath(os.path.join(cd, rel_path))
    INPUT_FILEPATH = filepath
    dat = np.load(filepath, allow_pickle=True)["data"]
    # for model_type in ['a','b','c']:
    for model_type in ["ref"]:
        print("Starting...")
        out = main(model_type=model_type)
        # save file
        pd = os.path.dirname(os.getcwd())
        output_folder = "out"
        base_filename = "out"
        tag = datetime.datetime.now().strftime("_%Y%m%d%H%M%S")
        filename = base_filename + tag + "_" + model_type + ".pkl"
        print("Saving seg maps for {} images".format(out.shape[0]))
        with open(os.path.join(pd, output_folder, filename), "wb") as file:
            pickle.dump(out, file)


#apply crop to BSD segmentation for each human segmentation map,  256 x 256
#apply crop to perceptual segmentation algos (skip model b) (only maps every 4th layer)
#rsc similarity = Pearson coeff. 

#decide on upper limit of segments 
#set minimum number of pixels per segment 