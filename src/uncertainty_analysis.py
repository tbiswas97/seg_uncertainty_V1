LAYER_START = 0
LAYER_STOP = 16
LAYER_STEP = 4
N_LAYERS = (LAYER_START-LAYER_STOP)//(LAYER_STEP)
from scipy.stats import entropy
import numpy as np  
import pandas as pd 
import toolbox as tb

#Single neuron analysis

#helper function for single coordinates
def _get_pmap_infolist(coord,pmap,labels) -> dict:
    p = list(pmap[:,coord[0],coord[1]])
    p.append(pmap[:,coord[0],coord[1]].argmax(0))
    d = {k:v for k,v in zip(labels,p)}

    return d

#function for list of coordinates
def get_pmap_infolist(coords,pmap,labels) -> pd.DataFrame:
    #helper function

    ld = [_get_pmap_infolist(coord,pmap,labels) for coord in coords]
    dl = {k:[d[k] for d in ld] for k in ld[0]}

    return pd.DataFrame.from_dict(dl)

def _calculate_p_same(coord1,coord2,_map):
    assert _map.ndim==3

    y1,x1 = coord1
    y2,x2 = coord2

    vec1 = _map[:,y1,x1]
    vec2 = _map[:,y2,x2]

    to_sum = vec1*vec2

    return np.sum(to_sum)

def _calculate_sameness_entropy(coord1,coord2,_map):
    y1,x1 = coord1
    y2,x2 = coord2

    vec1 = _map[:,y1,x1]
    vec2 = _map[:,y2,x2]

    vec1 = vec1[vec1>1e-5]
    vec2 = vec2[vec2>1e-5]

    to_sum = []
    for pn in vec1:
        assert pn>0
        for pm in vec2:
            assert pm>0
            to_sum.append(-pn*pm*np.log(pn*pm))
    
    assert ~np.isnan(np.sum(to_sum))
    return (np.sum(to_sum))

def _calculate_class_entropy(coord,_map):
    assert _map.ndim==3
    ny,nx = coord
    return entropy(_map[:,ny,nx])

def _reshape_model_weights(SM,layers_of_interest=None):
    d = SM.model_res
    SM.model = d.keys()[0]
    SM.n_components = d[SM.model].keys()[0]
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

def get_uci_at_layer(
        df,
        SM,
        layer=0,
        image_idx=0,
        im_shape=(256,256)
    ):
    """
    Gets uncertainty information and segment assignment for a give coordinate

    Params:
    -------
    df : pandas.DataFrame
        df output by Session object
    SM : src.SegmentationMap 
        output of SegmentationMap.py after fit_model has been called
    map_shape : tup of ints
        should match the array size used for numpy conversion 
    """

    pmap = SM.pmaps[model][n_components][layer]

    if pmap.shape[1:] != im_shape:
        pmap = tb.crop(pmap,size=im_shape)

    df_labels = [elem for elem in df.columns if "np_coord" in elem]

    entropy_labels = [elem.split("_")[0]+"_entropy" for elem in df_labels]
    segment_labels = [
        elem.split("_")[0]+"_p{}".format(i) for i in range(n_components) for elem in df_labels
    ]

    segment_labels.append("neuron_segment_argmax")

    coords_list = df.loc[(df.img_idx==image_idx),df_labels]

    segments = {
        k:[
            tb.get_coord_segment(coord,smap) for coord in coords_list[v]
        ] for k,v in zip(segment_labels,df_labels)
        
    }
    entropies = {
        k:[
            _calculate_class_entropy(coord,pmap) for coord in coords_list[v]
            ] for k,v in zip(entropy_labels,df_labels)
    }

    sameness = {
        "p_same":[_calculate_p_same(
                coords_list.iloc[i,0],coords_list.iloc[i,1],pmap) for i in range(len(coords_list))
        ],
        "sameness_entropy":[_calculate_sameness_entropy(
                coords_list.iloc[i,0],coords_list.iloc[i,1],pmap) for i in range(len(coords_list))
        ]
    }

    segments_df = pd.DataFrame.from_dict(segments)
    entropy_df = pd.DataFrame.from_dict(entropies)
    sameness_df = pd.DataFrame.from_dict(sameness)

    df_out = pd.concat([entropy_df,sameness_df,segments_df,],axis=1,ignore_index=False)
    df_out['layer']=layer

    return df_out

def get_uci_at_im(SM,df,analysis="pairwise"):

    entropies_all_layers = [get_uci_at_layer(df,SM,layer=k) for k in [0,1,2,3]]
    entropies_all_layers_df = pd.concat(entropies_all_layers,axis=0,ignore_index=True)

    return entropies_all_layers_df

def get_ucis(SMs,df,analysis="pairwise"):
    all_ims = [get_uci_at_im(SM,df) for SM in SMs]
    all_ims_df = pd.concat(all_ims,axis=0,ignore_index=True)

    return all_ims_df

def stitch_uci(df,SMs,analysis="pairwise"):
    df_repeated = pd.concat([df]*4,axis=0,ignore_index=True)
    uci = get_ucis(SMs,df)

    out = pd.concat([df_repeated,uci.iloc[:,0:]],axis=1)

    return out 