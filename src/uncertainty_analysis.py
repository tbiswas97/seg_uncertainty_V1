LAYER_START = 0
LAYER_STOP = 16
LAYER_STEP = 4
N_LAYERS = (LAYER_START-LAYER_STOP)//(LAYER_STEP)
from scipy.stats import entropy
import numpy as np  
import pandas as pd 
import toolbox as tb

def _reshape_model_weights(SM):
    d = SM.model_res
    SM.prob_maps = {}
    for key in d.keys():
        #self.seg_maps[key] = {}
        SM.prob_maps[key] = {}
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
            SM.prob_maps[key][smm.n_components] = []
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

                SM.prob_maps[key][smm.n_components].append(pmap)

    return None


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

def get_uci_at_layer(df,SM,layer=0):
    model = list(SM.model_res.keys())[0]
    n_components = SM.model_components[0]
    _reshape_model_weights(SM)

    pmap = SM.prob_maps[model][n_components][layer]
    smap = SM.seg_maps[model][n_components][layer]

    df_labels = ["neuron1_np_coord","neuron2_np_coord"]
    entropy_labels = ["neuron1_class_entropy","neuron2_class_entropy"]
    segment_labels = ["neuron1_segment","neuron2_segment"]

    coords_list = df.loc[(df.img_idx==SM.iid_idx),df_labels]

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

def get_uci_at_im(SM,df):

    entropies_all_layers = [get_uci_at_layer(df,SM,layer=k) for k in [0,1,2,3]]
    entropies_all_layers_df = pd.concat(entropies_all_layers,axis=0,ignore_index=True)

    return entropies_all_layers_df

def get_ucis(SMs,df):
    all_ims = [get_uci_at_im(SM,df) for SM in SMs]
    all_ims_df = pd.concat(all_ims,axis=0,ignore_index=True)

    return all_ims_df

def stitch_uci(df,SMs):
    df_repeated = pd.concat([df]*4,axis=0,ignore_index=True)
    uci = get_ucis(SMs,df)

    out = pd.concat([df_repeated,uci.iloc[:,0:]],axis=1)

    return out 