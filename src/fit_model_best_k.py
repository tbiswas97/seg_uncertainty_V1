import numpy as np
import import_utils
from SegmentationMap import SegmentationMap as SM 

for idx,iid in enumerate(import_utils.IIDS):
    a = SM(iid)
    k_arr = np.asarray(a.k)
    try:
        best_k = round(np.median(k_arr[k_arr<10]))
    except: 
        best_k = min(k_arr)
        if best_k > 20:
            best_k = 20
    print("Fitting image id: {} for {} components| idx {} out of {}".format(iid,best_k,idx,len(import_utils.IIDS)))
    a.fit_model(model="c",n_components=np.array([best_k]),max_components=21)

    import_utils._pickle(a,'SegmentationMap_kbest_{}_{}.pkl'.format(iid,idx))

#idxs = [2]
#for idx in idxs:
    #a = SM(import_utils.IIDS[idx])
    #print("fitting...")
    #a.fit_model(model='c',n_components=np.array([3,4,5]))
    #import_utils._pickle(a,'SegmentationMap_k345_{}_{}.pkl'.format(import_utils.IIDS[idx],idx))
