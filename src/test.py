import numpy as np
import import_utils
from SegmentationMap import SegmentationMap as SM 

for idx,iid in enumerate(import_utils.IIDS):
    a = SM(iid)
    print("Fitting image id: {} | idx {} out of {}".format(iid,idx,len(import_utils.IIDS)))
    a.fit_model(n_components=np.array([4]))

    import_utils._pickle(a,'SegmentationMap_{}_{}.pkl'.format(iid,idx))

#idxs = [2]
#for idx in idxs:
    #a = SM(import_utils.IIDS[idx])
    #print("fitting...")
    #a.fit_model(n_components=np.array([3]))
    #import_utils._pickle(a,'SegmentationMap_{}_{}.pkl'.format(import_utils.IIDS[idx],idx))
