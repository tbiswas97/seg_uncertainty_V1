import os
import import_utils
import toolbox as tb
import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
import seaborn as sns
import seg.segment as seg
from itertools import combinations
from Session import Session as Sess
from Session import DEFAULT_PROBES


class SegmentationMap:
    """
    Initiate a SegmentationMap object from an image ID of the BSDS500

    Atrributes:
    ----------

    iid : str
        Image ID of the BSDS500
    iid_idx : int
        index of the image in the particular experiment
    jpg_path : str
        path to .jpg file from BSDS500
    seg_path : str
        path to .mat seg file from BSDS500
    im : ndarray
        (h x w x 3) RGB image array
    gts : list of ndarrays
    model_res : dict
        segmentation model results keys of dict organize by parameters and model results are ndarrays
    k : list of ints
        number of segments across users
    users_d : dict
        key is n_components k and values are which users labeled for k components
    model_components : list of ints
        initialized as same as k but eventually reflects the number of components in the segmentation model
    seg_maps : dict
        segmentation maps from the model results keys of dict organize by parameters and segmaps are ndarrays
    cropped : bool
        whether the crop method has been used or not
    session_loaded : bool
        sessions are representations of neural data that can be loaded into memory, session_loaded is True if the session is loaded into memory
    primary_seg_map : ndarray
        the primary segmentation map is used to generate statistics for the segmentation
    c_im : ndarray
        same as im, but cropped
    c_gts : list of ndarrays
        same as gts, but cropped
    c_seg_maps : dict
        same as seg_maps, but cropped
    Session : Session
        Session object is defined in ./Session.py

    """

    def __init__(self, _in, mode="BSD"):
        """
        Initializes SegmentationMap class that interfaces with Session data

        Parameters:
        ------------

        _in : str or tup
        mode : str
            'array' : if mode is 'array' _in should be a tup of
                (array_idx, array)
            'BSD' : if mode is 'BSD' _in should be a str corresponding to the BSD image ID
        """
        self.gs_im = None
        if mode == "BSD":
            assert type(_in) == str or type(_in) == np.str_
            self.iid = _in  # image id in BSDS500
            try:
                self.iid_idx = import_utils.IIDS.index(self.iid)
            except:
                self.iid_idx = 0

            try:
                self.jpg_path = os.path.abspath(
                    os.path.join(import_utils.JPG_PATH_TRAIN, self.iid + ".jpg")
                )
                flag = "train"
                self.im = import_utils.import_jpg(self.jpg_path)
            except:
                try:
                    self.jpg_path = os.path.abspath(
                        os.path.join(import_utils.JPG_PATH_TEST, self.iid + ".jpg")
                    )
                    flag = "test"
                    self.im = import_utils.import_jpg(self.jpg_path)
                except:
                    self.jpg_path = os.path.abspath(
                        os.path.join(import_utils.JPG_PATH_VAL, self.iid + ".jpg")
                    )
                    flag = "val"
                    self.im = import_utils.import_jpg(self.jpg_path)

            if flag == "train":
                self.seg_path = os.path.abspath(
                    os.path.join(import_utils.SEG_PATH_TRAIN, self.iid + ".mat")
                )
            elif flag == "test":
                self.seg_path = os.path.abspath(
                    os.path.join(import_utils.SEG_PATH_TEST, self.iid + ".mat")
                )
            elif flag == "val":
                self.seg_path = os.path.abspath(
                    os.path.join(import_utils.SEG_PATH_VAL, self.iid + ".mat")
                )

            self.gts = import_utils.load_bsd_mat(self.seg_path)
            self.model_res = {}

            for gt in self.gts:
                assert self.im.shape[0:2] == self.gts[0].shape

            self.k = [
                len(np.unique(gt)) for gt in self.gts
            ]  # number of segments across users

            d = {}

            for i, val in enumerate(self.k):
                if val not in d.keys():
                    d[val] = []
                    d[val].append(i)
                else:
                    d[val].append(i)

            self.users_d = d
            self.model_components = np.sort(
                np.asarray(list(d.keys()))
            )  # intial values, changed when fit_model is called
            self.seg_maps = {}
            self.cropped = False
            self.session_loaded = False
            self.primary_seg_map = None
        elif mode == "array":
            assert type(_in) == tuple
            self.iid = str(_in[0])
            self.k = None
            self.iid_idx = _in[0]
            self.im = import_utils.norm_im(_in[1])
            self.model_res = {}
            self.seg_maps = {}
            self.cropped = False
            self.session_loaded = False
            self.primary_seg_map = None
            self.gts = None
        else:
            print("Invalid initiation")

    def __repr__(self) -> str:
        a = "IID:{}\n".format(self.iid)
        b = "n_users:{}\n".format(len(self.k))
        c = "n_components : [user(s)]\n"
        d = "--------------------"
        print(c + d)
        print(self.users_d)
        if len(self.seg_maps.keys()) > 0:
            fit_to_model = True
        else:
            fit_to_model = False
        print("fit_to_model : {}".format(fit_to_model))
        if self.primary_seg_map is not None:
            pass
        else:
            print("Primary segmentation map not defined")
        print("cropped : {}".format(self.cropped))
        print("session loaded : {}".format(self.session_loaded))

        return a + b

    def make_grayscale(self, im):
        out = tb.rgb2gray(im)
        out = import_utils.norm_im(out)
        self.gs_im = out

        return out

    def fit_model(
        self,
        model="ac",
        n_components=None,
        max_components=10,
        layer_start=0,
        layer_stop=16,
        layer_step=1,
        binning=False,
        use_crop=False,
        use_grayscale=False,
        keep=False,
        init=None,
        init_eps=None
    ):
        """
        Runs perceptual segmentation model on self.im
        Models are defined in models_deep_seg.py

        Parameters:
        -----------
        model : str
            string of options for which model to run, options are 'a','b','c', or combinations
            default behavior is to run model 'a' and model 'c'
        n_components : np.array
            Array of how many components for the model to return.
            If input is [4,5] the model will generate one result with 4 components, and one result with 5 components.
        max_components : int
            Maximum allowed number of components
        layer_start, layer_stop, layer_step : int
            layers will be assigned to the self.seg_maps variable according to indexes: [layer_start, layer_stop, layer_step]
        binning : bool
            Determines whether output segmentation maps at shallow layers are artificially downsampled (binned)
        use_crop : bool
            Determines whether to run the segmentation on the cropped image or the uncropped image
        use_grayscale : bool
            Determines whether to run the segmentation on a grayscale image or the original image
        keep : bool
            Set to True to keep the segmentation maps from every iteration of the EM algorithm
        init : np.array 
            Array of shape(image height, image width), this is the initial guess during segmentation fitting 
        init_eps: float
            This is the amount of uncertainty injected with the initial guess, if None 0.0001 is used as default 


        Raises:
        ------
        self.model_res : ndarray
            Model object defined in models_deep_seg.py
        self.seg_maps : dict
        """
        if keep:
            assert model=="c", "Must use model \"c\" if keep is True"


        if n_components is not None:
            pass
        else:
            n_components = self.model_components

        # if 3 not in n_components:
        # n_components = np.append(n_components, 3)

        n_components = n_components[n_components < max_components]

        self.model_components = n_components

        if init is not None: 
            assert type(init)==np.ndarray
            assert model=="c", "Must use model \"c\" if init is not None"
            if init_eps is not None: 
                k = self.model_components[-1]
                assert init_eps < (1/2)*(1/(k-1)), "Initialization epsilon value is too high for ground truth"

        if use_crop:
            if not self.cropped:
                raise ("Use crop method before calling with use_crop=True")
            else:
                model_im = self.c_im
        else:
            model_im = self.im

        if use_grayscale:
            model_im = self.make_grayscale(model_im)
            model_im = import_utils.norm_im(model_im)

        if keep:
            # run model 'c'
            if "c" in model:
                self.model_res["c"], self._res_iter = seg._fit_model(
                    model_im,
                    model_type="c",
                    n_components=n_components,
                    layer=layer_stop,
                    keep=keep,
                    init=init,
                    init_eps=init_eps
                )
        else:
            # run model 'a'
            if "a" in model:
                self.model_res["a"] = seg._fit_model(
                    model_im,
                    model_type="a",
                    n_components=n_components,
                    layer=layer_stop
                )
            # run model 'b'
            if "b" in model:
                self.model_res["b"] = seg._fit_model(
                    model_im,
                    model_type="b",
                    n_components=n_components,
                    layer=layer_stop
                )
            # run model 'c'
            if "c" in model:
                self.model_res["c"] = seg._fit_model(
                    model_im,
                    model_type="c",
                    n_components=n_components,
                    layer=layer_stop
                )
        d = self.model_res

        # gen nested dictionary for seg maps
        #for key in d.keys():
            #self.seg_maps[key] = {}
            #n = d[key].shape[1]

            ## different values of i will have different n_components
            #for i in range(n):
                ## index 2 below is the index for the smm object
                #smm = d[key][0, i, 2, 0]
                #smm_last = d[key][-1, i, 2, 0]
                #Ny, Nx = smm.im_shape
                #_Ny, _Nx = smm_last.im_shape
                #self.seg_maps[key][smm.n_components] = []
                #layers = d[key][
                    #layer_start:layer_stop:layer_step, i, 2, 0
                #]  # generates seg map from every 4th layer

                #for layer in layers:
                    #ny, nx = layer.im_shape
                    #smap = layer.weights_.argmax(1).reshape((ny, nx))

                    #if binning == True:
                        #smap = tb._bin(smap, binsize=(ny // _Ny, nx // _Nx))

                        #assert Ny // ny == Nx // nx

                        #m = Ny // ny
                        #smap = smap.repeat(m, 0).repeat(m, 1)

                    #else:
                        #if ny != Ny:
                            #assert Ny // ny == Nx // nx
                            #multiplier = Ny // ny
                            #m = multiplier
                            #smap = smap.repeat(m, 0).repeat(m, 1)

                    #self.seg_maps[key][smm.n_components].append(smap)

            #if use_crop:
                #self.c_seg_maps = self.seg_maps

        return None

    def crop(
        self,
        spec={"y": (23, 278), "x": (23, 278)},
        size=(256, 256),
        center=True,
        RGB=True,
    ):
        """
        Crops an image, parameters specify different methods of cropping, based on toolbox.py -> crop

        In SegmentationMap object crops the following attributes:

        self.im -> self.c_im (RGB image from BSDS500)
        self.gts -> self.c_gts(list of ground truth annotated images from BSDS500)
        self.seg_map -> self.c_seg_map (nested dict of model results)

        Parameters:
        -----------
        spec : dict
            {y:(y1,y2),x:(x1,x2)}
        size : tup
            crop
        center : bool
            Determines whether size parameter is calculated from the center (True) or from the origin

        """
        if RGB:
            self.c_im = tb.crop_RGB(self.im, spec, size, center)
        else:
            self.c_im = tb.crop(self.im, spec, size, center)
        if self.gts is not None:
            self.c_gts = np.asarray(tb.crop(self.gts, spec, size, center))
        self.c_seg_maps = dict.fromkeys(self.seg_maps)
        d = self.c_seg_maps
        for key in d.keys():
            d[key] = dict.fromkeys(self.seg_maps[key])
            for _key in d[key].keys():
                to_crop = np.asarray(self.seg_maps[key][_key][:])
                d[key][_key] = np.asarray(tb.crop(to_crop, spec, size, center))

        self.cropped = True

        return None

    def set_primary_seg_map(self, gt=None, model="c", n_components=None, layer=0):
        """
        Sets the primary segmentation map for the segmentation map object

        Parameters:
        -----------
        model : str
            specifies which model to use as a key
        n_components : int
            number of components to use as a key
        layer : int
            layer INDEX to use as a key, index 0 means the 16th layer
            each proceeding index corresponds to 4 layers down

        Returns:
        --------
        None
        """
        if gt is not None:
            if self.cropped:
                maps = self.c_gts
            else:
                maps = self.gts
            if type(gt) == int:
                self.primary_seg_map = maps[gt]
            elif type(gt) == bool:
                if n_components is not None:
                    idx = self.users_d[n_components][0]
                    self.primary_seg_map = maps[idx]
        else:
            if self.cropped:
                maps = self.c_seg_maps
            else:
                maps = self.seg_maps
            if model is not None:
                if n_components is not None:
                    self.primary_seg_map = maps[model][n_components][layer]
            else:
                temp = sorted(self.k)
                median_idx = len(temp) // 2
                median_n_components = temp[median_idx]
                self.primary_seg_map = maps["c"][median_n_components][layer]
        if self.session_loaded:
            self.get_neural_data(probe=self.probe, full=False)
        else:
            pass

    def get_neural_data(
        self, Session=None, probe=DEFAULT_PROBES, exists=False, full=False
    ) -> None:
        """
        Get neural response data from Session object

        Parameters:
        -----------
        Session : Session object defined in Session.py
            default behavior is to use self.Session
            self.Session should exist before calling this method
        probe : int, optional
            None by default will use data from all probes, otherwise will use data from specified probe

        Returns:
        --------
        self.neural_d : dict
            quantities that exist for all neurons (eg. segments, correlations matrices) are in self.neural_d
        self.neural_df : pandas.DataFrame
            quantities that exist for pairs of neurons (eg. entry from correlation matrix) are in self.neural_df
        """
        if Session is not None:
            S = Session
            self.Session = Session
        else:
            S = self.Session
        if full and self.primary_seg_map is None:
            self.get_full_df()

    # Display functions: if self.cropped is True, then display the cropped image

    def disp(self, scale=(2, 2)):
        if self.cropped:
            im = self.c_im
        else:
            im = self.im
        tb.disp(im, scale=scale)

        return None

    def disp_seg_maps(self, model=None, n_components=None, layer=None):
        if self.cropped:
            maps = self.c_seg_maps
        else:
            maps = self.seg_maps

        if model is not None:
            if n_components is not None:
                if layer is not None:
                    tb.disp(maps[model][n_components][layer])
                else:
                    tb.disp(maps[model][n_components][:], shape=(2, 2))
        else:
            if self.primary_seg_map is not None:
                tb.disp(self.primary_seg_map)
            else:
                for key in maps.keys():
                    print("Model {}".format(key))
                    for _key in maps[key].keys():
                        print("n_components: {}".format(_key))
                        tb.disp(
                            maps[key][_key][:], shape=(2, 2)
                        )  # outputs maps at all 4 layers

    def disp_gts(self):
        if self.cropped:
            gts = self.c_gts
        else:
            gts = self.gts
        for gt in range(len(self.gts)):
            tb.disp(gts[gt, :, :])
        return None

    def disp_neuron(self, neuron_list, type="im", transform=True, matplotlib=True):
        """
        Parameters:
        -----------
        transform : bool, optional
            determines whether the transform is applied to the coordinates or not, by default True
        matplotlib : bool, optional
            matplotlib uses (x,y) coordinates while numpy uses (y,x) coordinates, so determines whether to use matplotlib or numpy, by default True
        """
        neuron_coords = [
            self.neural_d["coords"][neuron_num] for neuron_num in neuron_list
        ]
        if transform:
            np_coords = neuron_coords
        else:
            np_coords = [tb.transform_coord_system(coord) for coord in neuron_coords]
        # MATPLOTLIB uses (x,y) coordinates while numpy uses (y,x) coordinates, so we need to transpose just for plotting
        if type == "im":
            if self.cropped:
                im = self.c_im
            else:
                im = self.im
            tb.disp(im, scale=(2, 2), marker=np_coords)
        if type == "gt":
            if self.cropped:
                gts = self.c_gts
            else:
                gts = self.gts
            for i in range(len(gts)):
                tb.disp(gts[i, :, :], scale=(2, 2), marker=np_coords)
        if type == "seg_map":
            if self.cropped:
                maps = self.c_seg_maps
            else:
                maps = self.seg_maps
