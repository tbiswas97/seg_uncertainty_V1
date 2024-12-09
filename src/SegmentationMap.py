import os
import import_utils
import toolbox as tb
import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
import seaborn as sns
import seg.segment as seg
import dynamics as dynamics
from numpy.lib.stride_tricks import sliding_window_view

# import analysis.dynamics as dynamics
from itertools import combinations
from Session import Session as Sess
from Session import DEFAULT_PROBES
import torchvision.models as models


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
        init_eps=None,
        spatial_smoothing=True,
        deepnet="vgg19",
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
        prior_weights: str
            determines the nature of the spatial smoothing
                "ext3" (default): uses Dirichlet hyperparameter
                None: no spatial smoothing
        deepnet : str
            determines which deep network is used for feature extraction, default is VGG19


        Raises:
        ------
        self.model_res : ndarray
            Model object defined in models_deep_seg.py
        self.seg_maps : dict
        """
        if deepnet is not None:
            if deepnet == "AlexNet":
                layer_stop = 1
        if keep:
            assert model == "c", 'Must use model "c" if keep is True'

        if n_components is not None:
            pass
        else:
            n_components = self.model_components

        # if 3 not in n_components:
        # n_components = np.append(n_components, 3)

        n_components = n_components[n_components < max_components]

        self.model_components = n_components

        if init is not None:
            assert type(init) == np.ndarray
            assert model == "c", 'Must use model "c" if init is not None'
            if init_eps is not None:
                k = self.model_components[-1]
                assert init_eps < (1 / 2) * (
                    1 / (k - 1)
                ), "Initialization epsilon value is too high for ground truth"

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

        # SEGMENTATION STEP:
        # calls files in seg/segment.py
        # TODO: put new arguments into model a and model c | keep=False
        if keep:
            assert model == "c"
            assert layer_stop == 1
            # run model 'c' keep results at each EM iteration
            if "c" in model:
                self.model_res["c"], self._res_iter = seg._fit_model(
                    model_im,
                    model_type="c",
                    n_components=n_components,
                    layer=layer_stop,
                    keep=keep,
                    init=init,
                    init_eps=init_eps,
                    spatial_smoothing=spatial_smoothing,
                    deepnet=deepnet,
                )

            make_array = lambda x: np.asarray([item for item in x if type(item) != int])
            weights = self._res_iter.T[0].squeeze()
            self.flat_weights = make_array(weights)
            self.weights_t = np.asarray(
                [
                    weight.reshape((*self.im.shape[:2], self.model_components[0]))
                    for weight in weights
                    if type(weight) != int
                ]
            )
            self.segmap = self.weights_t[-1, :, :, :].argmax(-1).astype("int")

            self.means_t = make_array(self._res_iter.T[1].squeeze())
            self.covars_t = make_array(self._res_iter.T[2].squeeze())
            self.degrees_t = make_array(self._res_iter.T[3].squeeze())

            responsibilities = self._res_iter.T[4].squeeze()
            self.responsibilities_t = np.asarray(
                [
                    resp.reshape((*self.im.shape[:2], self.model_components[0]))
                    for resp in responsibilities
                    if type(resp) != int
                ]
            )
            self.likelihoods = make_array(self._res_iter.T[5].squeeze())

            self.flat_pca = make_array(self._res_iter.T[6].squeeze()[0])
            self.data_pca = (
                self._res_iter.T[6].squeeze()[0].reshape((*self.im.shape[:2], -1))
            )

            self.data = (
                self._res_iter.T[7].squeeze()[0].reshape((*self.im.shape[:2], -1))
            )
            self.model_fitted = self.model_res["c"].squeeze()[2]
            self.test = self.model_fitted._posterior_proba(
                self.data_pca.reshape((-1, 6))
            )

        else:
            # run model 'a'
            if "a" in model:
                self.model_res["a"] = seg._fit_model(
                    model_im,
                    model_type="a",
                    n_components=n_components,
                    layer=layer_stop,
                    deepnet=deepnet,
                )
            # run model 'b'
            if "b" in model:
                self.model_res["b"] = seg._fit_model(
                    model_im,
                    model_type="b",
                    n_components=n_components,
                    layer=layer_stop,
                    deepnet=deepnet,
                )
            # run model 'c'
            if "c" in model:
                self.model_res["c"] = seg._fit_model(
                    model_im,
                    model_type="c",
                    n_components=n_components,
                    layer=layer_stop,
                    deepnet=deepnet,
                )
        d = self.model_res

        # CHANGED: used to build a nested dictionary in order to store data,
        # now use seg.segment._reshape_model_weights() instead
        # gen nested dictionary for seg maps
        # for key in d.keys():
        # self.seg_maps[key] = {}
        # n = d[key].shape[1]

        ## different values of i will have different n_components
        # for i in range(n):
        ## index 2 below is the index for the smm object
        # smm = d[key][0, i, 2, 0]
        # smm_last = d[key][-1, i, 2, 0]
        # Ny, Nx = smm.im_shape
        # _Ny, _Nx = smm_last.im_shape
        # self.seg_maps[key][smm.n_components] = []
        # layers = d[key][
        # layer_start:layer_stop:layer_step, i, 2, 0
        # ]  # generates seg map from every 4th layer

        # for layer in layers:
        # ny, nx = layer.im_shape
        # smap = layer.weights_.argmax(1).reshape((ny, nx))

        # if binning == True:
        # smap = tb._bin(smap, binsize=(ny // _Ny, nx // _Nx))

        # assert Ny // ny == Nx // nx

        # m = Ny // ny
        # smap = smap.repeat(m, 0).repeat(m, 1)

        # else:
        # if ny != Ny:
        # assert Ny // ny == Nx // nx
        # multiplier = Ny // ny
        # m = multiplier
        # smap = smap.repeat(m, 0).repeat(m, 1)

        # self.seg_maps[key][smm.n_components].append(smap)

        # if use_crop:
        # self.c_seg_maps = self.seg_maps

        return None

    # TODO: changed use_pseudocoords variable name to reflect that it is a window size
    def get_dynamic_map(
        self, coords, shape=None, use_pseudocoords=None, sample_size=10
    ):
        """
        coords : list to points to calculate rt at
        shape :

        """
        self.pseudocoords_sample_size = sample_size
        fpr = lambda x: dynamics.find_pointwise_rt(x, self)

        if shape is not None:
            shape = shape
        else:
            shape = int(np.sqrt(len(coords)))
        # use pseudocoords for binning:
        if use_pseudocoords is not None:
            canvas = np.zeros(self.im.shape[:-1])

            # window around points
            win = use_pseudocoords

            pseudocoords = np.zeros((len(coords), ((2 * win) ** 2) - 1, 2)).astype(
                "int"
            )
            pseudorts = np.zeros((len(coords), sample_size))
            for i, coord in enumerate(coords):
                canvas[
                    coord[0] - win : coord[0] + win, coord[1] - win : coord[1] + win
                ] = (i + 1)
                canvas[coord[0], coord[1]] = -(i + 1)

                pseudocoords[i, ...] = np.argwhere(canvas == (i + 1)).astype("int")

                samples = np.random.randint(0, (((2 * win) ** 2) - 1), size=sample_size)

                mask = np.ones((((2 * win) ** 2) - 1), dtype="bool")
                mask[samples] = False

                pseudocoords[i, mask, ...] = -1

                pseudorts[i, ...] = np.asarray(
                    [fpr(coord) for coord in pseudocoords[i, samples, ...]]
                )

            self.pseudocoords = pseudocoords
            pseudorts_binned = np.mean(pseudorts, axis=1)

        rts = [dynamics.find_pointwise_rt(coord, self) for coord in coords]

        if use_pseudocoords is not None:
            weight = 1 / (sample_size + 1)
            self.pointwise_rts = (
                weight * np.asarray(rts) + (1 - weight) * pseudorts_binned
            )
        else:
            self.pointwise_rts = np.asarray(rts)

        self.dynamic_map = self.pointwise_rts.reshape((shape, shape))

        return self.pointwise_rts

    def _pseudo_logits(self, coords_idx):

        pseudologits = []
        get_pos = lambda x: x[x > 0].reshape((-1, 2))

        for coord in coords_idx:
            neigh_a = get_pos(self.pseudocoords[coord[0] - 1])
            neigh_b = get_pos(self.pseudocoords[coord[1] - 1])
            all_pairs = [[a, b] for a in neigh_a for b in neigh_b]
            logits = np.asarray(
                [dynamics._get_evidence(pair[0], pair[1], self) for pair in all_pairs]
            )
            pseudologits.append(np.mean(logits, axis=0))

        self.pseudologits = np.asarray(pseudologits)

        return self.pseudologits

    def _pairwise_kernel_smooth_vs_distance(
        self, coords, coords_idx, kernel_size=10, use_pseudocoords=None
    ):
        distances = np.asarray(
            [tb.euclidean_distance(coord[0], coord[1]) for coord in coords]
        )
        sfs_t = np.asarray(
            [dynamics._get_seg_flag_t(coord[0], coord[1], self) for coord in coords]
        )
        logits = np.asarray(
            [dynamics._get_evidence(coord[0], coord[1], self) for coord in coords]
        )
        if use_pseudocoords is not None:
            pass
            weight = 1 / (self.pseudocoords_sample_size - 1)
            logits = weight * logits + (1 - weight) * self.pseudologits
        else:
            pass

        sfs_t = sfs_t.astype("bool")

        np_idxs = np.asarray(coords)

        np_idx_y = np_idxs[sfs_t[:, -1]]
        np_idx_n = np_idxs[~sfs_t[:, -1]]

        master_idx = np.asarray(list(range(len(coords_idx))))

        master_idx_y = master_idx[sfs_t[:, -1]]
        master_idx_n = master_idx[~sfs_t[:, -1]]

        coords_idx_y = coords_idx[sfs_t[:, -1]]
        coords_idx_n = coords_idx[~sfs_t[:, -1]]

        distances_y = distances[sfs_t[:, -1]]
        logits_y = logits[sfs_t[:, -1]]

        distances_n = distances[~sfs_t[:, -1]]
        logits_n = logits[~sfs_t[:, -1]]

        pairwise_rts = np.asarray(
            [
                np.array(
                    [
                        self.pointwise_rts[pair_idx[0] - 1],
                        self.pointwise_rts[pair_idx[1] - 1],
                    ]
                )
                for pair_idx in coords_idx
            ]
        )

        rts_y = pairwise_rts[sfs_t[:, -1]]
        rts_n = pairwise_rts[~sfs_t[:, -1]]

        rts = {"y": rts_y, "n": rts_n}

        distances = {"y": distances_y, "n": distances_n}
        logits = {"y": logits_y, "n": logits_n}

        if kernel_size is not None:
            for condition in ["y", "n"]:
                distances[condition + "s"] = np.asarray(
                    [
                        np.mean(temp)
                        for temp in sliding_window_view(
                            np.sort(distances[condition]), kernel_size
                        )
                    ]
                )

                logits[condition + "s"] = np.asarray(
                    [
                        np.mean(temp, axis=1)
                        for temp in sliding_window_view(
                            logits[condition][np.argsort(distances[condition])],
                            kernel_size,
                            axis=0,
                        )
                    ]
                )
                rts[condition + "s"] = np.asarray(
                    [
                        np.mean(temp, axis=1)
                        for temp in sliding_window_view(
                            rts[condition][np.argsort(distances[condition])],
                            kernel_size,
                            axis=0,
                        )
                    ]
                )

        self.pairwise_idxs = {"y": coords_idx_y, "n": coords_idx_n}
        self.master_idxs = {"y": master_idx_y, "n": master_idx_n}
        self.np_idxs = {"y": np_idx_y, "n": np_idx_n}
        self.pairwise_distances = distances
        self.pairwise_logits = logits
        self.pairwise_rts = rts

    def get_decision_rts(
        self,
        points,
        pairs,
        coords_idx,
        kernel_size=10,
        use_pointwise_rts=True,
        use_pseudocoords=None,
        boundary=None,
    ):

        coords = pairs
        if kernel_size is not None:
            kernel_size = kernel_size
            key = ["ys", "ns"]
        else:
            key = ["y", "n"]

        if boundary is not None:
            boundary = boundary
        else:
            boundary = [1, 2]

        if use_pseudocoords is not None:
            self.get_dynamic_map(
                points, use_pseudocoords=5, sample_size=use_pseudocoords
            )

            self._pseudo_logits(coords_idx)

            self._pairwise_kernel_smooth_vs_distance(
                coords, coords_idx, kernel_size, use_pseudocoords=True
            )

            pairwise_decision_rts_y = np.asarray(
                [
                    dynamics._get_decision_rt(
                        "yes", evidence, pointwise_rt=temp, boundary=boundary[0]
                    )
                    for evidence, temp in zip(
                        self.pairwise_logits["y"], self.pairwise_rts["y"]
                    )
                ]
            )

            pairwise_decision_rts_n = np.asarray(
                [
                    dynamics._get_decision_rt(
                        "no", evidence, pointwise_rt=temp, boundary=boundary[1]
                    )
                    for evidence, temp in zip(
                        self.pairwise_logits["n"], self.pairwise_rts["n"]
                    )
                ]
            )

            self.pairwise_decision_rts = {
                "y": pairwise_decision_rts_y,
                "n": pairwise_decision_rts_n,
            }

            boolean_decision = lambda x: 1 if x == "y" else 0

            dfs = []
            for _key in ["y", "n"]:
                d = {
                    "pair_idx": [idx for idx in self.master_idxs[_key]],
                    "grid_idx_0": [
                        grid_idx[0] for grid_idx in self.pairwise_idxs[_key]
                    ],
                    "grid_idx_1": [
                        grid_idx[1] for grid_idx in self.pairwise_idxs[_key]
                    ],
                    "np_idx_0": [np_idx[0] for np_idx in self.np_idxs[_key]],
                    "np_idx_1": [np_idx[1] for np_idx in self.np_idxs[_key]],
                    "image_distance": [dist for dist in self.pairwise_distances[_key]],
                    "model_rt": [rt for rt in self.pairwise_decision_rts[_key]],
                    "model_decision": [boolean_decision(_key)]
                    * len(self.master_idxs[_key]),
                }

                df = pd.DataFrame.from_dict(d)
                dfs.append(df)

            self.dynamics_df_pairwise = pd.concat(dfs, ignore_index=True)
        else:
            self.get_dynamic_map(points)

            self._pairwise_kernel_smooth_vs_distance(
                coords,
                coords_idx,
                kernel_size,
            )

        if use_pointwise_rts:
            decision_rts_y = np.asarray(
                [
                    dynamics._get_decision_rt(
                        "yes", evidence, pointwise_rt=temp, boundary=boundary[0]
                    )
                    for evidence, temp in zip(
                        self.pairwise_logits[key[0]], self.pairwise_rts[key[0]]
                    )
                ]
            )

            decision_rts_n = np.asarray(
                [
                    dynamics._get_decision_rt(
                        "no", evidence, pointwise_rt=temp, boundary=boundary[1]
                    )
                    for evidence, temp in zip(
                        self.pairwise_logits[key[1]], self.pairwise_rts[key[1]]
                    )
                ]
            )
        else:
            decision_rts_y = np.asarray(
                [
                    dynamics._get_decision_rt(
                        "yes", evidence, pointwise_rt=None, boundary=boundary[0]
                    )
                    for evidence, temp in zip(
                        self.pairwise_logits[key[0]], self.pairwise_rts[key[0]]
                    )
                ]
            )

            decision_rts_n = np.asarray(
                [
                    dynamics._get_decision_rt(
                        "no", evidence, pointwise_rt=None, boundary=boundary[1]
                    )
                    for evidence, temp in zip(
                        self.pairwise_logits[key[1]], self.pairwise_rts[key[1]]
                    )
                ]
            )

        self.decision_rts = {"y": decision_rts_y, "n": decision_rts_n}

        if kernel_size is not None:
            d = {
                "rt": [item for item in self.decision_rts["y"]]
                + [item for item in self.decision_rts["n"]],
                "distance": [item for item in self.pairwise_distances["ys"]]
                + [item for item in self.pairwise_distances["ns"]],
                "seg_flag": [1 for item in self.decision_rts["y"]]
                + [0 for item in self.decision_rts["n"]],
            }

            self.dynamics_df_distance = pd.DataFrame.from_dict(d)
        else:
            d = {
                "rt": [item for item in self.decision_rts["y"]]
                + [item for item in self.decision_rts["n"]],
                "distance": [item for item in self.pairwise_distances["y"]]
                + [item for item in self.pairwise_distances["n"]],
                "seg_flag": [1 for item in self.decision_rts["y"]]
                + [0 for item in self.decision_rts["n"]],
            }

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
