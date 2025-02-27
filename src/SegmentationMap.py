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
from numpy.random import default_rng

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
            self._model_res = {}

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
            self._model_res = {}
            self.seg_maps = {}
            self.cropped = False
            self.session_loaded = False
            self.primary_seg_map = None
            self.gts = None
        else:
            print("Invalid initiation")

    def __repr__(self) -> str:
        a = "IID:{}\n".format(self.iid)
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

        return a

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
        layer_normalization=True,
        reshape_deep_layers=True,
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
        binning (depr) : bool
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
        spatial_smoothing : bool
            True if using spatial smoothing (2D Gaussian finite kernel)
            False if no spatial smoothing (2D Gaussian infinite kernel)
        deepnet : str
            determines which deep network is used for feature extraction, default is VGG19


        Raises:
        ------
        self.model_res : ndarray
            Model object defined in models_deep_seg.py
        self.seg_maps : dict
        self.flat_weights : ndarray
        self.weights_t : ndarray
        self.segmap : ndarray
        self.means_t : ndarray
        self.covars_t : ndarray
        self.degrees_t : ndarray
        self.responsibilites_t : ndarray
        self.likelihoods : ndarray
        self.flat_pca : ndarray
        self.data_pca : ndarray
        self.data : ndarray
        self.model_fitted: model Object
        """
        if deepnet is not None:
            # only the first layer of AlexNet can be used
            if deepnet == "AlexNet":
                layer_stop = 1
        else:
            # default deepnet is VGG19
            ny, nx = self.im.shape[:2]
            # layer sizes for VGG (due to pooling)
            self.N_list = np.array(
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
        if keep:
            assert model == "c", 'Must use model "c" if keep is True'

        if n_components is not None:
            pass
        else:
            n_components = self.model_components

        # if 3 not in n_components:
        # n_components = np.append(n_components, 3)

        n_components = n_components[n_components < max_components]

        # model_components is the number of components used by the model at runtime
        self.model_components = n_components

        if init is not None:
            # ensure that the initial guess is an array
            assert type(init) == np.ndarray
            # init option only defined for model 'c'
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
        if keep:
            # run model 'c' keep results at each EM iteration
            # for layers until layer_stop
            if "c" in model:
                self._model_res["c"], self.__res_iter = seg._fit_model(
                    model_im,
                    model_type="c",
                    n_components=n_components,
                    layer=layer_stop,
                    keep=keep,
                    init=init,
                    init_eps=init_eps,
                    spatial_smoothing=spatial_smoothing,
                    deepnet=deepnet,
                    layer_normalization=layer_normalization,
                    reshape_deep_layers=reshape_deep_layers,
                )

            make_array = lambda x: np.asarray([item for item in x if type(item) != int])
            # single_layer results here
            self.model_res = {}
            if layer_stop > 1:
                self._res_iter = self.__res_iter[:, layer_stop - 1, :, :]
                self.model_res["c"] = self._model_res["c"][layer_stop - 1, ...]
                # self.active_layer is initially layer_stop but is changed with parse_layer
                self.active_layer = layer_stop
                # self.layer_stop is a record of all layers that were fixed
                self.layer_stop = layer_stop
            else:
                self._res_iter = self.__res_iter
                self.model_res["c"] = self._model_res["c"]

            # parse attributes of the model
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

        else:
            # non-keep option only saves the last EM iteration
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
                    init=init,
                    deepnet=deepnet,
                )
        d = self.model_res
        return None

    def parse_layer(self, layer):
        """
        Selects which layer in the model to parse output from
        Parsing output means reshaping data per layer

        Parameters:
        -----------
        layer : int
            layers are 1-indexed
        """
        assert layer > 0, "layers are 1-indexed"
        make_array = lambda x: np.asarray([item for item in x if type(item) != int])
        # set layer of interest here
        self._res_iter = self.__res_iter[:, layer - 1, :, :]
        self.model_res["c"] = self._model_res["c"][layer - 1, ...]
        # recalculate/reshape model outputs depending on layer
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

        self.data = self._res_iter.T[7].squeeze()[0].reshape((*self.im.shape[:2], -1))
        self.model_fitted = self.model_res["c"].squeeze()[2]
        self.active_layer = layer

    def _get_collapsing_bounds(self, boundary):
        """
        Creates collapsing boundary function as a linear function of model
        likelihood

        Parameters:
        ------------
        boundary : array like
            first element is boundary at t=0, last element is boundary at t=T

        Returns:
        ---------
        bounds : list of arrays
        """
        shape = self.likelihoods.mean(axis=1)
        shape = shape / shape.max()
        multiplier = boundary[0] - boundary[1]

        upper_bound = boundary[0] - (multiplier * shape)
        lower_bound = -upper_bound

        bounds = [upper_bound, lower_bound]

        return bounds

    def _create_pseudocoords(self, coords, window=10, sample_size=10):
        """
        Creates a set of pseudocoords around specified coordinates

        Parameters:
        ------------
        coords : array
            the coordinates to create pseudo-coordinates around
        window : int
            the radius (pixels) of a square window around coords
        sample_size : int
            the number of pseudocoords to create

        Returns:
        -------
        """
        self.pseudocoords_sample_size = sample_size
        canvas = np.zeros(self.im.shape[:-1]).astype("int")

        # window around points
        win = window

        # array of pseudocoord indices
        all_pseudocoords = np.zeros((len(coords), ((2 * win) ** 2) - 1, 2), "int")
        pseudocoords = np.zeros((len(coords), sample_size, 2), dtype="int")
        pseudorts = np.zeros((len(coords), sample_size))
        for i, coord in enumerate(coords):
            # HANDLE EDGE OF ARRAY CASES
            _win = [win, win]
            if coord[0] - win < 0:
                _win[0] = coord[0]
            elif coord[0] + win > self.im.shape[0]:
                _win[0] = self.im.shape[0] - coord[0]
            else:
                _win[0] = win
            if coord[1] - win < 0:
                _win[1] = coord[1]
            elif coord[1] + win > self.im.shape[1]:
                _win[1] = self.im.shape[1] - coord[1]
            else:
                _win[1] = win

            canvas[
                coord[0] - _win[0] : coord[0] + _win[0],
                coord[1] - _win[1] : coord[1] + _win[1],
            ] = (
                i + 1
            )
            canvas[coord[0], coord[1]] = -(i + 1)

            # selects pseudocoords while EXCLUDING original coord

            in_window = np.argwhere(canvas == (i + 1)).astype("int")

            all_pseudocoords[i, : len(in_window), :] = in_window

            # select a random number of pseudocoords within the window
            rng = default_rng()
            samples = rng.choice(len(in_window), size=sample_size, replace=False)

            pseudocoords[i, ...] = all_pseudocoords[i, samples, :]

        # self.pseudocoords is a 2d array of pseudocoords (cols) per true coord (index)
        self.pseudocoords = pseudocoords

    # TODO: changed use_pseudocoords variable name to reflect that it is a window size
    def get_dynamic_map(
        self, coords, shape=None, use_pseudocoords=None, sample_size=10
    ):
        """
        Creates a pointwise proxy for reaction time for each coord in coords. If
        using pseudocoords then the output is the average over all pseudocoords
        per coord

        Parameters:
        ------------
        coords : array like
            list to points to calculate rt at
        shape : int, default is None
            shape of the final square map (int x int) (same as input image
            shape) if None, assume shape = sqrt(len(coords))
        use_pseudocoords : bool, default False
            the radius of the window size defining the window from which
            pseudcoords are drawn
        sample_size : int
            the number of pseudocoords being used

        Returns:
        ---------
            self.pointwise_rts : array
                flattened array of pointwise reaction times (t_{pointwise,i})
            self.dynamic_map : array
                reshaped square array
        """
        fpr = lambda x: dynamics.find_pointwise_rt(x, self)

        self.coords = coords
        if shape is not None:
            shape = shape
        else:
            shape = int(np.sqrt(len(coords)))
        # use pseudocoords for binning:
        if use_pseudocoords is not None:

            self.pseudo_pointwise_rts = np.asarray(
                [
                    dynamics.find_pointwise_rt(coord, self)
                    for coord in self.pseudocoords.reshape((-1, 2))
                ]
            ).reshape(self.pseudocoords.shape[:-1])

            # pseudorts_binned = np.mean(pseudorts, axis=1)

        self.pointwise_rts = np.asarray(
            [dynamics.find_pointwise_rt(coord, self) for coord in coords]
        )

        if use_pseudocoords is not None:
            temp_avg = (1 / sample_size) * self.pointwise_rts + (
                (sample_size - 1) / sample_size
            ) * np.mean(self.pseudo_pointwise_rts, axis=1)

        # Reshape pointwise rts to a dynamic map
        self.dynamic_map = temp_avg.reshape((shape, shape))

        return self.pointwise_rts

    def _process_pseudocoords(
        self,
        pair_idx,
        grids_idx,
        use_pointwise_rts=False,
        use_evidence_integration=False,
        weighted_evidence_integration=True,
        out="df",
    ):
        """
        Processes pseudocoords, creating a dataframe for pseudocoord results
        alone

        Parameters:
        ------------
        pair_idx : array
            a one-dimensional index indicating the order of the pairs shown. All
            pseudocoord pairs have the same pair_idx as the true coordinate pair
            that the pseudocoords are spread around
        grid_idx : array
            an index indicating which grid pair
        use_pointwise_rts : bool
            if True, include pointwise_rts in the DataFrame, (takes much longer
            to run)
        use_evidence_integration: bool
            includes evidence integation rts in the pseudocoord dataframe
        out : str
            "logits" : returns logits for all pseudcoord pairs at a particular
                grid pair
            "smooth_logits" : returns smoothed logits for all
                pseudcoord pairs at a particular grid pair
            "logit_derivs" : returns first derivative logits for all
                pseudcoord pairs at a particular grid pair
            "ei_logits" : return evidence integration logits
            "wei_logits": return weighted evidence integration logits
            "df" :returns the DataFrame

        Returns:
        ---------
        df : pd.DataFrame logits : array
        """
        d = {}
        assert hasattr(self, "pseudocoords")

        neigh_a = self.pseudocoords[grids_idx[0] - 1]
        neigh_b = self.pseudocoords[grids_idx[1] - 1]
        all_pairs = np.asarray([[a, b] for a in neigh_a for b in neigh_b])

        if use_pointwise_rts is not None:
            pointwise_rts_a = self.pseudo_pointwise_rts[grids_idx[0] - 1]
            pointwise_rts_b = self.pseudo_pointwise_rts[grids_idx[1] - 1]
            all_pointwise_rt_pairs = np.asarray(
                [[a, b] for a in pointwise_rts_a for b in pointwise_rts_b]
            )

        psames_t = np.asarray(
            [dynamics._get_psame_t(pair[0], pair[1], self) for pair in all_pairs]
        )
        sfs_t = np.asarray(
            [dynamics._get_seg_flag_t(pair[0], pair[1], self) for pair in all_pairs]
        )

        logits = np.asarray(
            [
                dynamics._get_logit(pair[0], pair[1], psame_t)
                for pair, psame_t in zip(all_pairs, psames_t)
            ]
        )

        smooth_logits = np.asarray(
            [dynamics.sliding_window_mean(logit, 3) for logit in logits]
        )

        logit_derivs = [
            np.abs(dynamics.sliding_window_deriv1(logit, 3)) for logit in self.logits
        ]

        if out == "logits":
            return logits
        if out == "smooth_logits":
            return smooth_logits
        if out == "logit_derivs":
            return logit_derivs

        if use_evidence_integration:
            num_samples = psames_t.shape[1]
            drift_rate_arr = self.drift_rate_arr
            ei_logits = np.asarray(
                [
                    dynamics._get_ei_logits(
                        coord[0],
                        coord[1],
                        drift_rate=drift,
                        sample_size=num_samples,
                    )
                    for coord, drift in zip(all_pairs, drift_rate_arr)
                ]
            )

            if out == "ei_logits":
                return ei_logits

            ei_rts = np.asarray(
                [
                    dynamics._get_decision_rt(evidence, boundary=self.boundary)[0]
                    for evidence in ei_logits
                ]
            )

            ei_responses = np.asarray(
                [
                    dynamics._get_decision_rt(evidence, boundary=self.boundary)[1]
                    for evidence in ei_logits
                ]
            )

            if weighted_evidence_integration:
                drift_rate_arr = logits[:, -1]
                wei_logits = np.asarray(
                    [
                        dynamics._get_ei_logits(
                            coord[0],
                            coord[1],
                            drift_rate=drift,
                            sample_size=num_samples,
                        )
                        for coord, drift in zip(all_pairs, drift_rate_arr)
                    ]
                )

                if out == "wei_logits":
                    return wei_logits

                wei_rts = np.asarray(
                    [
                        dynamics._get_decision_rt(evidence, boundary=self.boundary)[0]
                        for evidence in wei_logits
                    ]
                )

                wei_responses = np.asarray(
                    [
                        dynamics._get_decision_rt(evidence, boundary=self.boundary)[1]
                        for evidence in wei_logits
                    ]
                )

        if out == "df":
            rts = [
                dynamics._get_decision_rt(evidence, boundary=self.boundary)[0]
                for evidence in logits
            ]

            auto_rts = [
                dynamics._get_decision_rt(
                    evidence, deriv=d, boundary="auto", c=self.auto_mult
                )[0]
                for evidence, d in zip(smooth_logits, logit_derivs)
            ]

            responses = [
                dynamics._get_decision_rt(evidence, boundary=self.boundary)[1]
                for evidence in logits
            ]

            auto_responses = [
                dynamics._get_decision_rt(
                    evidence, deriv=d, boundary="auto", c=self.auto_mult
                )[1]
                for evidence, d in zip(smooth_logits, logit_derivs)
            ]

            d["pseudo"] = list(range(1, len(all_pairs) + 1))
            d["pair_idx"] = [pair_idx] * len(all_pairs)
            d["grid_idx_0"] = [grids_idx[0]] * len(all_pairs)
            d["grid_idx_1"] = [grids_idx[1]] * len(all_pairs)
            if use_pointwise_rts is not None:
                d["pointwise_rt_0"] = [
                    all_pointwise_rt_pairs[i, 0] for i in range(len(all_pairs))
                ]
                d["pointwise_rt_1"] = [
                    all_pointwise_rt_pairs[i, 1] for i in range(len(all_pairs))
                ]
            d["np_idx_0"] = [all_pairs[i, 0] for i in range(len(all_pairs))]
            d["np_idx_1"] = [all_pairs[i, 1] for i in range(len(all_pairs))]
            d["image_distance"] = [
                tb.euclidean_distance(all_pairs[i, 0], all_pairs[i, 1])
                for i in range(len(all_pairs))
            ]
            d["psame_rt"] = [
                item[int(auto_rts[i])] if auto_rts[i] != len(item) else item[-1]
                for i, item in enumerate(psames_t)
            ]
            d["psame_final"] = [item[-1] for item in psames_t]
            d["online_rt"] = rts
            d["online_response"] = responses
            d["auto_rt"] = auto_rts
            d["auto_response"] = auto_responses
            if use_evidence_integration:
                d["ei_rt"] = ei_rts
                d["ei_response"] = ei_responses
                if weighted_evidence_integration:
                    d["wei_rt"] = wei_rts
                    d["wei_response"] = wei_responses
            d["rt_avg"] = rts
            d["response_avg"] = responses
            d["seg_flag"] = list(sfs_t[:, -1])

            df = pd.DataFrame.from_dict(d)

            return df

    # def _pseudo_logits(self, pair_idx, grid_idx):

    # d = {}
    # d["pair_idx"] = 0
    # pseudologits = []

    # for coord in coords_idx:
    # neigh_a = self.pseudocoords[coord[0] - 1]
    # neigh_b = self.pseudocoords[coord[1] - 1]
    # all_pairs = [[a, b] for a in neigh_a for b in neigh_b]
    # logits = np.asarray(
    # [dynamics._get_evidence(pair[0], pair[1], self) for pair in all_pairs]
    # )
    # pseudologits.append(np.mean(logits, axis=0))

    # self.pseudologits = np.asarray(pseudologits)

    # return self.pseudologits

    def get_decision_rts(
        self,
        points,
        pairs,
        grids_idx,
        auto_mult=1,
        boundary={"const": [1, -1]},
        use_pointwise_rts=None,
        use_pseudocoords=None,
        use_evidence_integration=True,
        weighted_evidence_integration=True,
        return_responses=True,
        return_df=True,
    ):
        """
        Returns pairwise decision times: $\hat{t}_p^*

        Parameters:
        ------------
        points : array like
            the set of np coords where pointwise rts will be calculated
        pairs : array like
            the set of pairs (ie pairs of np coords) for which decisions exist
            a subset of all possible pairs
        grids_idx : array like
            the grid index for each pair, 1-indexed, not 0!
        boundary : dict {str:list}
            "const" : uses a constant bound positive bound first in list then negative
            "collapsing" : uses a collapsing bound based on model likelihood
        use_pointwise_rts : bool
            calculates pointwise rts if true
        use_pseudocoords : bool
            use evidence calculated from pseudocoords as well
        use_evidence_integration : bool
            use evidence integration sampling to return an rt
        weighted_eidence_integration : bool
            use the strength of the decision in the evidence integration
        """

        self.auto_mult = auto_mult
        coords = pairs
        grids_idx = grids_idx.astype("int")
        self.grids_idx = grids_idx

        if boundary is not None:
            assert type(boundary) == dict
            boundary_key = list(boundary.keys())[0]
            if list(boundary.keys())[0] == "const":
                boundary = boundary["const"]
            elif list(boundary.keys())[0] == "collapsing":
                boundary = self._get_collapsing_bounds(boundary["collapsing"])
            else:
                raise ValueError(
                    "Only acceptable boundary keys are const or collapsing"
                )
        else:
            boundary_key = None
            boundary = None

        self.boundary = boundary

        distances = np.asarray(
            [tb.euclidean_distance(coord[0], coord[1]) for coord in coords]
        )
        psames_t = np.asarray(
            [dynamics._get_psame_t(coord[0], coord[1], self) for coord in coords]
        )

        self.psames_t = psames_t

        sfs_t = np.asarray(
            [dynamics._get_seg_flag_t(coord[0], coord[1], self) for coord in coords]
        )
        self.logits = np.asarray(
            [
                dynamics._get_logit(coord[0], coord[1], psame_t)
                for coord, psame_t in zip(coords, psames_t)
            ]
        )

        self.smooth_logits = [
            dynamics.sliding_window_mean(logit, 3) for logit in self.logits
        ]

        self.logit_deriv = [
            np.abs(dynamics.sliding_window_deriv1(logit, 3)) for logit in self.logits
        ]

        if use_evidence_integration:
            num_samples = self.logits.shape[1]
            pos_drift_rate = np.mean(self.logits[:, -1][sfs_t[:, -1]])
            neg_drift_rate = np.mean(self.logits[:, -1][~sfs_t[:, -1]])
            self.global_drift_rate = [pos_drift_rate, neg_drift_rate]
            drift_rate_arr = np.zeros(self.logits[:, -1].shape)
            drift_rate_arr[sfs_t[:, -1]] += pos_drift_rate
            drift_rate_arr[~sfs_t[:, -1]] += neg_drift_rate
            self.drift_rate_arr = drift_rate_arr

            self.ei_logits = np.asarray(
                [
                    dynamics._get_ei_logits(
                        coord[0],
                        coord[1],
                        drift_rate=drift / num_samples,
                        sample_size=num_samples,
                    )
                    for coord, drift in zip(pairs, drift_rate_arr)
                ]
            )

            self.ei_rts = np.asarray(
                [
                    dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                    for evidence in self.ei_logits
                ]
            )

            if return_responses:
                self.ei_responses = np.asarray(
                    [
                        dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                        for evidence in self.ei_logits
                    ]
                )
            if weighted_evidence_integration:
                drift_rate_arr = self.logits[:, -1]
                self.wei_logits = np.asarray(
                    [
                        dynamics._get_ei_logits(
                            coord[0],
                            coord[1],
                            drift_rate=drift / num_samples,
                            sample_size=num_samples,
                        )
                        for coord, drift in zip(pairs, drift_rate_arr)
                    ]
                )

                self.wei_rts = np.asarray(
                    [
                        dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                        for evidence in self.wei_logits
                    ]
                )

                if return_responses:
                    self.wei_responses = np.asarray(
                        [
                            dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                            for evidence in self.wei_logits
                        ]
                    )
        if use_pseudocoords is not None:
            # self.pseudocoords and self.coords created here
            # default window size of 5
            self._create_pseudocoords(points, sample_size=use_pseudocoords)

            if use_pointwise_rts is not None:
                self.get_dynamic_map(
                    points, use_pseudocoords=10, sample_size=use_pseudocoords
                )

            self.pseudo_logits = np.asarray(
                [
                    self._process_pseudocoords(
                        i,
                        grids_idx[i],
                        use_pointwise_rts=use_pointwise_rts,
                        out="logits",
                    )
                    for i in range(len(grids_idx))
                ]
            )

            self.pseudo_logits_smooth = np.asarray(
                [
                    self._process_pseudocoords(
                        i,
                        grids_idx[i],
                        use_pointwise_rts=use_pointwise_rts,
                        out="smooth_logits",
                    )
                    for i in range(len(grids_idx))
                ]
            )

            self.pseudo_logits_deriv = np.asarray(
                [
                    self._process_pseudocoords(
                        i,
                        grids_idx[i],
                        use_pointwise_rts=use_pointwise_rts,
                        out="logit_derivs",
                    )
                    for i in range(len(grids_idx))
                ]
            )

            if not return_df:
                auto_rts_pseudo = np.asarray(
                    [
                        dynamics._get_decision_rt(
                            evidence, deriv=d, boundary="auto", c=auto_mult
                        )[0]
                        for evidence, d in zip(
                            self.pseudo_logits_smooth.reshape(
                                (-1, self.pseudo_logits_smooth.shape[-1])
                            ),
                            self.pseudo_logits_deriv.reshape(
                                (-1, self.pseudo_logits_deriv.shape[-1])
                            ),
                        )
                    ]
                ).reshape(self.pseudo_logits_smooth.shape[:-1])
                auto_rts_pseudo_mean = np.mean(auto_rts_pseudo, axis=1)
                # if return_responses:
                # auto_responses_pseudo = [
                # dynamics._get_decision_rt(
                # evidence, deriv=d, boundary="auto", c=auto_mult
                # )[1]
                # for evidence, d in zip(
                # self.pseudo_logits_smooth.reshape(
                # (-1, self.pseudo_logits_smooth.shape[-1])
                # ),
                # self.pseudo_logits_deriv.reshape(
                # (-1, self.pseudo_logits_deriv.shape[-1])
                # ),
                # )
                # ]

            if use_evidence_integration:
                self.pseudo_ei_logits = np.asarray(
                    [
                        self._process_pseudocoords(
                            i,
                            grids_idx[i],
                            use_pointwise_rts=use_pointwise_rts,
                            use_evidence_integration=True,
                            out="ei_logits",
                        )
                        for i in range(len(grids_idx))
                    ]
                )
                if weighted_evidence_integration:
                    self.pseudo_wei_logits = np.asarray(
                        [
                            self._process_pseudocoords(
                                i,
                                grids_idx[i],
                                use_pointwise_rts=use_pointwise_rts,
                                use_evidence_integration=True,
                                out="wei_logits",
                            )
                            for i in range(len(grids_idx))
                        ]
                    )

            # NOTE: commented out for now to make code faster
            # mean_logits = (
            # (self.pseudocoords_sample_size - 1) / self.pseudocoords_sample_size
            # ) * np.mean(self.pseudo_logits, axis=1) + (
            # 1 / self.pseudocoords_sample_size
            # ) * self.logits

            # self.mean_logits = mean_logits

        else:
            if use_pointwise_rts is not None:
                self.get_dynamic_map(points)

        if boundary is not None:
            rts = [
                dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                for evidence in self.logits
            ]

            if return_responses:
                responses = [
                    dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                    for evidence in self.logits
                ]

        auto_rts = [
            dynamics._get_decision_rt(evidence, deriv=d, boundary="auto", c=auto_mult)[
                0
            ]
            for evidence, d in zip(self.smooth_logits, self.logit_deriv)
        ]
        if return_responses:
            auto_responses = [
                dynamics._get_decision_rt(
                    evidence, deriv=d, boundary="auto", c=auto_mult
                )[1]
                for evidence, d in zip(self.smooth_logits, self.logit_deriv)
            ]

        # NOTE: commented out for now to make code faster
        # if use_pseudocoords is not None:
        # rts_avg = [
        # dynamics._get_decision_rt(evidence, boundary=boundary)[0]
        # for evidence in mean_logits
        # ]

        # responses_avg = [
        # dynamics._get_decision_rt(evidence, boundary=boundary)[1]
        # for evidence in mean_logits
        # ]
        if not return_df:
            pseudo_weight = self.pseudocoords_sample_size / (
                self.pseudocoords_sample_size + 1
            )
            auto_rts = (1 - pseudo_weight) * np.asarray(
                auto_rts
            ) + pseudo_weight * np.asarray(auto_rts_pseudo_mean)

            if return_responses:
                return auto_rts, auto_responses
            else:
                return auto_rts
        if return_df:
            self.pseudo_df = pd.concat(
                [
                    self._process_pseudocoords(
                        i,
                        grids_idx[i],
                        use_pointwise_rts=use_pointwise_rts,
                        use_evidence_integration=use_evidence_integration,
                        weighted_evidence_integration=weighted_evidence_integration,
                        out="df",
                    )
                    for i in range(len(grids_idx))
                ],
                ignore_index=True,
            )
            d = {}
            # CREATE DF FROM PARAMETERS
            if use_pseudocoords is not None:
                d["pseudo"] = [0] * len(grids_idx)
            d["pair_idx"] = list(range(len(grids_idx)))
            d["grid_idx_0"] = [item[0] for item in grids_idx]
            d["grid_idx_1"] = [item[1] for item in grids_idx]
            if use_pointwise_rts is not None:
                d["pointwise_rt_0"] = [
                    self.pointwise_rts[idx - 1] for idx, _ in grids_idx
                ]
                d["pointwise_rt_1"] = [
                    self.pointwise_rts[idx - 1] for _, idx in grids_idx
                ]
            d["np_idx_0"] = [item[0] for item in pairs]
            d["np_idx_1"] = [item[1] for item in pairs]
            d["image_distance"] = distances
            d["psame_rt"] = [
                item[int(auto_rts[i])] if rts[i] != len(item) else item[-1]
                for i, item in enumerate(psames_t)
            ]
            d["psame_final"] = [item[-1] for item in psames_t]
            if boundary is not None:
                d["online_rt"] = rts
                d["online_response"] = responses
            d["auto_rt"] = auto_rts
            d["auto_response"] = auto_responses
            # if use_pseudocoords is not None:
            # d["rt_avg"] = rts_avg
            # d["response_avg"] = responses_avg
            if use_evidence_integration:
                d["ei_rt"] = self.ei_rts
                d["ei_response"] = self.ei_responses
                if weighted_evidence_integration:
                    d["wei_rt"] = self.wei_rts
                    d["wei_response"] = self.wei_responses

            d["seg_flag"] = list(sfs_t[:, -1])

            df = pd.DataFrame.from_dict(d)

            if use_pseudocoords is not None:
                df_out = pd.concat([df, self.pseudo_df], axis=0, ignore_index=True)
            else:
                df_out = df

            df_out["bounds"] = boundary_key
            self.dynamics_pairwise_df = df_out
            return df_out

    def reapply_bounds(self, boundary, col=None, key=None):
        # TODO: evidence integration does not work with collapsing bounds

        if key is not None:
            key = key
        else:
            boundary_key = list(boundary.keys())[0]
        if boundary is not None:
            assert type(boundary) == dict
            if list(boundary.keys())[0] == "const":
                boundary = boundary["const"]
            elif list(boundary.keys())[0] == "collapsing":
                boundary = self._get_collapsing_bounds(boundary["collapsing"])
        else:
            boundary = [-1, 1]

        self.boundary = boundary

        if col is not None:

            df_new_bounds = self.dynamics_pairwise_df.copy()

            if col == "online_rt":
                rts = [
                    dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                    for evidence in self.logits
                ]
                responses = [
                    dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                    for evidence in self.logits
                ]
                if hasattr(self, "pseudocoords") and hasattr(self, "pseudo_logits"):
                    rts_pseudo = [
                        dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                        for evidence in self.pseudo_logits.reshape(
                            (-1, self.pseudo_logits.shape[-1])
                        )
                    ]

                    responses_pseudo = [
                        dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                        for evidence in self.pseudo_logits.reshape(
                            (-1, self.pseudo_logits.shape[-1])
                        )
                    ]

                    df_new_bounds[col] = rts + rts_pseudo
                    df_new_bounds["online_response"] = responses + responses_pseudo

                    return df_new_bounds
            elif col == "ei_rt":
                if hasattr(self, "ei_logits"):
                    ei_rts = [
                        dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                        for evidence in self.ei_logits
                    ]

                    ei_responses = [
                        dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                        for evidence in self.ei_logits
                    ]

                    if hasattr(self, "pseudo_ei_logits"):
                        ei_rts_pseudo = [
                            dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                            for evidence in self.pseudo_ei_logits.reshape(
                                (-1, self.pseudo_ei_logits.shape[-1])
                            )
                        ]

                        ei_responses_pseudo = [
                            dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                            for evidence in self.pseudo_ei_logits.reshape(
                                (-1, self.pseudo_ei_logits.shape[-1])
                            )
                        ]

                        df_new_bounds[col] = ei_rts + ei_rts_pseudo
                        df_new_bounds["ei_response"] = (
                            ei_responses + ei_responses_pseudo
                        )
                        return df_new_bounds
            elif col == "wei_rt":
                if hasattr(self, "wei_logits"):
                    wei_rts = [
                        dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                        for evidence in self.wei_logits
                    ]

                    wei_responses = [
                        dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                        for evidence in self.wei_logits
                    ]
                    if hasattr(self, "pseudo_wei_logits"):
                        wei_rts_pseudo = [
                            dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                            for evidence in self.pseudo_wei_logits.reshape(
                                (-1, self.pseudo_wei_logits.shape[-1])
                            )
                        ]

                        wei_responses_pseudo = [
                            dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                            for evidence in self.pseudo_wei_logits.reshape(
                                (-1, self.pseudo_wei_logits.shape[-1])
                            )
                        ]
                        df_new_bounds[col] = wei_rts + wei_rts_pseudo
                        df_new_bounds["wei_response"] = (
                            wei_responses + wei_responses_pseudo
                        )

                        return df_new_bounds

        else:
            rts = [
                dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                for evidence in self.logits
            ]
            responses = [
                dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                for evidence in self.logits
            ]

            rts_avg = [
                dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                for evidence in self.mean_logits
            ]
            responses_avg = [
                dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                for evidence in self.mean_logits
            ]

            if hasattr(self, "pseudocoords") and hasattr(self, "pseudo_logits"):
                rts_pseudo = [
                    dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                    for evidence in self.pseudo_logits.reshape(
                        (-1, self.pseudo_logits.shape[-1])
                    )
                ]

                responses_pseudo = [
                    dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                    for evidence in self.pseudo_logits.reshape(
                        (-1, self.pseudo_logits.shape[-1])
                    )
                ]

            if hasattr(self, "ei_logits"):
                ei_rts = [
                    dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                    for evidence in self.ei_logits
                ]

                ei_responses = [
                    dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                    for evidence in self.ei_logits
                ]

                if hasattr(self, "wei_logits"):
                    wei_rts = [
                        dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                        for evidence in self.wei_logits
                    ]

                    wei_responses = [
                        dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                        for evidence in self.wei_logits
                    ]

                if hasattr(self, "pseudo_ei_logits"):
                    ei_rts_pseudo = [
                        dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                        for evidence in self.pseudo_ei_logits.reshape(
                            (-1, self.pseudo_ei_logits.shape[-1])
                        )
                    ]

                    ei_responses_pseudo = [
                        dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                        for evidence in self.pseudo_ei_logits.reshape(
                            (-1, self.pseudo_ei_logits.shape[-1])
                        )
                    ]
                    if hasattr(self, "pseudo_wei_logits"):
                        wei_rts_pseudo = [
                            dynamics._get_decision_rt(evidence, boundary=boundary)[0]
                            for evidence in self.pseudo_wei_logits.reshape(
                                (-1, self.pseudo_wei_logits.shape[-1])
                            )
                        ]

                        wei_responses_pseudo = [
                            dynamics._get_decision_rt(evidence, boundary=boundary)[1]
                            for evidence in self.pseudo_wei_logits.reshape(
                                (-1, self.pseudo_wei_logits.shape[-1])
                            )
                        ]

            df_new_bounds = self.dynamics_pairwise_df.copy()
            df_new_bounds["online_rt"] = rts + rts_pseudo
            df_new_bounds["online_response"] = responses + responses_pseudo
            if hasattr(self, "ei_logits"):
                df_new_bounds["ei_rt"] = ei_rts + ei_rts_pseudo
                df_new_bounds["ei_response"] = ei_responses + ei_responses_pseudo
                if hasattr(self, "wei_logits"):
                    df_new_bounds["wei_rt"] = wei_rts + wei_rts_pseudo
                    df_new_bounds["wei_response"] = wei_responses + wei_responses_pseudo
            df_new_bounds["rt_avg"] = rts_avg + rts_pseudo
            df_new_bounds["response_avg"] = responses_avg + rts_pseudo
            df_new_bounds["bounds"] = [key] * (len(rts) + len(rts_pseudo))

            return df_new_bounds

    def reapply_automult(self, automult, return_df=True, return_responses=True):
        self.auto_mult = automult

        if return_df:
            df_new = self.dynamics_pairwise_df.copy()

        auto_rts = [
            dynamics._get_decision_rt(evidence, deriv=d, boundary="auto", c=automult)[0]
            for evidence, d in zip(self.smooth_logits, self.logit_deriv)
        ]
        if return_responses:
            auto_responses = [
                dynamics._get_decision_rt(
                    evidence, deriv=d, boundary="auto", c=automult
                )[1]
                for evidence, d in zip(self.smooth_logits, self.logit_deriv)
            ]

        if hasattr(self, "pseudocoords") and hasattr(self, "pseudo_logits"):
            if return_df:
                auto_rts_pseudo = [
                    dynamics._get_decision_rt(
                        evidence, deriv=d, boundary="auto", c=automult
                    )[0]
                    for evidence, d in zip(
                        self.pseudo_logits_smooth.reshape(
                            (-1, self.pseudo_logits_smooth.shape[-1])
                        ),
                        self.pseudo_logits_deriv.reshape(
                            (-1, self.pseudo_logits_deriv.shape[-1])
                        ),
                    )
                ]
            else:
                auto_rts_pseudo = np.asarray(
                    [
                        dynamics._get_decision_rt(
                            evidence, deriv=d, boundary="auto", c=automult
                        )[0]
                        for evidence, d in zip(
                            self.pseudo_logits_smooth.reshape(
                                (-1, self.pseudo_logits_smooth.shape[-1])
                            ),
                            self.pseudo_logits_deriv.reshape(
                                (-1, self.pseudo_logits_deriv.shape[-1])
                            ),
                        )
                    ]
                ).reshape(self.pseudo_logits_smooth.shape[:-1])
                auto_rts_pseudo_mean = np.mean(auto_rts_pseudo, axis=1)

            if return_responses:
                auto_responses_pseudo = [
                    dynamics._get_decision_rt(
                        evidence, deriv=d, boundary="auto", c=automult
                    )[1]
                    for evidence, d in zip(
                        self.pseudo_logits_smooth.reshape(
                            (-1, self.pseudo_logits_smooth.shape[-1])
                        ),
                        self.pseudo_logits_deriv.reshape(
                            (-1, self.pseudo_logits_deriv.shape[-1])
                        ),
                    )
                ]

        if return_df:
            df_new["auto_rt"] = auto_rts + auto_rts_pseudo
            df_new["auto_response"] = auto_responses + auto_responses_pseudo

            return df_new
        else:
            pseudo_weight = self.pseudocoords_sample_size / (
                self.pseudocoords_sample_size + 1
            )
            auto_rts = (1 - pseudo_weight) * np.asarray(
                auto_rts
            ) + pseudo_weight * np.asarray(auto_rts_pseudo_mean)

    def get_decision_rts_layers(
        self,
        points,
        pairs,
        coords_idx,
        layers=None,
        kernel_size=10,
        use_pointwise_rts=True,
        use_pseudocoords=None,
        boundary=None,
    ):
        layer_dfs = []
        if layers is not None:
            layers = layers
        else:
            layers = range(self.layer_stop)
        for layer in layers:
            assert layer in range(
                1, self.layer_stop + 1
            ), "layer {} was not fit".format(layer)
            self.parse_layer(layer)
            self.get_decision_rts(
                points,
                pairs,
                coords_idx,
                kernel_size=kernel_size,
                use_pointwise_rts=use_pointwise_rts,
                use_pseudocoords=use_pseudocoords,
                boundary=boundary,
            )

            layer_dfs.append(self.dynamics_df_pairwise)

        out = pd.concat(layer_dfs)

        self.ddf_pairwise_all_layers = out

        return self.ddf_pairwise_all_layers

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
