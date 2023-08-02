import os
import import_utils
import toolbox as tb
import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
import seaborn as sns
import segment as seg
from itertools import permutations
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

    def __init__(self, iid):
        self.iid = iid  # image id in BSDS500
        self.iid_idx = import_utils.IIDS.index(self.iid)
        self.jpg_path = os.path.abspath(
            os.path.join(import_utils.JPG_PATH, iid + ".jpg")
        )
        self.seg_path = os.path.abspath(
            os.path.join(import_utils.SEG_PATH, iid + ".mat")
        )
        self.im = import_utils.import_jpg(self.jpg_path)
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
        self.model_components = np.sort(np.asarray(list(d.keys())))
        self.seg_maps = {}
        self.cropped = False
        self.session_loaded = False
        self.primary_seg_map = None

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

    def fit_model(self, model="ac", n_components=None, max_components=10):
        """
        Runs perceptual segmentation model on self.im
        Models are defined in models_deep_seg.py

        Parameters:
        -----------
        model : str
            string of options for which model to run, options are 'a','b','c', or combinations
            default behavior is to run model 'a' and model 'c'

        Raises:
        ------
        self.model_res : ndarray
            Model object defined in models_deep_seg.py
        self.seg_maps : dict
        """
        if n_components is not None:
            pass
        else:
            n_components = self.model_components

        #if 3 not in n_components:
            #n_components = np.append(n_components, 3)

        n_components = n_components[n_components < max_components]

        self.model_components = n_components
        # run model 'a'
        if "a" in model:
            self.model_res["a"] = seg._fit_model(
                self.im, model_type="a", n_components=n_components
            )
        # run model 'b'
        if "b" in model:
            self.model_res["b"] = seg._fit_model(
                self.im, model_type="b", n_components=n_components
            )
        # run model 'c'
        if "c" in model:
            self.model_res["c"] = seg._fit_model(
                self.im, model_type="c", n_components=n_components
            )
        d = self.model_res

        # gen nested dictionary for seg maps
        for key in d.keys():
            self.seg_maps[key] = {}
            n = d[key].shape[1]

            for i in range(n):
                smm = d[key][0, i, 2, 0]
                Ny, Nx = smm.im_shape
                self.seg_maps[key][smm.n_components] = []
                layers = d[key][
                    0:16:4, i, 2, 0
                ]  # generates seg map from every 4th layer

                for layer in layers:
                    ny, nx = layer.im_shape
                    smap = layer.weights_.argmax(1).reshape((ny, nx))

                    if ny != Ny:
                        assert Ny // ny == Nx // nx
                        multiplier = Ny // ny
                        m = multiplier
                        smap = smap.repeat(m, 0).repeat(m, 1)

                    self.seg_maps[key][smm.n_components].append(smap)

        return None

    def crop(self, spec={"y": (23, 278), "x": (23, 278)}, size=(256, 256), center=True):
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
        self.c_im = tb.crop_RGB(self.im, spec, size, center)
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
            self.get_neural_data()
        else:
            pass

    def get_neural_data(
        self, Session=None, probe=DEFAULT_PROBES, norm=True, exists=False, full=True
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
            # TODO: implement default behavior for loading Session which directly loads from a directory

        if probe is not None:
            self.probe = probe
            S.use_probe(probe)
            # TODO implement case where probe is not specified
        else:
            self.probe = [1, 3, 4]
            S.use_probe(probe)

        d = S.get_image_data(self.iid_idx)

        d["segments"] = [
            tb.get_coord_segment(coord, self.primary_seg_map) for coord in d["coords"]
        ]

        z_norm = lambda x: (x - np.median(x))
        if norm:
            d["resp_large"] = z_norm(d["resp_large"])
            d["resp_small"] = z_norm(d["resp_small"])
        try:
            d["corr_mat_large"] = np.corrcoef(d["resp_large"])
            d["cov_mat_large"] = np.cov(d["resp_large"])
            d["corr_mat_small"] = np.corrcoef(d["resp_small"])
            d["cov_mat_small"] = np.cov(d["resp_small"])
        except RuntimeWarning:  # suppresses division by zero warning
            pass

        self.neural_d = d
        self.neural_df = self._make_neural_df()
        df = self.neural_df
        self.centered_df = df.loc[
            (df.neuron1_centered == True) | (df.neuron2_centered == True)
        ]
        self.session_loaded = True
        if full:
            self.get_full_df()

    def _make_neural_df(self) -> pd.DataFrame:
        coords = self.neural_d["coords"]
        total_neurons = len(coords)
        pairs = list(permutations(range(total_neurons), 2))

        z_norm = lambda x: (x - np.mean(x)) / (np.std(x))

        dd = {}
        dd["pairs"] = pairs
        dd["neuron1"] = [pair[0] for pair in pairs]
        dd["neuron1_centered"] = [tb.is_centered(coords[pair[0]]) for pair in pairs]
        dd["neuron2"] = [pair[1] for pair in pairs]
        dd["neuron2_centered"] = [tb.is_centered(coords[pair[1]]) for pair in pairs]
        dd["rsc_large"] = [self.neural_d["corr_mat_large"][pair] for pair in pairs]
        dd["cov_large"] = [self.neural_d["cov_mat_large"][pair] for pair in pairs]
        dd["rsc_small"] = [self.neural_d["corr_mat_small"][pair] for pair in pairs]
        dd["cov_small"] = [self.neural_d["cov_mat_small"][pair] for pair in pairs]
        diff_mat = self.neural_d["corr_mat_small"] - self.neural_d["corr_mat_large"]
        dd["delta_rsc"] = [diff_mat[pair] for pair in pairs]
        dd["delta_rsc_pdc"] = tb.calculate_percent_change(
            dd["rsc_small"], dd["rsc_large"]
        )
        dd["distance"] = [
            tb.euclidean_distance(coords[pair[0]], coords[pair[1]]) for pair in pairs
        ]
        dd["segment_1"] = [self.neural_d["segments"][pair[0]] for pair in pairs]
        dd["segment_2"] = [self.neural_d["segments"][pair[1]] for pair in pairs]
        df = pd.DataFrame.from_dict(dd)
        df["seg_flag"] = df["segment_1"] == df["segment_2"]
        df.insert(6, "fisher_rsc_large", np.arctanh(df["rsc_large"]))
        df.insert(7, "fisher_rsc_small", np.arctanh(df["rsc_small"]))

        return df

    def _make_condition_df(self, gt, model, n_components, layer) -> pd.DataFrame:
        self.set_primary_seg_map(
            gt=gt, model=model, n_components=n_components, layer=layer
        )
        df = self.neural_df
        df["gt"] = gt
        df["model"] = model
        df["n_components"] = n_components
        

        return df

    def get_full_df(self, Session=None, models=["c"]) -> pd.DataFrame:
        if Session is not None:
            S = Session
            self.Session = Session
        else:
            S = self.Session
            # TODO: implement default behavior for loading Session which directly loads from a directory
        
        self.session_loaded = True
        models = models
        n_components = self.model_components
        layers = range(4)

        to_concat = []
        for model in models:
            for k in n_components:
                for layer in layers:
                    to_concat.append(self._make_condition_df(None, model, k, layer))

        df1 = pd.concat(to_concat).reset_index(drop=True)

        n_components_gts = list(self.users_d.keys())

        gt_concat = []

        for k in n_components_gts:
            gt_concat.append(self._make_condition_df(True, None, k, None))

        df2 = pd.concat(gt_concat).reset_index(drop=True)

        out = pd.concat([df1, df2]).reset_index(drop=True)

        out["img_idx"] = self.iid_idx
        out["iid"] = self.iid
        self.full_df = out
        self.full_centered_df = out.loc[
            (out.neuron1_centered == True) | (out.neuron2_centered == True)
        ]

        return out

    def get_all_mwu(self, stat="delta_rsc_pdc", models=["c"], probes=[1, 3, 4]):
        # TODO: %cleanup - maybe delete this function?
        """
        Calculates a p-value for each segmentation map
        """
        models = models
        n_components = self.model_components
        layers = range(4)
        probes = probes

        out = []
        params = []
        for probe in probes:
            for model in models:
                for k in n_components:
                    for layer in layers:
                        out.append(
                            self._mwu_seg_map(
                                stat=stat,
                                gt=None,
                                model=model,
                                n_components=k,
                                layer=layer,
                                probe=probe,
                            )
                        )
                        params.append((model, k, layer, probe))

        # %TODO: implement for ground-truth data too

        return out, params

    def _mwu_seg_map(
        self,
        stat="delta_rsc_pdc",
        alternative="two-sided",
        gt=None,
        model="c",
        n_components=None,
        layer=0,
        probe=1,
    ):
        """
        Sets the mann-whitney-u p-value for same segment vs. different segment delta_rsc

        Parameters:
        -----------
        stat : str
            must be a field in self.neural_df
        alternative : str
            from scipy.stats.mannwhitneyu:
            'two-sided': the distributions are not equal, i.e. *F(u) ≠ G(u)* for at least one *u*.
            'less': the distribution underlying x is stochastically less than the distribution underlying y, i.e. *F(u) > G(u)* for all *u*.
            'greater': the distribution underlying x is stochastically greater than the distribution underlying y, i.e. *F(u) < G(u)* for all *u*.
        gt : bool
            if gt is True, the ground-truth annotated image specified by n_components is used to generate data
        model : str
            specifies which model to use as a key
        n_components : int
            number of components to use as a key
        layer : int
            layer INDEX to use as a key, index 0 means the 16th layer
            each proceeding index corresponds to 4 layers down
        probe : int
            specifies which probe to use to generate data

        Returns:
        --------
        None
        """
        self.probe = probe
        self.set_primary_seg_map(
            gt=gt, model=model, n_components=n_components, layer=layer
        )

        df = self.neural_df
        res = self._mwu_df(df, stat=stat, alternative=alternative)

        return res

    def _mwu_df(self, df, stat="delta_rsc_pdc", alternative="two-sided"):
        df = df
        if stat == "delta_rsc" or stat == "delta_rsc_pdc":
            if "neuron1_centered" in df.columns:
                if "neuron2_centered" in df.columns:
                    df = df.loc[
                        (df["neuron1_centered"] == True)
                        | (df["neuron2_centered"] == True)
                    ]
            x = df[stat][df.seg_flag == True].dropna().values
            y = df[stat][df.seg_flag == False].dropna().values
        elif stat == "z_norm_rsc_large":
            x = df[stat][df.seg_flag == True].dropna().values
            y = df[stat][df.seg_flag == False].dropna().values
        try:
            res = mannwhitneyu(x, y, alternative=alternative)
        except ValueError:
            res = (None, None)
            print("All neurons are in one segment")

        return res

    def _calculate_rsc(self, neuron1, neuron2, flag=True):
        """
        Calculates the noise correlation between two neurons for the given image
        Parameters:
        -----------
        neuron1 : int
            index of neuron 1
        neuron2 : int
            index of neuron 2
        flag : bool
            if True, returns seg_flag

        Returns:
        --------
        rsc_large : float
            Pearson correlation coeff for large image neural data
        rsc_small : float
            Pearson correlation coeff for small image neural data
        seg_flag : bool
            if True the two neurons are part of the same segment, otherwise False
        """
        rsc_large = self.neural_d["corr_mat_large"][neuron1, neuron2]
        rsc_small = self.neural_d["corr_mat_small"][neuron1, neuron2]
        seg_flag = (
            self.neural_d["segments"][neuron1] == self.neural_d["segments"][neuron2]
        )

        return rsc_large, rsc_small, rsc_large - rsc_small, seg_flag

    def check_session(self):
        """
        Checks if Session object is loaded
        """
        if self.session_loaded:
            return True
        else:
            return False

    def _get_neuron_segments(self, neuron_num):
        """
        Returns the segment of a given neuron

        Parameters:
        -----------
        neuron_num : int
            index of neuron to return segment for
        """
        if self.check_session():
            coords = self.neural_d["coords"]
        else:
            raise (RuntimeError("Session not loaded"))

        if self.primary_seg_map is not None:
            _map = self.primary_seg_map
        else:
            raise (AttributeError("Primary segmentation map is NoneType"))

        neuron_coord = coords[neuron_num]

        return tb.get_coord_segment(neuron_coord, _map)

    def get_neuron_segments(self) -> None:
        neuron_coords = self.Session.coords
        segments = []
        for neuron_num in range(len(neuron_coords)):
            segments.append(self._get_neuron_segments(neuron_num))
        self.neuron_segments = segments

    def plot_delta_rsc_dist(
        self, gt=None, model="c", n_components=None, layer=None, probe=None, type="kde"
    ):
        if self.primary_seg_map is not None:
            df = self.neural_df
        else:
            probe = self.probe
            self.Session.use_probe(probe)
            self.set_primary_seg_map(
                gt=gt, model=model, n_components=n_components, layer=layer
            )
            df = self.neural_df

        if type == "kde":
            sns.displot(
                df, x="delta_rsc", hue="seg_flag", kind="kde", bw_adjust=0.2, fill=True
            )
        else:
            sns.displot(
                df, x="delta_rsc", hue="seg_flag", stat="density", multiple="stack"
            )

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
            tb.disp(self.primary_seg_map, scale=(2, 2), marker=np_coords)
