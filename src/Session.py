from import_utils import loadmat_h5, _load
import import_utils
import toolbox as tb
import numpy as np
import os

DEFAULT_PROBES = [1,3,4]

class Session:
    """
    Initiate from session .mat file
    """

    def __init__(self, mat, exists=True):
        """
        Parameters:
        -----------
        mat : str
            path to session.mat file
        probe : int
            if not None, use the given probe
        exists : bool
            if True, load from session.pkl file if it exists
        """
        if exists:
            cd = os.path.abspath(os.path.dirname(os.getcwd()))
            path = os.path.join(cd, "in", "Session.pkl")
            if os.path.isfile(path):
                temp = import_utils._load(path)
        else:
            temp = import_utils.loadmat_h5(mat)
        # probes are 1-indexed NOT 0-indexed so we must subtract 1
        self.d = temp["Session"]
        self.fields = list(self.d.keys())

    def use_probe(self, num=1) -> None:
        if type(num) is list:
            probe_nums = [val-1 for val in num ]
            trials = []
            to_concat_resp_large = []
            to_concat_resp_small = []
            to_concat_xy = []
            
            for probe in probe_nums:
                trials.append(self.d["T"][0][probe][0][0])
                to_concat_resp_large.append(self.d["resp_large"][0][probe])
                to_concat_resp_small.append(self.d["resp_small"][0][probe])
                to_concat_xy.append(self.d["XYch"][0][probe])
            self.n_trials = trials
            self.resp_large = np.concatenate(to_concat_resp_large,axis=0)            
            self.resp_small = np.concatenate(to_concat_resp_small,axis=0)
            self.xy_coords = np.concatenate(to_concat_xy,axis=0)
            self.np_coords = self._get_neuron_np_coords()

        else:
            self.probe = num - 1
            self.n_trials = self.d["T"][0][self.probe][0][0]
            # data from all probes below
            self.resp_large = self.d["resp_large"][0][self.probe]
            self.resp_small = self.d["resp_small"][0][self.probe]
            self.xy_coords = self.d["XYch"][0][self.probe]
            self.np_coords = self._get_neuron_np_coords()

    def _get_neuron_np_coords(self, transform=True):
        """
        Extract (x,y) coordinate from Session data

        Parameters:
        -----------
        transform : bool
            if True, transforms neuron coordinates (where origin is center of image) to numpy readable coordinates

        Raises:
        ------
        self.coords : dict
            if self.probe is None:
                {'probe1': list of coords, 'probe2':...} for however many probes there are
        self.coords : list
            if self.probe:
                list of coords for the given probe
        """
        _transform = lambda x: tb.transform_coord_system(x) if transform else x
        coords_list = self.xy_coords
        coords = [_transform(item) for item in coords_list]

        return coords

    def get_image_data(self, img_idx):
        resp_large = self.resp_large[:, img_idx, :]
        resp_small = self.resp_small[:, img_idx, :]
        neuron_coords = self._get_neuron_np_coords()
        d = {
            "resp_large": resp_large,
            "resp_small": resp_small,
            "coords": neuron_coords,
        }
        return d
