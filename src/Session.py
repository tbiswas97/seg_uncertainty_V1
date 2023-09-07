from import_utils import loadmat_h5, _load
import import_utils
import toolbox as tb
import numpy as np
import os

DEFAULT_PROBES = None


class Session:
    """
    Initiate from session .mat file
    """

    def __init__(self, mat, type="utah"):
        """
        Parameters:
        -----------
        mat : str
            path to session.mat file
        probe : int
            if not None, use the given probe
        type : str
            "utah" : if session has Utah array recordings
            "neuropixel" : if session has neuropixel recordings
        """
        temp = import_utils.loadmat_h5(mat)
        self.__dict__ = temp["Session"]
        self.d = temp["Session"]
        self.fields = list(self.d.keys())
        # probes are 1-indexed NOT 0-indexed so we must subtract 1
        if type == "utah":
            self.use_probe(num=None)
        elif type == "neuropixel":
            print("use use_probe function to select one or multiple probes")

    def use_probe(self, num=1) -> None:
        if num is not None:
            if type(num) is list:
                probe_nums = [val - 1 for val in num]
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
                self.resp_large = np.concatenate(to_concat_resp_large, axis=0)
                self.resp_small = np.concatenate(to_concat_resp_small, axis=0)
                self.xy_coords = np.concatenate(to_concat_xy, axis=0)
                self.np_coords = self._get_neuron_np_coords()
            else:
                self.probe = num - 1
                self.n_trials = self.d["T"][0][self.probe][0][0]
                # data from all probes below
                self.resp_large = self.d["resp_large"][0][self.probe]
                self.resp_small = self.d["resp_small"][0][self.probe]
                self.xy_coords = self.d["XYch"][0][self.probe]
                self.np_coords = self._get_neuron_np_coords()
        else:  # for utah array data
            self.n_trials = self.d["T"][0][0]
            self.resp_large = self.d["resp_large"]
            self.resp_small = self.d["resp_small"]
            self.xy_coords = self.d["XYch"]
            self.n_neurons = len(self.xy_coords)
            self.np_coords = self._get_neuron_np_coords()

    def _get_neuron_np_coords(self, transform=True):
        """
        Extract (x,y) coordinate from Session data

        Parameters:
        -----------
        transform : bool
            if True, transforms neuron coordinates (where origin is center of image) to numpy readable coordinates (where origin is top left)

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
        """
        Extracts large and small image neural responses from a particular image

        Parameters:
        ------------
        img_idx : int
            Index of which image to extract data from
        """
        resp_large = self.resp_large[:, img_idx, :]
        resp_small = self.resp_small[:, img_idx, :]
        neuron_coords = self._get_neuron_np_coords()
        d = {
            "resp_large": resp_large,
            "resp_small": resp_small,
            "coords": neuron_coords,
        }
        return d

    def get_spike_train(self, neuron=None, image=None, trials=None, blkwindow=None):
        """
        Gets a spike train from a particular neuron for a particular image averaged across all trials

        Parameters:
        ------------

        neuron : int
            The index of a particular neuron, if None: returns values for all neurons
        image : int
            The index of a particular image, if None: returns values for all images
        trials : int
            The index of a particular trial, if None: returns values averaged across trials
        blkwindow : list-like
            The time step to consider as the "blank" response, which is appended to the end of the evoked response.
            if None: use the class variable self.blkwindow from Session data

        Returns:
        --------
        train : np.array
            The spike train for a particular neuron at a particular image, blank train is appended to the end of the evoked response train
        """
        if blkwindow is not None:
            blk_window_1 = blkwindow[0]
            blk_window_2 = blkwindow[1]
        else:
            blk_window_1 = int(self.blkwindow[0][0])
            blk_window_2 = int(self.blkwindow[0][1])
            stim_train_large = self.resp_train[:, 1, :, :, :]
            blk_train_large = self.resp_train_blk[:, ::2, :, blk_window_1:blk_window_2]

        if trials is not None:
            # TODO: incorporate sampling at a particular trial
            stim_train_large_avg_T = np.mean(stim_train_large, axis=-2)
            blk_train_large_avg_T = np.mean(blk_train_large, axis=-2)
        else:
            stim_train_large_avg_T = np.mean(stim_train_large, axis=-2)
            blk_train_large_avg_T = np.mean(blk_train_large, axis=-2)

        if neuron is not None and image is not None:
            stim_train = stim_train_large_avg_T[neuron, image, :]
            blk_train = blk_train_large_avg_T[neuron, image, :]
            train = np.concatenate((stim_train, blk_train), axis=-1)
        elif neuron is None and image is not None:
            stim_train = stim_train_large_avg_T[:, image, :]
            blk_train = blk_train_large_avg_T[:, image, :]

            train = np.concatenate((stim_train, blk_train), axis=-1)

        elif neuron is not None and image is None:
            stim_train = stim_train_large_avg_T[neuron, :, :]
            blk_train = blk_train_large_avg_T[neuron, :, :]

            train = np.concatenate((stim_train, blk_train), axis=-1)

        return train

    #decide if static methods should be part of Session class or go somehwere else?  
    @staticmethod
    def _get_spike_counts(self,train):
        a = train 
        out = (a/a[np.nonzero(a)]).min()

        return out 
    
    @staticmethod
    def _get_firing_rate(self,train,window):
        counts = self._get_spike_counts(train)
        t1,t2 = window 
        r = sum(counts[t1:t2+1])/(t2-t1)

        return r 