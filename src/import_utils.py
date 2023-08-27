from scipy.io import loadmat
import numpy as np
from PIL import Image
import os
import glob
import h5py
import pickle

# CHANGE EXP NAME HERE
# EXP_NAME = "EXP150_NatImages_NeuroPixels"
EXP_NAME = "Sessions_NaturalEnsemble_136"

# iids of images associated with experiment
if EXP_NAME == "EXP150_NatImages_NeuroPixels":
    IID_MAT_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(os.getcwd()), EXP_NAME, "EXP150_NatImages_Names.mat"
        )
    )
    SESSION_MAT_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(os.getcwd()), EXP_NAME, "EXP150_NatImages_Sessions.mat"
        )
    )
    temp = np.concatenate(loadmat(IID_MAT_PATH)["IMAGENAME"]).tolist()
    IIDS = [elem[0] for elem in temp]
    JPGS = [iid + ".jpg" for iid in IIDS]
    SEGS = [iid + ".seg" for iid in IIDS]  # BSDS500 uses .mat

elif EXP_NAME == "Sessions_NaturalEnsemble_136":
    SESSION_MAT_PATH = os.path.abspath(
        os.path.join(os.path.dirname(os.getcwd()), EXP_NAME, EXP_NAME + ".mat")
    )
    SESSION_IMS = os.path.abspath(
        os.path.join(os.path.dirname(os.getcwd()), EXP_NAME, EXP_NAME + "_images.pkl")
    )


def get_SESSION_MAT_PATH(EXP_NAME):
    """
    Use to extract Session information given EXP_NAME input.
    This is because different experiments are formatted differently. 
    Useful if you need change variables without re-importing
    """
    if EXP_NAME == "EXP150_NatImages_NeuroPixels":
        SESSION_MAT_PATH = os.path.abspath(
            os.path.join(
                os.path.dirname(os.getcwd()), EXP_NAME, "EXP150_NatImages_Sessions.mat"
            )
        )
    elif EXP_NAME == "Sessions_NaturalEnsemble_136":
        SESSION_MAT_PATH = os.path.abspath(
            os.path.join(os.path.dirname(os.getcwd()), EXP_NAME, EXP_NAME + ".mat")
        )

    return SESSION_MAT_PATH


# Paths to BSD data
JPG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.getcwd()),
        "data",
        "BSR",
        "BSDS500",
        "data",
        "images",
        "train",
    )
)
SEG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.getcwd()),
        "data",
        "BSR",
        "BSDS500",
        "data",
        "groundTruth",
        "train",
    )
)

Dataset = h5py._hl.dataset.Dataset
Group = h5py._hl.group.Group
Reference = h5py.h5r.Reference


def read_file_lines(filename):
    with open(filename, "r") as file:
        lines = [line.rstrip("\n") for line in file.readlines()]
    return lines


def norm_im(arr, check_even=True):
    """
    Normalizes image input into segmentation model
    """
    #fit model function can only handle 3-slice color arrays
    #so we make input grayscale images as a triplicate of the original

    ndim = arr.ndim
    if ndim != 3:
        if ndim == 2:
            a = np.expand_dims(arr,2)
            arr = np.repeat(a,3,2)
        else:
            raise("Check dimensions of input image")

    ny, nx = arr.shape[0:2]

    if check_even:
        if ny % 2 == 1:
            arr = arr[1:, :, :]
        if nx % 2 == 1:
            arr = arr[:, 1:, :]

    m = arr.min()
    M = arr.max()

    arr = (arr - m) / (M - m)

    return arr


def import_jpg(filename, check_even=True):
    img = Image.open(filename)
    arr = np.array(img)
    arr = norm_im(arr, check_even)

    return arr


def load_bsd_mat(seg_path, check_even=True):
    gt_path = seg_path
    stop = loadmat(gt_path)["groundTruth"].size
    arr = np.asarray(
        [loadmat(gt_path)["groundTruth"][:, k][0][0][0][0] for k in range(stop)]
    )

    ny, nx = arr.shape[1:]

    if check_even:
        if ny % 2 == 1:
            arr = arr[:, 1:, :]
        if nx % 2 == 1:
            arr = arr[:, :, 1:]

    return arr


# from https://github.com/rkp8000/loadmat_h5.git
def loadmat_h5(file_name):
    """Loadmat equivalent for -v7.3 or greater .mat files, which break scipy.io.loadmat"""

    def deref_s(s, f, verbose=False):  # dereference struct
        keys = [k for k in s.keys() if k != "#refs#"]

        if verbose:
            print(f"\nStruct, keys = {keys}")

        d = {}

        for k in keys:
            v = s[k]

            if isinstance(v, Group):  # struct
                d[k] = deref_s(v, f, verbose=verbose)
            elif isinstance(v, Dataset) and v.size == 0:  # empty dataset
                d[k] = np.zeros(v.shape)
            elif isinstance(v, Dataset) and isinstance(
                np.array(v).flat[0], Reference
            ):  # cell
                d[k] = deref_c(v, f, verbose=verbose)
            elif isinstance(v, Dataset) and v.dtype == "uint16":
                d[k] = "".join(np.array(v).view("S2").flatten().astype(str))
                if verbose:
                    print(f"String, chars = {d[k]}")
            elif isinstance(v, Dataset):  # numerical array
                d[k] = np.array(v).T
                if verbose:
                    print(f"Numerical array, shape = {d[k].shape}")

        return d

    def deref_c(c, f, verbose=False):  # dereference cell
        n_v = c.size
        shape = c.shape

        if verbose:
            print(f"\nCell, shape = {shape}")

        a = np.zeros(n_v, dtype="O")

        for i in range(n_v):
            v = f["#refs#"][np.array(c).flat[i]]

            if isinstance(v, Group):  # struct
                a[i] = deref_s(v, f, verbose=verbose)
            elif isinstance(v, Dataset) and v.size == 0:  # empty dataset
                d[k] = np.zeros(v.shape)
            elif isinstance(v, Dataset) and isinstance(
                np.array(v).flat[0], Reference
            ):  # cell
                a[i] = deref_c(v, f, verbose=verbose)
            elif isinstance(v, Dataset) and v.dtype == "uint16":
                a[i] = "".join(np.array(v).view("S2").flatten().astype(str))
                if verbose:
                    print(f"String, chars = {a[i]}")
            elif isinstance(v, Dataset):  # numerical array
                a[i] = np.array(v).T
                if verbose:
                    print(f"Numerical array, shape = {a[i].shape}")

        return a.reshape(shape).T

    with h5py.File(file_name, "r+") as f:
        d = deref_s(f, f)

    return d


def _pickle(object, filename):
    """
    To quickly serialize an object
    """
    with open(filename, "wb") as f:
        pickle.dump(object, f)


def _load(filename):
    """
    To quickly load a serialized object
    """
    with open(filename, "rb") as f:
        return pickle.load(f)
