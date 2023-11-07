""" 
Toolbox for image processing

License CC BY-NC-SA 4.0 : https://creativecommons.org/licenses/by-nc-sa/4.0/
Jonathan Vacher, November 2017

"""


from re import A
import decorator
import numpy as np
import numpy.fft as npfft
import scipy as sp
import scipy.special as spec
import scipy.misc as misc
import imageio as io
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage import laplace, gaussian_filter, gaussian_laplace
from scipy.stats import bootstrap
from natsort import natsorted as ns
from glob import glob as glob
from matplotlib import image
from collections import Counter
import math

from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics.cluster import contingency_matrix

# pixel per centimeter
ppcm = 65

FILEPATH = "/Users/tridibbiswas/Documents/einstein/rotation2/seg-model-example/BSR/BSDS500/data/"
THRESHOLD = 20


# utils
def mat2dict(mat):
    vals = mat
    keys = mat.dtype.descr
    # Assemble the keys and values into variables with the same name as that used in MATLAB
    keys_ = np.zeros(len(keys), dtype=object)
    vals_ = np.zeros(len(keys), dtype=object)
    for i in range(len(keys)):
        keys_[i] = keys[i][0]
        if np.squeeze(vals[keys_[i]]).dtype.str[:2] == "<U":
            vals_[i] = str(np.squeeze(vals[keys_[i]]))
            # squeeze is used to covert matlat (1,n) arrays into numpy (1,) arrays.
        else:
            vals_[i] = np.squeeze(vals[keys_[i]])
    dico = {keys_[i]: vals_[i] for i in range(len(keys))}
    return dico


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gauss(x, s):
    return np.exp(-(x**2) / (2 * s**2))


def gauss2d(x, y, s):
    return np.exp(-(x**2 + y**2) / (2 * s**2))


def gabor(g, th, sx, sy, rho, a, x, y):
    res = (
        g
        * np.exp(
            -1.0
            / (2.0 * (1.0 - rho**2))
            * (
                (x * np.cos(th) + y * np.sin(th)) ** 2 / sx**2
                + (-x * np.sin(th) + y * np.cos(th)) ** 2 / sy**2
                - 2.0
                * rho
                * (x * np.cos(th) + y * np.sin(th))
                * (-x * np.sin(th) + y * np.cos(th))
                / (sx * sy)
            )
        )
        * np.cos(2.0 * np.pi * a * (x * np.cos(th) + y * np.sin(th)))
    )
    return res


def shift(x):
    return npfft.fftshift(x, axes=(0, 1))


def ishift(x):
    return npfft.ifftshift(x, axes=(0, 1))


def ft(x):
    return npfft.fft2(x, axes=(0, 1), norm="ortho")


def ift(x):
    return npfft.ifft2(x, axes=(0, 1), norm="ortho").real


def iftc(x):
    return npfft.ifft2(x, axes=(0, 1), norm="ortho")


def ft2logmodshift(x):
    return np.log(np.abs(shift(x)))


# precision - recall for object and parts
def comp_pr_op(pred_class, true_classes):
    # constant
    NOT_CLASSIFIED = 1
    OBJECT = 2
    PART = 3
    # Value adjusted to find the results (F,P,R)=(0.556, 0.6724, 0.4739) on the BSD500
    # Results presented in Pont-Tuset, J., & Marques, F. (2013).
    # Measures and meta-measures for the supervised evaluation of image segmentation.
    object_threshold = 0.9139
    part_threshold = 0.25
    beta = 0.1

    # init
    num_reg_pred = np.unique(pred_class).shape[0]
    classification_pred = np.ones(num_reg_pred, dtype=np.int32)
    prec_pred = np.zeros(num_reg_pred)
    num_reg_true = []
    classification_true = []
    recall_true = []

    n_true = true_classes.shape[0]
    # print(n_true)
    # contingency matrix to compute precision recall
    cont_mat = []
    for k in range(n_true):
        cont_mat.append(contingency_matrix(pred_class, true_classes[k]))
        num_reg_true.append(cont_mat[k].shape[1])
        classification_true.append(np.ones(num_reg_true[k], dtype=np.int32))
        recall_true.append(np.zeros(num_reg_true[k]))

    # percentile threshold to exclude small regions
    area_percent = 0.99

    # areas in both map and total area
    region_areas_pred = cont_mat[0].sum(axis=1)
    region_areas_true = []
    image_area = cont_mat[0].sum()

    # mask to exclude small regions
    area = region_areas_pred / image_area
    area_sort_ind = np.argsort(area)[::-1]

    candidate_pred = np.zeros(area.shape[0], dtype=bool)
    candidate_pred[area_sort_ind[(area[area_sort_ind].cumsum() < area_percent)]] = True
    if candidate_pred.sum() == 0:
        candidate_pred[candidate_pred.argmax()] = True

    candidate_true = []
    for k in range(n_true):
        region_areas_true.append(cont_mat[k].sum(axis=0))
        area = region_areas_true[k] / image_area
        area_sort_ind = np.argsort(area)[::-1]

        cand_true = np.zeros(area.shape[0], dtype=bool)
        cand_true[area_sort_ind[(area[area_sort_ind].cumsum() < area_percent)]] = True
        candidate_true.append(cand_true)
        if candidate_true[k].sum() == 0:
            candidate_true[k][candidate_true[k].argmax()] = True
            # np.ones(area.shape[0], dtype=bool)
            # print(candidate_true[k])

    for k in range(n_true):
        # can be done efficiently using index-wise operation
        for i in range(num_reg_true[k]):
            for j in range(num_reg_pred):
                recall = cont_mat[k][j, i] / region_areas_true[k][i]
                precision = cont_mat[k][j, i] / region_areas_pred[j]

                # exclude small regions
                if candidate_true[k][i] == True and candidate_pred[j] == True:
                    # classify regions as object or part
                    if (recall >= object_threshold) and (precision >= object_threshold):
                        classification_true[k][i] = OBJECT
                        classification_pred[j] = OBJECT
                        # mapping_gt[i] = j
                        # mapping_part[j] = i
                    elif (recall >= part_threshold) and (precision >= object_threshold):
                        if classification_pred[j] == NOT_CLASSIFIED:
                            classification_pred[j] = PART
                            # mapping_part[j] = i
                    elif (recall >= object_threshold) and (precision >= part_threshold):
                        # Cannot have a classification already */
                        classification_true[k][i] = PART
                        # mapping_gt[i] = j

                # Get recall_true and prec_pred (no matter if candidates or not), discarding objects
                if (precision >= object_threshold) and (recall < object_threshold):
                    recall_true[k][i] += recall
                elif (recall >= object_threshold) and (precision < object_threshold):
                    prec_pred[j] += precision

    num_objects_pred = 0
    num_objects_true = 0
    num_parts_pred = 0
    num_parts_true = 0
    num_underseg_pred = 0
    num_overseg_true = 0
    num_candidates_pred = 0
    num_candidates_true = 0

    for j in range(num_reg_pred):
        num_candidates_pred += candidate_pred[j]

        if classification_pred[j] == PART:
            num_parts_pred += 1
        elif classification_pred[j] == OBJECT:
            num_objects_pred += 1
        elif candidate_pred[j]:  # Compute degree of undersegmentation
            num_underseg_pred += prec_pred[j]

    num_underseg_pred /= n_true

    for k in range(n_true):
        for i in range(num_reg_true[k]):
            num_candidates_true += candidate_true[k][i]
            if classification_true[k][i] == PART:
                num_parts_true += 1
            elif classification_true[k][i] == OBJECT:
                num_objects_true += 1
            elif candidate_true[k][i]:
                num_overseg_true += recall_true[k][i]

    # Precision and recall
    precision = (
        num_objects_pred + num_underseg_pred + beta * num_parts_pred
    ) / num_candidates_pred
    recall = (
        num_objects_true + num_overseg_true + beta * num_parts_true
    ) / num_candidates_true

    # F-measure for Region Detection
    if precision == 0 and recall == 0:
        f_measure = np.array([0, 0, 0])
    else:
        f_measure = np.array(
            [2 * precision * recall / (precision + recall), precision, recall]
        )

    return f_measure


# rand index
def rand_index_score(clusters, classes):
    tp_plus_fp = spec.comb(np.bincount(np.int64(clusters)), 2).sum()
    tp_plus_fn = spec.comb(np.bincount(np.int64(classes)), 2).sum()
    A = np.c_[(clusters, classes)]
    A = np.int64(A)
    tp = sum(
        spec.comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
        for i in set(np.int64(clusters))
    )
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = spec.comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def get_contours(im, rm_borders=False):
    res = gaussian_filter(np.float32(laplace(im, mode="wrap") > 0), sigma=0.1) > 0.5
    if rm_borders:
        res[-1, :] = 0
        res[0, :] = 0
        res[:, -1] = 0
        res[:, 0] = 0
    return res


# computing precision recall values
def comp_pr(b1, b2, h, w, thres):
    cnt = 0
    for i in range(h):
        for j in range(w):
            if b1[i][j]:
                lower_x = max(0, i - thres)
                upper_x = min(h - 1, i + thres)
                lower_y = max(0, j - thres)
                upper_y = min(w - 1, j + thres)
                matrix_rows = b2[lower_x : upper_x + 1, :]
                matrix = matrix_rows[:, lower_y : upper_y + 1]
                if matrix.sum() > 0:
                    cnt = cnt + 1
    total = np.float32(b1.sum())
    return cnt / total


# computing precision recall values for contours
def comp_pr_c(mask1, mask2, thres):
    """Evaluate precision for boundary detection"""
    s1 = mask1.shape
    s2 = mask2.shape

    if s1 != s2:
        print("shape not match")
        return -1, -1
    if len(s1) == 3:
        b1 = mask1.reshape(s1[0], s1[1]) == 0
        b2 = mask2.reshape(s2[0], s2[1]) == 0
    else:
        b1 = mask1 == 0
        b2 = mask2 == 0

    h = s1[0]
    w = s1[1]
    precision = comp_pr(b1, b2, h, w, thres)
    recall = comp_pr(b2, b1, h, w, thres)
    return np.array(
        [2 * precision * recall / (precision + recall + 1e-13), precision, recall]
    )


# utils arithmetic
def get_prime_factors(number):
    """
    Return prime factor list for a given number
        number - an integer number
        Example: get_prime_factors(8) --> [2, 2, 2].
    """
    if number == 1:
        return []

    # We have to begin with 2 instead of 1 or 0
    # to avoid the calls infinite or the division by 0
    for i in range(2, number):
        # Get remainder and quotient
        rd, qt = divmod(number, i)
        if not qt:  # if equal to zero
            return [i] + get_prime_factors(rd)

    return [number]


# display image
def disp(imList, shape=(1, 1), scale=(1, 1), marker=None, np_coords=True):
    """
    Displays a list of images in a figure. Shape must match number of images in list.
    Parameters:
    -----------
    imList : list or arr
    shape : tuple
    scale : tuple
    marker : tuple or None
        used to display markers on top of an image
    np_coords : bool
        only if marker is not None, deicdes whether to use numpy indexing or matplotlib indexing when plotting marker

    """
    fig, ax = plt.subplots(
        nrows=shape[0],
        ncols=shape[1],
        figsize=(scale[1] * 3 * shape[0], scale[0] * 3 * shape[1]),
    )
    if shape[0] == 1 and shape[1] == 1:
        ax.imshow(imList, cmap="gray")
        if marker is not None:
            if type(marker) == list:
                z = np.random.rand(len(marker))
                for i, item in enumerate(marker):
                    if np_coords:
                        item = item[::-1]
                    ax.scatter(item[0], item[1], marker="o", s=50, c=z[i], cmap="hsv")
            else:
                if np_coords:
                    marker = marker[::-1]
                ax.scatter(marker[0], marker[1], marker="o", s=50, c="r")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    elif shape[0] == 1 or shape[1] == 1:
        for i in range(np.max(np.array(shape))):
            ax[i].imshow(imList[i], cmap="gray")
            ax[i].xaxis.set_visible(False)
            ax[i].yaxis.set_visible(False)
    else:
        k = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                ax[i, j].imshow(imList[k], cmap="gray")
                ax[i, j].xaxis.set_visible(False)
                ax[i, j].yaxis.set_visible(False)
                k += 1
    return fig, ax


# steerable pyramid
def low_filter(r):
    return np.complex64(
        (r <= np.pi / 4)
        + (r > np.pi / 4)
        * (r < np.pi / 2)
        * np.cos(np.pi / 2 * (np.log(4 * r / np.pi) / np.log(2)))
    )


def high_filter(r):
    return np.complex64(
        (r >= np.pi / 2)
        + (r > np.pi / 4)
        * (r < np.pi / 2)
        * np.cos(np.pi / 2 * (np.log(2 * r / np.pi) / np.log(2)))
    )


def steer_filter(t, k, n):
    alpha = (
        2 ** (n - 1)
        * spec.factorial(n - 1)
        / (n * spec.factorial(2 * (n - 1))) ** (0.5)
    )
    return np.complex64(
        (alpha * np.cos(t - np.pi * k / n) ** (n - 1))
        * (np.abs(np.mod(t + np.pi - np.pi * k / n, 2 * np.pi) - np.pi) < np.pi / 2)
    )


# def mask(t,k,n):
#     return np.complex64(2.0*(np.abs(np.mod(t+np.pi-np.pi*k/n,2*np.pi)-np.pi)<np.pi/2) \
#     + 1.0*(np.abs(np.mod(t+np.pi-np.pi*k/n,2*np.pi)-np.pi)==np.pi/2))


def mask(t, t0):
    return np.complex64(
        2.0 * (np.abs(np.mod(t - t0, 2 * np.pi) - np.pi) < np.pi / 2)
        + 1.0 * (np.abs(np.mod(t - t0, 2 * np.pi) - np.pi) == np.pi / 2)
    )


# pyramid decomposition
def build_pyr(img, Ns, Np, upSamp=0, cplx=1, freq=0):
    M = img.shape[0]
    N = img.shape[1]
    Fimg = ft(img)
    imgL = np.empty(Np + 1, dtype=object)
    imgH = np.empty(Np * Ns + 2, dtype=object)

    Lx = np.concatenate(
        (np.linspace(0, N // 2 - 1, N // 2), np.linspace(-N // 2, -1, N // 2))
    )
    Ly = np.concatenate(
        (np.linspace(0, M // 2 - 1, M // 2), np.linspace(-M // 2, -1, M // 2))
    )
    lx = np.concatenate(
        (np.linspace(0, N // 4 - 1, N // 4), np.linspace(-N // 4, -1, N // 4))
    )
    ly = np.concatenate(
        (np.linspace(0, M // 4 - 1, M // 4), np.linspace(-M // 4, -1, M // 4))
    )

    X, Y = np.meshgrid(Lx, Ly)
    x, y = np.meshgrid(lx, ly)

    R = np.sqrt((X * 2 * np.pi / N) ** 2 + (Y * 2 * np.pi / M) ** 2)
    R[0, 0] = 10 ** (-16)
    T = np.arctan2(Y, X)

    if Fimg.ndim == 3:
        R = R[:, :, np.newaxis]
        T = T[:, :, np.newaxis]

    L0 = low_filter(R / 2.0)
    H0 = high_filter(R / 2.0)

    L = low_filter(R)
    H = high_filter(R)

    low = L0 * Fimg
    imgL[0] = low

    imgH[0] = H0 * Fimg

    for i in range(Np):
        if i == 0:
            for k in range(Ns):
                imgH[1 + Ns * i + k] = 2.0 * steer_filter(T, k, Ns) * H * low
        else:
            if upSamp == 0:
                for k in range(Ns):
                    imgH[1 + Ns * i + k] = 2.0 * steer_filter(T, k, Ns) * H * low
            else:
                extx = np.int32((Fimg.shape[1] - N) // 2)
                exty = np.int32((Fimg.shape[0] - M) // 2)

                for k in range(Ns):
                    if Fimg.ndim == 3:
                        imgH[1 + Ns * i + k] = 2**i * shift(
                            np.pad(
                                shift(2.0 * steer_filter(T, k, Ns) * H * low),
                                pad_width=((exty, exty), (extx, extx), (0, 0)),
                                mode="constant",
                            )
                        )
                    else:
                        imgH[1 + Ns * i + k] = 2**i * shift(
                            np.pad(
                                shift(2.0 * steer_filter(T, k, Ns) * H * low),
                                pad_width=((exty, exty), (extx, extx)),
                                mode="constant",
                            )
                        )

        low = L * low
        if upSamp == 0:
            if i < Np - 1:
                low = low[np.int64(y), np.int64(x)] / 2.0
            imgL[1 + i] = low
        else:
            low = low[np.int64(y), np.int64(x)] / 2.0
            extx = np.int32((Fimg.shape[1] - N // 2) // 2)
            exty = np.int32((Fimg.shape[0] - M // 2) // 2)
            if Fimg.ndim == 3:
                imgL[1 + i] = 2 ** (i + 1) * shift(
                    np.pad(
                        shift(low),
                        pad_width=((exty, exty), (extx, extx), (0, 0)),
                        mode="constant",
                    )
                )
            else:
                imgL[1 + i] = 2 ** (i + 1) * shift(
                    np.pad(
                        shift(low),
                        pad_width=((exty, exty), (extx, extx)),
                        mode="constant",
                    )
                )

        M = M // 2
        N = N // 2

        Lx = np.concatenate(
            (np.linspace(0, N // 2 - 1, N // 2), np.linspace(-N // 2, -1, N // 2))
        )
        Ly = np.concatenate(
            (np.linspace(0, M // 2 - 1, M // 2), np.linspace(-M // 2, -1, M // 2))
        )
        lx = np.concatenate(
            (np.linspace(0, N // 4 - 1, N // 4), np.linspace(-N // 4, -1, N // 4))
        )
        ly = np.concatenate(
            (np.linspace(0, M // 4 - 1, M // 4), np.linspace(-M // 4, -1, M // 4))
        )

        X, Y = np.meshgrid(Lx, Ly)
        x, y = np.meshgrid(lx, ly)

        R = np.sqrt((X * 2 * np.pi / N) ** 2 + (Y * 2 * np.pi / M) ** 2)
        R[0, 0] = 10 ** (-16)
        T = np.arctan2(Y, X)

        if Fimg.ndim == 3:
            R = R[:, :, np.newaxis]
            T = T[:, :, np.newaxis]

        L = low_filter(R)
        H = high_filter(R)

    imgH[Np * Ns + 1] = imgL[Np]

    if freq == 0:
        for i in range(Np + 1):
            imgL[i] = ift(imgL[i])
        for i in range(Ns * Np + 2):
            if cplx == 0:
                imgH[i] = iftc(imgH[i]).real
            else:
                imgH[i] = iftc(imgH[i])

    return imgL, imgH


# pyramid reconstruction
def collapse_pyr(imH, Ns, Np, downSamp=0, freq=0):
    if freq == 0:
        imgH = np.empty(Np * Ns + 2, dtype=object)
        for i in range(Ns * Np + 2):
            imgH[i] = ft(imH[i].real)
    else:
        imgH = np.copy(imH)

    imgHH = np.copy(imgH)

    if downSamp == 1:
        N = imgH[0].shape[1]
        M = imgH[0].shape[0]
        for i in range(1, Np):
            extx = np.int32((N - N // 2**i) // 2)
            exty = np.int32((M - M // 2**i) // 2)
            for k in range(Ns):
                imgHH[1 + Ns * i + k] = (
                    shift(
                        shift(imgH[1 + Ns * i + k])[
                            exty : exty + M // 2**i, extx : extx + N // 2**i
                        ]
                    )
                    / 2**i
                )

            imgHH[Np * Ns + 1] = (
                shift(
                    shift(imgH[Np * Ns + 1])[
                        exty : exty + M // 2**i, extx : extx + N // 2**i
                    ]
                )
                / 2**i
            )

    N = imgHH[Np * Ns + 1].shape[1]
    M = imgHH[Np * Ns + 1].shape[0]
    Lx = np.concatenate(
        (np.linspace(0, N // 2 - 1, N // 2), np.linspace(-N // 2, -1, N // 2))
    )
    Ly = np.concatenate(
        (np.linspace(0, M // 2 - 1, M // 2), np.linspace(-M // 2, -1, M // 2))
    )

    X, Y = np.meshgrid(Lx, Ly)

    R = np.sqrt((X * 2 * np.pi / N) ** 2 + (Y * 2 * np.pi / M) ** 2)
    R[0, 0] = 10 ** (-16)
    T = np.arctan2(Y, X)

    if imgHH[Np * Ns + 1].ndim == 3:
        R = R[:, :, np.newaxis]
        T = T[:, :, np.newaxis]

    L = low_filter(R)
    H = high_filter(R)

    imgF = L * imgHH[Np * Ns + 1]

    for i in range(Np):
        for k in range(Ns):
            imgF = imgF + (
                steer_filter(T + np.pi, Ns - 1.0 - k, Ns)
                + steer_filter(T, Ns - 1.0 - k, Ns)
            ) * H * np.fft.fft2(
                np.real(
                    np.fft.ifft2(imgHH[Np * Ns - Ns * i - k], axes=(0, 1), norm="ortho")
                ),
                axes=(0, 1),
                norm="ortho",
            )

            # (imgH[Np*Ns-Ns*i-k]+shift(np.rot90(shift(imgH[Np*Ns-Ns*i-k]),2)))/2

        if i < Np - 1:
            if imgHH[Np * Ns + 1].ndim == 3:
                imgFF = np.zeros((2 * M, 2 * N, imgF.shape[2]), dtype=np.complex64)
                for j in range(imgF.shape[2]):
                    imgFF[:, :, j] = shift(
                        np.pad(
                            shift(imgF[:, :, j]),
                            ((M // 2, M // 2), (N // 2, N // 2)),
                            "constant",
                        )
                    )
                imgF = imgFF
                # imgF = shift(np.pad(shift(imgF), ((N/2,N/2),(N/2,N/2),(0,0)),
                #'constant', constant_values=((0,0),(0,0),(0,0)) ))
            else:
                # print imgF.dtype
                imgF = 2 * shift(
                    np.pad(
                        shift(imgF), ((M // 2, M // 2), (N // 2, N // 2)), "constant"
                    )
                )

            N = 2 * N
            M = 2 * M

            Lx = np.concatenate(
                (np.linspace(0, N // 2 - 1, N // 2), np.linspace(-N // 2, -1, N // 2))
            )
            Ly = np.concatenate(
                (np.linspace(0, M // 2 - 1, M // 2), np.linspace(-M // 2, -1, M // 2))
            )
            X, Y = np.meshgrid(Lx, Ly)

            R = np.sqrt((X * 2 * np.pi / N) ** 2 + (Y * 2 * np.pi / M) ** 2)
            R[0, 0] = 10 ** (-16)
            T = np.arctan2(Y, X)

            if imgHH[Np * Ns + 1].ndim == 3:
                R = R[:, :, np.newaxis]
                T = T[:, :, np.newaxis]

            L = low_filter(R)
            H = high_filter(R)

            imgF = L * imgF

    imgF = ift(low_filter(R / 2.0) * imgF + high_filter(R / 2.0) * imgHH[0]).real

    return imgF


# periodic component
#


def per_comp(x):
    M, N = x.shape
    x = np.float64(x)
    # energy border

    v1 = np.zeros((M, N))
    v1[0, :] = x[M - 1, :] - x[0, :]
    v1[M - 1, :] = x[0, :] - x[M - 1, :]

    v2 = np.zeros((M, N))
    v2[:, 0] = x[:, N - 1] - x[:, 0]
    v2[:, N - 1] = x[:, 0] - x[:, N - 1]

    v = v1 + v2

    # compute the discrete laplacian of u

    lapx = -4.0 * x
    lapx[:, 0 : N - 1] = lapx[:, 0 : N - 1] + x[:, 1:N]
    lapx[:, 1:N] = lapx[:, 1:N] + x[:, 0 : N - 1]
    lapx[0 : M - 1, :] = lapx[0 : M - 1, :] + x[1:M, :]
    lapx[1:M, :] = lapx[1:M, :] + x[0 : M - 1, :]
    lapx[0, :] = lapx[0, :] + x[M - 1, :]
    lapx[M - 1, :] = lapx[M - 1, :] + x[0, :]
    lapx[:, 0] = lapx[:, 0] + x[:, N - 1]
    lapx[:, N - 1] = lapx[:, N - 1] + x[:, 0]

    # Fourier transform of ((lapx) - v)
    Dx_v = np.fft.fft2(lapx - v, norm="ortho")

    # compute the fourier transform of the periodic component
    Lx = np.linspace(
        0, N - 1, N
    )  # np.concatenate((np.linspace(0,N/2-1,N/2),np.linspace(-N/2,-1,N/2)))
    Ly = np.linspace(0, M - 1, M)
    X, Y = np.meshgrid(Lx, Ly)

    div = 2.0 * np.cos(2 * np.pi * X / N) + 2.0 * np.cos(2 * np.pi * Y / M) - 4.0
    div[0, 0] = 1.0
    perufft = Dx_v / div
    perufft[0, 0] = np.sum(x)

    # per = np.real(np.fft.ifft2(perufft, norm='ortho'))

    return perufft


# mc spatial kernel
def mc_spatial_kernel(N, fM, fS, th, thS, stdContrast, octa, cplx=0):
    theta = (90 - th) * np.pi / 180
    thetaSpread = thS * np.pi / 180
    fMode = N / ppcm * fM
    octave = octa
    if octave == 1:
        fSpread = fS
        u = np.sqrt(np.exp((fSpread / np.sqrt(8) * np.sqrt(np.log(2))) ** 2) - 1)
    elif octave == 0:
        fSpread = N / ppcm * fS  # /N
        u = np.roots([1, 0, 3, 0, 3, 0, 1, 0, -(fSpread**2) / fMode**2])  #
        u = u[np.where(np.isreal(u))]
        u = np.real(u[np.where(u > 0)])
        u = u[0]

    rho = fMode * (1.0 + u**2)
    srho = u

    if np.mod(N, 2) == 0:
        Lx = np.concatenate(
            (np.linspace(0, N // 2 - 1, N // 2), np.linspace(-N // 2, -1, N // 2))
        )
    else:
        Lx = np.concatenate(
            (
                np.linspace(0, (N - 1) // 2, (N + 1) // 2),
                np.linspace((-N + 1) // 2, -1, (N - 1) // 2),
            )
        )

    x, y = np.meshgrid(Lx, Lx)
    R = np.sqrt(x**2 + y**2)
    R[0, 0] = 10 ** (-6)
    Theta = np.arctan2(y, x)

    # Spacial kernel
    angular = np.exp(np.cos(2 * (Theta - theta)) / (4 * thetaSpread**2))
    radial = (
        np.exp(-(np.log(R / rho) ** 2 / np.log(1 + srho**2)) / 2)
        * (1.0 / R)
        * (R < 0.5 * N)
    )
    spatialKernel = angular * radial
    if cplx == 1:
        spatialKernel = spatialKernel  # .astype(np.complex128)
        spatialKernel *= mask(Theta, theta).real

    # Compute normalization constant
    spatialKernel /= np.mean(spatialKernel)
    spatialKernel = stdContrast * np.sqrt(spatialKernel)

    return spatialKernel


def mc_3D_kernel(N, fM, fS, th, thS, fT, v, stdContrast, octa):
    theta = th * np.pi / 180
    thetaSpread = thS * np.pi / 180
    fMode = N / ppcm * fM
    LifeTime = 1.0 / fT
    octave = octa
    if octave == 1:
        fSpread = fS
        u = np.sqrt(np.exp((fSpread / np.sqrt(8) * np.sqrt(np.log(2))) ** 2) - 1)
    elif octave == 0:
        fSpread = N / ppcm * fS  # /N
        u = np.roots([1, 0, 3, 0, 3, 0, 1, 0, -(fSpread**2) / fMode**2])  #
        u = u[np.where(np.isreal(u))]
        u = np.real(u[np.where(u > 0)])
        u = u[0]

    rho = fMode * (1.0 + u**2)
    srho = u
    sv = 1 / (rho * LifeTime)

    if np.mod(N, 2) == 0:
        Lx = np.concatenate(
            (np.linspace(0, N // 2 - 1, N // 2), np.linspace(-N // 2, -1, N // 2))
        )
    else:
        Lx = np.concatenate(
            (
                np.linspace(0, (N - 1) // 2, (N + 1) // 2),
                np.linspace((-N + 1) // 2, -1, (N - 1) // 2),
            )
        )

    x, y, t = np.meshgrid(Lx, Lx, Lx)
    R = np.sqrt(x**2 + y**2)
    R[0, 0] = 10 ** (-6)
    Theta = np.arctan2(y, x)
    nuxi = 1.0 / sv * R

    # Spacial kernel
    angular = np.exp(np.cos(2 * (Theta - theta)) / (4 * thetaSpread**2))
    radial = (
        np.exp(-(np.log(R / rho) ** 2 / np.log(1 + srho**2)) / 2)
        * (1.0 / R) ** 3
        * (R < 0.5 * N)
    )

    temp = (
        1.0
        / (1.0 + (nuxi * (t + np.linalg.norm(v) * R * np.cos(Theta + theta))) ** 2) ** 2
    )

    mc3DKer = angular * radial * temp

    # Compute normalization constant
    mc3DKer /= np.sum(mc3DKer)
    mc3DKer = stdContrast * np.sqrt(mc3DKer)

    return mc3DKer


def histo2d(X, Y, xbins, ybins):
    mainFig = plt.figure(1, figsize=(6, 6), facecolor="white")

    # define some gridding.
    axHist2d = plt.subplot2grid((9, 9), (1, 0), colspan=8, rowspan=8)
    #     axHistx  = plt.subplot2grid( (9,9), (0,0), colspan=8 )
    #     axHisty  = plt.subplot2grid( (9,9), (1,8), rowspan=8 )

    H, xedges, yedges = np.histogram2d(X.ravel(), Y.ravel(), bins=(xbins, ybins))

    H = H / H.sum(axis=1)
    axHist2d.imshow(H, interpolation="nearest", aspect="auto", cmap="gray")
    #     axHistx.hist(Y.ravel(), bins=xedges, facecolor='k', alpha=0.5, edgecolor='None' );
    #     axHisty.hist(X.ravel(), bins=yedges, facecolor='k', alpha=0.5, orientation='horizontal', edgecolor='None');

    #     axHistx.set_xlim( [xedges.min(), xedges.max()] )
    #     axHisty.set_ylim( [yedges.min(), yedges.max()] )
    axHist2d.set_ylim([axHist2d.get_ylim()[1], axHist2d.get_ylim()[0]])

    nullfmt = plt.NullFormatter()
    #     axHistx.xaxis.set_major_formatter(nullfmt)
    #     axHistx.yaxis.set_major_formatter(nullfmt)
    #     axHisty.xaxis.set_major_formatter(nullfmt)
    #     axHisty.yaxis.set_major_formatter(nullfmt)

    #     axHistx.spines['top'].set_visible(False)
    #     axHistx.spines['right'].set_visible(False)
    #     axHistx.spines['left'].set_visible(False)
    #     axHisty.spines['top'].set_visible(False)
    #     axHisty.spines['bottom'].set_visible(False)
    #     axHisty.spines['right'].set_visible(False)

    #     axHistx.set_xticks()
    #     axHistx.set_yticks([])
    #     axHisty.set_xticks([])
    #     axHisty.set_yticks([])

    plt.show()


def load_seg_file(filename):
    # return res, im
    # im contains the segmented image by the user
    # res return a vector containing image ID number, user ID number, number of segments, invert, flipflop
    # see https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/seg-format.txt

    with open(filename, "r") as f:
        lines = f.read().split("\n")
        segFile = []
        for l in lines:
            segFile.append(l)

    # res = np.empty(6,dtype=object)
    if segFile[1][:4] == "date":
        k = 0
    else:
        k = -1

    res = np.array(
        [
            np.int32(segFile[2 + k][6:]),
            np.int32(segFile[3 + k][5:]),
            np.int32(segFile[6 + k][9:]),
            np.int32(segFile[8 + k][7:]),
            np.int32(segFile[9 + k][9:]),
        ]
    )

    im = np.zeros((int(segFile[5 + k][7:]), int(segFile[4 + k][6:])))

    for i in range(11 + k, len(segFile) - 2):
        u = np.int32(segFile[i].split(" "))
        im[u[1], u[2] : u[3]] = u[0]

    return res, im


# get V1 orientation feature vector at each scale
def get_features(
    im, w_b, n_th, n_sc, cplx=1, neigh_size=1, sub_samp=1, fuse_sc=True, lum_col=0
):
    # im: color image
    # w_b: wavelet bands

    if np.mod(neigh_size, 2) == 0:
        raise ValueError('"neigh_size" must be odd.')

    N = w_b[0].shape[1]
    M = w_b[0].shape[0]

    d_neigh = (neigh_size - 1) // 2
    n_patch = (M - 2 * sub_samp * d_neigh) * (N - 2 * sub_samp * d_neigh)
    if fuse_sc:
        x = np.zeros(
            (1, n_patch, ((1 + cplx) * n_sc * n_th + lum_col) * neigh_size**2)
        )
        all_bands = np.zeros((M, N, n_th * n_sc), dtype=np.complex128)
        for s in range(n_sc):
            for k in range(n_th):
                all_bands[:, :, k + s * n_th] = w_b[1 + k + s * n_th]
        if lum_col == 3:
            all_bands = np.concatenate((all_bands, 1j * im), axis=2)
        elif lum_col == 1:
            all_bands = np.concatenate(
                (all_bands, 1j * rgb2gray(im)[..., np.newaxis]), axis=2
            )

        xr = np.moveaxis(
            extract_patches_2d(
                all_bands.real, (2 * d_neigh * sub_samp + 1, 2 * d_neigh * sub_samp + 1)
            )[:, ::sub_samp, ::sub_samp, :],
            [1, 2, 3],
            [2, 3, 1],
        ).reshape((n_patch, (n_sc * n_th + lum_col) * neigh_size**2))
        xi = np.moveaxis(
            extract_patches_2d(
                all_bands.imag, (2 * d_neigh * sub_samp + 1, 2 * d_neigh * sub_samp + 1)
            )[:, ::sub_samp, ::sub_samp, :],
            [1, 2, 3],
            [2, 3, 1],
        ).reshape((n_patch, (n_sc * n_th + lum_col) * neigh_size**2))
        if cplx == 1:
            if lum_col == 3:
                x[0] = np.concatenate((xr[:, : -3 * neigh_size**2], xi), axis=1)
            elif lum_col == 1:
                x[0] = np.concatenate((xr[:, : -1 * neigh_size**2], xi), axis=1)
            else:
                x[0] = np.concatenate((xr, xi), axis=1)
        else:
            if lum_col == 3:
                x[0] = np.concatenate(
                    (xr[:, : -3 * neigh_size**2], xi[:, -3 * neigh_size**2 :]),
                    axis=1,
                )
            elif lum_col == 1:
                x[0] = np.concatenate(
                    (xr[:, : -1 * neigh_size**2], xi[:, -1 * neigh_size**2 :]),
                    axis=1,
                )
            else:
                x[0] = xr

    else:
        x = np.zeros((1, n_patch, ((1 + cplx) * n_th + lum_col) * neigh_size**2))
        all_bands = np.zeros((M, N, n_th), dtype=np.complex128)
        s = n_sc - 1
        for k in range(n_th):
            all_bands[:, :, k] = w_b[1 + k + s * n_th]

        if lum_col == 3:
            all_bands = np.concatenate((all_bands, 1j * im), axis=2)
        elif lum_col == 1:
            all_bands = np.concatenate(
                (all_bands, 1j * rgb2gray(im)[..., np.newaxis]), axis=2
            )

        xr = np.moveaxis(
            extract_patches_2d(
                all_bands.real, (2 * d_neigh * sub_samp + 1, 2 * d_neigh * sub_samp + 1)
            )[:, ::sub_samp, ::sub_samp, :],
            [1, 2, 3],
            [2, 3, 1],
        ).reshape((n_patch, (n_th + lum_col) * neigh_size**2))
        print(
            extract_patches_2d(
                all_bands.real, (2 * d_neigh * sub_samp + 1, 2 * d_neigh * sub_samp + 1)
            )[:, ::sub_samp, ::sub_samp, :].shape
        )
        print(
            np.moveaxis(
                extract_patches_2d(
                    all_bands.real,
                    (2 * d_neigh * sub_samp + 1, 2 * d_neigh * sub_samp + 1),
                )[:, ::sub_samp, ::sub_samp, :],
                [1, 2, 3],
                [2, 3, 1],
            ).shape
        )

        xi = np.moveaxis(
            extract_patches_2d(
                all_bands.imag, (2 * d_neigh * sub_samp + 1, 2 * d_neigh * sub_samp + 1)
            )[:, ::sub_samp, ::sub_samp, :],
            [1, 2, 3],
            [2, 3, 1],
        ).reshape((n_patch, (n_th + lum_col) * neigh_size**2))

        if cplx == 1:
            if lum_col == 3:
                x[0] = np.concatenate((xr[:, : -3 * neigh_size**2], xi), axis=1)
            elif lum_col == 1:
                x[0] = np.concatenate((xr[:, : -1 * neigh_size**2], xi), axis=1)
            else:
                x[0] = np.concatenate((xr, xi), axis=1)
        else:
            if lum_col == 3:
                x[0] = np.concatenate(
                    (xr[:, : -3 * neigh_size**2], xi[:, -3 * neigh_size**2 :]),
                    axis=1,
                )
            elif lum_col == 1:
                x[0] = np.concatenate(
                    (xr[:, : -1 * neigh_size**2], xi[:, -1 * neigh_size**2 :]),
                    axis=1,
                )
            else:
                x[0] = xr

    return x


# spatial stationary auto-correlation
def corr2d(a, b):
    N = a.shape[2]
    if len(a.shape) == 2:
        res = np.roll(np.fft.ifft2(a * np.conj(b), norm="ortho"), (1, 1), (0, 1))
        # res /= N**2
    else:
        res = np.roll(
            np.fft.ifft2(np.sum(a * np.conj(b), axis=0), norm="ortho").real,
            (1, 1),
            (0, 1),
        )
        # res /= N**2
        # (a.shape[0]*N**2-1)
    return res


# turn 2d autocovariance into regular vectorial covariance
def cov2d2cov(A):
    N = A.shape[0]
    Ablocks = ()
    for i in range(A.shape[0]):
        if np.mod(N, 2) == 1:
            Ablocks += (sp.linalg.circulant(np.roll(A[i, :], N - 1)),)  #
        else:
            Ablocks += (sp.linalg.circulant(np.roll(A[i, :], N - 1)),)

    Arow = np.concatenate(Ablocks, axis=0)
    Amat = np.copy(Arow)
    for i in range(A.shape[1] - 1):
        Amat = np.concatenate(
            (Amat, np.roll(Arow, (i + 1) * A.shape[0], axis=0)), axis=1
        )

    if np.mod(N, 2) == 1:
        Amat = np.roll(Amat, N * (N - 1), axis=0)
    else:
        Amat = np.roll(Amat, N * (N - 1), axis=0)

    return np.matrix(Amat)


# compute covariance with spatial stationarity
# assume zero mean
def stationary_cov(fx, w=1):
    N = fx.shape[-1] ** 2
    D = fx.shape[1]
    blocks = ()
    block_d = ()

    if np.shape(w) != ():
        w = w[:, np.newaxis, np.newaxis]

    for i in range(D):
        block = ()
        for j in range(D):
            if j < i:
                block += (np.zeros((N, N)),)
            elif j == i:
                block += (np.zeros((N, N)),)
                block_d += (cov2d2cov(corr2d(w * fx[:, i, :, :], fx[:, j, :, :])),)
            else:
                block += (cov2d2cov(corr2d(w * fx[:, i, :, :], fx[:, j, :, :])),)

        blocks += (block,)

    block_lines = ()
    for i in range(len(blocks)):
        block_lines += (np.concatenate(blocks[i], axis=1),)

    cov_mat = np.concatenate(block_lines, axis=0)
    cov_mat += np.conj(cov_mat.T)
    cov_mat += sp.linalg.block_diag(*block_d)
    cov_mat /= fx.shape[-1]
    return cov_mat


def stationary_mean(x, w=1):
    mu = np.zeros(x.shape[1:])
    if np.shape(w) == ():
        mu += x.sum(axis=(0, 2, 3), keepdims=True)[0] / (x.shape[0] * x.shape[2] ** 2)
    else:
        mu += (w[:, np.newaxis, np.newaxis, np.newaxis] * x).sum(
            axis=(0, 2, 3), keepdims=True
        )[0] / (w.sum() * x.shape[2] ** 2)
    return mu.reshape((x.shape[1] * x.shape[2] * x.shape[3]))


def gt_pca_cluster_centers(Xpca, gt):
    """
    Xpca: array of pixels x PCA components
    gt: ground truth 2d array
    """
    p = Xpca.shape[0]
    f = Xpca.shape[1]
    gt = np.reshape(gt, p)
    vals, counts = np.asarray(np.unique(gt, return_counts=True))
    vals[counts < THRESHOLD] = 0
    adj_vals = vals[np.nonzero(vals)]
    indicators = []
    for val in adj_vals:
        indicator = np.zeros(gt.shape)
        indicator[gt == val] = 1
        indicators.append(indicator)
    labeled_arrs = []
    for indicator in indicators:
        labeled_arrs.append(
            np.multiply(Xpca, np.repeat(indicator[:, np.newaxis], f, axis=1))
        )
    cluster_centers = []
    for labeled_arr in labeled_arrs:
        labeled_arr[labeled_arr == 0] = np.nan
        cluster_centers.append(np.nanmean(labeled_arr, axis=0))

    cluster_centers = np.asarray(cluster_centers)
    n_components = len(adj_vals)

    return cluster_centers, n_components


def load_bsd(n, filepath, subject=None, gt_only=False, im_only=False):
    # loads images from the test dataset from a particular subject
    # default behavior loads all subjects
    gt_path = ns(glob(filepath + "groundTruth/test/*.mat"))[n]
    im_path = ns(glob(filepath + "images/test/*.jpg"))[n]
    im = image.imread(im_path)
    if subject is not None:
        gt = loadmat(gt_path)["groundTruth"][:, subject][0][0][0][0]
        gt = gt[:-1, :-1]
    else:
        stop = loadmat(gt_path)["groundTruth"].size
        gt = np.asarray(
            [loadmat(gt_path)["groundTruth"][:, k][0][0][0][0] for k in range(stop)]
        )
        gt = gt[:, :-1, :-1]

    im = im[:-1, :-1]

    if gt_only and im_only:
        return im, gt
    elif im_only:
        return im
    elif gt_only:
        return gt
    else:
        return im, gt


@decorator.decorator
def slicer(fxn, arr, *args, **kwargs):
    arr_ = [fxn(frame, *args, **kwargs) for frame in arr]

    output = arr_

    return output


@slicer
def get_n_components(gt, threshold, return_outliers=False):
    #
    vals, counts = np.asarray(np.unique(gt, return_counts=True))
    trash = vals[counts < threshold]
    vals[counts < threshold] = 0
    if return_outliers:
        return np.count_nonzero(vals), trash
    else:
        return np.count_nonzero(vals)


def crop_RGB(im, spec=None, size=(256, 256), center=True, RGB=True):
    color_channel = np.argmin(np.array(im.shape))
    im = np.moveaxis(im, color_channel, -1)
    if spec is not None:
        y, Y = spec["y"]
        x, X = spec["x"]
        cropped = im[y:Y, x:X, :]
    else:
        if center:
            margins = np.array(im.shape[:-1]) - np.array(size)
            my = margins[0] // 2
            My = im.shape[0] - my
            mx = margins[1] // 2
            Mx = im.shape[1] - mx
            cropped = im[my:My, mx:Mx, :]
        else:
            cropped = im[0 : size[0], 0 : size[2], :]

    return cropped


@slicer
def crop(im, spec=None, size=(256, 256), center=True):
    """
    Crops an image, parameters specify different methods of cropping, based on toolbox.py -> crop


    Parameters:
    -----------
    im : array or ndarray
        can handle array or multiple arrays because of slicer
    spec : dict
        {y:(y1,y2),x:(x1,x2)}
    size : tup
        crop
    center : bool
        Determines whether size parameter is calculated from the center (True) or from the origin

    """
    if spec is not None:
        y, Y = spec["y"]
        x, X = spec["x"]
        cropped = im[y:Y, x:X]
    else:
        if center:
            margins = np.array(im.shape) - np.array(size)
            my = margins[0] // 2
            My = im.shape[0] - my
            mx = margins[1] // 2
            Mx = im.shape[1] - mx
            cropped = im[my:My, mx:Mx]
        else:
            cropped = im[0 : size[0], 0 : size[1]]

    return cropped

def transform_coord_system_mpl(offset,size=(256,256)):
    i,j = offset
    transform = lambda x,y: tuple(transform_coord_system((-y,x),size=size))

    out = transform(i,j)
    
    return out

def transform_coord_system(offset, size=(256, 256), origin="center"):
    """
    Transforms coordinates given as offset from center to numpy index coordinates
    Parameters:
    -----------
    offset : tuple
    size : tuple
    origin : str

    Returns:
    --------
    arr : numpy array
        can be used for numpy indexing
    """
    ny, nx = size
    # TODO: cases where origin is not center
    if origin == "center":
        if ny // 2 == 0:  # if segmap height is even
            segmap = segmap[1:, :]
        if nx // 2 == 0:  # if segmap width is even
            segmap = segmap[:, 1:]
        og = (ny // 2, nx // 2)
    else:
        pass
    query_y = og[0] + offset[1]
    query_x = og[1] + offset[0]

    return np.array([int(query_y), int(query_x)])


def get_coord_segment(coord, segmap, origin="center"):
    coord = [int(item) for item in coord]
    ny, nx = coord
    try:
        return segmap[ny, nx]
    except IndexError:
        print("Probe is outside of image area")


def euclidean_distance(coord1, coord2):
    """
    Calculate the Euclidean distance between two coordinates (x1, y1) and (x2, y2).

    Parameters:
        x1 (float): x-coordinate of the first point.
        y1 (float): y-coordinate of the first point.
        x2 (float): x-coordinate of the second point.
        y2 (float): y-coordinate of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return math.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)


@slicer
#put a maximum distance (size of large image)
def is_centered_xy(coord, origin=[0, 0], thresh=25, d=0):
    """
    Determines whether a coordinate is in the center of an image.
    """

    if origin is not None:
        cy, cx = origin
    else:
        cy, cx = (0, 0)

    distance_from_origin = euclidean_distance(coord, origin)

    if distance_from_origin <= thresh:
        return 1
    elif distance_from_origin >= (thresh + d):
        return 2
    else:
        return 0


def is_centered_np(coord, size=(256, 256), thresh=25, d=0, origin=None):
    """
    Determines whether a coordinate is in the center of an image."""

    if origin is not None:
        cy, cx = origin
    else:
        cy = size[0] // 2
        cx = size[1] // 2

    _is_centered_x = lambda x: not ((x < cx - thresh) or (x > cx + thresh))
    _is_centered_y = lambda y: not ((y < cy - thresh) or (y > cy + thresh))

    return _is_centered_x(coord[1]) and _is_centered_y(coord[0])


def calculate_percent_change(x_series, y_series, c=1):
    out = [((x - y) / (abs(x) + c)) for x, y in zip(x_series, y_series)]
    return out


def _bin(im,binsize=(16,16)):
    """
    A function that downsamples segmentation maps by outputting the most common element 
    for a sliding window with size binsize

    Parameters: 
    ------------
    im : array 
        A Ny x Nx array 

    binsize : tup of int (by,bx)
        The size of the sliding window used

    Returns: 
    ------------
    new_im : array of size (Ny/by) x (Nx/bx)
    """
    #original dimensions of array
    by,bx = binsize
    Ny,Nx = im.shape
    #new shape
    ny,nx = Ny//by, Nx//bx
    r_im = im.reshape(ny,by,nx,bx)
    
    new_im = np.zeros(shape=(ny,nx))
    
    most_common = lambda arr: (Counter(np.ravel(arr)).most_common())[0][0]
    
    for i in range(ny):
        for j in range(nx):
            new_im[i,j] = most_common(r_im[i,:,j,:])
    
    return new_im

def clean_df(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

def divide_2f(x,y):
    return np.round(x/y,2)

def pearson_r(x, y, invalid_value=np.nan):
    # Check if the input vectors have the same length
    if len(x) != len(y):
        raise ValueError("Input vectors must have the same length")

    # Calculate the mean of x and y
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    # Calculate the numerator and denominators for Pearson correlation coefficient
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x)
    denominator_y = sum((yi - mean_y) ** 2 for yi in y)

    if denominator_x == 0 or denominator_y == 0:
        # Handle the case where one of the vectors has zero variance
        return invalid_value
    else:
        return numerator / (denominator_x**0.5 * denominator_y**0.5)


def nan_softmax(x):
    """
    Computes the softmax function (see scipy.special.softmax)
    Works with a vector that includes np.nan values
    """
    not_nan = (x==x)
    x_nn = x[not_nan] #x not nan
    x_n = x[~not_nan] #x nan
    #if all values are nan, max weight is assigned to the first element
    if len(x_nn)==0: 
        x_n[0] = 1
        z = np.array([])
    #else calculate the softmax for non-nan values
    else:
        z = spec.softmax(x_nn)
    #concatenate the softmax values with the nan values
    out = np.concatenate([z,x_n],axis=0)

    assert x.shape == out.shape

    return out

def mean_match(data1, data2, nboot, nbin):


    new_edges = np.linspace(
        np.amin(np.array([data1, data2])),
        np.amax(np.array([data1, data2])),
        num=nbin + 1,
    )
    new_counts = np.zeros((nbin, 1))

    samples_data1 = []
    samples_data2 = []

    for i in range(nbin + 1):
        bin_min = new_edges[i]
        bin_max = new_edges[i + 1]

        bin_idx_data_1 = np.where(((data1 > bin_min) & (data1 < bin_max)))[0]
        bin_idx_data_2 = np.where(((data2 > bin_min) & (data2 < bin_max)))[0]

        num_samples = min((len(bin_idx_data_1), len(bin_idx_data_2)))

        def return_n_samples(x,**kwargs):
            return x[:num_samples]

        if num_samples > 0:
            if len(bin_idx_data_1) == 1:
                idx1_samples = np.ones((nboot, 1)) * bin_idx_data_1
            else:
                idx1_samples = bootstrap(
                    (bin_idx_data_1,), return_n_samples, n_resamples=nboot
                ).bootstrap_distribution

            if len(bin_idx_data_2) == 1:
                idx2_samples = np.ones((nboot, 1)) * bin_idx_data_2
            else:
                idx2_samples = bootstrap(
                    (bin_idx_data_2,), x, n_resamples=nboot
                ).bootstrap_distribution

            samples_data1.append(idx1_samples)
            samples_data2.append(idx2_samples)

            new_counts[i, :] = num_samples

    d = {
        "samples_data1": samples_data1,
        "samples_data2": samples_data2,
        "new_edges": new_edges,
        "new_counts": new_counts,
    }

    return d

