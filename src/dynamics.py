import numpy as np
from scipy.stats import entropy
from numpy.lib.stride_tricks import sliding_window_view


def point_likelihood(coord, SegMap):
    """
    Find the likelihood of an observation (pixel) in the SegMap object
    uses *unnormalized* probability
    """
    h, w = SegMap.im.shape[:2]
    weights = SegMap.weights_t
    model_ = SegMap.model_fitted
    data = SegMap.flat_pca
    flat_index = np.ravel_multi_index(
        (np.array([coord[0]]), np.array([coord[1]])), (h, w)
    )

    temp = []
    for i in range(len(weights)):
        model_.weights_ = SegMap.flat_weights[i, flat_index, :]
        sample_score = model_.score_samples(data[flat_index].reshape(1, -1))[
            0
        ].squeeze()
        temp.append(sample_score)

    return np.asarray(temp)


def point_convergence(coord, SegMap):
    """
    Finds convergence based on the KLD(pi^(t)||pi^(t-1))
    """
    h, w = SegMap.im.shape[:2]
    weights = SegMap.weights_t
    flat_index = np.ravel_multi_index(
        (np.array([coord[0]]), np.array([coord[1]])), (h, w)
    )
    temp = []
    for i in range(len(weights) - 1):
        curr_pi = SegMap.flat_weights[i, flat_index, :]
        next_pi = SegMap.flat_weights[i + 1, flat_index, :]
        ent = entropy(next_pi.squeeze(), qk=curr_pi.squeeze())
        temp.append(ent)

    return np.asarray(temp)


def sliding_window_deriv1(array, kernel_size):
    """
    1st-order finite difference method:

    f'(x) = {f(x+\delta) - f(x-\delta)}/(\delta)
    """
    diff = lambda x: (x[-1] - x[0]) / len(x)

    out = [diff(bin) for bin in sliding_window_view(array, kernel_size)]

    return np.asarray(out)


def sliding_window_deriv2(array, kernel_size):
    """
    2nd-order finite difference method:

    f''(x) = {f(x + \delta) - 2f(x) + f(x - \delta)}/(\delta)^2

    """

    diff = lambda x: (x[-1] - 2 * x[len(x) // 2] + x[0]) / (len(x) ** 2)

    out = [diff(bin) for bin in sliding_window_view(array, kernel_size)]

    return np.asarray(out)


def sliding_window_mean(array, kernel_size):

    out = [np.mean(bin) for bin in sliding_window_view(array, kernel_size)]

    return np.asarray(out)


def check_derivatives(index, d1, d2, epsilon=1):
    if d1[index] > epsilon:
        return 0
    elif d1[index] < epsilon and d1[index] > 0:
        if d2[index] > epsilon:  # local minimum
            return 0
        else:
            return 1
    elif d1[index] < epsilon and d1[index] < -(epsilon):
        return 0


def find_pointwise_rt(coord, SegMap, kern_size=3, kld_tol=0.005, use_lkl=0.01):

    h, w = SegMap.im.shape[:2]

    flat_index = np.ravel_multi_index(
        (np.array([coord[0]]), np.array([coord[1]])), (h, w)
    )

    smooth_conv = sliding_window_mean(
        point_convergence(coord, SegMap), kernel_size=kern_size - 1
    )

    smooth_lkl = sliding_window_mean(
        point_likelihood(coord, SegMap), kernel_size=kern_size
    )

    d1 = sliding_window_deriv1(point_likelihood(coord, SegMap), kernel_size=kern_size)

    d2 = sliding_window_deriv2(point_likelihood(coord, SegMap), kernel_size=kern_size)

    converged_weights = SegMap.flat_weights[-1]

    SegMap.model_fitted.weights_ = converged_weights

    possible_ind = np.where(smooth_conv < kld_tol)[0]

    total_proba = SegMap.model_fitted.score(SegMap.flat_pca).sum()

    if use_lkl is not None:
        lkl_thresh = (use_lkl * total_proba) / (SegMap.im.size)
        out = -1

        for ind in possible_ind:
            if check_derivatives(ind, d1, d2, epsilon=lkl_thresh):
                out = ind
                break
            else:
                continue

        if out == -1:
            try:
                out = np.argmax(smooth_lkl[possible_ind])
            except:
                out = np.argmax(smooth_lkl)
        elif out == 0:
            out = 1
    else:
        try:
            out = np.where(smooth_conv < kld_tol)[0][0]
        except:
            out = len(smooth_conv)

    return out


def _get_psame_t(coord1, coord2, SegMap):

    pmap = np.moveaxis(SegMap.weights_t, -1, 1)

    n_iter = pmap.shape[0]

    pmap_a = pmap[:, :, coord1[0], coord1[1]]

    pmap_b = pmap[:, :, coord2[0], coord2[1]]

    psame_t = np.asarray([np.dot(pmap_a[i], pmap_b[i]) for i in range(n_iter)])

    return psame_t


def _get_seg_flag_t(coord1, coord2, SegMap):

    pmap = np.moveaxis(SegMap.weights_t, -1, 1)
    n_iter = pmap.shape[0]

    pmap_a = pmap[:, :, coord1[0], coord1[1]]

    pmap_b = pmap[:, :, coord2[0], coord2[1]]

    seg_a = pmap_a.argmax(1)

    seg_b = pmap_b.argmax(1)

    seg_flag_t = seg_a == seg_b

    return seg_flag_t


def _get_entropy(coord1, coord2, SegMap):

    pmap = np.moveaxis(SegMap.weights_t, -1, 1)
    n_iter = pmap.shape[0]

    pmap_a = pmap[:, :, coord1[0], coord1[1]]

    seg_a = pmap_a.argmax(1)

    pmap_b = pmap[:, :, coord2[0], coord2[1]]

    seg_b = pmap_b.argmax(1)

    psame_t = np.asarray([np.dot(pmap_a[i], pmap_b[i]) for i in range(n_iter)])
    seg_flag_t = seg_a == seg_b

    assert len(psame_t) == len(seg_flag_t)

    return entropy(psame_t[..., np.newaxis], axis=1)


def _get_evidence(coord1, coord2, SegMap, evidence_type="logit"):
    if evidence_type == "logit":
        get_logit = lambda x: np.log(x) - np.log(1 - x)

        psame = _get_psame_t(coord1, coord2, SegMap)
        sf_t = _get_seg_flag_t(coord1, coord2, SegMap)
        logit = get_logit(psame)

        return logit


def _get_decision_rt(yes_no, evidence, pointwise_rt=None, boundary=None):
    if boundary is not None:
        boundary = boundary
    else:
        boundary = 0.8 * np.max(evidence)

    if pointwise_rt is not None:
        slow_point = np.max(pointwise_rt)
        slow_point_idx = np.ceil(slow_point).astype("int")
        try:
            if yes_no == "yes":
                bound_idx = np.where(evidence > boundary)[0][0]
            elif yes_no == "no":
                bound_idx = np.where(evidence < -boundary)[0][0]
            if bound_idx < np.ceil(slow_point_idx):
                rt = slow_point_idx
            else:
                rt = bound_idx

        except:
            rt = len(evidence)
    else:
        try:
            if yes_no == "yes":
                bound_idx = np.where(evidence > boundary)[0][0]
            elif yes_no == "no":
                bound_idx = np.where(evidence < -boundary)[0][0]
            rt = bound_idx
        except:
            rt = len(evidence)

    if rt == 0:
        rt = 1

    return rt
