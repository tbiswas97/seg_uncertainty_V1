import numpy as np
from scipy.stats import entropy
from matplotlib import pyplot as plt
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
    """
    find the pointwise reaction time proxy t_{pointwise} for a coordinate given
    a segmentation map

    Parameters:
    ------------
    coord : array
        the coordinate of interest (in numpy coordinates)
    SegMap : SegmentationMap object
        SegmentationMap object with likelihood attributes
    kern_size : int
        the kernel size for *temporal* smoothing
    kld_tol : float
        the threshold where KLD(\vec{\pi_i^{(t)}}||\vec{\pi_i^{(t)}}) is
        considered to have coverged
    use_lkl : float, default None

    """

    h, w = SegMap.im.shape[:2]

    flat_index = np.ravel_multi_index(
        (np.array([coord[0]]), np.array([coord[1]])), (h, w)
    )

    # kern size is kern_size minus 1 to ensure smooth_conv and smooth_lkl are the same shape
    smooth_conv = sliding_window_mean(
        point_convergence(coord, SegMap), kernel_size=kern_size - 1
    )

    smooth_lkl = sliding_window_mean(
        point_likelihood(coord, SegMap), kernel_size=kern_size
    )

    # calculates the derivatives of the likelihood function
    d1 = sliding_window_deriv1(point_likelihood(coord, SegMap), kernel_size=kern_size)

    d2 = sliding_window_deriv2(point_likelihood(coord, SegMap), kernel_size=kern_size)

    converged_weights = SegMap.flat_weights[-1]

    SegMap.model_fitted.weights_ = converged_weights

    possible_ind = np.where(smooth_conv < kld_tol)[0]

    # The total probability of all observations (pixels) belonging to the fit model
    total_proba = SegMap.model_fitted.score(SegMap.flat_pca).sum()

    if use_lkl is not None:
        # default value for likelihood threshold is 1% of the total probability per pixel
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
        # if use_lkl is None then only use smooth_conv
        try:
            out = np.where(smooth_conv < kld_tol)[0][0]
        except:
            out = len(smooth_conv)

    return out


def _get_psame_t(coord1, coord2, SegMap):
    """
    Calculates \pi_{ij}^{(t)}
    """

    pmap = np.moveaxis(SegMap.weights_t, -1, 1)

    n_iter = pmap.shape[0]

    pmap_a = pmap[:, :, coord1[0], coord1[1]]

    pmap_b = pmap[:, :, coord2[0], coord2[1]]

    psame_t = np.asarray([np.dot(pmap_a[i], pmap_b[i]) for i in range(n_iter)])

    return psame_t


def _get_seg_flag_t(coord1, coord2, SegMap):
    """
    Calculates f_{ij}^{(t)}
    """

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


def _get_logit(coord1, coord2, psame_t, evidence_type="logit"):
    """
    Calculate E_{ij}^{(t)}

    Parameters:
    ------------
    coord1 : array like using np coordinates
    coord2 : array like using np coordinates
    psame_t : array like
    evidence_type : string
    """
    if evidence_type == "logit":
        get_logit = lambda x: np.log(x) - np.log(1 - x)

        logit = get_logit(psame_t)

        return logit


def _get_decision_rt(yes_no, evidence, pointwise_rt=None, boundary=None):
    """
    Calculate decision reaction time from evidence and boundary

    Parameters:
    -----------
    yes_no : str or None
        "yes" : if decision is "yes" a priori then use the positive boundary
        "no" : if decision is "no" a priori then use the negative boundary
        None : if decision is not known a priori then use *either* boundary
    evidence : array
        logits per algorithm iteration
    pointwise_rt : bool
        if True, use the slowest pointwise reaction time if the pairwise time is
        faster than either point
    boundary : int
        positive int, if yes_no is "no" then use the negative of boundary
        #TODO: change this later to handle asymmetric boundaries

    Returns:
    ---------
    rt : int
        The iteration where the evidence crosses the boundary
    response : bool, None if yes_no is not None

    """
    if boundary is not None:
        boundary = boundary
    else:
        # default value for boundary
        boundary = 0.8 * np.max(evidence)

    response = None

    if pointwise_rt is not None:
        slow_point = np.max(pointwise_rt)
        slow_point_idx = np.ceil(slow_point).astype("int")
        try:
            if yes_no == "yes":
                bound_idx = np.where(evidence > boundary)[0][0]
            elif yes_no == "no":
                bound_idx = np.where(evidence < -boundary)[0][0]
            elif yes_no == None:
                bound_idx = np.where(
                    (evidence > boundary[0]) | (evidence < boundary[-1])
                )[0][0]
                decision = evidence[bound_idx]
                if decision > boundary:
                    response = True
                elif decision < -boundary:
                    response = False
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
            elif yes_no == None:
                bound_idx = np.where((evidence > boundary) | (evidence < -boundary))[0][
                    0
                ]
                decision = evidence[bound_idx]
                if decision > boundary:
                    response = True
                elif decision < -boundary:
                    response = False
            rt = bound_idx
        except:
            rt = len(evidence)

    if rt == 0:
        rt = 1

    if response is not None:
        return rt, response
    else:
        return rt, np.nan


def df_to_rt_vs_distance(df, rt_col="model_rt", kernel_size=10, groupby="seg_flag"):
    dist_y = (
        df.sort_values("image_distance")
        .loc[(df[groupby] == True), "image_distance"]
        .values
    )
    rt_y = df.sort_values("image_distance").loc[(df[groupby] == True), rt_col].values
    dist_n = (
        df.sort_values("image_distance")
        .loc[(df[groupby] == False), "image_distance"]
        .values
    )
    rt_n = df.sort_values("image_distance").loc[(df[groupby] == False), rt_col].values

    dist_smooth_y = [np.mean(_bin) for _bin in sliding_window_view(dist_y, kernel_size)]
    rt_smooth_y = [np.mean(_bin) for _bin in sliding_window_view(rt_y, kernel_size)]

    dist_smooth_n = [np.mean(_bin) for _bin in sliding_window_view(dist_n, kernel_size)]
    rt_smooth_n = [np.mean(_bin) for _bin in sliding_window_view(rt_n, kernel_size)]

    d = {}

    d["plot_yes"] = (dist_smooth_y, rt_smooth_y)
    d["plot_no"] = (dist_smooth_n, rt_smooth_n)

    plt.plot(d["plot_yes"][0], d["plot_yes"][1])
    plt.plot(d["plot_no"][0], d["plot_no"][1])

    return d
