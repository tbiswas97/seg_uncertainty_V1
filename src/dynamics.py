import numpy as np
from scipy.stats import entropy, beta
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
        elif d2[index] < 0:
            return 1
        else:
            return 0
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


# EVIDENCE INTEGRATION FUNCTIONS
# def estimate_beta_param(mu, var=None, eps=1e6):

# if mu == 0.0:
# mu += eps
# elif not (mu < 1.0):
# mu = 1 - eps

# if var is not None:
# var = var
# else:
# var = 0.1 * (mu * (1 - mu))
# alpha = (((1 - mu) / var) - (1 / mu)) * mu**2
# beta = alpha * ((1 / mu) - 1)
# return {"alpha": alpha, "beta": beta, "var": var, "std": np.sqrt(var)}


# def draw_beta_samples(mu, num_samples=10, var=None):
# if var is not None:
# var = var
# else:
# var = 0.1 * (mu * (1 - mu))

# out = estimate_beta_param(mu)

# try:
# samples = beta.rvs(out["alpha"], out["beta"], size=num_samples)
# except ValueError:
# samples = beta.rvs(out["alpha"], out["beta"], size=num_samples)

# return samples


# def evidence_integration(coord1, coord2, SegMap, num_samples=None):
# segmap = SegMap.segmap

# seg_a = segmap[coord1[0], coord1[1]]
# seg_b = segmap[coord2[0], coord2[1]]

# flag = seg_a == seg_b

# pmap = np.moveaxis(SegMap.weights_t, -1, 1)

# n_iter = pmap.shape[0]

# if num_samples is not None:
# num_samples = num_samples
# else:
# num_samples = n_iter

# pmap_a = pmap[:, :, coord1[0], coord1[1]]
# pmap_b = pmap[:, :, coord2[0], coord2[1]]

# psame = np.dot(pmap_a[-1], pmap_b[-1])

# samples = draw_beta_samples(psame, num_samples=num_samples)
# integration = np.cumsum(samples) / np.arange(1, len(samples) + 1)

# return integration


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


def _get_ei_logits(
    starting_point=0,
    drift_rate=None,
    noise=5,
    eps=1e-4,
    sample_size=20,
):

    if drift_rate is not None:
        drift_rate = drift_rate

    starting_point_arr = np.zeros(sample_size) + starting_point

    samples = drift_rate + noise * np.random.normal(0, 1, size=sample_size)
    integrated_evidence = np.cumsum(samples)

    assert len(starting_point_arr) == len(integrated_evidence)

    starting_point_arr[1:] = integrated_evidence[:-1]

    out = starting_point_arr

    return out


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

        psame_t[psame_t > 1] = 1

        logit = get_logit(psame_t)

        return logit


def _get_decision_rt(
    evidence,
    deriv=None,
    boundary=None,
    c=1,
    window_size=3,
    return_responses=False,
    failure="argmax",
):
    """
    Calculate decision reaction time from evidence and boundary

    Parameters:
    -----------
    evidence : array
        logits per algorithm iteration
    boundary : array-like
        element 1 is the positive boundary, element 2 is the negative boundary
    c : array
        multiplier of evidence minimum that is used as a threshold for the first
        derivative
    return_responses : bool
        if True, then return the decision (response) at rt
    fit_params : dict
        dict of form {"condition":boundary}, only applicable if boundary == "fit"
    fit_param_key : str:
        string to use in fit_params, only applicable if boundary == "fit"

    Returns:
    ---------
    rt : int
        The iteration where the evidence crosses the boundary
    response : bool
    """
    if boundary == "auto":
        assert type(c) == np.ndarray
        abs_evidence = np.abs(evidence)
        thresh = c[0] * (np.mean(abs_evidence))
        smooth_evidence = evidence
        assert deriv is not None
        smooth_evidence_d1 = deriv

        d1_windows = sliding_window_view(smooth_evidence_d1, window_size)
        check_deriv = [(window < thresh).all() for window in d1_windows[:]]

        possible_rts = np.where(check_deriv)[0]

        if len(c) > 1:
            if len(possible_rts) > 0:
                static_evidence = abs_evidence[possible_rts]
                more_possible_rts = np.where(static_evidence > c[1])[0]
                if len(more_possible_rts) > 0:
                    possible_rts = possible_rts[more_possible_rts]
            else:
                if failure == "argmax":
                    rt = np.argmax(np.abs(smooth_evidence))
                else:
                    rt = len(smooth_evidence)

        if len(possible_rts) > 0:
            rt = possible_rts[0]
        else:
            if failure == "argmax":
                rt = np.argmax(np.abs(smooth_evidence))
            else:
                rt = len(smooth_evidence)

        response = None
        if return_responses:
            if smooth_evidence[rt] > 0:
                response = True
            else:
                response = False

        return rt, response

        # return rt, response
        # possible_idxs = range(len(smooth_evidence))
        # out = -1
        # thresh = np.min(abs_evidence)
        # for idx in possible_idxs:
        # if check_derivatives(
        # idx, smooth_evidence_d1, smooth_evidence_d2, epsilon=thresh
        # ):
        # out = idx
        # break
        # else:
        # continue
        # if out == -1:
        # out = np.argmax(smooth_evidence)

        # rt = out
        # if evidence[rt] > 0:
        # response = True
        # else:
        # response = False

        # return rt, response
    else:
        if boundary is not None:
            boundary = boundary
        else:
            # default value for boundary
            boundary = [1, -1]

        response = None

        try:
            bound_idx = np.where((evidence > boundary[0]) | (evidence < boundary[1]))[
                0
            ][0]
            decision = evidence[bound_idx]
            if decision > 0:
                response = True
            elif decision < 0:
                response = False
            rt = bound_idx
        except:
            # if failure == "argmax":
            # rt = np.argmax(np.abs(evidence))
            # else:
            rt = len(evidence) - 1
            decision = evidence[-1]
            if decision > 0:
                response = True
            elif decision < 0:
                response = False
            else:
                response = None

        if response is not None:
            return rt, response
        else:
            return rt, np.nan


def _get_rt_from_boundary(
    logits,
    boundary,
    output_flat=True,
    return_mean=True,
    mean_axis=(0, -1),
    add_one=False,
):
    """
    Calculates reaction times using a boundary on the array (vectorized)
    """
    times = np.abs(logits) > boundary
    times[..., -1] = True

    rts = np.argmax(times, axis=-1)
    if add_one:
        rts += 1
    if return_mean:
        rts = rts.mean(axis=mean_axis)
    if output_flat:
        rts = np.ravel(rts)

    return rts


def _get_rt_from_deriv(
    smooth_logits,
    logit_deriv,
    thresh,
    output_flat=True,
    return_mean=True,
    failure_mode="argmax",
    mean_axis=(0, -1),
    use_boundary=None,
    add_one=False,
):

    abs_evidence = np.abs(smooth_logits)
    abs_deriv = np.abs(logit_deriv)

    if use_boundary is not None:
        cond = np.logical_or(
            abs_deriv < (thresh * abs_evidence), (abs_evidence > use_boundary)
        )
    else:
        cond = abs_deriv < (thresh * abs_evidence)

    failure_to_conv = np.nonzero((~cond).all(axis=-1))
    conv_cond = sliding_window_view(cond, 3, axis=-1).all(axis=-1)

    rt_arr = conv_cond.argmax(-1)
    if failure_mode == "argmax":
        rt_arr[failure_to_conv] = abs_evidence[failure_to_conv].argmax(-1)
    else:
        rt_arr[failure_to_conv] = conv_cond.shape[-1]

    rts = rt_arr

    if add_one:
        rts += 1

    if return_mean:
        rts = rt_arr.mean(axis=mean_axis)

    if output_flat:
        rts = np.ravel(rts)

    return rts


def _get_responses_from_rt_arr(rt_arr, logits):

    assert rt_arr.shape == logits.shape[:-1]

    indxs = np.indices(rt_arr.shape)

    responses = (logits[indxs[0], indxs[1], indxs[2], rt_arr]) > 0

    return responses


def _get_errors_from_rt_arr(rt_arr, logits, sfs_t):

    assert rt_arr.shape == logits.shape[:-1]

    indxs = np.indices(rt_arr.shape)

    responses = (logits[indxs[0], indxs[1], indxs[2], rt_arr]) > 0
    segflags = sfs_t[..., -1]

    errors = np.logical_xor(responses, segflags)

    return errors


def df_to_rt_hist(df, rt_col="online_rt", groupby="seg_flag", nbins=20):
    all_yes = df.loc[(df[groupby] == 1), [rt_col]]
    all_no = df.loc[(df[groupby] == 0), [rt_col]]
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    axs[0].hist(all_yes.values, bins=nbins, density=True, facecolor="green")
    axs[1].hist(all_no.values, bins=nbins, density=True, facecolor="orange")


def df_to_rt_vs_distance(
    df,
    rt_col="model_rt",
    kernel_size=10,
    groupby="seg_flag",
    error_mode="fuzzy",
    _sample=None,
    ax=None,
    colors=["#40539F", "#db3b31"],
    plot_args = None
):

    sem = lambda x: np.std(x) / (np.sqrt(len(x)))

    df = df.loc[:, ["image_distance", rt_col, groupby]]

    if _sample is not None:
        df = df.sample(frac=_sample)
    else:
        pass

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

    dist_smooth_y_err = [sem(_bin) for _bin in sliding_window_view(dist_y, kernel_size)]
    rt_smooth_y_err = [sem(_bin) for _bin in sliding_window_view(rt_y, kernel_size)]

    dist_smooth_n_err = [sem(_bin) for _bin in sliding_window_view(dist_n, kernel_size)]
    rt_smooth_n_err = [sem(_bin) for _bin in sliding_window_view(rt_n, kernel_size)]

    d = {}

    d["plot_yes"] = (dist_smooth_y, rt_smooth_y)
    d["plot_no"] = (dist_smooth_n, rt_smooth_n)

    d["plot_yes_err"] = (dist_smooth_y_err, rt_smooth_y_err)
    d["plot_no_err"] = (dist_smooth_n_err, rt_smooth_n_err)

    if ax is not None:
        if plot_args is not None:
            if error_mode == "fuzzy":
                #ax.plot(d["plot_yes"][0], d["plot_yes"][1], c=colors[0], label="yes",**plot_args)

                ax.errorbar(
                    x=d["plot_yes"][0],
                    y=d["plot_yes"][1],
                    xerr=d["plot_yes_err"][0],
                    yerr=d["plot_yes_err"][1],
                    c=colors[0],
                    **plot_args
                )

                #ax.plot(d["plot_no"][0], d["plot_no"][1], c=colors[1], label="no",**plot_args)

                ax.errorbar(
                    x=d["plot_no"][0],
                    y=d["plot_no"][1],
                    xerr=d["plot_no_err"][0],
                    yerr=d["plot_no_err"][1],
                    c=colors[1],
                    **plot_args
                )
            elif error_mode=="shaded":
                y_dist = np.asarray(d["plot_yes"][0])
                y_rt = np.asarray(d["plot_yes"][1])
                n_dist = np.asarray(d["plot_no"][0])
                n_rt = np.asarray(d["plot_no"][1])
                y_err = np.asarray(d["plot_yes_err"][1])
                n_err = np.asarray(d["plot_no_err"][1])

                #ax.plot(y_dist, y_rt, c=colors[0], label="yes",**plot_args)
                #ax.plot(n_dist, n_rt, c=colors[1], label="no",**plot_args)
                ax.fill_between(y_dist, y1=y_rt+y_err, y2=y_rt-y_err,facecolor="none",color=colors[0],lw=0,**plot_args)
                ax.fill_between(n_dist, y1=n_rt+n_err, y2=n_rt-n_err,facecolor="none",color=colors[1],lw=0,**plot_args)
        else:
            if error_mode == "fuzzy":
                #ax.plot(d["plot_yes"][0], d["plot_yes"][1], c=colors[0], label="yes",**plot_args)

                ax.errorbar(
                    x=d["plot_yes"][0],
                    y=d["plot_yes"][1],
                    xerr=d["plot_yes_err"][0],
                    yerr=d["plot_yes_err"][1],
                    c=colors[0],
                )

                #ax.plot(d["plot_no"][0], d["plot_no"][1], c=colors[1], label="no",**plot_args)

                ax.errorbar(
                    x=d["plot_no"][0],
                    y=d["plot_no"][1],
                    xerr=d["plot_no_err"][0],
                    yerr=d["plot_no_err"][1],
                    c=colors[1],
                )
            elif error_mode=="shaded":
                y_dist = np.asarray(d["plot_yes"][0])
                y_rt = np.asarray(d["plot_yes"][1])
                n_dist = np.asarray(d["plot_no"][0])
                n_rt = np.asarray(d["plot_no"][1])
                y_err = np.asarray(d["plot_yes_err"][1])
                n_err = np.asarray(d["plot_no_err"][1])

                #ax.plot(y_dist, y_rt, c=colors[0], label="yes")
                #ax.plot(n_dist, n_rt, c=colors[0], label="no")
                ax.fill_between(y_dist, y1=y_rt+y_err, y2=y_rt-y_err,alpha=0.4,facecolor="none",color=colors[0],lw=0)
                ax.fill_between(n_dist, y1=n_rt+n_err, y2=n_rt-n_err,alpha=0.4,facecolor="none",color=colors[1],lw=0)

    else:
        plt.plot(d["plot_yes"][0], d["plot_yes"][1], c=colors[0])
        plt.plot(d["plot_no"][0], d["plot_no"][1], c=colors[1])

    return d
