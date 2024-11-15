def point_likelihood(coord, SegMap):
    """
    Find the likelihood of an observation (pixel) in the SegMap object
    uses *unnormalized* probability
    """
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
