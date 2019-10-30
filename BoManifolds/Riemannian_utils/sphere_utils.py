import numpy as np

'''
Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
'''


def expmap(u, x0):
    """
    This function maps a vector u lying on the tangent space of x0 into the manifold.

    Parameters
    ----------
    :param u: vector in the tangent space
    :param x0: basis point of the tangent space

    Returns
    -------
    :return: point on the manifold
    """
    if np.ndim(x0) < 2:
        x0 = x0[:, None]

    if np.ndim(u) < 2:
        u = u[:, None]

    norm_u = np.sqrt(np.sum(u*u, axis=0))
    x = x0 * np.cos(norm_u) + u * np.sin(norm_u)/norm_u

    x[:, norm_u < 1e-16] = x0

    return x


def logmap(x, x0):
    """
    This functions maps a point lying on the manifold into the tangent space of a second point of the manifold.

    Parameters
    ----------
    :param x: point on the manifold
    :param x0: basis point of the tangent space where x will be mapped

    Returns
    -------
    :return: vector in the tangent space of x0
    """
    if np.ndim(x0) < 2:
        x0 = x0[:, None]

    if np.ndim(x) < 2:
        x = x[:, None]

    theta = np.arccos(np.dot(x0.T, x))
    u = (x - x0 * np.cos(theta)) * theta/np.sin(theta)

    u[:, theta[0] < 1e-16] = np.zeros((u.shape[0], 1))

    return u


def sphere_distance(x, y):
    """
    This function computes the Riemannian distance between two points on the manifold.

    Parameters
    ----------
    :param x: point on the manifold
    :param y: point on the manifold

    Returns
    -------
    :return: manifold distance between x and y
    """
    if np.ndim(x) < 2:
        x = x[:, None]

    if np.ndim(y) < 2:
        y = y[:, None]

    # Compute the inner product (should be [-1,1])
    inner_product = np.dot(x.T, y)
    inner_product = max(min(inner_product, 1), -1)
    return np.arccos(inner_product)


def parallel_transport_operator(x1, x2):
    """
    This function compute the parallel transport operator from x1 to x2. Transported vectors can be computed as u.dot(v)

    Parameters
    ----------
    :param x1: point on the manifold
    :param x2: point on the manifold

    Returns
    -------
    :return: parallel transport operator
    """
    if np.sum(x1-x2) == 0.:
        return np.eye(x1.shape[0])
    else:
        if np.ndim(x1) < 2:
            x1 = x1[:, None]

        if np.ndim(x2) < 2:
            x2 = x2[:, None]

        x_dir = logmap(x2, x1)
        norm_x_dir = np.sqrt(np.sum(x_dir*x_dir, axis=0))
        normalized_x_dir = x_dir / norm_x_dir
        u = np.dot(-x1 * np.sin(norm_x_dir), normalized_x_dir.T) \
            + np.dot(normalized_x_dir * np.cos(norm_x_dir), normalized_x_dir.T) \
            + np.eye(x_dir.shape[0]) - np.dot(normalized_x_dir, normalized_x_dir.T)

        return u


def karcher_mean_sphere(data, nb_iter=10):
    """
    This function computes the mean of points lying on the manifold (Fréchet/Karcher mean).

    Parameters
    ----------
    :param data: data points lying on the manifold

    Optional parameters
    -------------------
    :param nb_iter: number of iterations

    Returns
    -------
    :return: mean of the datapoints
    """
    # Initialize the mean as equal to the first datapoint
    m = data[:, 0]
    for i in range(nb_iter):
        data_tgt = logmap(data, m)
        m_tgt = np.mean(data_tgt, axis=1)
        m = expmap(m_tgt, m)

    return m


def in_domain(domain, x):
    """
    Check if x is in domain

    Parameters
    ----------
    :param domain: (gpflowopt) domain
    :param x: data

    Returns
    -------
    :return: True is x is in domain, False otherwise
    """
    return x in domain


def project_to_domain(domain, x0):
    x = np.copy(x0)
    indices = np.arange(x.shape[0])
    indices_to_ignore = []

    # Check if bound-constraints are respected for each dimension and change the value of the indices that do not
    for d in range(x.shape[0]):
        if x[d] < domain.lower[d]:
            x[d] = domain.lower[d]
            indices_to_ignore.append(d)
        elif x[d] > domain.upper[d]:
            x[d] = domain.upper[d]
            indices_to_ignore.append(d)

    # Adapt remaining indices to have a point on the unit sphere
    indices_to_change = list(set(indices) - set(indices_to_ignore))
    x[indices_to_change] = x[indices_to_change] / np.linalg.norm(x[indices_to_change]) \
                           * np.sqrt((1-np.sum(x[indices_to_ignore]**2)))

    # Check if changed indices respect the bound constraints
    in_domain = 0
    # ! If the constraints are not feasible, we may stay stucked here.
    while in_domain is 0:
        in_domain = 1
        # If not change the value of the indices that are out of the bound
        for d in indices_to_change:
            if x[d] < domain.lower[d]:
                x[d] = domain.lower[d]
                indices_to_ignore.append(d)
                in_domain *= 0
            elif x[d] > domain.upper[d]:
                x[d] = domain.upper[d]
                indices_to_ignore.append(d)
                in_domain *= 0

        # Adapt remaining indices to have a point on the unit sphere
        indices_to_change = list(set(indices) - set(indices_to_ignore))
        x[indices_to_change] = x[indices_to_change] / np.linalg.norm(x[indices_to_change]) \
                               * np.sqrt((1 - np.sum(x[indices_to_ignore] ** 2)))

    return x


# OLD FUNCTION (not robust)
# def project_to_domain(domain, x, nelems=100):
#     """
#     Project a point on the sphere to a domain contained in the manifold.
#
#     Parameters
#     ----------
#     :param domain: (gpflowopt) domain
#     :param x: datapoint
#
#     Optional parameters
#     -------------------
#     :param nelems: number of elements for "linesearch"
#
#     Returns
#     -------
#     :return: project point
#     """
#
#     factors = np.linspace(0, 1, nelems)
#     for d in range(x.shape[0]):
#         if x[d] < domain.lower[d]:
#             pole = np.zeros(x.shape)
#             pole[d] = 1.
#             dir_geodesic = logmap(pole, x)
#             # dir_geodesic = self.log(x, pole)
#             for i in range(nelems):
#                 newx = expmap(factors[i] * dir_geodesic, x)
#                 # newx = self.exp(x, factors[i] * dir_geodesic)
#                 if newx[d] > domain.lower[d]:
#                     x = newx
#                     break
#         elif x[d] > domain.upper[d]:
#             pole = np.zeros(x.shape)
#             pole[d] = -1.
#             dir_geodesic = logmap(pole, x)
#             # dir_geodesic = self.log(x, pole)
#             for i in range(nelems):
#                 newx = expmap(factors[i] * dir_geodesic, x)
#                 if newx[d] < domain.upper[d]:
#                     x = newx
#                     break
#     return x[:, 0]


def get_axisangle(d):
    """
    Gets axis-angle representation of a point lying on the sphere
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param d: point on the sphere

    Returns
    -------
    :return: axis, angle: corresponding axis and angle representation
    """
    norm = np.sqrt(d[0]**2 + d[1]**2)
    if norm < 1e-6:
        return np.array([0,0,1]), 0
    else:
        vec = np.array( [-d[1], d[0], 0])
        return vec/norm,np.arccos(d[2])

