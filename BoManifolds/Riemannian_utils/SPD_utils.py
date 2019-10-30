#! /usr/bin/env python

import numpy as np
import scipy.linalg as sc_la

from BoManifolds.Riemannian_utils.geometry_utils import rotation_matrix_from_axis_angle

'''
Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
'''


def tensor_matrix_product(T, U, mode):
    """
    Tensor-matrix product

    Parameters
    ----------
    :param T: tensor
    :param U: matrix
    :param mode: mode of the product

    Returns
    -------
    :return: tensor T x_mode U
    """
    # Mode-n tensor-matrix product
    N = len(T.shape)

    # Compute the complement of the set of modes
    modec = range(0, N)
    modec.remove(mode)

    # Permutation of the tensor
    perm = [mode] + modec
    S = np.transpose(T, perm)
    sizeS = S.shape
    S = S.reshape((sizeS[0], -1), order='F')

    # n-mode product
    S = np.dot(U, S)
    sizeS = U.shape[0:1] + sizeS[1:]
    S = S.reshape(sizeS, order='F')

    # Inverse permutation
    inv_perm = [0]*N
    for i in range(0, N):
        inv_perm[perm[i]] = i

    S = np.transpose(S, inv_perm)

    return S


def symmetric_matrix_to_vector_mandel(M):
    """
    Symmetric matrix to vector using Mandel notation

    Parameters
    ----------
    :param M: symmetric matrix

    Returns
    -------
    :return: vector
    """
    N = M.shape[0]

    v = np.copy(M.diagonal())

    for i in range(1, N):
        v = np.concatenate((v, 2.0**0.5*M.diagonal(i)))

    return v


def vector_to_symmetric_matrix_mandel(v):
    """
    Vector to symmetric matrix using Mandel notation

    Parameters
    ----------
    :param v: vector

    Returns
    -------
    :return: symmetric matrix M
    """
    n = v.shape[0]
    N = int((-1.0 + (1.0+8.0*n)**0.5)/2.0)

    M = np.copy(np.diag(v[0:N]))

    id = np.cumsum(range(N,0,-1))

    for i in range(0, N-1):
        M += np.diag(v[range(id[i], id[i+1])], i+1) / 2.0**0.5 + np.diag(v[range(id[i], id[i+1])], -i-1) / 2.0**0.5

    return M


def expmap(U, S):
    """
    Exponential map

    Parameters
    ----------
    :param U: symmetric matrix
    :param S: SPD matrix

    Returns
    -------
    :return: SPD matrix computed as Expmap_S(U)
    """
    D, V = np.linalg.eig(np.linalg.solve(S, U))
    X = S.dot(V.dot(np.diag(np.exp(D))).dot(np.linalg.inv(V)))

    return X


def logmap(X, S):
    """
    Logarithm map

    Parameters
    ----------
    :param X: SPD matrix
    :param S: SPD matrix

    Returns
    -------
    :return: symmetric matrix computed as Logmap_S(X)
    """
    D, V = np.linalg.eig(np.linalg.solve(S, X))
    U = S.dot(V.dot(np.diag(np.log(D))).dot(np.linalg.inv(V)))

    return U


def expmap_mandel_vector(u, s):
    """
    Exponential map using Mandel notation

    Parameters
    ----------
    :param u: symmetrix matrix in Mandel notation form
    :param s: SPD matrix in Mandel notation form

    Returns
    -------
    :return: SPD matrix computed as Expmap_S(U) in Mandel notation form
    """
    U = vector_to_symmetric_matrix_mandel(u)
    S = vector_to_symmetric_matrix_mandel(s)

    return symmetric_matrix_to_vector_mandel(expmap(U, S))


def logmap_mandel_vector(x, s):
    """
    Logarithm map using Mandel notation

    Parameters
    ----------
    :param x: SPD matrix in Mandel notation form
    :param s: SPD matrix in Mandel notation form

    Returns
    -------
    :return: symmetric matrix computed as Logmap_S(X) in Mandel notation form
    """
    X = vector_to_symmetric_matrix_mandel(x)
    S = vector_to_symmetric_matrix_mandel(s)

    return symmetric_matrix_to_vector_mandel(logmap(X, S))


def affine_invariant_distance(S1, S2):
    """
    SPD affine invariant distance

    Parameters
    ----------
    :param S1: SPD matrix
    :param S2: SPD matrix

    Returns
    -------
    :return: affine invariant distance between S1 and S2
    """
    # S1_pow = sc_la.fractional_matrix_power(S1, -0.5)
    # return np.linalg.norm(sc_la.logm(np.dot(np.dot(S1_pow, S2), S1_pow)), 'fro')

    eigv, _ = np.linalg.eig(np.dot(np.linalg.inv(S1), S2))
    return np.sqrt(np.sum(np.log(eigv)*np.log(eigv)))


def parallel_transport_operator(S1, S2):
    """
    Parallel transport operation

    Parameters
    ----------
    :param S1: SPD matrix
    :param S2: SPD matrix

    Returns
    -------
    :return: parallel transport operator
    """
    return sc_la.fractional_matrix_power(np.dot(S2, np.linalg.inv(S1)), 0.5)


def parallel_transport_operator_mandel_vector(s1, s2):
    """
    Parallel transport operation

    Parameters
    ----------
    :param S1: SPD matrix
    :param S2: SPD matrix

    Returns
    -------
    :return: parallel transport operator
    """
    S1 = vector_to_symmetric_matrix_mandel(s1)
    S2 = vector_to_symmetric_matrix_mandel(s2)

    return parallel_transport_operator(S1, S2)


def karcher_mean_spd(data, nb_iter=10):
    """
    This function computes the mean of points lying on the manifold (Fréchet/Karcher mean).

    Parameters
    ----------
    :param data: data points lying on the manifold      N x nb_dim x nb_dim

    Optional parameters
    -------------------
    :param nb_iter: number of iterations

    Returns
    -------
    :return: mean of the datapoints
    """
    nb_data = data.shape[0]

    # Initialize the mean as equal to the first datapoint
    m = data[0]
    for i in range(nb_iter):
        data_tgt = logmap(data[0], m)
        for n in range(1, nb_data):
            data_tgt += logmap(data[n], m)
        m_tgt = data_tgt / nb_data
        m = expmap(m_tgt, m)

    return m


def karcher_mean_spd_mandel_vector(data, nb_iter=10):
    """
    This function computes the mean of points lying on the manifold (Fréchet/Karcher mean).

    Parameters
    ----------
    :param data: data points lying on the manifold      nb_dim_vec x N

    Optional parameters
    -------------------
    :param nb_iter: number of iterations

    Returns
    -------
    :return: mean of the datapoints
    """
    nb_data = data.shape[1]

    # Initialize the mean as equal to the first datapoint
    m = data[:, 0]
    for i in range(nb_iter):
        data_tgt = logmap_mandel_vector(data[:, 0], m)
        for n in range(1, nb_data):
            data_tgt += logmap_mandel_vector(data[:, n], m)
        m_tgt = data_tgt / nb_data
        m = expmap_mandel_vector(m_tgt, m)

    return m

# OLD FUNCTION
# def in_domain(domain, x):
#     """
#     This function checks if a symmetric matrix is in a domain defined by the Mandel notation
#
#     Parameters
#     ----------
#     :param domain: domain.upper and domain.lower contains the bounds of the domain
#     :param x: symmetric matrix
#
#     Returns
#     -------
#     :return: True if the matrix is in the domain, False otherwise
#     """
#     if symmetric_matrix_to_vector_mandel(x) in domain:
#         return True
#     else:
#         return False


def in_domain(domain, x):
    """
    This function checks if a SPD matrix is in a domain defined by minimum and maximum eigenvalues

    Parameters
    ----------
    :param domain: domain.upper and domain.lower contains the bounds of the domain
    :param x: SPD matrix

    Returns
    -------
    :return: True if the matrix is in the domain, False otherwise
    """
    max_eig = domain.upper[0]
    min_eig = domain.lower[0]

    D = np.linalg.eigvals(x)

    if np.min(D) < min_eig:
        return False
    elif np.max(D) > max_eig:
        return False
    else:
        return True


def project_to_domain(domain, x):
    """
    This function scales the eigenvalue of a matrix so that they are constrained in a specific domain

    Parameters
    ----------
    :param domain: domain.upper[0] and domain.lower[0] contain the maximum and minimum eigenvalue, respectively.
    :param x: matrix to project in the domain

    Returns
    -------
    :return: matrix with rescaled eigenvalues
    """
    max_eig = domain.upper[0]
    min_eig = domain.lower[0]

    D, V = np.linalg.eig(x)
    D[D < min_eig] = min_eig
    D[D > max_eig] = max_eig
    return np.dot(V, np.dot(np.diag(D), V.T))
