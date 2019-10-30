import numpy as np
import tensorflow as tf

'''
Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
'''


def vector_to_symmetric_matrix_tf(vect, dim):
    """
    Vector to symmetric matrix using Mandel notation

    Parameters
    ----------
    :param vect: vectors                                    nb_data x dim_vect
    :param dim: side dimension of the symmetric matrix

    Returns
    -------
    :return: symmetric matrices
    """
    if np.size(vect.get_shape()) == 1:
        mat = tf.expand_dims(vect, 0)

    id = np.cumsum(range(dim, 0, -1))

    T = tf.constant(np.zeros((dim, dim)))
    T = tf.expand_dims(T, 0)
    T = tf.tile(T, [tf.shape(vect)[0], 1, 1])

    # Upper diagonal part of the reconstructed matrix
    for i in range(0, dim - 1):
        idlist = list(range(id[i], id[i + 1]))
        dim_tmp = len(idlist)

        # Create a small diagonal matrix
        Ttmp = tf.constant(np.zeros((dim_tmp, dim_tmp)))
        Ttmp = tf.expand_dims(Ttmp, 0)
        Ttmp = tf.tile(Ttmp, [tf.shape(vect)[0], 1, 1])

        Ttmp = tf.matrix_set_diag(Ttmp, tf.gather(vect, idlist, axis=1))

        # Matrix for columns expension
        Ttmp_col = tf.constant(np.zeros((dim_tmp, dim - dim_tmp)))
        Ttmp_col = tf.expand_dims(Ttmp_col, 0)
        Ttmp_col = tf.tile(Ttmp_col, [tf.shape(vect)[0], 1, 1])

        # Matrix for rows expension
        Ttmp_row = tf.constant(np.zeros((dim - dim_tmp, dim)))
        Ttmp_row = tf.expand_dims(Ttmp_row, 0)
        Ttmp_row = tf.tile(Ttmp_row, [tf.shape(vect)[0], 1, 1])

        # Concatenate the matrices and add them to the reconstructed symmetric one
        T = tf.add(T, tf.concat([tf.concat([Ttmp_col, Ttmp], axis=2), Ttmp_row], axis=1))

    # Scale and set lower diagonal part
    T = tf.divide(T, np.sqrt(2))
    T = tf.add(tf.transpose(T, perm=[0, 2, 1]), T)

    # Add main diagonal
    T = tf.matrix_set_diag(T, tf.gather(vect, list(range(dim)), axis=1))

    return T


def symmetric_matrix_to_vector_tf(mat, dim):
    """
    Symmetric matrix to vector using Mandel notation

    Parameters
    ----------
    :param mat: symmetric matrices                          nb_data x dim x dim
    :param dim: side dimension of the symmetric matrix

    Returns
    -------
    :return: vectors
    """
    if np.size(mat.get_shape()) == 2:
        mat = tf.expand_dims(mat, 0)

    # Extract main diagonal
    vect_comp = [tf.matrix_diag_part(mat)]

    # Extract side diagonals, multiplied by sqrt(2)
    for i in range(1, dim):
        vect_comp.append(tf.multiply(tf.matrix_diag_part(mat[:, i:dim, 0:dim - i]), np.sqrt(2)))

    # Concatenate the different diagonals
    return tf.concat(vect_comp, axis=1)


def affine_invariant_distance_tf(S1, S2, full_dist_mat=False):
    """
    SPD affine invariant distance

    Parameters
    ----------
    :param S1: SPD matrices                 nb_data x dim x dim
    :param S2: SPD matrices                 nb_data x dim x dim

    Optional parameters
    -------------------
    :param full_dist_mat: if True return a matrix of the distances of each matrix of S1 with each one of S2

    Returns
    -------
    :return: affine invariant distance between S1 and S2
    """
    if full_dist_mat is True:
        # In this case the distance between each matrix of S1 is computed with each matrix of S2
        # Useful for example for kernel computation
        # In this case, the preprocessing step extends the dimensions of the input tensors to compute the distances
        if np.size(S1.get_shape()) == 2:
            S1 = tf.expand_dims(S1, 0)
        if np.size(S2.get_shape()) == 2:
            S2 = tf.expand_dims(S2, 0)

        S1 = tf.expand_dims(S1, 1)
        S2 = tf.expand_dims(S2, 0)

        # Repeat x and y data along 1- and 0- dimensions to have ndata_S1 x ndata_S2 x dim x dim arrays
        S1 = tf.tile(S1, [1, tf.shape(S2)[1], 1, 1])
        S2 = tf.tile(S2, [tf.shape(S1)[0], 1, 1, 1])

    # Compute the distance between each pair of matrices
    S1_invhalf = tf.linalg.sqrtm(tf.matrix_inverse(S1))

    mult_tens = tf.matmul(tf.matmul(S1_invhalf, S2), S1_invhalf)

    # eigval, _ = tf.self_adjoint_eig(mult_tens)  # does not give eigenvalues, maybe due to need of self_adjoint matrices
    eigval, _, _ = tf.svd(mult_tens)  # still not sure why svd seems to give directly the eigenvalues here...

    log_eig = tf.log(eigval)

    sum_log2_eig = tf.reduce_sum(tf.multiply(log_eig, log_eig), -1)
    return tf.sqrt(sum_log2_eig)


def expmap_tf(U, S):
    """
    Exponential map

    Parameters
    ----------
    :param U: symmetric matrices                nb_data x dim x dim or dim x dim
    :param S: SPD matrix                        dim x dim

    Returns
    -------
    :return: SPD matrix computed as Expmap_S(U)
    """
    # Expend S dimensions if U contains multiple matrices
    if np.size(U.get_shape()) > 2:
        S = tf.expand_dims(S, 0)
        S = tf.tile(S, [tf.shape(U)[0], 1, 1])

    # Compute expmap
    invSU = tf.linalg.solve(S, U)

    expinvSU = tf.cast(tf.linalg.expm(tf.cast(invSU, tf.complex64)), tf.float64)

    X = tf.matmul(S, expinvSU)

    return X


def logmap_tf(X, S):
    """
    Logarithm map

    Parameters
    ----------
    :param X: SPD matrices                      nb_data x dim x dim or dim x dim
    :param S: SPD matrix                        dim x dim

    Returns
    -------
    :return: symmetric matrix computed as Logmap_S(X)
    """
    # Expend S dimensions if X contains multiple matrices
    if np.size(X.get_shape()) > 2:
        S = tf.expand_dims(S, 0)
        S = tf.tile(S, [tf.shape(X)[0], 1, 1])

    # Compute expmap
    invSX = tf.linalg.solve(S, X)

    loginvSX = tf.cast(tf.linalg.logm(tf.cast(invSX, tf.complex64)), tf.float64)

    U = tf.matmul(S, loginvSX)

    return U
