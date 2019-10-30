import numpy as np
import tensorflow as tf

'''
Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
'''


def sphere_distance_tf(x1, x2, diag=False):
    """
    This function computes the Riemannian distance between points on a sphere manifold.

    Parameters
    ----------
    :param x1: points on the sphere                                             N1 x dim or b1 x ... x bk x N1 x dim
    :param x2: points on the sphere                                             N2 x dim or b1 x ... x bk x N2 x dim

    Optional parameters
    -------------------
    :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.

    Returns
    -------
    :return: matrix of manifold distance between the points in x1 and x2         N1 x N2 or b1 x ... x bk x N1 x N2
    """
    # If only one data is provided, put it in format 1 x ndims
    # if np.size(x.get_shape()) == 1:
    #     x = tf.expand_dims(x, 0)
    # if np.size(y.get_shape()) == 1:
    #     y = tf.expand_dims(y, 0)

    if diag is False:
        # In this case the distance between each vector of x is computed with each matrix of y
        # Useful for example for kernel computation
        # In this case, the preprocessing step extends the dimensions of the input tensors to compute the distances

        # Expand dimensions to compute all vector-vector distances
        x1 = tf.expand_dims(x1, 1)
        x2 = tf.expand_dims(x2, 0)

        # Repeat x and y data along 1- and 0- dimensions to have ndata_x x ndata_y x dim arrays
        x1 = tf.tile(x1, [1, tf.shape(x2)[1], 1])
        x2 = tf.tile(x2, [tf.shape(x1)[0], 1, 1])

        # Expand dimension to perform inner product
        x1 = tf.expand_dims(x1, 2)
        x2 = tf.expand_dims(x2, 3)

        # Compute the inner product (should be [-1,1])
        inner_product = tf.squeeze(tf.matmul(x1, x2), [2, 3])

    else:
        # Compute the inner product (should be [-1,1])
        inner_product = tf.linalg.tensor_diag_part(tf.matmul(x1, tf.transpose(x2)))

    # Clamp in case any value is not in the interval [-1,1]
    inner_product = tf.maximum(tf.minimum(inner_product, 1), -1)
    return tf.acos(inner_product)