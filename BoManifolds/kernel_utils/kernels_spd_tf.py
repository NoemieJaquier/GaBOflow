import numpy as np
import gpflow
import tensorflow as tf
from BoManifolds.Riemannian_utils.SPD_utils_tf import vector_to_symmetric_matrix_tf, affine_invariant_distance_tf

'''
Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
'''

# The SPD kernels are implemented here for GPflow version = 0.5 (used by GPflowOpt) and version >=1.2.0.
if gpflow.__version__ == '0.5':
    class SpdSteinGaussianKernel(gpflow.kernels.Kern):
        """
        Instances of this class represent a Stein covariance matrix between input points on the SPD manifold
        using the affine-invariant distance.

        Attributes
        ----------
        self.matrix_dim: SPD matrix dimension, computed from the dimension of the inputs (given with Mandel notation)
        self.continuous_param_space_limit: low limit of the continous parameter space where the inverse square
        lengthscale parameter beta results in PD kernels
        self.beta_shifted: equal to beta-continuous_param_space_limit and used to optimize beta in the space of
        beta values in the continuous parameter space resulting in PD kernels
        self.variance: variance parameter of the kernel

        Methods
        -------
        K(point1_in_SPD, point2_in_SPD):
        Kdiag(point1_in_SPD):
        update_beta(new_beta_value):
        get_beta():

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta, variance=1.0):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension (in Mandel notation form)
            :param active_dims: dimensions of the input used for kernel computation (in Mandel notation form),
                                defined as range(input_dim) if all the input dimensions are considered.

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Matrix dimension from input vector dimension
            self.matrix_dim = int((-1.0 + (1.0 + 8.0 * input_dim) ** 0.5) / 2.0)

            # Lower limit of the continuous space where beta can be optimized continuously
            # beta \in [j/2: 1 <= j <= n-1] U ]0.5(n-1), +inf[
            self.continuous_param_space_limit = 0.5 * (self.matrix_dim - 1)

            # Parameter initialization
            # Beta shifted is used to optimize beta in the continuous part of its space
            # The values of beta in the discrete part of the space have to be tested separetely, i.e. by comparing the
            # obtained log marginal likelihood with the discrete value to the one obtained by optimizing in the
            # continous part of the space. Therefore, do not optimize the kernel for initial values of the parameter
            # beta smaller than self.continuous_param_space_limit.
            self.beta_shifted = gpflow.param.Param(beta - self.continuous_param_space_limit, transform=gpflow.transforms.positive)
            self.variance = gpflow.param.Param(variance, transform=gpflow.transforms.positive)

        def K(self, X, X2=None):
            """
            Computes the Stein kernel matrix between inputs X (and X2) belonging to a SPD manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold (Mandel notation)

            Optional parameters
            -------------------
            :param X2: input points on the SPD manifold (Mandel notation)

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            # Transform input vector to matrices
            X = vector_to_symmetric_matrix_tf(X, self.matrix_dim)

            if X2 is None:
                X2 = X
            else:
                X2 = vector_to_symmetric_matrix_tf(X2, self.matrix_dim)

            # Compute beta value from beta_shifted
            beta = self.beta_shifted + self.continuous_param_space_limit

            # Compute the kernel
            X = tf.expand_dims(X, 1)
            X2 = tf.expand_dims(X2, 0)

            X = tf.tile(X, [1, tf.shape(X2)[1], 1, 1])
            X2 = tf.tile(X2, [tf.shape(X)[0], 1, 1, 1])

            mult_XX2 = tf.matmul(X, X2)

            add_halfXX2 = 0.5 * tf.add(X, X2)

            detmult_XX2 = tf.linalg.det(mult_XX2)
            detadd_halfXX2 = tf.linalg.det(add_halfXX2)

            dist = tf.divide(tf.math.pow(detmult_XX2, beta), tf.math.pow(detadd_halfXX2, beta))
            return tf.multiply(self.variance, dist)

        def Kdiag(self, X):
            """
            Computes the diagonal of Gaussian kernel matrix of inputs X belonging to a sphere manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold

            Optional parameters
            -------------------

            Returns
            -------
            :return: diagonal of the kernel matrix of X
            """
            return tf.linalg.tensor_diag_part(self.K(X))

        def update_beta(self, beta):
            """
            Update the parameter beta of the class.

            Parameters
            ----------
            :param beta: new value of beta

            Optional parameters
            -------------------

            Returns
            -------
            :return:
            """
            self.beta_shifted = (beta - self.continuous_param_space_limit)

        def get_beta(self):
            """
            Return the parameter beta of the class.

            Parameters
            ----------

            Optional parameters
            -------------------

            Returns
            -------
            :return: value of beta
            """
            return self.beta_shifted.value + self.continuous_param_space_limit


    class SpdAffineInvariantGaussianKernel(gpflow.kernels.Kern):
        """
        Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the SPD manifold
        using the affine-invariant distance.

        Attributes
        ----------
        self.matrix_dim: SPD matrix dimension, computed from the dimension of the inputs (given with Mandel notation)
        self.beta_min: minimum value of the inverse square lengthscale parameter beta
        self.beta_shifted: equal to beta-beta_min and used to optimize beta in the space of beta values resulting in
        PD kernels
        self.variance: variance parameter of the kernel

        Methods
        -------
        K(point1_in_SPD, point2_in_SPD):
        Kdiag(point1_in_SPD):
        update_beta(new_beta_value):
        get_beta():

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta_min, beta = 1., variance=1.):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension (in Mandel notation form)
            :param active_dims: dimensions of the input used for kernel computation (in Mandel notation form),
                                defined as range(input_dim) if all the input dimensions are considered.
            :param beta_min: minimum value of the inverse square lengthscale parameter beta

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Matrix dimension from input vector dimension
            self.matrix_dim = int((-1.0 + (1.0 + 8.0 * input_dim) ** 0.5) / 2.0)

            # Parameter initialization
            # Beta shifted is used to optimize beta in the space of beta values resulting in PD kernels
            self.beta_shifted = gpflow.param.Param(beta, transform=gpflow.transforms.positive)
            self.variance = gpflow.param.Param(variance, transform=gpflow.transforms.positive)
            self.beta_min_value = beta_min

        def K(self, X, X2=None):
            """
            Computes the Gaussian kernel matrix between inputs X (and X2) belonging to a SPD manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold (Mandel notation)

            Optional parameters
            -------------------
            :param X2: input points on the SPD manifold (Mandel notation)

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            # Transform input vector to matrices
            X = vector_to_symmetric_matrix_tf(X, self.matrix_dim)

            if X2 is None:
                X2 = X
            else:
                X2 = vector_to_symmetric_matrix_tf(X2, self.matrix_dim)

            # Compute beta value from beta_shifted
            beta = self.beta_shifted + self.beta_min_value

            # Compute the kernel
            aff_inv_dist = affine_invariant_distance_tf(X, X2, full_dist_mat=True)
            aff_inv_dist2 = tf.square(aff_inv_dist)

            # aff_inv_dist = tf.divide(aff_inv_dist, tf.multiply(self.beta_param, self.beta_param))
            aff_inv_dist2 = tf.multiply(aff_inv_dist2, beta)

            return tf.multiply(self.variance, tf.exp(-aff_inv_dist2))

        def Kdiag(self, X):
            """
            Computes the diagonal of Gaussian kernel matrix of inputs X belonging to a sphere manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold

            Optional parameters
            -------------------

            Returns
            -------
            :return: diagonal of the kernel matrix of X
            """
            return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

        def update_beta(self, beta):
            """
            Update the parameter beta of the class.

            Parameters
            ----------
            :param beta: new value of beta

            Optional parameters
            -------------------

            Returns
            -------
            :return:
            """
            self.beta_shifted = beta - self.beta_min_value

        def get_beta(self):
            """
            Return the parameter beta of the class.

            Parameters
            ----------

            Optional parameters
            -------------------

            Returns
            -------
            :return: value of beta
            """
            return self.beta_shifted.value + self.beta_min_value

        # Checks for affine-invariant kernel
        # X = tf.convert_to_tensor(y_man_mat)
        # X2 = tf.convert_to_tensor(y_man_mat_test)
        #
        # aff_inv_dist = affine_inv_distance_tf(X, X2, full_dist_mat=True)
        #
        # with tf.Session() as sess:
        #     x_np = sess.run(X)
        #     x2_np = sess.run(X2)
        #     affinv_np = sess.run(aff_inv_dist)
        #
        # affinv_check = np.zeros((y_man_mat.shape[0], y_man_mat_test.shape[0]))
        # for m in range(y_man_mat.shape[0]):
        #     for n in range(y_man_mat_test.shape[0]):
        #         affinv_check[m, n] = aff_invariant_distance(y_man_mat[m], y_man_mat_test[n])
        #
        # kernel_check = np.exp(- affinv_check**2 / 1.0)


    class SpdAffineInvariantLaplaceKernel(gpflow.kernels.Kern):
        """
        Instances of this class represent a Laplace covariance matrix between input points on the SPD manifold
        using the affine-invariant distance.

        Attributes
        ----------
        self.matrix_dim: SPD matrix dimension, computed from the dimension of the inputs (given with Mandel notation)
        self.beta_min: minimum value of the inverse square lengthscale parameter beta
        self.beta_shifted: equal to beta-beta_min and used to optimize beta in the space of beta values resulting in
        PD kernels
        self.variance: variance parameter of the kernel

        Methods
        -------
        K(point1_in_SPD, point2_in_SPD):
        Kdiag(point1_in_SPD):
        update_beta(new_beta_value):
        get_beta():

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta_min=0., beta = 1., variance=1.):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension (in Mandel notation form)
            :param active_dims: dimensions of the input used for kernel computation (in Mandel notation form),
                                defined as range(input_dim) if all the input dimensions are considered.
            :param beta_min: minimum value of the inverse square lengthscale parameter beta

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Matrix dimension from input vector dimension
            self.matrix_dim = int((-1.0 + (1.0 + 8.0 * input_dim) ** 0.5) / 2.0)

            # Parameter initialization
            # Beta shifted is used to optimize beta in the space of beta values resulting in PD kernels
            self.beta_shifted = gpflow.param.Param(beta, transform=gpflow.transforms.positive)
            self.variance = gpflow.param.Param(variance, transform=gpflow.transforms.positive)
            self.beta_min_value = beta_min

        def K(self, X, X2=None):
            """
            Computes the Laplace kernel matrix between inputs X (and X2) belonging to a SPD manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold (Mandel notation)

            Optional parameters
            -------------------
            :param X2: input points on the SPD manifold (Mandel notation)

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            # Transform input vector to matrices
            X = vector_to_symmetric_matrix_tf(X, self.matrix_dim)

            if X2 is None:
                X2 = X
            else:
                X2 = vector_to_symmetric_matrix_tf(X2, self.matrix_dim)

            # Compute beta value from beta_shifted
            beta = self.beta_shifted + self.beta_min_value

            # Compute the kernel
            aff_inv_dist = affine_invariant_distance_tf(X, X2, full_dist_mat=True)

            aff_inv_dist = tf.multiply(aff_inv_dist, beta)

            return tf.multiply(self.variance, tf.exp(-aff_inv_dist))

        def Kdiag(self, X):
            """
            Computes the diagonal of Gaussian kernel matrix of inputs X belonging to a sphere manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold

            Optional parameters
            -------------------

            Returns
            -------
            :return: diagonal of the kernel matrix of X
            """
            return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

        def update_beta(self, beta):
            """
            Update the parameter beta of the class.

            Parameters
            ----------
            :param beta: new value of beta

            Optional parameters
            -------------------

            Returns
            -------
            :return:
            """
            self.beta_shifted = beta - self.beta_min_value

        def get_beta(self):
            """
            Return the parameter beta of the class.

            Parameters
            ----------

            Optional parameters
            -------------------

            Returns
            -------
            :return: value of beta
            """
            return self.beta_shifted.value + self.beta_min_value


    class SpdFrobeniusGaussianKernel(gpflow.kernels.Kern):
        """
        Instances of this class represent a Frobenius covariance matrix between input points on the SPD manifold
        using the affine-invariant distance.

        Attributes
        ----------
        self.matrix_dim: SPD matrix dimension, computed from the dimension of the inputs (given with Mandel notation)
        self.beta_param: inverse square lengthscale parameter beta
        self.variance: variance parameter of the kernel

        Methods
        -------
        K(point1_in_SPD, point2_in_SPD):
        Kdiag(point1_in_SPD):

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta = 1., variance=1.):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension (in Mandel notation form)
            :param active_dims: dimensions of the input used for kernel computation (in Mandel notation form),
                                defined as range(input_dim) if all the input dimensions are considered.

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Matrix dimension from input vector dimension
            self.matrix_dim = int((-1.0 + (1.0 + 8.0 * input_dim) ** 0.5) / 2.0)

            # Parameter initialization
            self.beta_param = gpflow.param.Param(beta, transform=gpflow.transforms.positive)
            self.variance = gpflow.param.Param(variance, transform=gpflow.transforms.positive)

        def K(self, X, X2=None):
            """
            Computes the Frobenius kernel matrix between inputs X (and X2) belonging to a SPD manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold (Mandel notation)

            Optional parameters
            -------------------
            :param X2: input points on the SPD manifold (Mandel notation)

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            # Transform input vector to matrices
            X = vector_to_symmetric_matrix_tf(X, self.matrix_dim)

            if X2 is None:
                X2 = X
            else:
                X2 = vector_to_symmetric_matrix_tf(X2, self.matrix_dim)

            # Compute the kernel
            X = tf.expand_dims(X, 1)
            X2 = tf.expand_dims(X2, 0)

            X = tf.tile(X, [1, tf.shape(X2)[1], 1, 1])
            X2 = tf.tile(X2, [tf.shape(X)[0], 1, 1, 1])

            diff_XX2 = tf.subtract(X, X2)

            frob_dist = tf.norm(diff_XX2, axis=(-2, -1))
            frob_dist2 = tf.square(frob_dist)

            frob_dist2 = tf.multiply(frob_dist2, self.beta_param)

            return tf.multiply(self.variance, tf.exp(-frob_dist2))

        def Kdiag(self, X):
            """
            Computes the diagonal of Gaussian kernel matrix of inputs X belonging to a sphere manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold

            Optional parameters
            -------------------

            Returns
            -------
            :return: diagonal of the kernel matrix of X
            """
            return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

        # Numpy check
        # frobdist = np.zeros((y_man_mat.shape[0], y_man_mat_test.shape[0]))
        # for m in range(y_man_mat.shape[0]):
        #     for n in range(y_man_mat_test.shape[0]):
        #         frobdist[m, n] = np.linalg.norm(y_man_mat[m] - y_man_mat_test[n])
        #
        # kernel_check = np.exp(- frobdist**2 / 1.0)


    class SpdLogEuclideanGaussianKernel(gpflow.kernels.Kern):
        """
        Instances of this class represent a Log-Euclidean covariance matrix between input points on the SPD manifold
        using the affine-invariant distance.

        Attributes
        ----------
        self.matrix_dim: SPD matrix dimension, computed from the dimension of the inputs (given with Mandel notation)
        self.beta_param: inverse square lengthscale parameter beta
        self.variance: variance parameter of the kernel

        Methods
        -------
        K(point1_in_SPD, point2_in_SPD):
        Kdiag(point1_in_SPD):

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta = 1., variance=1.):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension (in Mandel notation form)
            :param active_dims: dimensions of the input used for kernel computation (in Mandel notation form),
                                defined as range(input_dim) if all the input dimensions are considered.

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Matrix dimension from input vector dimension
            self.matrix_dim = int((-1.0 + (1.0 + 8.0 * input_dim) ** 0.5) / 2.0)

            # Parameter initialization
            self.beta_param = gpflow.param.Param(beta, transform=gpflow.transforms.positive)
            self.variance = gpflow.param.Param(variance, transform=gpflow.transforms.positive)

        def K(self, X, X2=None):
            """
            Computes the Log-Euclidean kernel matrix between inputs X (and X2) belonging to a SPD manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold (Mandel notation)

            Optional parameters
            -------------------
            :param X2: input points on the SPD manifold (Mandel notation)

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            # Transform input vector to matrices
            X = vector_to_symmetric_matrix_tf(X, self.matrix_dim)

            if X2 is None:
                X2 = X
            else:
                X2 = vector_to_symmetric_matrix_tf(X2, self.matrix_dim)

            # Compute the kernel
            X = tf.expand_dims(X, 1)
            X2 = tf.expand_dims(X2, 0)

            X = tf.tile(X, [1, tf.shape(X2)[1], 1, 1])
            X2 = tf.tile(X2, [tf.shape(X)[0], 1, 1, 1])

            diff_XX2 = tf.cast(tf.subtract(tf.linalg.logm(tf.cast(X, dtype=tf.complex64)), tf.linalg.logm(tf.cast(X2, dtype=tf.complex64))), dtype=tf.float64)

            logeucl_dist = tf.norm(diff_XX2, axis=(-2, -1))
            logeucl_dist2 = tf.square(logeucl_dist)

            logeucl_dist2 = tf.multiply(logeucl_dist2, self.beta_param)

            return tf.multiply(self.variance, tf.exp(-logeucl_dist2))

        def Kdiag(self, X):
            """
            Computes the diagonal of Gaussian kernel matrix of inputs X belonging to a sphere manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold

            Optional parameters
            -------------------

            Returns
            -------
            :return: diagonal of the kernel matrix of X
            """
            return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

        # Numpy/scipy check
        # logeucldist = np.zeros((y_man_mat.shape[0], y_man_mat_test.shape[0]))
        # for m in range(y_man_mat.shape[0]):
        #     for n in range(y_man_mat_test.shape[0]):
        #         logeucldist[m, n] = np.linalg.norm(sc.linalg.logm(y_man_mat[m]) - sc.linalg.logm(y_man_mat_test[n]))
        #
        # kernel_check = np.exp(- logeucldist**2 / 1.0)

else:
    class SpdSteinGaussianKernel(gpflow.kernels.Kernel):
        """
        Instances of this class represent a Stein covariance matrix between input points on the SPD manifold
        using the affine-invariant distance.

        Attributes
        ----------
        self.matrix_dim: SPD matrix dimension, computed from the dimension of the inputs (given with Mandel notation)
        self.continuous_param_space_limit: low limit of the continous parameter space where the inverse square
        lengthscale parameter beta results in PD kernels
        self.beta_shifted: equal to beta-continuous_param_space_limit and used to optimize beta in the space of
        beta values in the continuous parameter space resulting in PD kernels
        self.variance: variance parameter of the kernel

        Methods
        -------
        K(point1_in_SPD, point2_in_SPD):
        update_beta(new_beta_value):
        get_beta():

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta, variance=1.0):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension (in Mandel notation form)
            :param active_dims: dimensions of the input used for kernel computation (in Mandel notation form),
                                defined as range(input_dim) if all the input dimensions are considered.

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Matrix dimension from input vector dimension
            self.matrix_dim = int((-1.0 + (1.0 + 8.0 * input_dim) ** 0.5) / 2.0)

            # Lower limit of the continuous space where beta can be optimized continuously
            # beta \in [j/2: 1 <= j <= n-1] U ]0.5(n-1), +inf[
            self.continuous_param_space_limit = 0.5 * (self.matrix_dim - 1)

            # Parameter initialization
            # Beta shifted is used to optimize beta in the continuous part of its space
            # The values of beta in the discrete part of the space have to be tested separetely, i.e. by comparing the
            # obtained log marginal likelihood with the discrete value to the one obtained by optimizing in the continous
            # part of the space. Therefore, do not optimize the kernel for initial values of the parameter beta smaller than
            # self.continuous_param_space_limit.
            self.beta_shifted = gpflow.Param(beta - self.continuous_param_space_limit, transform=gpflow.transforms.positive)
            self.variance = gpflow.Param(variance, transform=gpflow.transforms.positive)

        @gpflow.params_as_tensors
        def K(self, X, X2=None):
            """
            Computes the Stein kernel matrix between inputs X (and X2) belonging to a SPD manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold (Mandel notation)

            Optional parameters
            -------------------
            :param X2: input points on the SPD manifold (Mandel notation)

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            # Transform input vector to matrices
            X = vector_to_symmetric_matrix_tf(X, self.matrix_dim)

            if X2 is None:
                X2 = X
            else:
                X2 = vector_to_symmetric_matrix_tf(X2, self.matrix_dim)

            # Compute beta value from beta_shifted
            beta = self.beta_shifted + self.continuous_param_space_limit

            # Compute the kernel
            X = tf.expand_dims(X, 1)
            X2 = tf.expand_dims(X2, 0)

            X = tf.tile(X, [1, tf.shape(X2)[1], 1, 1])
            X2 = tf.tile(X2, [tf.shape(X)[0], 1, 1, 1])

            mult_XX2 = tf.matmul(X, X2)

            add_halfXX2 = 0.5 * tf.add(X, X2)

            detmult_XX2 = tf.linalg.det(mult_XX2)
            detadd_halfXX2 = tf.linalg.det(add_halfXX2)

            dist = tf.divide(tf.math.pow(detmult_XX2, beta), tf.math.pow(detadd_halfXX2, beta))
            return tf.multiply(self.variance, dist)

        def update_beta(self, beta):
            """
            Update the parameter beta of the class.

            Parameters
            ----------
            :param beta: new value of beta

            Optional parameters
            -------------------

            Returns
            -------
            :return:
            """
            self.beta_shifted.assign(beta-self.continuous_param_space_limit)

        def get_beta(self):
            """
            Return the parameter beta of the class.

            Parameters
            ----------

            Optional parameters
            -------------------

            Returns
            -------
            :return: value of beta
            """
            return self.read_trainables()['GPR/kern/beta_shifted'] + self.continuous_param_space_limit


    class SpdAffineInvariantGaussianKernel(gpflow.kernels.Kernel):
        """
        Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the SPD manifold
        using the affine-invariant distance.

        Attributes
        ----------
        self.matrix_dim: SPD matrix dimension, computed from the dimension of the inputs (given with Mandel notation)
        self.beta_min: minimum value of the inverse square lengthscale parameter beta
        self.beta_shifted: equal to beta-beta_min and used to optimize beta in the space of beta values resulting in
        PD kernels
        self.variance: variance parameter of the kernel

        Methods
        -------
        K(point1_in_SPD, point2_in_SPD):
        update_beta(new_beta_value):
        get_beta():

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta_min, beta = 1., variance=1.):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension (in Mandel notation form)
            :param active_dims: dimensions of the input used for kernel computation (in Mandel notation form),
                                defined as range(input_dim) if all the input dimensions are considered.
            :param beta_min: minimum value of the inverse square lengthscale parameter beta

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Matrix dimension from input vector dimension
            self.matrix_dim = int((-1.0 + (1.0 + 8.0 * input_dim) ** 0.5) / 2.0)

            # Parameter initialization
            # Beta shifted is used to optimize beta in the space of beta values resulting in PD kernels
            self.beta_shifted = gpflow.Param(beta, transform=gpflow.transforms.positive)
            self.variance = gpflow.Param(variance, transform=gpflow.transforms.positive)
            self.beta_min_value = beta_min

        @gpflow.params_as_tensors
        def K(self, X, X2=None):
            """
            Computes the Gaussian kernel matrix between inputs X (and X2) belonging to a SPD manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold (Mandel notation)

            Optional parameters
            -------------------
            :param X2: input points on the SPD manifold (Mandel notation)

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            # Transform input vector to matrices
            X = vector_to_symmetric_matrix_tf(X, self.matrix_dim)

            if X2 is None:
                X2 = X
            else:
                X2 = vector_to_symmetric_matrix_tf(X2, self.matrix_dim)

            # Compute beta value from beta_shifted
            beta = self.beta_shifted + self.beta_min_value

            # Compute the kernel
            aff_inv_dist = affine_invariant_distance_tf(X, X2, full_dist_mat=True)
            aff_inv_dist2 = tf.square(aff_inv_dist)

            # aff_inv_dist = tf.divide(aff_inv_dist, tf.multiply(self.beta_param, self.beta_param))
            aff_inv_dist2 = tf.multiply(aff_inv_dist2, beta)

            return tf.multiply(self.variance, tf.exp(-aff_inv_dist2))

        def update_beta(self, beta):
            """
            Update the parameter beta of the class.

            Parameters
            ----------
            :param beta: new value of beta

            Optional parameters
            -------------------

            Returns
            -------
            :return:
            """
            self.beta_shifted.assign(beta-self.beta_min_value)

        def get_beta(self):
            """
            Return the parameter beta of the class.

            Parameters
            ----------

            Optional parameters
            -------------------

            Returns
            -------
            :return: value of beta
            """
            return self.read_trainables()['GPR/kern/beta_shifted'] + self.beta_min_value

        # Checks for affine-invariant kernel
        # X = tf.convert_to_tensor(y_man_mat)
        # X2 = tf.convert_to_tensor(y_man_mat_test)
        #
        # aff_inv_dist = affine_inv_distance_tf(X, X2, full_dist_mat=True)
        #
        # with tf.Session() as sess:
        #     x_np = sess.run(X)
        #     x2_np = sess.run(X2)
        #     affinv_np = sess.run(aff_inv_dist)
        #
        # affinv_check = np.zeros((y_man_mat.shape[0], y_man_mat_test.shape[0]))
        # for m in range(y_man_mat.shape[0]):
        #     for n in range(y_man_mat_test.shape[0]):
        #         affinv_check[m, n] = aff_invariant_distance(y_man_mat[m], y_man_mat_test[n])
        #
        # kernel_check = np.exp(- affinv_check**2 / 1.0)


    class SpdAffineInvariantLaplaceKernel(gpflow.kernels.Kernel):
        """
        Instances of this class represent a Laplace covariance matrix between input points on the SPD manifold
        using the affine-invariant distance.

        Attributes
        ----------
        self.matrix_dim: SPD matrix dimension, computed from the dimension of the inputs (given with Mandel notation)
        self.beta_min: minimum value of the inverse square lengthscale parameter beta
        self.beta_shifted: equal to beta-beta_min and used to optimize beta in the space of beta values resulting in
        PD kernels
        self.variance: variance parameter of the kernel

        Methods
        -------
        K(point1_in_SPD, point2_in_SPD):
        update_beta(new_beta_value):
        get_beta():

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta_min, beta = 1., variance=1.):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension (in Mandel notation form)
            :param active_dims: dimensions of the input used for kernel computation (in Mandel notation form),
                                defined as range(input_dim) if all the input dimensions are considered.
            :param beta_min: minimum value of the inverse square lengthscale parameter beta

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Matrix dimension from input vector dimension
            self.matrix_dim = int((-1.0 + (1.0 + 8.0 * input_dim) ** 0.5) / 2.0)

            # Parameter initialization
            # Beta shifted is used to optimize beta in the space of beta values resulting in PD kernels
            self.beta_shifted = gpflow.Param(beta, transform=gpflow.transforms.positive)
            self.variance = gpflow.Param(variance, transform=gpflow.transforms.positive)
            self.beta_min_value = beta_min

        @gpflow.params_as_tensors
        def K(self, X, X2=None):
            """
            Computes the Laplace kernel matrix between inputs X (and X2) belonging to a SPD manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold (Mandel notation)

            Optional parameters
            -------------------
            :param X2: input points on the SPD manifold (Mandel notation)

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            # Transform input vector to matrices
            X = vector_to_symmetric_matrix_tf(X, self.matrix_dim)

            if X2 is None:
                X2 = X
            else:
                X2 = vector_to_symmetric_matrix_tf(X2, self.matrix_dim)

            # Compute beta value from beta_shifted
            beta = self.beta_shifted + self.beta_min_value

            # Compute the kernel
            aff_inv_dist = affine_invariant_distance_tf(X, X2, full_dist_mat=True)

            aff_inv_dist = tf.multiply(aff_inv_dist, beta)

            return tf.multiply(self.variance, tf.exp(-aff_inv_dist))

        def update_beta(self, beta):
            """
            Update the parameter beta of the class.

            Parameters
            ----------
            :param beta: new value of beta

            Optional parameters
            -------------------

            Returns
            -------
            :return:
            """
            self.beta_shifted.assign(beta-self.beta_min_value)

        def get_beta(self):
            """
            Return the parameter beta of the class.

            Parameters
            ----------

            Optional parameters
            -------------------

            Returns
            -------
            :return: value of beta
            """
            return self.read_trainables()['GPR/kern/beta_shifted'] + self.beta_min_value


    class SpdFrobeniusGaussianKernel(gpflow.kernels.Kernel):
        """
        Instances of this class represent a Frobenius covariance matrix between input points on the SPD manifold
        using the affine-invariant distance.

        Attributes
        ----------
        self.matrix_dim: SPD matrix dimension, computed from the dimension of the inputs (given with Mandel notation)
        self.beta_param: inverse square lengthscale parameter beta
        self.variance: variance parameter of the kernel

        Methods
        -------
        K(point1_in_SPD, point2_in_SPD):

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta = 1., variance=1.):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension (in Mandel notation form)
            :param active_dims: dimensions of the input used for kernel computation (in Mandel notation form),
                                defined as range(input_dim) if all the input dimensions are considered.

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Matrix dimension from input vector dimension
            self.matrix_dim = int((-1.0 + (1.0 + 8.0 * input_dim) ** 0.5) / 2.0)

            # Parameter initialization
            self.beta_param = gpflow.Param(beta, transform=gpflow.transforms.positive)
            self.variance = gpflow.Param(variance, transform=gpflow.transforms.positive)

        @gpflow.params_as_tensors
        def K(self, X, X2=None):
            """
            Computes the Frobenius kernel matrix between inputs X (and X2) belonging to a SPD manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold (Mandel notation)

            Optional parameters
            -------------------
            :param X2: input points on the SPD manifold (Mandel notation)

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            # Transform input vector to matrices
            X = vector_to_symmetric_matrix_tf(X, self.matrix_dim)

            if X2 is None:
                X2 = X
            else:
                X2 = vector_to_symmetric_matrix_tf(X2, self.matrix_dim)

            # Compute the kernel
            X = tf.expand_dims(X, 1)
            X2 = tf.expand_dims(X2, 0)

            X = tf.tile(X, [1, tf.shape(X2)[1], 1, 1])
            X2 = tf.tile(X2, [tf.shape(X)[0], 1, 1, 1])

            diff_XX2 = tf.subtract(X, X2)

            frob_dist = tf.norm(diff_XX2, axis=(-2, -1))
            frob_dist2 = tf.square(frob_dist)

            frob_dist2 = tf.multiply(frob_dist2, self.beta_param)

            return tf.multiply(self.variance, tf.exp(-frob_dist2))

        # Numpy check
        # frobdist = np.zeros((y_man_mat.shape[0], y_man_mat_test.shape[0]))
        # for m in range(y_man_mat.shape[0]):
        #     for n in range(y_man_mat_test.shape[0]):
        #         frobdist[m, n] = np.linalg.norm(y_man_mat[m] - y_man_mat_test[n])
        #
        # kernel_check = np.exp(- frobdist**2 / 1.0)


    class SpdLogEuclideanGaussianKernel(gpflow.kernels.Kernel):
        """
        Instances of this class represent a Log-Euclidean covariance matrix between input points on the SPD manifold
        using the affine-invariant distance.

        Attributes
        ----------
        self.matrix_dim: SPD matrix dimension, computed from the dimension of the inputs (given with Mandel notation)
        self.beta_param: inverse square lengthscale parameter beta
        self.variance: variance parameter of the kernel

        Methods
        -------
        K(point1_in_SPD, point2_in_SPD):

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta = 1., variance=1.):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension (in Mandel notation form)
            :param active_dims: dimensions of the input used for kernel computation (in Mandel notation form),
                                defined as range(input_dim) if all the input dimensions are considered.

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Matrix dimension from input vector dimension
            self.matrix_dim = int((-1.0 + (1.0 + 8.0 * input_dim) ** 0.5) / 2.0)

            # Parameter initialization
            self.beta_param = gpflow.Param(beta, transform=gpflow.transforms.positive)
            self.variance = gpflow.Param(variance, transform=gpflow.transforms.positive)

        @gpflow.params_as_tensors
        def K(self, X, X2=None):
            """
            Computes the Log-Euclidean kernel matrix between inputs X (and X2) belonging to a SPD manifold.

            Parameters
            ----------
            :param X: input points on the SPD manifold (Mandel notation)

            Optional parameters
            -------------------
            :param X2: input points on the SPD manifold (Mandel notation)

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            # Transform input vector to matrices
            X = vector_to_symmetric_matrix_tf(X, self.matrix_dim)

            if X2 is None:
                X2 = X
            else:
                X2 = vector_to_symmetric_matrix_tf(X2, self.matrix_dim)

            # Compute the kernel
            X = tf.expand_dims(X, 1)
            X2 = tf.expand_dims(X2, 0)

            X = tf.tile(X, [1, tf.shape(X2)[1], 1, 1])
            X2 = tf.tile(X2, [tf.shape(X)[0], 1, 1, 1])

            diff_XX2 = tf.cast(tf.subtract(tf.linalg.logm(tf.cast(X, dtype=tf.complex64)), tf.linalg.logm(tf.cast(X2, dtype=tf.complex64))), dtype=tf.float64)

            logeucl_dist = tf.norm(diff_XX2, axis=(-2, -1))
            logeucl_dist2 = tf.square(logeucl_dist)

            logeucl_dist2 = tf.multiply(logeucl_dist2, self.beta_param)

            return tf.multiply(self.variance, tf.exp(-logeucl_dist2))

        # Numpy/scipy check
        # logeucldist = np.zeros((y_man_mat.shape[0], y_man_mat_test.shape[0]))
        # for m in range(y_man_mat.shape[0]):
        #     for n in range(y_man_mat_test.shape[0]):
        #         logeucldist[m, n] = np.linalg.norm(sc.linalg.logm(y_man_mat[m]) - sc.linalg.logm(y_man_mat_test[n]))
        #
        # kernel_check = np.exp(- logeucldist**2 / 1.0)
