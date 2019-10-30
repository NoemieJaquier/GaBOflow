import numpy as np
import gpflow
import tensorflow as tf
from BoManifolds.Riemannian_utils.sphere_utils_tf import sphere_distance_tf

'''
Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
'''

# The Gaussian and Laplace kernels are implemented here for GPflow version = 0.5 (used by GPflowOpt)
# and version >=1.2.0.
if gpflow.__version__ == '0.5':
    class SphereGaussianKernel(gpflow.kernels.Kern):
        """
        Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the sphere manifold.

        Attributes
        ----------
        self.beta_min_value: minimum value of the inverse square lengthscale parameter beta
        self.beta_shifted: equal to beta-beta_min and used to optimize beta in the space of beta values resulting in
        PD kernels
        self.variance: variance parameter of the kernel

        Methods
        -------
        K(point1_in_the_sphere, point2_in_the_sphere)
        K_diag(point1_in_the_sphere)
        update_beta(new_beta_value)
        get_beta

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta_min, beta=1., variance=1.):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension
            :param active_dims: dimensions of the input used for kernel computation,
                                defined as range(input_dim) if all the input dimensions are considered.
            :param beta_min: minimum value of the inverse square lengthscale parameter beta

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Parameter initialization
            # Beta shifted is used to optimize beta in the space of beta values resulting in PD kernels
            self.beta_shifted = gpflow.param.Param(beta, transform=gpflow.transforms.positive)
            self.variance = gpflow.param.Param(variance, transform=gpflow.transforms.positive)
            self.beta_min_value = beta_min

        def K(self, X, X2=None):
            """
            Computes the Gaussian kernel matrix between inputs X (and X2) belonging to a sphere manifold.

            Parameters
            ----------
            :param X: input points on the sphere

            Optional parameters
            -------------------
            :param X2: input points on the sphere

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            if X2 is None:
                X2 = X

            # Compute beta value from beta_shifted
            beta = self.beta_shifted + self.beta_min_value

            # Compute the kernel
            sphere_dist = sphere_distance_tf(X, X2)
            sphere_dist2 = tf.square(sphere_dist)

            # sphere_dist = tf.divide(sphere_dist, tf.multiply(self.beta_param, self.beta_param))
            sphere_dist2 = tf.multiply(sphere_dist2, beta)

            return tf.multiply(self.variance, tf.exp(-sphere_dist2))

        def Kdiag(self, X):
            """
            Computes the diagonal of Gaussian kernel matrix of inputs X belonging to a sphere manifold.

            Parameters
            ----------
            :param X: input points on the sphere

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


    class SphereLaplaceKernel(gpflow.kernels.Kern):
        """
        Instances of this class represent a Laplace covariance matrix between input points on the sphere manifold.

        Attributes
        ----------
        self.beta: inverse square lengthscale parameter beta
        self.variance: variance parameter of the kernel

        Methods
        -------
        K(point1_in_the_sphere, point2_in_the_sphere)
        K_diag(point1_in_the_sphere)

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta=1., variance=1.):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension
            :param active_dims: dimensions of the input used for kernel computation,
                                defined as range(input_dim) if all the input dimensions are considered.

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Parameter initialization
            # Beta shifted is used to optimize beta in the space of beta values resulting in PD kernels
            self.beta = gpflow.param.Param(beta, transform=gpflow.transforms.positive)
            self.variance = gpflow.param.Param(variance, transform=gpflow.transforms.positive)

        def K(self, X, X2=None):
            """
            Computes the Laplace kernel matrix between inputs X (and X2) belonging to a sphere manifold.

            Parameters
            ----------
            :param X: input points on the sphere

            Optional parameters
            -------------------
            :param X2: input points on the sphere

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            if X2 is None:
                X2 = X

            # Compute the kernel
            sphere_dist = sphere_distance_tf(X, X2)

            # sphere_dist = tf.divide(sphere_dist, tf.multiply(self.beta_param, self.beta_param))
            sphere_dist = tf.multiply(sphere_dist, self.beta)

            return tf.multiply(self.variance, tf.exp(-sphere_dist))

        def Kdiag(self, X):
            """
            Computes the diagonal of Gaussian kernel matrix of inputs X belonging to a sphere manifold.

            Parameters
            ----------
            :param X: input points on the sphere

            Optional parameters
            -------------------

            Returns
            -------
            :return: diagonal of the kernel matrix of X
            """
            return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


else:
    class SphereGaussianKernel(gpflow.kernels.Kernel):
        """
        Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the sphere manifold.

        Attributes
        ----------
        self.beta_min_value: minimum value of the inverse square lengthscale parameter beta
        self.beta_shifted: equal to beta-beta_min and used to optimize beta in the space of beta values resulting in
        PD kernels
        self.variance, variance parameter of the kernel

        Methods
        -------
        K(point1_in_the_sphere, point2_in_the_sphere)
        K_diag(point1_in_the_sphere)
        update_beta(new_beta_value)
        get_beta

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta_min, beta = 1., variance=1.):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension
            :param active_dims: dimensions of the input used for kernel computation,
                                defined as range(input_dim) if all the input dimensions are considered.
            :param beta_min: minimum value of the inverse square lengthscale parameter beta

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Parameter initialization
            # Beta shifted is used to optimize beta in the space of beta values resulting in PD kernels
            self.beta_shifted = gpflow.Param(beta, transform=gpflow.transforms.positive)
            self.variance = gpflow.Param(variance, transform=gpflow.transforms.positive)
            self.beta_min_value = beta_min

        @gpflow.params_as_tensors
        def K(self, X, X2=None):
            """
            Computes the Gaussian kernel matrix between inputs X (and X2) belonging to a sphere manifold.

            Parameters
            ----------
            :param X: input points on the sphere

            Optional parameters
            -------------------
            :param X2: input points on the sphere

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            if X2 is None:
                X2 = X

            # Compute beta value from beta_shifted
            beta = self.beta_shifted + self.beta_min_value

            # Compute the kernel
            sphere_dist = sphere_distance_tf(X, X2)
            sphere_dist2 = tf.square(sphere_dist)

            # sphere_dist = tf.divide(sphere_dist, tf.multiply(self.beta_param, self.beta_param))
            sphere_dist2 = tf.multiply(sphere_dist2, beta)

            return tf.multiply(self.variance, tf.exp(-sphere_dist2))

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


    class SphereLaplaceKernel(gpflow.kernels.Kernel):
        """
        Instances of this class represent a Laplace covariance matrix between input points on the sphere manifold.

        Attributes
        ----------
        self.beta: inverse square lengthscale parameter beta
        self.variance: variance parameter of the kernel

        Methods
        -------
        K(point1_in_the_sphere, point2_in_the_sphere)
        K_diag(point1_in_the_sphere)

        Static methods
        --------------
        """
        def __init__(self, input_dim, active_dims, beta = 1., variance=1.):
            """
            Initialisation.

            Parameters
            ----------
            :param input_dim: input dimension
            :param active_dims: dimensions of the input used for kernel computation,
                                defined as range(input_dim) if all the input dimensions are considered.

            Optional parameters
            -------------------
            :param beta: value of beta
            :param variance: value of the variance
            """
            super().__init__(input_dim=input_dim, active_dims=active_dims)
            # Parameter initialization
            # Beta shifted is used to optimize beta in the space of beta values resulting in PD kernels
            self.beta = gpflow.Param(beta, transform=gpflow.transforms.positive)
            self.variance = gpflow.Param(variance, transform=gpflow.transforms.positive)

        @gpflow.params_as_tensors
        def K(self, X, X2=None):
            """
            Computes the Laplace kernel matrix between inputs X (and X2) belonging to a sphere manifold.

            Parameters
            ----------
            :param X: input points on the sphere

            Optional parameters
            -------------------
            :param X2: input points on the sphere

            Returns
            -------
            :return: kernel matrix of X or between X and X2
            """
            if X2 is None:
                X2 = X

            # Compute the kernel
            sphere_dist = sphere_distance_tf(X, X2)

            # sphere_dist = tf.divide(sphere_dist, tf.multiply(self.beta_param, self.beta_param))
            sphere_dist = tf.multiply(sphere_dist, self.beta)

            return tf.multiply(self.variance, tf.exp(-sphere_dist))
