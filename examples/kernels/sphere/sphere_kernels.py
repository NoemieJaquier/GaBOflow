import numpy as np
import gpflow
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import axes3d, Axes3D
from BoManifolds.Riemannian_utils.sphere_utils import logmap
from BoManifolds.kernel_utils.kernels_sphere_tf import SphereGaussianKernel, SphereLaplaceKernel

from BoManifolds.plot_utils.manifold_plots import plot_sphere

plt.rcParams['text.usetex'] = True  # use Latex font for plots
plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']
"""
This example shows the use of different kernels for the hypershere manifold S^n , used for Gaussian process regression.
The tested function corresponds to a Gaussian distribution with a mean defined on the sphere and a covariance defined on 
the tangent space of the mean. Training data are generated "far" from the mean. The trained Gaussian process is then 
used to determine the value of the function from test data sampled around the mean of the test function. 
The kernels used are:
    - Manifold-RBF kernel (geometry-aware)
    - Laplace kernel (geometry-aware)
    - Euclidean kernel (classical geometry-unaware)
This example works with GPflow version = 0.5 (used by GPflowOpt).

Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
"""


def test_function(x, mu_test_function):

    # Parameters
    sigma_test_fct = np.array([[0.6, 0.2, 0], [0.2, 0.3, -0.01], [0, -0.01, 0.2]])
    inv_sigma_test_fct = np.linalg.inv(sigma_test_fct)
    det_sigma_test_fct = np.linalg.det(sigma_test_fct)

    # Function value
    x_proj = logmap(x, mu_test_function)
    return np.exp(- 0.5 * np.dot(x_proj.T, np.dot(inv_sigma_test_fct, x_proj))) / np.sqrt(
        (2 * np.pi) ** dim * det_sigma_test_fct)


def plot_gaussian_process_prediction(figure_handle, mu, test_data, mean_est, mu_test_fct, title):
    ax = Axes3D(figure_handle)

    # Make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    # Remove axis
    ax._axis3don = False

    # Initial view
    # ax.view_init(elev=10, azim=-20.)  # (default: elev=30, azim=-60)
    ax.view_init(elev=10, azim=30.)  # (default: elev=30, azim=-60)

    # Plot sphere
    plot_sphere(ax, alpha=0.4)

    # Plot training data on the manifold
    plt_scale_fact = test_function(mu_test_fct, mu_test_fct)[0, 0]
    nb_data_test = test_data.shape[0]
    for n in range(nb_data_test):
        ax.scatter(test_data[n, 0], test_data[n, 1], test_data[n, 2], c=pl.cm.inferno(mean_est[n] / plt_scale_fact))

    # Plot mean of Gaussian test function
    ax.scatter(mu[0], mu[1], mu[2], c='g', marker='D')

    plt.title(title, size=25)


if __name__ == "__main__":
    np.random.seed(1234)

    # Define the test function mean
    mu_test_fct = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0])

    # Generate random data on the sphere
    nb_data = 20
    dim = 3

    mean = np.array([1, 0, 0])
    mean = mean / np.linalg.norm(mean)
    fact_cov = 0.1
    cov = fact_cov * np.eye(dim)

    data = np.random.multivariate_normal(mean, cov, nb_data)
    x_man = data / np.linalg.norm(data, axis=1)[:, None]

    y_train = np.zeros((nb_data,1))
    for n in range(nb_data):
        y_train[n] = test_function(x_man[n], mu_test_fct)

    # Generate test data on the sphere
    nb_data_test = 10

    mean_test = mu_test_fct
    mean_test = mean_test / np.linalg.norm(mean)
    fact_cov = 0.1
    cov_test = fact_cov * np.eye(dim)

    data = np.random.multivariate_normal(mean_test, cov_test, nb_data_test)
    x_man_test = data / np.linalg.norm(data, axis=1)[:, None]

    y_test = np.zeros((nb_data_test, 1))
    for n in range(nb_data_test):
        y_test[n] = test_function(x_man_test[n], mu_test_fct)

    # Plot training data - 3D figure
    fig = plt.figure(figsize=(5, 5))
    plot_gaussian_process_prediction(fig, mu_test_fct, x_man, y_train, mu_test_fct, r'Training data')

    # Plot true test data - 3D figure
    fig = plt.figure(figsize=(5, 5))
    plot_gaussian_process_prediction(fig, mu_test_fct, x_man_test, y_test, mu_test_fct, r'Test data (ground truth)')

    # ### Gaussian kernel
    # Define the kernel
    k_gauss = SphereGaussianKernel(input_dim=dim, active_dims=range(dim), beta_min=7.0, beta=10.0, variance=1.)
    # Kernel computation
    K1 = k_gauss.compute_K_symm(x_man)
    K12 = k_gauss.compute_K(x_man, x_man_test)
    K2 = k_gauss.compute_K_symm(x_man_test)
    # GPR model
    m_gauss = gpflow.gpr.GPR(x_man, y_train, kern=k_gauss, mean_function=None)
    # Optimization of the model parameters
    m_gauss.optimize()
    # Compute posterior samples
    # Does not always work due to Cholesky decomposition used in gpflow
    # nb_samples_post = 10
    # posterior_samples = m.predict_f_samples(y_man_test.T, nb_samples_post)
    # Prediction
    mean_est_gauss, cov_est_gauss = m_gauss.predict_f_full_cov(x_man_test)
    # mean, cov = m.predict_y(x_new)  # includes noise variance (seems not to be included in predict_f functions
    var_est_gauss = np.diag(cov_est_gauss[0])[:, None]
    # Error computation
    error_gauss = np.sqrt(np.sum((y_test - mean_est_gauss) ** 2) / nb_data_test)
    print('Estimation error (Manifold-RBF kernel) = ', error_gauss)
    # Plot test data
    fig = plt.figure(figsize=(5, 5))
    plot_gaussian_process_prediction(fig, mu_test_fct, x_man_test, mean_est_gauss, mu_test_fct, r'Manifold-RBF kernel')

    # ### Laplace kernel
    # Define the kernel
    k_laplace = SphereLaplaceKernel(input_dim=dim, active_dims=range(dim), beta=10.0, variance=1.)
    # Kernel computation
    K1 = k_laplace.compute_K_symm(x_man)
    K12 = k_laplace.compute_K(x_man, x_man_test)
    K2 = k_laplace.compute_K_symm(x_man_test)
    # GPR model
    m_laplace = gpflow.gpr.GPR(x_man, y_train, kern=k_laplace, mean_function=None)
    # Optimization of the model parameters
    m_laplace.optimize()
    # Compute posterior samples
    # Does not always work due to Cholesky decomposition used in gpflow
    # nb_samples_post = 10
    # posterior_samples = m.predict_f_samples(y_man_test.T, nb_samples_post)
    # Prediction
    mean_est_laplace, cov_est_laplace = m_laplace.predict_f_full_cov(x_man_test)
    # mean, cov = m.predict_y(x_new)  # includes noise variance (seems not to be included in predict_f functions
    var_est_laplace = np.diag(cov_est_laplace[0])[:, None]
    # Error computation
    error_laplace = np.sqrt(np.sum((y_test - mean_est_laplace) ** 2) / nb_data_test)
    print('Estimation error (Laplace kernel) = ', error_laplace)
    # Plot test data
    fig = plt.figure(figsize=(5, 5))
    plot_gaussian_process_prediction(fig, mu_test_fct, x_man_test, mean_est_laplace, mu_test_fct, r'Laplace kernel')

    # ### Euclidean RBF
    # Define the kernel
    k_eucl = gpflow.kernels.RBF(input_dim=dim, ARD=False)
    # Kernel computation
    K1 = k_eucl.compute_K_symm(x_man)
    K12 = k_eucl.compute_K(x_man, x_man_test)
    K2 = k_eucl.compute_K_symm(x_man_test)
    # GPR model
    m_eucl = gpflow.gpr.GPR(x_man, y_train, kern=k_eucl, mean_function=None)
    # Optimization of the model parameters
    m_eucl.optimize()
    # Compute posterior samples
    # Does not always work due to Cholesky decomposition used in gpflow
    # nb_samples_post = 10
    # posterior_samples = m.predict_f_samples(y_man_test.T, nb_samples_post)
    # Prediction
    mean_est_eucl, cov_est_eucl = m_eucl.predict_f_full_cov(x_man_test)
    # mean, cov = m_eucl.predict_y(x_new)  # includes noise variance (seems not to be included in predict_f functions
    var_est_eucl = np.diag(cov_est_eucl[0])[:, None]
    # Error computation
    error_eucl = np.sqrt(np.sum((y_test - mean_est_eucl) ** 2) / nb_data_test)
    print('Estimation error (Euclidean-RBF kernel) = ', error_eucl)
    # Plot test data
    fig = plt.figure(figsize=(5, 5))
    plot_gaussian_process_prediction(fig, mu_test_fct, x_man_test, mean_est_eucl, mu_test_fct, r'Euclidean-RBF kernel')

    plt.show()

