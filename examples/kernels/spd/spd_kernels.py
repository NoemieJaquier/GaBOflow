import numpy as np
import gpflow
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from BoManifolds.kernel_utils.kernels_spd_tf import SpdSteinGaussianKernel, SpdAffineInvariantGaussianKernel, SpdFrobeniusGaussianKernel, SpdLogEuclideanGaussianKernel
from BoManifolds.Riemannian_utils.SPD_utils import expmap, symmetric_matrix_to_vector_mandel, vector_to_symmetric_matrix_mandel

from BoManifolds.plot_utils.manifold_plots import plot_spd_cone

plt.rcParams['text.usetex'] = True  # use Latex font for plots
plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']
"""
This example shows the use of different kernels for the SPD manifold, used for Gaussian process regression.
Artificial data are created from the time t and positions x of C-shape trajectory. The input data corresponds to the 
symmetric matrices xx' projected to the SPD manifold and the output data correspond to the time. Only a part of the 
trajectory is used to form the training data, while the whole trajectory is used to form the test data. 
Gaussian processes are trained on the training data and used to predict the output of the test data.
The kernels used are:
    - Stein divergence Gaussian kernel (geometry-aware)
    - Affine-Invariant Gaussian Kernel (geometry-aware)
    - Frobenius Gaussian Kernel 
    - Log-Euclidean kernel (this kernel is sometimes negative definite, likely due to round errors)
This example works with GPflow version = 0.5 (used by GPflowOpt).

Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
"""


def plot_training_test_data_spd_cone(training_spd_data, test_spd_data, figure_handle):
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
    ax.view_init(elev=10, azim=-20.)  # (default: elev=30, azim=-60)
    # ax.view_init(elev=10, azim=50.)  # (default: elev=30, azim=-60)

    # Plot SPD cone
    plot_spd_cone(ax, r=2.5, lim_fact=0.8)

    # Plot testing data on the manifold
    plt.plot(test_spd_data[0], test_spd_data[1], test_spd_data[2] / np.sqrt(2), color='b', marker='.', linewidth=0.,
             markersize=9.)

    # Plot training data on the manifold
    plt.plot(training_spd_data[0], training_spd_data[1], training_spd_data[2] / np.sqrt(2), color='k', marker='.',
             linewidth=0., markersize=9.)

    plt.title(r'Input data', size=25)


def plot_gaussian_process(true_output, mean, variance, posterior_samples, plot_title):
    number_data = true_output.shape[0]
    t = np.array(range(0, number_data))
    plt.figure(figsize=(12, 6))
    plt.plot(t, true_output, 'kx', mew=2)
    plt.plot(t, mean, 'C0', lw=2)
    plt.fill_between(t, mean - 1.96 * np.sqrt(variance), mean + 1.96 * np.sqrt(variance), color='C0', alpha=0.2)

    for s in range(nb_samples_post):
        plt.plot(t, posterior_samples[s], 'C0', linewidth=0.5)

    plt.title(plot_title, size=25)


if __name__ == "__main__":
    # Load data from 2D letters
    nb_samples = 1

    data_demos = loadmat('../../../data/2Dletters/C.mat')

    data_demos = data_demos['demos'][0]
    demos = [data_demos[i]['pos'][0][0] for i in range(data_demos.shape[0])]

    nb_data_init = demos[0].shape[1]
    dt = 1.

    time = np.hstack([np.arange(0, nb_data_init) * dt] * data_demos.shape[0])
    demos_np = np.hstack(demos)

    # Euclidean vector data
    data_eucl = np.vstack((time, demos_np))
    data_eucl = data_eucl[:, :nb_data_init * nb_samples]

    # Create artificial SPD matrices from demonstrations and store them in Mandel notation (along with time)
    data_spd_mandel = [symmetric_matrix_to_vector_mandel(expmap(0.01 * np.dot(data_eucl[1:, n][:, None], data_eucl[1:, n][None]),
                                                                np.eye(2)))[:, None] for n in range(data_eucl.shape[1])]
    data_spd_mandel = np.vstack((data_eucl[0], np.concatenate(data_spd_mandel, axis=1)))

    # Training data
    data = data_spd_mandel[:, ::2]
    # Removing data to show GP uncertainty
    # id_to_remove = np.hstack((np.arange(12, 27), np.arange(34, 38)))
    # id_to_remove = np.hstack((np.arange(24, 54), np.arange(68, 76)))
    id_to_remove = np.hstack((np.arange(24, 37), np.arange(68, 76)))
    # id_to_remove = np.hstack((np.arange(12, 24), np.arange(76, 84)))
    data = np.delete(data, id_to_remove, axis=1)
    nb_data = data.shape[1]
    dim = 2
    dim_vec = 3

    # Training data in SPD form
    y = data[0][:, None]
    x_man = data[1:]

    x_man_mat = np.zeros((nb_data, dim, dim))
    for n in range(nb_data):
        x_man_mat[n] = vector_to_symmetric_matrix_mandel(x_man[:, n])

    # New output (test) vector
    y_test = data_spd_mandel[0, ::2][:, None]
    nb_data_test = y_test.shape[0]

    # Test data in SPD form
    x_man_test = data_spd_mandel[1:, ::2]
    x_man_mat_test = np.zeros((nb_data_test, dim, dim))
    for n in range(nb_data_test):
        x_man_mat_test[n] = vector_to_symmetric_matrix_mandel(x_man_test[:, n])

    # Plot input data - 3D figure
    fig = plt.figure(figsize=(5, 5))
    plot_training_test_data_spd_cone(x_man, x_man_test, fig)

    # ### Stein kernel
    # Define the kernel
    k_stein = SpdSteinGaussianKernel(input_dim=dim_vec, active_dims=range(dim_vec), beta=1.0, variance=1.)
    # Kernel computation
    K1 = k_stein.compute_K_symm(x_man.T)
    K12 = k_stein.compute_K(x_man.T, x_man_test.T)
    K2 = k_stein.compute_K_symm(x_man_test.T)
    # GPR model
    m_stein = gpflow.gpr.GPR(x_man.T, y, kern=k_stein, mean_function=None)
    # Optimization of the model parameters
    # First check discrete part of beta space
    log_marg_lik = []
    beta_list = []
    for j in range(1, int(2 * m_stein.kern.low_lim_continuous_param_space + 1)):
        beta = j / 2.
        beta_list.append(beta)
        m_stein.kern.update_beta(beta)
        log_marg_lik.append(m_stein.compute_log_likelihood())
    # Then optimize for the continous part
    m_stein.kern.update_beta(1.)
    m_stein.optimize()
    # List of parameters and log marginal likelihood
    beta_list.append(m_stein.kern.get_beta())
    log_marg_lik.append(m_stein.compute_log_likelihood())
    # Compare log marginal likelihoods and choose beta that maximize it
    beta_opt = beta_list[log_marg_lik.index(max(log_marg_lik))]
    m_stein.kern.update_beta(beta_opt)
    # Compute posterior samples
    # Does not always work due to Cholesky decomposition used in gpflow
    nb_samples_post = 10
    posterior_samples_stein = m_stein.predict_f_samples(x_man_test.T, nb_samples_post)
    # Prediction
    mean_stein, cov_ai = m_stein.predict_f_full_cov(x_man_test.T)
    var_stein = np.diag(cov_ai[:, :, 0])
    # Plot
    plot_gaussian_process(y_test, mean_stein[:, 0], var_stein, posterior_samples_stein, r'Stein divergence kernel')

    # ### Affine invariant kernel
    # Define the kernel
    k_ai = SpdAffineInvariantGaussianKernel(input_dim=dim_vec, active_dims=range(dim_vec), beta=1.0, variance=1., beta_min=0.6)
    # Kernel computation
    K1 = k_ai.compute_K_symm(x_man.T)
    K12 = k_ai.compute_K(x_man.T, x_man_test.T)
    K2 = k_ai.compute_K_symm(x_man_test.T)
    # GPR model
    m_ai = gpflow.gpr.GPR(x_man.T, y, kern=k_ai, mean_function=None)
    # Optimization of the model parameters
    m_ai.optimize()
    # Parameters and log marginal likelihood
    log_marg_lik = m_ai.compute_log_likelihood()
    beta = m_ai.kern.get_beta()
    # Compute posterior samples
    # Does not always work due to Cholesky decomposition used in gpflow
    nb_samples_post = 10
    posterior_samples_ai = m_ai.predict_f_samples(x_man_test.T, nb_samples_post)
    # Prediction
    mean_ai, cov_ai = m_ai.predict_f_full_cov(x_man_test.T)
    var_ai = np.diag(cov_ai[:, :, 0])
    # Plot
    plot_gaussian_process(y_test, mean_ai[:, 0], var_ai, posterior_samples_ai, r'Affine-invariant kernel')

    # ### Frobenius kernel
    # Define the kernel
    k_frob = SpdFrobeniusGaussianKernel(input_dim=dim_vec, active_dims=range(dim_vec), beta=1.0, variance=1.)
    # Kernel computation
    K1 = k_frob.compute_K_symm(x_man.T)
    K12 = k_frob.compute_K(x_man.T, x_man_test.T)
    K2 = k_frob.compute_K_symm(x_man_test.T)
    # GPR model
    m_frob = gpflow.gpr.GPR(x_man.T, y, kern=k_frob, mean_function=None)
    # Optimization of the model parameters
    m_frob.optimize()
    # Log marginal likelihood
    log_marg_lik = m_frob.compute_log_likelihood()
    # Compute posterior samples
    # Does not always work due to Cholesky decomposition used in gpflow
    nb_samples_post = 10
    posterior_samples_frob = m_frob.predict_f_samples(x_man_test.T, nb_samples_post)
    # Prediction
    mean_frob, cov_frob = m_frob.predict_f_full_cov(x_man_test.T)
    var_frob = np.diag(cov_frob[:, :, 0])
    # Plot
    plot_gaussian_process(y_test, mean_frob[:, 0], var_frob, posterior_samples_frob, r'Frobenius kernel')

    # # ### Log-Euclidean kernel
    # # Define the kernel
    # k_logE = LogEuclKern(input_dim=dim_vec, active_dims=range(dim_vec), beta=1.0, variance=1.)
    # # Kernel computation
    # K1 = k_logE.compute_K_symm(x_man.T)
    # K12 = k_logE.compute_K(x_man.T, x_man_test.T)
    # K2 = k_logE.compute_K_symm(x_man_test.T)
    # # GPR model
    # m_logE = gpflow.gpr.GPR(x_man.T, y, kern=k_logE, mean_function=None)
    # # Optimization of the model parameters
    # m_logE.optimize()
    # # Log marginal likelihood
    # log_marg_lik = m_logE.compute_log_likelihood()
    # # Compute posterior samples
    # # Does not always work due to Cholesky decomposition used in gpflow
    # nb_samples_post = 10
    # posterior_samples_logE = m_logE.predict_f_samples(x_man_test.T, nb_samples_post)
    # # Prediction
    # mean_logE, cov_logE = m_logE.predict_f_full_cov(x_man_test.T)
    # var_logE = np.diag(cov_logE[:, :, 0])
    # # Plot
    # plot_gaussian_process(y_test, mean_logE[:, 0], var_logE, posterior_samples_logE, 'Log-Euclidean kernel')

    plt.show()
