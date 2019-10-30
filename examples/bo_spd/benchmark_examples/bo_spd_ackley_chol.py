import numpy as np
import gpflow
import gpflowopt

import pymanopt.manifolds as pyman_man

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from BoManifolds.Riemannian_utils.SPD_utils import symmetric_matrix_to_vector_mandel, vector_to_symmetric_matrix_mandel

from BoManifolds.BO_utils.multistarted_optimization import MultistartedOptimizer, AnchorPointsGenerator
from BoManifolds.BO_utils.constrained_optimization import AnchorPointsGeneratorWithSpdConstraint, SciPyOptimizerWithConstraints

from BoManifolds.plot_utils.manifold_plots import plot_spd_cone
from BoManifolds.plot_utils.bo_plots import bo_plot_function_spd, bo_plot_acquisition_spd, bo_plot_gp_spd

plt.rcParams['text.usetex'] = True  # use Latex font for plots
"""
This example shows the use of Cholesky Bayesian optimization on the SPD manifold S2_++ to optimize the Ackley function.
An Euclidean BO is applied on the Cholesky decomposition of the SPD matrices. 

The Ackley function, defined on the tangent space of the north pole, is projected on the SPD manifold with the 
exponential map (i.e. the logarithm map is used to determine the function value). 
The search space is defined as a subspace of the SPD manifold bounded by minimum and maximum eigenvalues. 
The Euclidean BO uses a Gaussian kernel for comparisons with GaBO.
The acquisition function is optimized with on the Cholesky decomposition to obtain points lying on the SPD manifold. 
The domain of the optimization is defined for the Cholesky decomposition. Constraints on minimum and maximum eigenvalues 
are added to satisfy the bound constraints on the eigenvalues.

The dimension of the manifold is set by the variable 'dim'. Note that the following element must be adapted when the 
dimension is modified:
- the domain must be updated to have gpflowopt domain per dimension. This domain is not used in GaBO, 
    but is required by gpflowopt;
- if the dimension is not 3, 'display_figures' must be set to 'False'.
The number of BO iterations is set by the user by changing the variable 'nb_iter_bo'.

The current optimum value of the function is printed at each BO iteration and the optimal estimate of the optimizer 
(on the sphere) is printed at the end of the queries. 
The following graphs are produced by this example:
- the convergence graph shows the distance between two consecutive iterations and the best function value found by the 
    BO at each iteration. Note that the randomly generated initial data are not displayed, so that the iterations number 
    starts at the number of initial data + 1.
The following graphs are produced by this example if 'display_figures' is 'True':
- the true function graph is displayed on S2_++;
- the acquisition function at the end of the optimization is displayed on S2_++;
- the GP mean at the end of the optimization is displayed on S2_++;
- the GP mean and variances are displayed on 2D projections of S2_++;
- the BO observations are displayed on S2_++.
For all the graphs, the optimum parameter is displayed with a star, the current best estimation with a diamond and all 
the BO observation with dots.

Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
"""

if __name__ == "__main__":
    np.random.seed(1234)

    # Define the dimension
    # If the dimension is changed:
    # - the optimization domain must be updated as one domain per dimension is required by gpflowopt
    dim = 2

    # Define the dimension of the Mandel vector notation
    dim_vec = int(dim + dim * (dim - 1) / 2)

    # True to display sphere figures (possible only if the dimension is 3 (3D graphs))
    display_figures = True

    # Number of BO iterations
    nb_iter_bo = 25

    # Instantiate the manifold (for the objective function and for initial observations, to have the same as GaBO)
    spd_manifold = pyman_man.PositiveDefinite(dim)

    # Bounding eigenvalues
    min_eigenvalue = 0.001
    max_eigenvalue = 5.

    # Define minimum and maximum eigenvalue constraints
    def minimum_eigenvalue_constraint(x_chol):
        indices = np.tril_indices(dim)
        xL = np.zeros((dim, dim))
        xL[indices] = x_chol

        x = np.dot(xL, xL.T)
        eig, _ = np.linalg.eig(x)
        return np.min(eig) - min_eigenvalue

    def maximum_eigenvalue_constraint(x_chol):
        indices = np.tril_indices(dim)
        xL = np.zeros((dim, dim))
        xL[indices] = x_chol

        x = np.dot(xL, xL.T)
        eig, _ = np.linalg.eig(x)
        return max_eigenvalue - np.max(eig)

    eigenvalue_constraints = [{'type': 'ineq', 'fun': minimum_eigenvalue_constraint},
                              {'type': 'ineq', 'fun': maximum_eigenvalue_constraint}]

    # Function to optimize
    base = np.array([[2.5, -0.7], [-0.7, 2.3]])

    # Define the function to optimize with BO
    # Must output a numpy [1,1] shaped array
    # Minus likelihood of covariance sigma for the distribution of data_test_fct (assumed centered)
    def test_function_chol(x_chol):
        # Verify that Cholesky decomposition does not have zero
        if x_chol.size - np.count_nonzero(x_chol):
            x_chol += 1e-6

        # Reshape matrix
        indices = np.tril_indices(dim)
        xL = np.zeros((dim, dim))
        xL[indices] = x_chol

        x = np.dot(xL, xL.T)

        x_proj = spd_manifold.log(base, x)

        # Ackley function
        a = 20
        b = 0.2
        c = 2 * np.pi
        y = -a * np.exp(-b * np.sqrt((x_proj[0, 0] ** 2 + x_proj[1, 1] ** 2 + x_proj[1, 0] ** 2) / 3.)) \
            - np.exp(
            (np.cos(c * x_proj[0, 0]) + np.cos(c * x_proj[1, 1]) + np.cos(c * x_proj[1, 0])) / 3.) + a + np.exp(1.)

        return y[None, None]

    # Optimal parameter
    true_sigma = base

    # Optimal parameter in Cholesky form
    true_sigma_chol = np.linalg.cholesky(true_sigma)
    true_sigma_chol = true_sigma_chol[np.tril_indices(dim)]

    # Optimal function value
    true_opt_val = test_function_chol(true_sigma_chol)[0]

    if display_figures:
        # Plot test function with inputs in the SPD manifold
        # 3D figure
        r_cone = 5.
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        max_colors = bo_plot_function_spd(ax, test_function_chol, r_cone=r_cone, true_opt_x=true_sigma,
                                          true_opt_y=true_opt_val, alpha=0.25, chol=True)
        ax.set_title('True function', fontsize=50)
        plt.show()

    # ### Optimization of the test function
    # Specify the optimization domain (this domain is used only to generate the same initial points as GaBO and
    # Euclidean BO, the Cholesky domain is defined later)
    domain = gpflowopt.domain.ContinuousParameter('x1', min_eigenvalue, max_eigenvalue) \
             + gpflowopt.domain.ContinuousParameter('x2', min_eigenvalue, max_eigenvalue) \
             + gpflowopt.domain.ContinuousParameter('x3', -max_eigenvalue, max_eigenvalue)

    # Define SPD data generator
    spd_random_generator = AnchorPointsGeneratorWithSpdConstraint(domain=domain, dim=dim, min_eigenvalue=0.1,
                                                                  max_eigenvalue=max_eigenvalue)

    # Generate random data in the SPD cone
    nb_data_init = 5
    # Vector form
    x_init_vec = spd_random_generator.generate_samples(nb_samples=nb_data_init)
    # Matrix form
    x_init = np.array([vector_to_symmetric_matrix_mandel(x_init_vec[i]) for i in range(nb_data_init)])
    # Cholesky data
    x_init_chol = np.array([np.linalg.cholesky(x_init[i])[np.tril_indices(dim)] for i in range(nb_data_init)])
    y_init = np.zeros((nb_data_init, 1))
    for n in range(nb_data_init):
        y_init[n] = test_function_chol(x_init_chol[n][None])

    # Create gpflow model
    k = gpflow.kernels.RBF(input_dim=dim_vec, ARD=False)
    # Constant mean function
    mean_fct = gpflow.mean_functions.Constant(25.)
    model = gpflow.gpr.GPR(x_init_chol, y_init, kern=k, mean_function=mean_fct)

    # Specify the optimization domain for Cholesky
    # Compared to manifold/euclidean: x1_chol = sqrt(x1), x2_chol = x3/sqrt(x1), x3_chol = sqrt(x2 - x3**2/x1)
    domain = gpflowopt.domain.ContinuousParameter('x1', np.sqrt(min_eigenvalue), np.sqrt(max_eigenvalue)) \
             + gpflowopt.domain.ContinuousParameter('x2', -np.sqrt(max_eigenvalue), np.sqrt(max_eigenvalue)) \
             + gpflowopt.domain.ContinuousParameter('x3', np.sqrt(min_eigenvalue), np.sqrt(max_eigenvalue))

    # Define the acquisition function
    acq_fct = gpflowopt.acquisition.ExpectedImprovement(model=model)

    # GPflowOpt optimizers
    acq_fct_opt = MultistartedOptimizer(domain, SciPyOptimizerWithConstraints(domain=domain,
                                                                              constraints=eigenvalue_constraints),
                                        AnchorPointsGenerator(domain=domain))

    # ### Bayesian optimization
    # Define the Bayesian optimization
    # An optimizer for the acquisition function can additionally be specified with the "optimizer" parameter
    bo_optimizer = gpflowopt.bo.BayesianOptimizer(domain=domain, acquisition=acq_fct, optimizer=acq_fct_opt,
                                                  scaling=False, verbose=True)

    # Run the Bayessian optimization
    Bopt = bo_optimizer.optimize(test_function_chol, n_iter=nb_iter_bo)
    print(Bopt)

    # Evaluated points
    x_eval_chol = bo_optimizer.acquisition.data[0]
    x_eval = np.zeros((x_eval_chol.shape[0], dim, dim))
    x_eval_vec = np.zeros((x_eval_chol.shape[0], dim_vec))
    indices = np.tril_indices(dim)
    for n in range(x_eval.shape[0]):
        sigmaL = np.zeros((dim, dim))
        sigmaL[indices] = x_eval_chol[n]
        x_eval[n] = np.dot(sigmaL, sigmaL.T)
        x_eval_vec[n] = symmetric_matrix_to_vector_mandel(x_eval[n])
    y_eval = bo_optimizer.acquisition.data[1]

    # BO optimum to SPD
    opt_x_chol = np.zeros((dim, dim))
    opt_x_chol[indices] = Bopt.x
    opt_x_mat = np.dot(opt_x_chol, opt_x_chol.T)
    opt_x_vec = symmetric_matrix_to_vector_mandel(opt_x_mat)[None]

    if display_figures:
        # Plot the acquisition function
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        bo_plot_acquisition_spd(ax, acq_fct, r_cone=r_cone, xs=x_eval_vec, opt_x=opt_x_vec,
                                true_opt_x=symmetric_matrix_to_vector_mandel(true_sigma)[None], chol=True, alpha=0.25,
                                n_elems=20, n_elems_h=10)
        ax.set_title('Acquisition function', fontsize=50)
        plt.show()

        # Plot the GP
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        bo_plot_gp_spd(ax, model, r_cone=r_cone, xs=x_eval_vec, opt_x=opt_x_vec,
                       true_opt_x=symmetric_matrix_to_vector_mandel(true_sigma)[None], true_opt_y=true_opt_val,
                       chol=True, alpha=0.25, max_colors=max_colors, n_elems=20, n_elems_h=10)
        ax.set_title('GP mean', fontsize=50)
        plt.show()

    if display_figures:
        # Plot test function with SPD inputs
        # 3D figure
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)
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
        # Plot SPD cone
        plot_spd_cone(ax, r=r_cone, lim_fact=0.8)
        # Plot evaluated points
        for n in range(x_eval.shape[0]):
            ax.scatter(x_eval[n, 0, 0], x_eval[n, 1, 1], x_eval[n, 0, 1],
                       c=pl.cm.inferno(1. - (y_eval[n] - true_opt_val) / max_colors))
        # Plot true minimum
        ax.scatter(true_sigma[0, 0], true_sigma[1, 1], true_sigma[0, 1], s=40, c='g', marker='P')
        # Plot BO minimum
        ax.scatter(opt_x_mat[0, 0], opt_x_mat[1, 1], opt_x_mat[0, 1], s=20, c='r', marker='D')
        ax.set_title('BO observations', fontsize=30)
        plt.show()

    # Compute distances between consecutive x's
    nb_eval = x_eval.shape[0]
    distances = np.zeros(nb_eval - 1)
    for n in range(nb_eval - 1):
        distances[n] = np.linalg.norm(x_eval_chol[n + 1, :] - x_eval_chol[n, :])
    # Compute best evaluation for each iteration
    y_best = np.ones(nb_eval)
    for i in range(nb_eval):
        y_best[i] = y_eval[:(i + 1)].min()

    #  Plot distances between consecutive x's
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.array(range(nb_data_init, nb_eval - 1)), distances[nb_data_init:], '-ro')
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('d(x[n], x[n-1])', fontsize=18)
    plt.title('Distance between consecutive observations', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    # Plot best estimation at each iteration
    plt.subplot(1, 2, 2)
    plt.plot(np.array(range(nb_data_init, nb_eval)), y_best[nb_data_init:], '-o')
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Best y', fontsize=18)
    plt.title('Value of the best selected sample', fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
