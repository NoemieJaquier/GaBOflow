import numpy as np

import gpflow
import gpflowopt

import pymanopt.manifolds as pyman_man

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from BoManifolds.BO_utils.constrained_optimization import SciPyOptimizerWithConstraints, \
    AnchorPointsGeneratorWithNormConstraint
from BoManifolds.BO_utils.multistarted_optimization import MultistartedOptimizer

from BoManifolds.plot_utils.manifold_plots import plot_sphere
from BoManifolds.plot_utils.bo_plots import bo_plot_function_sphere, bo_plot_acquisition_sphere, bo_plot_gp_sphere, \
    bo_plot_gp_sphere_planar

plt.rcParams['text.usetex'] = True  # use Latex font for plots
"""
This example shows the use of Euclidean Bayesian optimization on the sphere S2 to optimize the Ackley function. 

The Ackley function, defined on the tangent space of the north pole, is projected on the sphere with the exponential 
map (i.e. the logarithm map is used to determine the function value). 
The Euclidean BO uses a Gaussian kernel for comparisons with GaBO.
The acquisition function is optimized with a constrained optimization to obtain points lying on the sphere.
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
- the true function graph is displayed on S2;
- the acquisition function at the end of the optimization is displayed on S2;
- the GP mean at the end of the optimization is displayed on S2;
- the GP mean and variances are displayed on 2D projections of S2;
- the BO observations are displayed on S2.
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
    dim = 3

    # True to display sphere figures (possible only if the dimension is 3 (3D graphs))
    display_figures = True

    # Number of BO iterations
    nb_iter_bo = 25

    # Instantiate the manifold (for the objective function and for initial observations, to have the same as GaBO)
    sphere_manifold = pyman_man.Sphere(dim)

    # Define unit norm constraint (point on the sphere)
    def unit_norm_constraint(x):
        return np.linalg.norm(x) - 1.

    sphere_constraints = [{'type': 'eq', 'fun': unit_norm_constraint}]

    # Define the function to optimize with BO
    # Must output a numpy [1,1] shaped array
    def test_function(x):
        if np.ndim(x) < 2:
            x = x[None]

        # Projection in tangent space of the base.
        # The base is fixed at (1, 0, 0, ...) for simplicity. Therefore, the tangent plane is aligned with the axis x.
        # The first coordinate of x_proj is always 0, so that vectors in the tangent space can be expressed in a dim-1
        # dimensional space by simply ignoring the first coordinate.
        base = np.zeros((1, dim))
        base[0, 0] = 1.
        x_proj = sphere_manifold.log(base, x)[0]

        # Remove first dim
        x_proj_red = x_proj[1:]
        dim_red = dim - 1

        # Ackley function
        a = 20
        b = 0.2
        c = 2 * np.pi

        aexp_term = -a * np.exp(-b * np.sqrt(np.sum(x_proj_red ** 2) / dim_red))
        expcos_term = - np.exp(np.sum(np.cos(c * x_proj_red) / dim_red))
        y = aexp_term + expcos_term + a + np.exp(1.)

        return y[None, None]

    # Optimal parameter
    true_min = np.zeros((1, dim))
    true_min[0, 0] = 1

    # Optimal function value
    true_opt_val = test_function(true_min)

    if display_figures:
        # Plot test function with inputs on the sphere
        # 3D figure
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        max_colors = bo_plot_function_sphere(ax, test_function, true_opt_x=true_min, true_opt_y=true_opt_val, elev=10,
                                             azim=30, n_elems=300)
        ax.set_title('True function', fontsize=50)
        plt.show()

    # Generate random data on the sphere
    nb_data = 5
    x_init = np.array([sphere_manifold.rand() for n in range(nb_data)])
    y_init = np.zeros((nb_data, 1))
    for n in range(nb_data):
        y_init[n] = test_function(x_init[n])

    # Create gpflow model
    k = gpflow.kernels.RBF(input_dim=dim)
    # Add a constant mean function.
    mean_fct = gpflow.mean_functions.Constant(25.)
    model = gpflow.gpr.GPR(x_init, y_init, kern=k, mean_function=mean_fct)

    # Specify the optimization domain
    domain = gpflowopt.domain.ContinuousParameter('x1', -1, 1) + gpflowopt.domain.ContinuousParameter('x2', -1, 1) + \
             gpflowopt.domain.ContinuousParameter('x3', -1, 1)

    # Define the acquisition function
    acq_fct = gpflowopt.acquisition.ExpectedImprovement(model=model)

    # GPflowOpt optimizers
    acq_fct_opt = MultistartedOptimizer(domain, SciPyOptimizerWithConstraints(domain=domain,
                                                                              constraints=sphere_constraints,
                                                                              method='SLSQP'),
                                        AnchorPointsGeneratorWithNormConstraint(domain=domain, norm=1.))

    # ### Bayesian optimization
    # Define the Bayesian optimization
    # An optimizer for the acquisition function can additionally be specified with the "optimizer" parameter
    bo_optimizer = gpflowopt.bo.BayesianOptimizer(domain=domain, acquisition=acq_fct, optimizer=acq_fct_opt,
                                                  scaling=False, verbose=True)

    # Run the Bayessian optimization
    Bopt = bo_optimizer.optimize(test_function, n_iter=nb_iter_bo)
    print(Bopt)

    # Evaluated points
    x_eval = bo_optimizer.acquisition.data[0]
    y_eval = bo_optimizer.acquisition.data[1]

    if display_figures:
        # Plot acquisition function
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        bo_plot_acquisition_sphere(ax, acq_fct, xs=bo_optimizer.acquisition.data[0], opt_x=Bopt.x, true_opt_x=true_min,
                                   elev=10, azim=30, n_elems=100)
        ax.set_title('Acquisition function', fontsize=50)
        plt.show()

        # Plot GP
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        bo_plot_gp_sphere(ax, model, xs=bo_optimizer.acquisition.data[0], opt_x=Bopt.x, true_opt_x=true_min,
                          true_opt_y=true_opt_val, max_colors=max_colors, elev=10, azim=30, n_elems=100)
        ax.set_title('GP mean', fontsize=50)
        plt.show()

        # Plot GP projected on planes
        fig = plt.figure(figsize=(20, 10))
        bo_plot_gp_sphere_planar(fig, model, var_fact=2., xs=bo_optimizer.acquisition.data[0],
                                 ys=bo_optimizer.acquisition.data[1], opt_x=Bopt.x, opt_y=test_function(Bopt.x),
                                 true_opt_x=true_min, true_opt_y=true_opt_val, max_colors=max_colors, n_elems=100)
        fig.suptitle('GP mean and variance', fontsize=50)
        plt.show()

    if display_figures:
        # Plot observations on the sphere
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
        ax.view_init(elev=10, azim=30.)  # (default: elev=30, azim=-60)
        # Plot sphere
        plot_sphere(ax, alpha=0.4)
        # Plot evaluated points
        plt_scale_fact = test_function(true_min)[0, 0]  # optimal value
        for n in range(x_eval.shape[0]):
            ax.scatter(x_eval[n, 0], x_eval[n, 1], x_eval[n, 2],
                       c=pl.cm.inferno(1. - (y_eval[n] - true_opt_val[0]) / max_colors))
        # Plot true minimum
        ax.scatter(true_min[0, 0], true_min[0, 1], true_min[0, 2], s=40, c='g', marker='P')
        # Plot BO minimum
        ax.scatter(Bopt.x[0, 0], Bopt.x[0, 1], Bopt.x[0, 2], s=20, c='r', marker='D')
        ax.set_title('BO observations', fontsize=30)
        plt.show()

    # Convergence plots
    # Compute distances between consecutive x's
    neval = x_eval.shape[0]
    distances = np.zeros(neval-1)
    for n in range(neval-1):
        distances[n] = np.linalg.norm(x_eval[n + 1, :] - x_eval[n, :])
    # Compute best evaluation for each iteration
    Y_best = np.ones(neval)
    for i in range(neval):
        Y_best[i] = y_eval[:(i + 1)].min()

    #  Plot distances between consecutive x's
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.array(range(neval - 1)), distances, '-ro')
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('d(x[n], x[n-1])', fontsize=18)
    plt.title('Distance between consecutive observations', fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    # Plot best estimation at each iteration
    plt.subplot(1, 2, 2)
    plt.plot(np.array(range(neval)), Y_best, '-o')
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Best y', fontsize=18)
    plt.title('Value of the best selected sample', fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
