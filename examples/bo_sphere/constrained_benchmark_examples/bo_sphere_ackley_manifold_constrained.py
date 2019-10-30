import numpy as np
import gpflow
import gpflowopt
import pymanopt.manifolds as pyman_man
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from BoManifolds.Riemannian_utils.sphere_utils import in_domain, project_to_domain
from BoManifolds.kernel_utils.kernels_sphere_tf import SphereGaussianKernel
from BoManifolds.BO_utils.manifold_optimization import ManifoldOptimizer, ManifoldAnchorPointsGenerator
from BoManifolds.BO_utils.multistarted_optimization import MultistartedOptimizer

from BoManifolds.plot_utils.manifold_plots import plot_sphere
from BoManifolds.plot_utils.bo_plots import bo_plot_function_sphere, bo_plot_acquisition_sphere, bo_plot_gp_sphere, \
    bo_plot_gp_sphere_planar

plt.rcParams['text.usetex'] = True  # use Latex font for plots
"""
This example shows the use of Geometry-aware Bayesian optimization (GaBO) on the sphere S2 to optimize the Ackley 
function. In this example, the search domain is bounded and represents a subspace of the manifold.

The Ackley function, defined on the tangent space of the north pole, is projected on the sphere with the exponential 
map (i.e. the logarithm map is used to determine the function value). 
GaBO uses a Gaussian kernel with the geodesic distance. To guarantee the positive-definiteness of the kernel, the 
lengthscale beta must be above the beta min value. This value can be determined by using the example 
kernels/sphere_gaussian_kernel_parameters.py for each sphere manifold.
The acquisition function is optimized on the manifold with the constrained conjugate gradient descent method on 
Riemannian manifold. The conjugate gradient descent is originally implemented in pymanopt. A constrained version 
is used here to handle bound constraints.
The dimension of the manifold is set by the variable 'dim'. Note that the following element must be adapted when the 
dimension is modified:
- beta_min must be recomputed for the new manifold;
- the constraints must be updated;
- the domain must be updated to have gpflowopt domain per dimension. This domain is not used in GaBO, 
    but is required by gpflowopt;
- if the dimension is not 3, 'display_figures' must be set to 'False'.
The bound constraints are defined by a higher (maximum) and a lower (minimum) value for each dimension. These are 
specified when defining the gpflowopt domain. No constraints are applied to a particular dimension if its maximum and 
minimum values are set to -1 and 1 (i.e. it corresponds to the whole surface of the sphere along this dimension).
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
    # - beta min must be adapted
    # - the optimization domain must be updated as one domain per dimension is required by gpflowopt
    dim = 3

    # Beta min value
    beta_min = 6.5

    # True to display sphere figures (possible only if the dimension is 3 (3D graphs))
    display_figures = True

    # Number of BO iterations
    nb_iter_bo = 25

    # Instantiate the manifold
    sphere_manifold = pyman_man.Sphere(dim)

    # Instantiate manifold domain functions (needed for the constrained optimization with CG on manifold)
    sphere_manifold.in_domain = in_domain
    sphere_manifold.project_to_domain = project_to_domain

    # Define bound constraints
    xl = 0.
    xu = 1.
    yl = -0.6
    yu = 0.6
    zl = -0.6
    zu = 0.6

    # Sample function for the sphere with constraints
    def sample_sphere_constrained():
        not_in_domain = True
        while not_in_domain:
            sample = np.array([np.random.uniform(xl, xu), np.random.uniform(yl, yu), np.random.uniform(zl, zu)])
            sample = sample / np.linalg.norm(sample)
            if sample[0] < xu and sample[0] > xl and sample[1] < yu and sample[1] > yl and sample[2] < zu and sample[
                2] > zl:
                not_in_domain = False
        return sample

    # Replace sample function of the manifold by the constrained sampling
    sphere_manifold.rand = sample_sphere_constrained

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
        # Plot surface
        max_colors = bo_plot_function_sphere(ax, test_function, true_opt_x=true_min, true_opt_y=true_opt_val, elev=10,
                                             azim=30, n_elems=300)
        # Plot constraints
        nbDrawingSeg = 35
        t = np.linspace(-np.pi, np.pi, nbDrawingSeg)
        circle = np.vstack((np.cos(t), np.sin(t)))
        # Define constraint curves
        xl_circle = np.vstack((xl*np.ones(nbDrawingSeg), circle))
        r_circle = np.sqrt(1. - yl**2)
        yl_circle = np.vstack((r_circle*circle[0], yl*np.ones(nbDrawingSeg), r_circle*circle[1]))
        r_circle = np.sqrt(1. - yu ** 2)
        yu_circle = np.vstack((r_circle * circle[0], yu * np.ones(nbDrawingSeg), r_circle * circle[1]))
        r_circle = np.sqrt(1. - zl ** 2)
        zl_circle = np.vstack((r_circle * circle[0], r_circle * circle[1], zl * np.ones(nbDrawingSeg)))
        r_circle = np.sqrt(1. - zu ** 2)
        zu_circle = np.vstack((r_circle * circle[0], r_circle * circle[1], zu * np.ones(nbDrawingSeg)))
        # Restrict curves to plot only the borders of the domain
        yl_circle = yl_circle[:, yl_circle[0] > xl - 0.05]
        yl_circle = yl_circle[:, yl_circle[2] > zl - 0.05]
        yl_circle = yl_circle[:, yl_circle[2] < zu + 0.05]
        yu_circle = yu_circle[:, yu_circle[0] > xl - 0.05]
        yu_circle = yu_circle[:, yu_circle[2] > zl - 0.05]
        yu_circle = yu_circle[:, yu_circle[2] < zu + 0.05]
        zl_circle = zl_circle[:, zl_circle[0] > xl - 0.05]
        zl_circle = zl_circle[:, zl_circle[1] > yl - 0.05]
        zl_circle = zl_circle[:, zl_circle[1] < yu + 0.05]
        zu_circle = zu_circle[:, zu_circle[0] > xl - 0.05]
        zu_circle = zu_circle[:, zu_circle[1] > yl - 0.05]
        zu_circle = zu_circle[:, zu_circle[1] < yu + 0.05]
        # Plot constraints
        # ax.plot(xs=xl_circle[0, :], ys=xl_circle[1, :], zs=xl_circle[2, :], color='k')
        ax.plot(xs=yl_circle[0, :], ys=yl_circle[1, :], zs=yl_circle[2, :], color='k')
        ax.plot(xs=yu_circle[0, :], ys=yu_circle[1, :], zs=yu_circle[2, :], color='k')
        ax.plot(xs=zl_circle[0, :], ys=zl_circle[1, :], zs=zl_circle[2, :], color='k')
        ax.plot(xs=zu_circle[0, :], ys=zu_circle[1, :], zs=zu_circle[2, :], color='k')
        ax.set_title('True function', fontsize=50)
        plt.show()

    # Generate random data on the sphere
    nb_data = 5
    x_init = np.array([sample_sphere_constrained() for i in range(nb_data)])
    y_init = np.zeros((nb_data, 1))
    for n in range(nb_data):
        y_init[n] = test_function(x_init[n])

    # Create gpflow model
    k = SphereGaussianKernel(input_dim=dim, active_dims=range(dim), beta=10.0, variance=1., beta_min=beta_min)
    # Constant mean
    mean_fct = gpflow.mean_functions.Constant(25.)
    model = gpflow.gpr.GPR(x_init, y_init, kern=k, mean_function=mean_fct)

    # Specify the optimization domain
    domain = gpflowopt.domain.ContinuousParameter('x1', xl, xu) + gpflowopt.domain.ContinuousParameter('x2', yl, yu) + \
             gpflowopt.domain.ContinuousParameter('x3', zl, zu)

    # Define the acquisition function
    acq_fct = gpflowopt.acquisition.ExpectedImprovement(model=model)

    # Set the acquisition function optimizer
    acq_fct_opt = MultistartedOptimizer(domain, ManifoldOptimizer(domain=domain, manifold=sphere_manifold,
                                                                  manifold_dim=dim,
                                                                  solver_type='BoundConstrainedConjugateGradient'),
                                        ManifoldAnchorPointsGenerator(manifold=sphere_manifold))

    # ### Bayesian optimization
    # Define the Bayesian optimization
    # An optimizer for the acquisition function can additionally be specified with the "optimizer" parameter
    bo_optimizer = gpflowopt.bo.BayesianOptimizer(domain=domain, acquisition=acq_fct, optimizer=acq_fct_opt,
                                                  scaling=False, verbose=True)

    # Run the Bayesian optimization
    Bopt = bo_optimizer.optimize(test_function, n_iter=nb_iter_bo)
    print(Bopt)

    if display_figures:
        # Plot acquisition function
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        bo_plot_acquisition_sphere(ax, acq_fct, xs=bo_optimizer.acquisition.data[0], opt_x=Bopt.x, true_opt_x=true_min,
                                   elev=10, azim=30, n_elems=100)
        # Plot bounds
        ax.plot(xs=yl_circle[0, :], ys=yl_circle[1, :], zs=yl_circle[2, :], color='k')
        ax.plot(xs=yu_circle[0, :], ys=yu_circle[1, :], zs=yu_circle[2, :], color='k')
        ax.plot(xs=zl_circle[0, :], ys=zl_circle[1, :], zs=zl_circle[2, :], color='k')
        ax.plot(xs=zu_circle[0, :], ys=zu_circle[1, :], zs=zu_circle[2, :], color='k')
        ax.set_title('Acquisition function', fontsize=50)
        plt.show()

        # Plot GP
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        bo_plot_gp_sphere(ax, model, xs=bo_optimizer.acquisition.data[0], opt_x=Bopt.x, true_opt_x=true_min,
                          max_colors=max_colors, elev=10, azim=30, n_elems=100)
        # Plot bounds
        ax.plot(xs=yl_circle[0, :], ys=yl_circle[1, :], zs=yl_circle[2, :], color='k')
        ax.plot(xs=yu_circle[0, :], ys=yu_circle[1, :], zs=yu_circle[2, :], color='k')
        ax.plot(xs=zl_circle[0, :], ys=zl_circle[1, :], zs=zl_circle[2, :], color='k')
        ax.plot(xs=zu_circle[0, :], ys=zu_circle[1, :], zs=zu_circle[2, :], color='k')
        ax.set_title('GP mean', fontsize=50)
        plt.show()

        # Plot GP projected on planes
        fig = plt.figure(figsize=(20, 10))
        ax1, ax2 = bo_plot_gp_sphere_planar(fig, model, var_fact=2., xs=bo_optimizer.acquisition.data[0],
                                            ys=bo_optimizer.acquisition.data[1], opt_x=Bopt.x,
                                            opt_y=test_function(Bopt.x), true_opt_x=true_min, true_opt_y=true_opt_val,
                                            max_colors=max_colors, n_elems=100)
        # Plot bounds
        ax1.plot(xs=[xl, xl], ys=[yl, yu], zs=[true_opt_val[0, 0], true_opt_val[0, 0]], color='k')
        ax1.plot([xu, xu], [yl, yu], zs=[true_opt_val[0, 0], true_opt_val[0, 0]], color='k')
        ax1.plot([xl, xu], [yl, yl], zs=[true_opt_val[0, 0], true_opt_val[0, 0]], color='k')
        ax1.plot([xl, xu], [yu, yu], zs=[true_opt_val[0, 0], true_opt_val[0, 0]], color='k')

        ax2.plot([yl, yl], [zl, zu], zs=[true_opt_val[0, 0], true_opt_val[0, 0]], color='k')
        ax2.plot([yu, yu], [zl, zu], zs=[true_opt_val[0, 0], true_opt_val[0, 0]], color='k')
        ax2.plot([yl, yu], [zl, zl], zs=[true_opt_val[0, 0], true_opt_val[0, 0]], color='k')
        ax2.plot([yl, yu], [zu, zu], zs=[true_opt_val[0, 0], true_opt_val[0, 0]], color='k')
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
        x_eval = bo_optimizer.acquisition.data[0]
        y_eval = bo_optimizer.acquisition.data[1]
        plt_scale_fact = test_function(true_min)[0, 0]  # optimal value
        for n in range(x_eval.shape[0]):
            ax.scatter(x_eval[n, 0], x_eval[n, 1], x_eval[n, 2],
                       c=pl.cm.inferno(1. - (y_eval[n] - true_opt_val[0]) / max_colors))
        # Plot true minimum
        ax.scatter(true_min[0, 0], true_min[0, 1], true_min[0, 2], s=40, c='g', marker='P')
        # Plot BO minimum
        ax.scatter(Bopt.x[0, 0], Bopt.x[0, 1], Bopt.x[0, 2], s=20, c='r', marker='D')
        # Plot bounds
        ax.plot(xs=yl_circle[0, :], ys=yl_circle[1, :], zs=yl_circle[2, :], color='k')
        ax.plot(xs=yu_circle[0, :], ys=yu_circle[1, :], zs=yu_circle[2, :], color='k')
        ax.plot(xs=zl_circle[0, :], ys=zl_circle[1, :], zs=zl_circle[2, :], color='k')
        ax.plot(xs=zu_circle[0, :], ys=zu_circle[1, :], zs=zu_circle[2, :], color='k')
        ax.set_title('BO observations', fontsize=30)
        plt.show()

    # Convergence plots
    # Compute distances between consecutive x's
    nb_eval = x_eval.shape[0]
    distances = np.zeros(nb_eval - 1)
    for n in range(nb_eval - 1):
        distances[n] = sphere_manifold.dist(x_eval[n + 1, :], x_eval[n, :])
    # Compute best evaluation for each iteration
    y_best = np.ones(nb_eval)
    for i in range(nb_eval):
        y_best[i] = y_eval[:(i + 1)].min()

    #  Plot distances between consecutive x's
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.array(range(nb_eval - 1)), distances, '-ro')
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('d(x[n], x[n-1])', fontsize=18)
    plt.title('Distance between consecutive observations', fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    # Plot best estimation at each iteration
    plt.subplot(1, 2, 2)
    plt.plot(np.array(range(nb_eval)), y_best, '-o')
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Best y', fontsize=18)
    plt.title('Value of the best selected sample', fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
