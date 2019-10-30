import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from BoManifolds.Riemannian_utils.geometry_utils import rotation_matrix_from_axis_angle
from BoManifolds.plot_utils.manifold_plots import plot_spd_cone

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']

'''
Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
'''


def bo_plot_function_sphere(ax, function, true_opt_x=None, true_opt_y=None, max_colors=None, alpha=0.4, elev=30,
                            azim=-60, n_elems=100, half_sphere=False):
    """
    Plot a function on the surface of a unit 2-sphere

    Parameters
    ----------
    :param ax: figure axis
    :param function: function to represent

    Optional parameters
    -------------------
    :param true_opt_x: true minimum point on the sphere                 [1 x 3]
    :param true_opt_y: true minimum value
    :param max_colors: maximum value (to bound the colors)
    :param alpha: transparency
    :param elev: axis elevation
    :param azim: axis azimut
    :param n_elems: number of elements to approximate the sphere

    Returns
    -------
    :return: maximum value used to bound the colors
        (typically used as scaling value when calling bo_plot_gp_sphere to have the same color scale)
    """
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
    ax.view_init(elev=elev, azim=azim)

    # Sphere
    u = np.linspace(0, 2 * np.pi, n_elems)
    if half_sphere:
        u = np.linspace(3*np.pi/2, 5* np.pi/2, n_elems)
    v = np.linspace(0, np.pi, n_elems)

    r = 0.99
    x_sphere = r * np.outer(np.cos(u), np.sin(v))
    y_sphere = r * np.outer(np.sin(u), np.sin(v))
    z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Colors in function the output of the test function
    colors = np.zeros(x_sphere.shape)
    for i in range(x_sphere.shape[0]):
        for j in range(x_sphere.shape[1]):
            data_tmp = np.array([[x_sphere[i, j], y_sphere[i, j], z_sphere[i, j]]])
            colors[i, j] = function(data_tmp)

    # Rescale the colors
    if true_opt_y is not None:
        min_colors = true_opt_y
    else:
        min_colors = np.min(colors)

    colors = colors - min_colors

    if max_colors is None:
        max_colors = np.max(colors)
    else:
        np.min([colors, max_colors * np.ones(colors.shape)], axis=0)

    colors = pl.cm.inferno(np.ones(colors.shape) - colors / max_colors)

    # Plot sphere with colored surface
    ax.plot_surface(x_sphere, y_sphere, z_sphere, rstride=4, cstride=4, facecolors=colors, linewidth=0., alpha=alpha)

    # Plot true minimum
    ax.scatter(true_opt_x[0, 0], true_opt_x[0, 1], true_opt_x[0, 2], s=300, c='lime', marker='*')
    # ax.scatter(true_opt_x[0, 0], true_opt_x[0, 1], true_opt_x[0, 2], s=300, c='k', marker='*')

    # Limits
    lim = 1.1
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    return max_colors


def bo_plot_function_sphere_planar(fig, function, true_opt_x=None, true_opt_y = None, max_colors=None, n_elems=100):
    """
    Plot a function on 2d-projections of a unit 2-sphere

    Parameters
    ----------
    :param fig: figure
    :param function: function to represent

    Optional parameters
    -------------------
    :param true_opt_x: true optimum                                     [1 x 3]
    :param true_opt_y: true optimum value
    :param max_colors: maximum value (to bound the colors)
    :param n_elems: number of elements to approximate the sphere

    Returns
    -------
    :return: axis of the two subplots
    """
    # Sphere
    u = np.linspace(0, 2 * np.pi, n_elems)
    v = np.linspace(0, np.pi, n_elems)

    r = 1
    x_sphere = r * np.outer(np.cos(u), np.sin(v))
    y_sphere = r * np.outer(np.sin(u), np.sin(v))
    z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Values and color in function of function
    function_values = np.zeros(x_sphere.shape)
    for i in range(x_sphere.shape[0]):
        for j in range(x_sphere.shape[1]):
            data_tmp = np.array([[x_sphere[i, j], y_sphere[i, j], z_sphere[i, j]]])
            function_values[i, j] = function(data_tmp)

    # Rescale the colors
    if true_opt_y is not None:
        min_colors = true_opt_y
    else:
        min_colors = np.min(function_values)

    colors = np.max([function_values - min_colors, min_colors * np.ones(function_values.shape)], axis=0)

    if max_colors is None:
        max_colors = np.max(colors)
    else:
        colors = np.min([colors, max_colors * np.ones(colors.shape)], axis=0)

    colors = colors / max_colors
    colors = pl.cm.inferno(np.ones(colors.shape) - colors)

    # Plot x vs y
    ax1 = fig.add_subplot(121, projection='3d')

    # Make the panes transparent
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Plot projected function
    ax1.plot_surface(x_sphere, y_sphere, function_values, rstride=8, cstride=8, facecolors=colors, alpha=0.2,
                     edgecolor='white', linewidth=0.3)

    # Plot true minimum
    if true_opt_x is not None and true_opt_y is not None:
        ax1.scatter(true_opt_x[0, 0], true_opt_x[0, 1], true_opt_y, s=200, c='k', marker='*')

    # ax1.locator_params(axis='x', nbins=4)
    # ax1.locator_params(axis='y', nbins=4)
    ax1.locator_params(axis='z', nbins=4)
    ax1.set_xticks([-1., 1.])
    ax1.set_yticks([-1., 1.])
    ax1.set_zticks([0., 10.])

    ax1.tick_params(labelsize=32)

    ax1.set_xlabel(r'$x_1$', fontsize=48)
    ax1.set_ylabel(r'$x_2$', fontsize=48)
    ax1.set_zlabel(r'$f(\mathbf{x})$', fontsize=48)
    # ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
    # ax1.set_zlabel('Cost value', fontsize=20, rotation=90)

    # Plot y vs z
    ax2 = fig.add_subplot(122, projection='3d')

    # Make the panes transparent
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Plot projected function
    ax2.plot_surface(y_sphere, z_sphere, function_values, rstride=8, cstride=8, facecolors=colors, alpha=0.2,
                     edgecolor='white', linewidth=0.3)

    # Plot true minimum
    if true_opt_x is not None and true_opt_y is not None:
        ax2.scatter(true_opt_x[0, 1], true_opt_x[0, 2], true_opt_y, s=200, c='k', marker='*')

    # ax2.locator_params(axis='x', nbins=4)
    # ax2.locator_params(axis='y', nbins=4)
    ax2.locator_params(axis='z', nbins=4)
    ax2.set_xticks([-1., 1.])
    ax2.set_yticks([-1., 1.])
    ax2.set_zticks([0., 10.])

    ax2.tick_params(labelsize=32)

    ax2.set_xlabel(r'$x_2$', fontsize=48)
    ax2.set_ylabel(r'$x_3$', fontsize=48)
    ax2.set_zlabel(r'$f(\mathbf{x})$', fontsize=48)
    # ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
    # ax2.set_zlabel('Cost value', fontsize=20, rotation=90)

    return ax1, ax2


def bo_plot_acquisition_sphere(ax, acq_fct, xs=None, opt_x=None, true_opt_x=None, alpha=0.4, elev=30, azim=-60,
                               n_elems=100):
    """
    Plot an acquisition function at the surface of a unit 2-sphere

    Parameters
    ----------
    :param ax: figure axis
    :param acq_fct: acquisition function

    Optional parameters
    -------------------
    :param xs: samples of the BO                                    [n x 3]
    :param opt_x: current best optimizer of the BO                  [1 x 3]
    :param true_opt_x: true best optimizer                          [1 x 3]
    :param alpha: transparency
    :param elev: axis elevation
    :param azim: axis azimut
    :param n_elems: number of elements to approximate the sphere

    Returns
    -------
    :return: -
    """
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
    ax.view_init(elev=elev, azim=azim)

    # Sphere
    u = np.linspace(0, 2 * np.pi, n_elems)
    v = np.linspace(0, np.pi, n_elems)

    r = 1
    x_sphere = r * np.outer(np.cos(u), np.sin(v))
    y_sphere = r * np.outer(np.sin(u), np.sin(v))
    z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Colors in function of acquisition function
    colors = np.zeros(x_sphere.shape)
    for i in range(x_sphere.shape[0]):
        for j in range(x_sphere.shape[1]):
            data_tmp = np.array([[x_sphere[i, j], y_sphere[i, j], z_sphere[i, j]]])
            colors[i, j] = acq_fct.evaluate(data_tmp)

    # Rescale the colors
    colors = colors - np.min(colors)
    colors = colors / np.max(colors)
    colors = pl.cm.inferno(np.ones(colors.shape) - colors)

    # Plot colored sphere
    ax.plot_surface(x_sphere, y_sphere, z_sphere, rstride=4, cstride=4, facecolors=colors, linewidth=0., alpha=alpha)

    # Plots observations xs
    if xs is not None:
        for n in range(xs.shape[0]):
            ax.scatter(xs[n, 0], xs[n, 1], xs[n, 2], c='k', s=80)

    # Plot current optimum
    if opt_x is not None:
        ax.scatter(opt_x[0, 0], opt_x[0, 1], opt_x[0, 2], s=200, c='deepskyblue', marker='D')

    # Plot true minimum
    if true_opt_x is not None:
        ax.scatter(true_opt_x[0, 0], true_opt_x[0, 1], true_opt_x[0, 2], s=300, c='limegreen', marker='*')

    # Limits
    lim = 1.1
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])


def bo_plot_gp_sphere(ax, model, xs=None, ys=None, opt_x=None, true_opt_x=None, true_opt_y=None, alpha=0.4,
                      max_colors=None, elev=30, azim=-60, n_elems=100, half_sphere=False):
    """
    Plot a GP at the surface of a unit 2-sphere

    Parameters
    ----------
    :param ax: figure axis
    :param model: GP model (gpflowopt model)

    Optional parameters
    -------------------
    :param xs: observations of the GP (samples of the BO)               [n x 3]
    :param opt_x: current best optimizer (of the BO)                    [1 x 3]
    :param true_opt_x: true optimum                                     [1 x 3]
    :param true_opt_y: true optimum value
    :param alpha: transparency
    :param max_colors: maximum value (to bound the colors, e.g. equal to the true function maximum value)
    :param elev: axis elevation
    :param azim: axis azimut
    :param n_elems: number of elements to approximate the sphere

    Returns
    -------
    :return: -
    """
    # Make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    # Remove axis
    # ax._axis3don = False

    # Initial view
    ax.view_init(elev=elev, azim=azim)

    # Sphere
    u = np.linspace(0, 2 * np.pi, n_elems)
    if half_sphere:
        u = np.linspace(3*np.pi/2, 5* np.pi/2, n_elems)
    v = np.linspace(0, np.pi, n_elems)

    r = 0.99
    x_sphere = r * np.outer(np.cos(u), np.sin(v))
    y_sphere = r * np.outer(np.sin(u), np.sin(v))
    z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Colors in function of acquisition function
    colors = np.zeros(x_sphere.shape)
    for i in range(x_sphere.shape[0]):
        for j in range(x_sphere.shape[1]):
            data_tmp = np.array([[x_sphere[i, j], y_sphere[i, j], z_sphere[i, j]]])
            colors[i, j] = model.predict_f(data_tmp)[0]

    # Rescale the colors
    if true_opt_y is not None:
        min_colors = true_opt_y
    else:
        min_colors = np.min(colors)

    colors = np.max([colors - min_colors, min_colors * np.ones(colors.shape)], axis=0)

    if max_colors is None:
        max_colors = np.max(colors)
    else:
        np.min([colors, max_colors * np.ones(colors.shape)], axis=0)
    colors = colors/max_colors

    colors = pl.cm.inferno(np.ones(colors.shape) - colors)

    # Plot colored surface
    ax.plot_surface(x_sphere, y_sphere, z_sphere, rstride=4, cstride=4, facecolors=colors, linewidth=0., alpha=alpha)
    # Plot gray surface (if the point are plotted with colors later)
    # ax.plot_surface(x_sphere, y_sphere, z_sphere, rstride=4, cstride=4, color=[0.8, 0.8, 0.8], linewidth=0.,
    #                 alpha=alpha)

    # Plots observations xs
    if ys is not None:
        # For colored observations
        colors_ys = pl.cm.inferno(np.ones(ys.shape) - (ys - min_colors) / max_colors)

    if xs is not None:
        for n in range(xs.shape[0]):
            # Plot observations in black
            ax.scatter(xs[n, 0], xs[n, 1], xs[n, 2], c='k', s=80)
            # Plot colored observations
            # ax.scatter(xs[n, 0], xs[n, 1], xs[n, 2], s=80, c=colors_ys[n])

    # Plot current optimum
    if opt_x is not None:
        ax.scatter(opt_x[0, 0], opt_x[0, 1], opt_x[0, 2], s=200, c='deepskyblue', marker='D')

    # Plot true minimum
    if true_opt_x is not None:
        ax.scatter(true_opt_x[0, 0], true_opt_x[0, 1], true_opt_x[0, 2], s=300, c='lime', marker='*')
        # ax.scatter(true_opt_x[0, 0], true_opt_x[0, 1], true_opt_x[0, 2], s=200, c='k', marker='*')

    # Limits
    lim = 1.1
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    # Labels
    # ax.locator_params(axis='x', nbins=2)
    # ax.locator_params(axis='y', nbins=2)
    # ax.locator_params(axis='z', nbins=2)
    plt.xticks([-1., 1.])
    plt.yticks([-1., 1.])
    ax.set_zticks([-1., 1.])

    ax.tick_params(labelsize=32)
    ax.set_xlabel(r'$x_1$', fontsize=48)
    ax.set_ylabel(r'$x_2$', fontsize=48)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$x_3$', fontsize=48, rotation=90)


def bo_plot_gp_sphere_planar(fig, model, var_fact=2., xs=None, ys=None, opt_x=None, opt_y=None, true_opt_x=None,
                             true_opt_y = None, max_colors=None, n_elems=100):
    """
    Plot a GP on 2d-projections of a unit 2-sphere

    Parameters
    ----------
    :param fig: figure
    :param model: GP model

    Optional parameters
    -------------------
    :param var_fact: displayed variance factor
    :param xs: observations of the GP (samples of the BO)               [n x 3]
    :param ys: value of the observations
    :param opt_x: current best optimizer (of the BO)                    [1 x 3]
    :param opt_y: value of the current best optimizer (of the BO)
    :param true_opt_x: true optimum                                     [1 x 3]
    :param true_opt_y: true optimum value
    :param max_colors: maximum value (to bound the colors)
    :param n_elems: number of elements to approximate the sphere

    Returns
    -------
    :return: axis of the two subplots
    """
    # Sphere
    u = np.linspace(0, 2 * np.pi, n_elems)
    v = np.linspace(0, np.pi, n_elems)

    r = 1
    x_sphere = r * np.outer(np.cos(u), np.sin(v))
    y_sphere = r * np.outer(np.sin(u), np.sin(v))
    z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Values and color in function of function
    fmean = np.zeros(x_sphere.shape)
    fvar = np.zeros(x_sphere.shape)
    for i in range(x_sphere.shape[0]):
        for j in range(x_sphere.shape[1]):
            data_tmp = np.array([[x_sphere[i, j], y_sphere[i, j], z_sphere[i, j]]])
            fmean[i, j], fvar[i, j] = model.predict_f(data_tmp)

    # Rescale the colors
    if true_opt_y is not None:
        min_colors = true_opt_y
    else:
        min_colors = np.min(fmean)

    colors = np.max([fmean - min_colors, min_colors * np.ones(fmean.shape)], axis=0)

    if max_colors is None:
        max_colors = np.max(colors)
    else:
        colors = np.min([colors, max_colors * np.ones(colors.shape)], axis=0)

    colors = colors / max_colors
    colors = pl.cm.inferno(np.ones(colors.shape) - colors)

    if ys is not None:
        colors_ys = pl.cm.inferno(np.ones(ys.shape) - (ys - min_colors) / max_colors)

    # Plot x vs y
    ax1 = fig.add_subplot(121, projection='3d')

    # Make the panes transparent
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Plot projected mean and variance of the GP
    ax1.plot_surface(x_sphere, y_sphere, fmean, rstride=8, cstride=8, facecolors=colors, alpha=0.2, edgecolor='white', linewidth=0.3)
    # ax1.plot_surface(x_sphere, y_sphere, fmean, rstride=8, cstride=8, color='deepskyblue', alpha=0.2, edgecolor='gray', linewidth=0.3)
    ax1.plot_surface(x_sphere, y_sphere, fmean + var_fact*fvar, rstride=4, cstride=4, color=[0.5, 0.5, 0.5], alpha=0.1)
    ax1.plot_surface(x_sphere, y_sphere, fmean - var_fact*fvar, rstride=4, cstride=4, color=[0.5, 0.5, 0.5], alpha=0.1)

    # Plots observation xs
    if xs is not None and ys is not None:
        for n in range(xs.shape[0]):
            # ax1.scatter(xs[n, 0], xs[n, 1], ys[n], c='darkblue', s=25)
            ax1.scatter(xs[n, 0], xs[n, 1], ys[n], c=colors_ys[n], s=50)

    # Plot current optimum
    if opt_x is not None and opt_y is not None:
        ax1.scatter(opt_x[0, 0], opt_x[0, 1], opt_y, s=100, c='deepskyblue', marker='D')

    # Plot true minimum
    if true_opt_x is not None and true_opt_y is not None:
        ax1.scatter(true_opt_x[0, 0], true_opt_x[0, 1], true_opt_y, s=200, c='limegreen', marker='*')

    # ax1.locator_params(axis='x', nbins=4)
    # ax1.locator_params(axis='y', nbins=4)
    ax1.locator_params(axis='z', nbins=4)
    ax1.set_xticks([-1., 1.])
    ax1.set_yticks([-1., 1.])
    ax1.set_zticks([0., 10.])

    ax1.tick_params(labelsize=32)

    ax1.set_xlabel(r'$x_1$', fontsize=48)
    ax1.set_ylabel(r'$x_2$', fontsize=48)
    ax1.set_zlabel(r'$\mu(\mathbf{x})$', fontsize=48)
    # ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
    # ax1.set_zlabel('Cost value', fontsize=20, rotation=90)

    # Plot y vs z
    ax2 = fig.add_subplot(122, projection='3d')

    # Make the panes transparent
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Plot projected mean and variance of the GP
    ax2.plot_surface(y_sphere, z_sphere, fmean, rstride=8, cstride=8, facecolors=colors, alpha=0.2, edgecolor='white', linewidth=0.3)
    # ax2.plot_surface(y_sphere, z_sphere, fmean, rstride=8, cstride=8, color='deepskyblue', alpha=0.2, edgecolor='gray', linewidth=0.3)
    ax2.plot_surface(y_sphere, z_sphere, fmean + var_fact*fvar, rstride=4, cstride=4, color=[0.5, 0.5, 0.5], alpha=0.1)
    ax2.plot_surface(y_sphere, z_sphere, fmean - var_fact*fvar, rstride=4, cstride=4, color=[0.5, 0.5, 0.5], alpha=0.1)

    # Plots observations xs
    if xs is not None and ys is not None:
        for n in range(xs.shape[0]):
            # ax2.scatter(xs[n, 1], xs[n, 2], ys[n], c='darkblue', s=25)
            ax2.scatter(xs[n, 1], xs[n, 2], ys[n], c=colors_ys[n], s=50)

    # Plot current optimum
    if opt_x is not None and opt_y is not None:
        ax2.scatter(opt_x[0, 1], opt_x[0, 2], opt_y, s=100, c='deepskyblue', marker='D')

    # Plot true minimum
    if true_opt_x is not None and true_opt_y is not None:
        ax2.scatter(true_opt_x[0, 1], true_opt_x[0, 2], true_opt_y, s=200, c='limegreen', marker='*')

    # ax2.locator_params(axis='x', nbins=4)
    # ax2.locator_params(axis='y', nbins=4)
    ax2.locator_params(axis='z', nbins=4)
    ax2.set_xticks([-1., 1.])
    ax2.set_yticks([-1., 1.])
    ax2.set_zticks([0., 10.])

    ax2.tick_params(labelsize=32)

    ax2.set_xlabel(r'$x_2$', fontsize=48)
    ax2.set_ylabel(r'$x_3$', fontsize=48)
    ax2.set_zlabel(r'$\mu(\mathbf{x})$', fontsize=48)
    # ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
    # ax2.set_zlabel('Cost value', fontsize=20, rotation=90)

    return ax1, ax2


def bo_plot_function_spd(ax, function, r_cone, true_opt_x=None, true_opt_y=None, chol=False, max_colors=None,
                         alpha=0.3, elev=10, azim=-20, n_elems=100, n_elems_h=10):
    """
    Plot a function in the SPD cone

    Parameters
    ----------
    :param ax: figure axis
    :param function: function
    :param r_cone: cone radius

    Optional parameters
    -------------------
    :param true_opt_x: true minimum point on the manifold                 [1 x 3]
    :param true_opt_y: true minimum value
    :param chol: if True, the Cholesky decomposition is used
    :param max_colors: maximum value (to bound the colors)
    :param alpha: transparency
    :param elev: axis elevation
    :param azim: axis azimut
    :param n_elems: number of elements to plot in a slice of the cone
    :param n_elems_h: number of slices of the cone to plot

    Returns
    -------
    :return: maximum value used to bound the colors
        (typically used as scaling value when calling bo_plot_gp_spd to have the same color scale)
    """
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
    ax.view_init(elev=elev, azim=azim)
    # ax.view_init(elev=10, azim=50.)

    # Plot SPD cone
    plot_spd_cone(ax, r=r_cone, lim_fact=0.8)

    # Values of test function for points on the manifold
    phi = np.linspace(0, 2 * np.pi, n_elems)

    # Matrix for rotation of 45° of the cone
    dir = np.cross(np.array([1, 0, 0]), np.array([1., 1., 0.]))
    R = rotation_matrix_from_axis_angle(dir, np.pi / 4.)

    # Points of the cone
    h = np.linspace(0.01, r_cone, n_elems_h)
    x_cone = np.zeros((n_elems_h, n_elems, n_elems))
    y_cone = np.zeros((n_elems_h, n_elems, n_elems))
    z_cone = np.zeros((n_elems_h, n_elems, n_elems))
    colors = np.zeros((n_elems_h, n_elems, n_elems))
    for k in range(n_elems_h):
        r = np.linspace(0, h[k] - 0.01, n_elems)
        for i in range(n_elems):
            # Points on a plane cutting the cone
            xyz = np.vstack((h[k] * np.ones(n_elems), r[i] * np.sin(phi), r[i] / np.sqrt(2) * np.cos(phi)))

            # Rotation
            xyz = R.dot(xyz)

            # Coordinates
            x_cone[k, i] = xyz[0]
            y_cone[k, i] = xyz[1]
            z_cone[k, i] = xyz[2]

        # Compute the function values at given points
        for i in range(n_elems):
            for j in range(n_elems):
                if not chol:
                    data_tmp = np.array([[x_cone[k, i, j], y_cone[k, i, j], z_cone[k, i, j] * np.sqrt(2)]])
                    colors[k, i, j] = function(data_tmp)
                else:
                    indices = np.tril_indices(2)
                    data_tmp = np.array([[x_cone[k, i, j], z_cone[k, i, j]], [z_cone[k, i, j], y_cone[k, i, j]]])
                    data_chol_tmp = np.linalg.cholesky(data_tmp)
                    colors[k, i, j] = function(data_chol_tmp[indices])

    # Rescale the colors
    if true_opt_y is not None:
        min_colors = true_opt_y
    else:
        min_colors = np.min(colors)
    colors = (colors - min_colors)
    if max_colors is None:
        max_colors = np.max(colors)
    else:
        colors = np.min([colors, max_colors * np.ones(colors.shape)], axis=0)
    colors = colors / max_colors

    # Plot colored cone
    for k in range(n_elems_h):
        colors_plot = pl.cm.inferno(np.ones((n_elems, n_elems)) - colors[k])

        ax.plot_surface(x_cone[k], y_cone[k], z_cone[k], rstride=4, cstride=4, facecolors=colors_plot, linewidth=0., alpha=alpha)

    # Plot optimal point
    if true_opt_x is not None:
        ax.scatter(true_opt_x[0, 0], true_opt_x[1, 1], true_opt_x[0, 1], s=300, c='g', marker='*')

    return max_colors


def bo_plot_acquisition_spd(ax, acq_fct, r_cone, xs=None, ys=None, opt_x=None, true_opt_x=None, chol=False, alpha=0.3,
                            elev=10, azim=-20, n_elems=100, n_elems_h=10):
    """
    Plot an acquisition function in the SPD cone

    Parameters
    ----------
    :param ax: figure axis
    :param acq_fct: acquisition function
    :param r_cone: cone radius

    Optional parameters
    -------------------
    :param xs: samples of the BO                        [n x 3]
    :param ys: value of the samples of the BO
    :param opt_x: current best optimizer of the BO      [1 x 3]
    :param true_opt_x: true minimum point               [1 x 3]
    :param chol: if True, the Cholesky decomposition is used
    :param alpha: transparency
    :param elev: axis elevation
    :param azim: axis azimut
    :param n_elems: number of elements to plot in a slice of the cone
    :param n_elems_h: number of slices of the cone to plot

    Returns
    -------
    :return: -
    """
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
    ax.view_init(elev=elev, azim=azim)

    # Plot SPD cone
    plot_spd_cone(ax, r=r_cone, lim_fact=0.8)

    # Values of test function for points on the manifold
    phi = np.linspace(0, 2 * np.pi, n_elems)

    # Matrix for rotation of 45° of the cone
    dir = np.cross(np.array([1, 0, 0]), np.array([1., 1., 0.]))
    R = rotation_matrix_from_axis_angle(dir, np.pi / 4.)

    # Points of the cone
    h = np.linspace(0.01, r_cone, n_elems_h)
    x_cone = np.zeros((n_elems_h, n_elems, n_elems))
    y_cone = np.zeros((n_elems_h, n_elems, n_elems))
    z_cone = np.zeros((n_elems_h, n_elems, n_elems))
    colors = np.zeros((n_elems_h, n_elems, n_elems))
    for k in range(n_elems_h):
        r = np.linspace(0, h[k] - 0.01, n_elems)
        for i in range(n_elems):
            # Points on a plane cutting the cone
            xyz = np.vstack((h[k] * np.ones(n_elems), r[i] * np.sin(phi), r[i] / np.sqrt(2) * np.cos(phi)))

            # Rotation
            xyz = R.dot(xyz)

            # Coordinates
            x_cone[k, i] = xyz[0]
            y_cone[k, i] = xyz[1]
            z_cone[k, i] = xyz[2]

        for i in range(n_elems):
            for j in range(n_elems):
                if not chol:
                    data_tmp = np.array([[x_cone[k, i, j], y_cone[k, i, j], z_cone[k, i, j] * np.sqrt(2)]])
                    colors[k, i, j] = acq_fct.evaluate(data_tmp)
                else:
                    indices = np.tril_indices(2)
                    data_tmp = np.array([[x_cone[k, i, j], z_cone[k, i, j]], [z_cone[k, i, j], y_cone[k, i, j]]])
                    data_chol_tmp = np.linalg.cholesky(data_tmp)
                    colors[k, i, j] = acq_fct.evaluate(data_chol_tmp[indices][None])

    # Rescale the colors
    min_colors = np.min(colors)
    colors = (colors - min_colors)
    max_colors = np.max(colors)
    colors = np.min([max_colors * np.ones(colors.shape), colors], axis=0)
    colors = colors / max_colors

    # Plot colored cone
    for k in range(n_elems_h):
        colors_plot = pl.cm.inferno(np.ones((n_elems, n_elems)) - colors[k])
        ax.plot_surface(x_cone[k], y_cone[k], z_cone[k], rstride=4, cstride=4, facecolors=colors_plot, linewidth=0., alpha=alpha)

    # Plots observations xs
    if ys is not None:
        # For colored observations
        colors_ys = pl.cm.inferno(np.ones(ys.shape) - (ys - min_colors) / max_colors)

    if xs is not None:
        for n in range(xs.shape[0]):
            # Plot black observations
            ax.scatter(xs[n, 0], xs[n, 1], xs[n, 2] / np.sqrt(2), s=80, c='k')
            # Plot colored observations
            # ax.scatter(xs[n, 0], xs[n, 1], xs[n, 2] / np.sqrt(2), s=80, c=colors_ys[n])

    # Plot opt x
    if opt_x is not None:
        ax.scatter(opt_x[0, 0], opt_x[0, 1], opt_x[0, 2] / np.sqrt(2), s=120, c='deepskyblue', marker='D')

    # Plot true minimum
    if true_opt_x is not None:
        ax.scatter(true_opt_x[0, 0], true_opt_x[0, 1], true_opt_x[0, 2] / np.sqrt(2), s=300, c='g', marker='*')


def bo_plot_gp_spd(ax, model, r_cone, xs=None, ys=None, opt_x=None, true_opt_x=None, true_opt_y=None, chol=False,
                   max_colors=None, alpha=0.3, elev=10, azim=-20, n_elems=100, n_elems_h=10):
    """
    Plot a GP in the SPD cone

    Parameters
    ----------
    :param ax: figure axis
    :param model: GP model
    :param r_cone: cone radius

    Optional parameters
    -------------------
    :param xs: samples of the BO                        [n x 3]
    :param ys: value of the samples of the BO
    :param opt_x: current best optimizer of the BO      [1 x 3]
    :param true_opt_x: true minimum point               [1 x 3]
    :param true_opt_y: true minimum value
    :param chol: if True, the Cholesky decomposition is used
    :param max_colors: maximum value (to bound the colors)
    :param alpha: transparency
    :param elev: axis elevation
    :param azim: axis azimut
    :param n_elems: number of elements to plot in a slice of the cone
    :param n_elems_h: number of slices of the cone to plot

    Returns
    -------
    :return: -
    """
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
    ax.view_init(elev=elev, azim=azim)

    # Plot SPD cone
    plot_spd_cone(ax, r=r_cone, lim_fact=0.8)

    # Value of test function for points on the manifold
    # Values of test function for points on the manifold
    phi = np.linspace(0, 2 * np.pi, n_elems)

    # Matrix for rotation of 45° of the cone
    dir = np.cross(np.array([1, 0, 0]), np.array([1., 1., 0.]))
    R = rotation_matrix_from_axis_angle(dir, np.pi / 4.)

    # Points of the cone
    h = np.linspace(0.01, r_cone, n_elems_h)
    x_cone = np.zeros((n_elems_h, n_elems, n_elems))
    y_cone = np.zeros((n_elems_h, n_elems, n_elems))
    z_cone = np.zeros((n_elems_h, n_elems, n_elems))
    colors = np.zeros((n_elems_h, n_elems, n_elems))
    var = np.zeros((n_elems_h, n_elems, n_elems))
    for k in range(n_elems_h):
        r = np.linspace(0, h[k] - 0.01, n_elems)
        for i in range(n_elems):
            # Points on a plane cutting the cone
            xyz = np.vstack((h[k] * np.ones(n_elems), r[i] * np.sin(phi), r[i] / np.sqrt(2) * np.cos(phi)))

            # Rotation
            xyz = R.dot(xyz)

            # Coordinates
            x_cone[k, i] = xyz[0]
            y_cone[k, i] = xyz[1]
            z_cone[k, i] = xyz[2]

        for i in range(n_elems):
            for j in range(n_elems):
                if not chol:
                    data_tmp = np.array([[x_cone[k, i, j], y_cone[k, i, j], z_cone[k, i, j] * np.sqrt(2)]])
                    colors[k, i, j], var[k, i, j] = model.predict_f(data_tmp)
                else:
                    indices = np.tril_indices(2)
                    data_tmp = np.array([[x_cone[k, i, j], z_cone[k, i, j]], [z_cone[k, i, j], y_cone[k, i, j]]])
                    data_chol_tmp = np.linalg.cholesky(data_tmp)
                    colors[k, i, j], var[k, i, j] = model.predict_f(data_chol_tmp[indices][None])

    # Rescale the colors
    if true_opt_y is not None:
        min_colors = true_opt_y
    else:
        min_colors = np.min(colors)

    colors = np.max([colors - min_colors, min_colors * np.ones(colors.shape)], axis=0)

    if max_colors is None:
        max_colors = np.max(colors)
    else:
        colors = np.min([colors, max_colors * np.ones(colors.shape)], axis=0)
    colors = colors / max_colors

    # Plot colored cone
    for k in range(n_elems_h):
        colors_plot = pl.cm.inferno(np.ones((n_elems, n_elems)) - colors[k])

        ax.plot_surface(x_cone[k], y_cone[k], z_cone[k], rstride=4, cstride=4, facecolors=colors_plot, linewidth=0.,
                        alpha=alpha)

    # Plots observations xs
    if ys is not None:
        # For colored observations
        colors_ys = pl.cm.inferno(np.ones(ys.shape) - (ys - min_colors) / max_colors)

    if xs is not None:
        for n in range(xs.shape[0]):
            # Plot black observations
            ax.scatter(xs[n, 0], xs[n, 1], xs[n, 2] / np.sqrt(2), s=80, c='k')
            # Plot colored observations
            # ax.scatter(xs[n, 0], xs[n, 1], xs[n, 2] / np.sqrt(2), s=80, c=colors_ys[n])

    # Plot opt x
    if opt_x is not None:
        ax.scatter(opt_x[0, 0], opt_x[0, 1], opt_x[0, 2] / np.sqrt(2), s=120, c='deepskyblue', marker='D')

    # Plot true minimum
    if true_opt_x is not None:
        ax.scatter(true_opt_x[0, 0], true_opt_x[0, 1], true_opt_x[0, 2] / np.sqrt(2), s=300, c='g', marker='*')

