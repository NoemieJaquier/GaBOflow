import numpy as np
import scipy.optimize as sc_opt
import gpflow
import gpflowopt

from BoManifolds.Riemannian_utils.SPD_utils import symmetric_matrix_to_vector_mandel

'''
Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
'''


class SciPyOptimizerWithConstraints(gpflowopt.optim.Optimizer):
    """
    Instances of this class are optimizer that wraps SciPy's minimize function with constraints.

    Attributes
    ----------
    self.config:
    self.constraints:

    Methods
    -------
    _optimize(objective):

    Static methods
    --------------
    """

    def __init__(self, domain, constraints, method='SLSQP', tol=None, maxiter=1000):
        """
        Initialisation

        Parameters
        ----------
        :param domain: gpflowopt domain
        :param constraints: list of constraints in scipy format

        Optional parameters
        -------------------
        :param method: optimization method
                (!! only SLSQP, COBYLA (only inequalities) and trus-constr support constraints !!)
        :param tol: scipy tolerance parameter
        :param maxiter: maximum number of iterations
        """
        super(SciPyOptimizerWithConstraints, self).__init__(domain)
        options = dict(disp=gpflow.settings.verbosity.optimisation_verb,
                       maxiter=maxiter)
        self.config = dict(tol=tol,
                           method=method,
                           options=options)
        self.constraints = constraints

    def _optimize(self, objective):
        """
        Optimize the objective function by calling scipy.optimize.minimize.

        Parameters
        ----------
        :param objective: objective function to minimize

        Returns
        -------
        :return: optimal point found by the optimizer
        """
        objective1d = lambda X: tuple(map(lambda arr: arr.ravel(), objective(X)))
        result = sc_opt.minimize(fun=objective1d,
                                 x0=self.get_initial(),
                                 jac=self.gradient_enabled(),
                                 bounds=list(zip(self.domain.lower, self.domain.upper)),
                                 constraints=self.constraints,
                                 **self.config)
        return result


class AnchorPointsGeneratorWithNormConstraint:
    """
    Instances of this class are generators of anchor points with a specific norm (1 for the sphere manifold).

    Note: for sphere manifolds, manifold_optimization.ManifoldAnchorPointsGen() should be preferred over the following
    anchor points generator (ManifoldAnchorPointsGen is more general).

    Attributes
    ----------
    self.domain:
    self.norm:

    Methods
    -------
    generate(objective, nb_anchor_points, nb_samples):

    Static methods
    --------------
    """
    def __init__(self, domain, norm):
        """
        Initialisation

        Parameters
        ----------
        :param domain: gpflowopt domain
        :param norm: norm of the anchor point (1 for unit spheres)
        """
        self.domain = domain
        self.norm = norm

    def generate(self, objective, nb_anchor_points=5, nb_samples=1000):
        """
        Generate anchor points with the given norm. Samples are generated and the points with the best score are
        returned.

        Parameters
        ----------
        :param objective: objective function to minimize
        :param nb_anchor_points: number of anchor points to return

        Optional parameters
        -------------------
        :param nb_samples: number of samples where the objective function is evaluated

        Returns
        -------
        :return: sample points with the specified norm with the lowest objective function value
        """
        # No checks are made for duplicate points here. We could try to include something to ensure that the points
        # are somehow separated from each other.
        points = gpflowopt.design.RandomDesign(nb_samples, self.domain).generate()
        points = points*self.norm/np.linalg.norm(points, axis=1)[:, None]
        scores = objective(points)[0][:, 0]

        anchor_points = points[np.argsort(scores)[:min(len(scores), nb_anchor_points)], :]

        return anchor_points


class AnchorPointsGeneratorWithNormConstraintInDomain:
    """
    Instances of this class are generators of anchor points with a specific norm in a reduced domain
    (norm 1 for the sphere manifold).

    Note: for sphere manifolds, manifold_optimization.ManifoldAnchorPointsGen() should be preferred over the following
    anchor points generator (ManifoldAnchorPointsGen is more general).

    Attributes
    ----------
    self.domain:
    self.norm:

    Methods
    -------
    generate(objective, nb_anchor_points, nb_samples):

    Static methods
    --------------
    """
    def __init__(self, domain, norm):
        """
        Initialisation

        Parameters
        ----------
        :param domain: gpflowopt domain
        :param norm: norm of the anchor point (1 for unit spheres)
        """
        self.domain = domain
        self.norm = norm

    def generate(self, objective, nb_anchor_points=5, nb_samples=1000):
        """
        Generate anchor points with the given norm in the given domain. Samples are generated and the points with
        the best score are returned.

        Parameters
        ----------
        :param objective: objective function to minimize

        Optional parameters
        -------------------
        :param nb_anchor_points: number of anchor points to return
        :param nb_samples: number of samples where the objective function is evaluated

        Returns
        -------
        :return: sample points with the lowest objective function value
        """
        # No checks are made for duplicate points here. We could try to include something to ensure that the points
        # are somehow separated from each other.
        points = gpflowopt.design.RandomDesign(nb_samples, self.domain).generate()
        points = points*self.norm/np.linalg.norm(points, axis=1)[:, None]
        for n in range(nb_samples):
            while not points[n] in self.domain:
                points[n] = gpflowopt.design.RandomDesign(1, self.domain).generate()
                points[n] = points[n] * self.norm / np.linalg.norm(points[n])
        scores = objective(points)[0][:, 0]

        anchor_points = points[np.argsort(scores)[:min(len(scores), nb_anchor_points)], :]

        return anchor_points


class AnchorPointsGeneratorWithSpdConstraint:
    """
    Instances of this class are generators of symmetric positive definite (SPD) anchor points.

    This class was created to overcome the limitation in the rand function of the psd class of pymanopt (as it samples
    for eigenvalues from 0 to 1, and therefore does not cover so well the domain).
    Another way to achieve the same result is to set psd.rand as equal to the function generate_samples below and to use
    manifold_optimization.ManifoldAnchorPointsGen() (to be more general).

    Attributes
    ----------
    self.domain:
    self.dim:
    self.min_eigenvalue:
    self.max_eigenvalue:

    Methods
    -------
    generate_samples(nb_samples)
    generate(objective, nb_anchor_points, nb_samples):

    Static methods
    --------------
    """
    def __init__(self, domain, dim, min_eigenvalue=1., max_eigenvalue=2.):
        """
        Initialization

        Parameters
        ----------
        :param domain: gpflowopt domain
        :param dim: dimension of the spd manifold

        Optional parameters
        -------------------
        :param min_eigenvalue: minimum eigenvalue of the samples
        :param max_eigenvalue: maximum eigenvalue of the samples
        """
        self.domain = domain
        self.dim = dim
        self.min_eigenvalue = min_eigenvalue
        self.max_eigenvalue = max_eigenvalue

    def generate_samples(self, nb_samples=1000):
        """
        Generate SPD samples.

        Parameters
        ----------
        :param nb_samples: number of samples

        Returns
        -------
        :return: SPD samples
        """
        # Generate eigenvalues between min_eigenvalue and max_eigenvalue
        d = self.min_eigenvalue * np.ones((nb_samples, self.dim)) \
            + (self.max_eigenvalue - self.min_eigenvalue) * np.random.rand(nb_samples, self.dim)

        # Generate an orthogonal matrix. Annoyingly qr decomp isn't
        # vectorized so need to use a for loop. Could be done using
        # svd but this is slower for bigger matrices (from pymanopt).
        point_mat = np.zeros((nb_samples, self.dim, self.dim))
        points = []
        for n in range(nb_samples):
            u, _ = np.linalg.qr(np.random.randn(self.dim, self.dim))
            point_mat[n] = np.dot(u, np.dot(np.diag(d[n]), u.T))
            points.append(symmetric_matrix_to_vector_mandel(point_mat[n]))

        points = np.array(points)

        return points

    def generate(self, objective, nb_anchor_points=5, nb_samples=1000):
        """
        Generate SPD anchor points. Samples are generated and the points with the best score are returned.

        Parameters
        ----------
        :param objective: objective function to minimize

        Optional parameters
        -------------------
        :param nb_anchor_points: number of anchor points to return
        :param nb_samples: number of samples where the objective function is evaluated

        Returns
        -------
        :return: SPD sample points with the lowest objective function value
        """
        # No checks are made for duplicate points here. We could try to include something to ensure that the points
        # are somehow separated from each other.
        points = self.generate_samples(nb_samples)

        scores = objective(points)[0][:, 0]

        anchor_points = points[np.argsort(scores)[:min(len(scores), nb_anchor_points)], :]

        return anchor_points
