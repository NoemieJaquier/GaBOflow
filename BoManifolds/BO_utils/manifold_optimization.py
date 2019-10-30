import numpy as np
import scipy.optimize as sc_opt

import gpflowopt
import tensorflow as tf

import pymanopt as pyman
import pymanopt.solvers as pyman_solv

from BoManifolds.BO_utils.manifold_conjugate_gradient import ConjugateGradientWithBetaLimit, ConjugateGradientRobust
from BoManifolds.BO_utils.manifold_bound_constrained_conjugate_gradient import BoundConstrainedConjugateGradient

'''
Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
'''


class ManifoldOptimizer(gpflowopt.optim.Optimizer):
    """
    Instances of this class represent an optimizer on the manifold (using pymanopt optimizers).

    Attributes
    ----------
    self.domain:
    self.manifold:
    self.dimension:
    self.matrix_manifold_dimension:
    self.matrix_to_vector_transform:
    self.vector_to_matrix_transform:
    self.matrix_to_vector_transform_tf:
    self.vector_to_matrix_transform_tf:
    self.linesearch:
    self.solver_type:

    Methods
    -------
    _optimize(objective):

    Static methods
    --------------
    """
    def __init__(self, domain, manifold, manifold_dim=None, matrix_manifold_dim=None, matrix_to_vector_transform=None,
                 vector_to_matrix_transform=None, matrix_to_vector_transform_tf=None,
                 vector_to_matrix_transform_tf=None, solver_type='ConjugateGradient', linesearch_obj=None,
                 logverbosity=1, **kwargs):
        """
        Initialization

        Parameters
        ----------
        :param domain: domain for the optimization, (convenient to use a gpflowopt domain)
        :param manifold: manifold where the optimization is carried (pymanopt manifold)

        Optional parameters
        -------------------
        :param manifold_dim: manifold dimension. None if the vector dimension of the parameter to optimize
                            corresponds to the default manifold.dim
        :param matrix_manifold_dim: matrix-manifold dimension. None if the parameter belong to a vector-valued
                                    manifold (e.g. sphere)
        :param matrix_to_vector_transform: matrix to vector transformation function
            (not None for matrix manifolds only)
        :param vector_to_matrix_transform: vector to matrix transformation function
            (not None for matrix manifolds only)
        :param matrix_to_vector_transform_tf: tensorflow matrix to vector transformation function
            (not None for matrix manifolds only)
        :param vector_to_matrix_transform_tf: tensorflow vector to matrix transformation function
            (not None for matrix manifolds only)
        :param solver_type: type of solver
            Options are TrustRegions, SteepestDescent, ConjugateGradient, ConjugateGradientWithBetaLim,
            BoundConstrainedConjugateGradient, NelderMead and ParticleSwarm
        :param linesearch_obj: linesearch object
            Options are LineSearchAdaptive and LineSearchBackTracking
        :param logverbosity: characterise the output format (MUST be >=1)
        :param kwargs: parameters for the linesearch_obj and for the solver
        """
        super(ManifoldOptimizer, self).__init__(domain)

        # Domain
        self.domain = domain

        # Initialize the manifold
        self.manifold = manifold

        # TODO check if there is a more general way to do this.
        # This should be the number of variables that the optimizer has to find
        if manifold_dim is None:
            self.dimension = self.manifold.dim
        else:
            self.dimension = manifold_dim

        # For matrix-manifolds
        self.matrix_manifold_dimension = matrix_manifold_dim

        # For matrix-manifolds, if the objective function expects a vector as input
        self.matrix_to_vector_transform = matrix_to_vector_transform
        self.vector_to_matrix_transform = vector_to_matrix_transform
        self.matrix_to_vector_transform_tf = matrix_to_vector_transform_tf
        self.vector_to_matrix_transform_tf = vector_to_matrix_transform_tf

        # Necessary to output the good result format.
        if logverbosity < 1:
            logverbosity = 1

        # Initialize linesearch object is any is given (used for steepest descent and conjugate gradient solvers)
        if linesearch_obj is 'LineSearchAdaptive':
            if 'contraction_factor' in kwargs:
                contraction_factor = kwargs['contraction_factor']
            else:
                contraction_factor = .5
            if 'suff_decr' in kwargs:
                suff_decr = kwargs['suff_decr']
            else:
                suff_decr = .5
            if 'maxiter' in kwargs:
                maxiter = kwargs['maxiter']
            else:
                maxiter = 10
            if 'initial_stepsize' in kwargs:
                initial_stepsize = kwargs['initial_stepsize']
            else:
                initial_stepsize = 1
            self.linesearch = pyman_solv.linesearch.LineSearchAdaptive(contraction_factor=contraction_factor,
                                                                       suff_decr=suff_decr, maxiter=maxiter,
                                                                       initial_stepsize=initial_stepsize,
                                                                       logverbosity=logverbosity)

        elif linesearch_obj is 'LineSearchBackTracking':
            if 'contraction_factor' in kwargs:
                contraction_factor = kwargs['contraction_factor']
            else:
                contraction_factor = .5
            if 'optimism' in kwargs:
                optimism = kwargs['optimism']
            else:
                optimism = 2
            if 'suff_decr' in kwargs:
                suff_decr = kwargs['suff_decr']
            else:
                suff_decr = 1e-4
            if 'maxiter' in kwargs:
                maxiter = kwargs['maxiter']
            else:
                maxiter = 25
            if 'initial_stepsize' in kwargs:
                initial_stepsize = kwargs['initial_stepsize']
            else:
                initial_stepsize = 1
            self.linesearch = pyman_solv.linesearch.LineSearchBackTracking(contraction_factor=contraction_factor,
                                                                           optimism=optimism, suff_decr=suff_decr,
                                                                           maxiter=maxiter,
                                                                           initial_stepsize=initial_stepsize,
                                                                           logverbosity=logverbosity)
        else:
            self.linesearch = None

        # Initialize solver
        self.solver_type = solver_type

        if 'mingradnorm' in kwargs:
            mingradnorm = kwargs['mingradnorm']
        else:
            mingradnorm = 1e-6

        if solver_type is 'ConjugateGradient':
            if 'beta_type' in kwargs:
                beta_type = kwargs['beta_type']
            else:
                beta_type = 2
            if 'orth_value' in kwargs:
                orth_value = kwargs['orth_value']
            else:
                orth_value = np.inf
            self.solver = pyman_solv.ConjugateGradient(linesearch=self.linesearch, beta_type=beta_type,
                                                       orth_value=orth_value, logverbosity=logverbosity,
                                                       mingradnorm=mingradnorm)

        elif solver_type is 'ConjugateGradientWithBetaLim':
            if 'beta_type' in kwargs:
                beta_type = kwargs['beta_type']
            else:
                beta_type = 2
            if 'orth_value' in kwargs:
                orth_value = kwargs['orth_value']
            else:
                orth_value = np.inf
            self.solver = ConjugateGradientWithBetaLimit(linesearch=self.linesearch, beta_type=beta_type,
                                                         orth_value=orth_value, logverbosity=logverbosity,
                                                         mingradnorm=mingradnorm)

        elif solver_type is 'ConjugateGradientRobust':
            if 'beta_type' in kwargs:
                beta_type = kwargs['beta_type']
            else:
                beta_type = 2
            if 'orth_value' in kwargs:
                orth_value = kwargs['orth_value']
            else:
                orth_value = np.inf
            self.solver = ConjugateGradientRobust(linesearch=self.linesearch, beta_type=beta_type,
                                            orth_value=orth_value, logverbosity=logverbosity,
                                            mingradnorm=mingradnorm)

        elif solver_type is 'BoundConstrainedConjugateGradient':
            if 'beta_type' in kwargs:
                beta_type = kwargs['beta_type']
            else:
                beta_type = 2
            if 'orth_value' in kwargs:
                orth_value = kwargs['orth_value']
            else:
                orth_value = np.inf
            self.solver = BoundConstrainedConjugateGradient(self.domain, linesearch=self.linesearch, beta_type=beta_type,
                                                            orth_value=orth_value, logverbosity=logverbosity,
                                                            mingradnorm=mingradnorm)

        else:
            raise ValueError('Solver options are ConjugateGradient, ConjugateGradientRobust'
                             'ConjugateGradientWithBetaLim and BoundConstrainedConjugateGradient.')

    def _optimize(self, objective):
        """
        Minimize the objective function

        Parameters
        ----------
        :param objective: objective function to minimize

        Returns
        -------
        :return: optimal parameter found by the optimization (scipy format)
        """
        # Initial value
        initial = self.get_initial()[0]

        if self.vector_to_matrix_transform is not None:
            initial = self.vector_to_matrix_transform(initial)

        if self.solver_type is 'NelderMead' or self.solver_type is 'ParticleSwarm':
            initial = None

        # Create tensorflow variable
        if self.matrix_manifold_dimension is None:
            x_tf = tf.Variable(tf.zeros(self.dimension, dtype=tf.float64))
        else:
            x_tf = tf.Variable(tf.zeros([self.matrix_manifold_dimension, self.matrix_manifold_dimension], dtype=tf.float64))

        # Cost function for pymanopt
        def objective_fct(x):
            if self.matrix_to_vector_transform_tf is not None:
                # Reshape x from matrix to vector form to compute the objective function (tensorflow format)
                x = self.matrix_to_vector_transform_tf(x, self.matrix_manifold_dimension)
            return objective(x)[0]

        # Transform the cost function to tensorflow function
        cost = tf.py_function(objective_fct, [x_tf], tf.float64)

        # Gradient function for pymanopt
        def objective_grad(x):
            if self.matrix_to_vector_transform is not None:
                # Reshape x from matrix to vector form to compute the gradient
                x = self.matrix_to_vector_transform(x)

            # Compute the gradient
            grad = np.array(objective(x)[1])[0]

            if self.vector_to_matrix_transform is not None:
                # Reshape the gradient in matrix form for the optimization on the manifold
                grad = self.vector_to_matrix_transform(grad)
            return grad

        # Define pymanopt problem
        problem = pyman.Problem(manifold=self.manifold, cost=cost, egrad=objective_grad, arg=x_tf, verbosity=2)

        # Optimize the parameters of the problem
        opt_x, opt_log = self.solver.solve(problem, x=initial)

        if self.matrix_to_vector_transform is not None:
            # Reshape the optimum from matrix to vector form
            opt_x = self.matrix_to_vector_transform(opt_x)

        # Format the result to fit with GPflowOpt
        result = sc_opt.OptimizeResult(x=opt_x, fun=opt_log['final_values']['f(x)'], nit=opt_log['final_values']['iterations'], message=opt_log['stoppingreason'], success=True)

        return result


class MCManifoldOptimizer(gpflowopt.optim.Optimizer):
    """
    Instances of this class are optimizers that optimize the function by evaluating a number of samples on the manifold
    and return the best.

    Attributes
    ----------
    self.manifold:
    self._nb_samples:
    self.matrix_to_vector_transform:

    Methods
    -------
    _get_eval_points():
    _optimize(objective):

    Static methods
    --------------
    """
    def __init__(self, domain, manifold, nb_samples, matrix_to_vector_transform=None):
        """
        Initialisation.

        Parameters
        ----------
        :param domain: domain for the optimization, (convenient to use a gpflowopt domain)
        :param manifold: manifold where the optimization is carried (pymanopt manifold)
        :param nb_samples: number of samples considered in the optimization

        Optional parameters
        -------------------
        :param matrix_to_vector_transform: matrix to vector transformation function
            (not None for matrix manifolds only)
        """
        super(MCManifoldOptimizer, self).__init__(domain, exclude_gradient=True)
        self.manifold = manifold
        self._nb_samples = nb_samples

        # Clear the initial data points
        self.set_initial(np.empty((0, self.domain.size)))

        # For matrix-manifolds, if the objective function expects a vector as input
        self.matrix_to_vector_transform = matrix_to_vector_transform

    def _get_eval_points(self):
        """
        Generate random points on the manifold.

        Returns
        -------
        :return: random points on the manifold
        """
        points = [self.manifold.rand() for i in range(self._nb_samples)]
        return np.array(points)

    def _optimize(self, objective):
        """
        Select the random point with the minimum objective function value.

        Parameters
        ----------
        :param objective: objective function to minimize

        Returns
        -------
        :return: optimal parameter found by the optimizer (scipy format)
        """
        points = self._get_eval_points()

        if self.matrix_to_vector_transform is not None:
            # Transform the sampled matrix points in vectors
            points = np.array([self.matrix_to_vector_transform(points[i]) for i in range(self._nb_samples)])

        evaluations = objective(points)
        idx_best = np.argmin(evaluations, axis=0)

        return sc_opt.OptimizeResult(x=points[idx_best, :], success=True, fun=evaluations[idx_best, :],
                                     nfev=points.shape[0], message="OK")


class ManifoldAnchorPointsGenerator:
    """
    Instances of this class are generators of anchor points on a manifold.

    Attributes
    ----------
    self.manifold:
    self.matrix_to_vector_transform:

    Methods
    -------
    generate(objective, nb_anchor_points, nb_samples):

    Static methods
    --------------
    """
    def __init__(self, manifold, matrix_to_vector_transform=None):
        """
        Initialization

        Parameters
        ----------
        :param manifold: manifold (pymanopt class)

        Optional parameters
        -------------------
        :param matrix_to_vector_transform: transformation from matrix to vector (if the manifold is a matrix manifold)
        """
        self.manifold = manifold

        # For matrix-manifolds, if the objective function expects a vector as input
        self.matrix_to_vector_transform = matrix_to_vector_transform

    def generate(self, objective, nb_anchor_points=10, nb_samples=1000):
        """
        Generate anchor points on the manifold

        Parameters
        ----------
        :param objective: objective function to minimize (takes a vector as input)
        :param nb_anchor_points: number of anchor points to return

        Optional parameters
        -------------------
        :param nb_samples: number of samples where the objective function is evaluated

        Returns
        -------
        :return: sample points on the manifold with the lowest objective function value (vector form)
        """
        # No checks are made for duplicate points here. We could try to include something to ensure that the points
        # are somehow separated from each other.
        points = np.array([self.manifold.rand() for i in range(nb_samples)])

        if self.matrix_to_vector_transform is not None:
            # Transform the sampled matrix points in vectors
            points = np.array([self.matrix_to_vector_transform(points[i]) for i in range(nb_samples)])

        scores = objective(points)[0][:, 0]

        anchor_points = points[np.argsort(scores)[:min(len(scores), nb_anchor_points)], :]

        return anchor_points
