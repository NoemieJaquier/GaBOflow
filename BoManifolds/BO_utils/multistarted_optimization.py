import numpy as np
import gpflowopt

'''
Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
'''


class MultistartedOptimizer(gpflowopt.optim.Optimizer):
    """
    Instances of this class represent a multistarted optimizer.
    During the optimization, several optimizations are carried with different anchor points as starting points.
    The best optimum is returned.

    Attributes
    ----------
    self.optimizer:
    self.anchor_points_generator:
    self.nb_anchor_points:
    self.nb_samples

    Methods
    -------
    _optimize(objective):

    Static methods
    --------------
    """
    def __init__(self, domain, optimizer, anchor_points_generator, nb_anchor_points=5, nb_samples=100):
        """
        Initialisation.

        Parameters
        ----------
        :param domain: optimization domain
        :param optimizer: optimizer instance
        :param anchor_points_generator: instance generating starting points
        :param nb_anchor_points: number of starting points
        :param nb_samples: number of samples considered for generating starting point
        """
        super(MultistartedOptimizer, self).__init__(domain)

        self.optimizer = optimizer
        self.anchor_points_generator = anchor_points_generator
        self.nb_anchor_points = nb_anchor_points
        self.nb_samples = nb_samples

    def _optimize(self, objective):
        """
        Optimize the objective function

        Parameters
        ----------
        :param objective: objective function to be minimized

        Returns
        -------
        :return: optimal parameter found by the optimizer
        """
        # Generate nanchors anchor points
        anchor_points = self.anchor_points_generator.generate(objective, self.nb_anchor_points, self.nb_samples)

        # Optimize with each anchor point as starting point
        results = []
        for point in anchor_points:
            self.optimizer.set_initial(point)
            try:  #TODO: this is tempory and should be removed. It allows the continuation of the programm is (Scipy) optimizer bugs.
                results.append(self.optimizer.optimize(objective))
            except Exception as e:
                print(e)

        # Take best result
        idx_best = np.argmin(np.array([result.fun for result in results]))

        return results[idx_best]


class AnchorPointsGenerator:
    """
    Instances of this class are generators of anchor points in a domain.

    Attributes
    ----------
    self.domain:

    Methods
    -------
    generate(objective, nb_anchor_points, nb_samples):

    Static methods
    --------------
    """
    def __init__(self, domain):
        """
        Initialisation

        Parameters
        ----------
        :param domain: gpflowopt domain
        """
        self.domain = domain

    def generate(self, objective, nb_anchor_points=5, nb_samples=1000):
        """
        Generate anchor points in the domain. Samples are generated and the points with the best score are returned.

        Parameters
        ----------
        :param objective: objective function to minimize
        :param nb_anchor_points: number of anchor points to return
        :param nb_samples: number of samples where the objective function is evaluated

        Returns
        -------
        :return: sample points with the lowest objective function value
        """
        # No checks are made for duplicate points here. We could try to include something to ensure that the points
        # are somehow separated from each other.
        points = gpflowopt.design.RandomDesign(nb_samples, self.domain).generate()
        scores = objective(points)[0][:, 0]

        anchor_points = points[np.argsort(scores)[:min(len(scores), nb_anchor_points)], :]

        return anchor_points
