# GaBO

This repository contains the source code to perform Geometry-aware Bayesian Optimization (GaBO) on Riemannian manifolds.

### Important note: 
GaBOflow depends on GPflow and GPflowOpt. However, GPflowOpt has not been updated since a while and is not compatible with the latest version of GPflow.
Therefore, we created [GaBOtorch](https://github.com/NoemieJaquier/GaBOtorch), which is based on PyTorch and contains the source code of GaBO. 
New examples and updates will be available in [GaBOtorch](https://github.com/NoemieJaquier/GaBOtorch).

# Requirements
This code was tested with Python 3.5 and 3.6. It requires the following packages:
- Numpy
- Scipy
- Matplotlib
- Tensorflow
- GPflow 0.5
- GPflowOpt
- Pymanopt


# Installation 
To install GaBOflow, first clone the repository and install the related packages, as explained below.
```
pip install numpy scipy matplotlib pymanopt
```

To install GPflowOpt and its dependencies (e.g. tensorflow and gpflow) follow the instructions given in [GPflowOpt repository](https://github.com/GPflow/GPflowOpt).

Finally, from the GaBOflow folder, run
```
pip install -e .
```

# Examples
The following examples are available in GaBOflow:
### Kernels
| Sphere manifold      |           | 
|:------------- |:-------------| 
| sphere_kernels      | This example shows the use of different kernels for the hypershere manifold S^n , used for Gaussian process regression. | 
| sphere_gaussian_kernel_parameters      | This example shows the experimental selection of parameters for the Sphere Gaussian kernel.      |

| SPD manifold       |           | 
|:------------- |:-------------| 
| spd_kernels      | This example shows the use of different kernels for the SPD manifold, used for Gaussian process regression | 
| spd_gaussian_kernel_parameters      | This example shows the experimental selection of parameters for the SPD Affine-Invariant Gaussian kernel.  |


### BO on the sphere
| Benchmark examples      |           | 
|:------------- |:-------------| 
| bo_sphere_ackley_manifold      | This example shows the use of Geometry-aware Bayesian optimization (GaBO) on the sphere S2 to optimize the Ackley function. | 
| bo_sphere_ackley_eucl      | This example shows the use of Euclidean Bayesian optimization on the sphere S2 to optimize the Ackley function.  |

| Constrained benchmark examples      |           | 
|:------------- |:-------------| 
| bo_sphere_ackley_manifold_constrained      | This example shows the use of Geometry-aware Bayesian optimization (GaBO) on the sphere S2 to optimize the Ackley function. In this example, the search domain is bounded and represents a subspace of the manifold. | 
| bo_sphere_ackley_eucl_constrained      | This example shows the use of Euclidean Bayesian optimization on the sphere S2 to optimize the Ackley function.  In this example, the search domain is bounded and represents a subspace of the manifold. |


### BO on the SPD manifold
| Benchmark examples      |           | 
|:------------- |:-------------| 
| bo_spd_ackley_manifold      | This example shows the use of Geometry-aware Bayesian optimization (GaBO) on the SPD manifold S2_++ to optimize the Ackley function. | 
| bo_spd_ackley_chol      | This example shows the use of Cholesky Bayesian optimization on the SPD manifold S2_++ to optimize the Ackley function. An Euclidean BO is applied on the Cholesky decomposition of the SPD matrices.  | 
| bo_spd_ackley_eucl      | This example shows the use of Euclidean Bayesian optimization on the SPD manifold S2_++ to optimize the Ackley function. |

# References
If you found GaBOtorch useful, we would be grateful if you cite the following [reference](http://njaquier.ch/files/CoRL19_Jaquier_GaBO.pdf):

[1] N. Jaquier, L. Rozo, S. Calinon and M. Bürger (2019). Bayesian Optimization meets Riemannian Manifolds in Robot Learning. In Conference on Robot Learning (CoRL).
```
@inproceedings{Jaquier19GaBO,
	author="Jaquier, N and Rozo, L. and Calinon, S. and B\"urger, M.", 
	title="Bayesian Optimization meets Riemannian Manifolds in Robot Learning",
	booktitle="In Conference on Robot Learning (CoRL)",
	year="2019",
	month="October",
	address="Osaka, Japan",
	pages=""
}
```
```
