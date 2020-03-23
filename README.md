# fbm_sim
Simulate short fractional Brownian motion (FBM) trajectories in pure Python

## Purpose

A simple tool to simulate many FBM trajectories, quickly. This repository uses the version of the FBM based on the Riemann-Liouville integral, rather than Mandelbrot's Weyl integral-based version. Numerically, simulation is achieved by filtering white Gaussian noise through the Cholesky decomposition of the FBM covariance matrix.

## Kinds of diffusion coefficient

`fbm_sim` uses one of two distinct parameterizations of the diffusion coefficient. See the notebook `definitions.ipynb` for a description of each kind.

## Example usage

```
	import fbm_sim

	# Create the simulator object
	fbm_obj = fbm_sim.FractionalBrownianMotion(
		128,         # simulate 128 steps of this FBM
		0.6,         # Hurst parameter 0.6
		D=2.0,       # Diffusion coefficient 2.0 m^2 s^-1
		dt=0.00548,  # 5.48 millisecond intervals
		D_type=2     # type of diffusion coefficient used
	)

	# Simulate 100000 instances of the FBM 
	trajs = fbm_obj(100000)

```
