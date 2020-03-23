"""
fbm.py -- simulation methods for fractional Brownian
motions in particular and Gaussian processes in 
general

"""
import numpy as np 
import matplotlib.pyplot as plt 

class MultivariateNormal(object):
    """
    A multivariate normal random vector X.

    init
    ----
        u :  1D ndarray of shape (N,), the mean of
            the random vector

        C :  2D ndarray of shape (N, N), the covariance
            matrix of the random vector

    """
    def __init__(self, u, C):
        u = np.asarray(u)
        C = np.asarray(C)

        assert len(u.shape) == 1
        assert len(C.shape) == 2
        assert u.shape[0] == C.shape[0]
        assert C.shape[0] == C.shape[1]

        self.u = u 
        self.C = C 
        self.N = u.shape[0]

    @property
    def cholesky(self):
        """Cholesky decomposition of covariance matrix"""
        if not hasattr(self, 'E'):
            self.E = np.linalg.cholesky(self.C)
        return self.E 

    @property 
    def inv_covariance(self):
        """Inverse of covariance matrix"""
        if not hasattr(self, 'C_inv'):
            self.C_inv = np.linalg.inv(self.C)
        return self.C_inv 

    @property
    def det_covariance(self):
        """Determinant of the covariance matrix"""
        if not hasattr(self, 'C_dev'):
            self.C_det = np.linalg.det(self.C)
        return self.C_det 
    
    def __call__(self, n=1):
        """
        Simulate *n* instances of the random
        vector X.

        args
        ----
            n :  int

        returns
        -------
            2D ndarray of shape (self.N, n),
                the simulated vectors

        """
        E = self.cholesky
        z = np.random.normal(size=(self.N, n))
        return ((E @ z).T + self.u).T 

    def f(self, x):
        """
        Return the probability density function (PDF) of
        the random vector at the point x.

        args
        ----
            x :  1D ndarray of shape (self.N), a point, or
                2D ndarray of shape (m, self.N), a set of
                points

        returns
        -------
            float or 1D ndarray of shape (m,), the PDF

        """
        norm = np.power(2.0 * np.pi, self.N / 2.0) * \
            np.power(self.det_covariance, 0.5)
        x_shift = x - self.u 

        if len(x.shape) == 1:
            return np.exp(-0.5 * x_shift @ self.inv_covariance \
                 @ x_shift) / norm 
        else:
            return np.exp(-0.5 * (x_shift * (self.inv_covariance @ \
                x_shift.T).T).sum(1)) / norm

    def check_compatible(self, MN):
        """
        Check that another MultivariateNormal is compatible
        for addition with the self random vector X.
        """
        assert self.N == MN.N 

    def add(self, MN):
        """
        Add another multivariate random vector to the 
        random vector X.

        args
        ----
            MN :  Multivariate Normal, the variable to
                be added

        returns
        -------
            MultivariateNormal, the distribution of the
                sum

        """
        self.check_compatible(MN)
        return MultivariateNormal(
            self.u + MN.u,
            self.C + MN.C 
        )

    def linear_transformation(self, A):
        """
        Return the distribution that describes
        the random variable A @ X, where A is a
        matrix operator.

        args
        ----
            A :  2D ndarray of shape (m, self.N)

        returns
        -------
            MultivariateNormal, the distribution
                of the transformed random vector

        """
        assert A.shape[0] == self.N 
        return MultivariateNormal(A.T @ self.u, A.T @ self.C @ A)

    def marginalize(self, indices):
        """
        Marginalize the distribution on some subset of 
        the components of the random vector X.

        args
        ----
            indices :  list of int, the indices of the
                vectors on X on which to marginalize.
                For instance, if indices == [0], then
                we'll marginalize on X[0].

        returns
        -------
            MultivariateNormal, the marginal distribution

        """ 
        if isinstance(indices, int):
            indices = [indices]
        out = [i for i in range(self.N) if i not in indices]
        m = len(out)
        P = np.zeros((self.N, m))
        P[tuple(out), tuple(range(m))] = 1
        return self.linear_transformation(P)

    def condition(self, indices, values):
        """Return the MultivariateNormal() given by
        conditioning on some subset of the elements
        of the random vector X.

        args
        ----
            indices :  list of int, the indices of the
                components of X on which to condition
            values :  list of float, the corresponding
                values of each component

        returns
        -------
            MultivariateNormal, the conditional
                distribution

        """
        if isinstance(indices, int):
            indices = [indices]
        indices = tuple(indices)
        keep = tuple([i for i in range(self.N) if i not in indices])

        # Partition the covariance matrix according to 
        # conditioned and unconditioned variables
        C11 = self.C[keep, :][:, keep]
        C22 = self.C[indices, :][:, indices]
        C12 = self.C[keep, :][:, indices]
        C21 = self.C[indices, :][:, keep]

        # Do the same for the offset vector
        u1 = self.u[list(keep)]
        u2 = self.u[list(indices)]

        # Compute the offset and covariance matrix
        # for the conditional distribution
        A = C12 @ np.linalg.inv(C22)
        sub_C = C11 - A @ C21 
        sub_u = u1 + A @ (np.asarray(values) - u2)

        return MultivariateNormal(sub_u, sub_C)

class BrownianMotion(MultivariateNormal):
    """
    A regular Brownian motion.

    initialization
    --------------
        N :  int, the number of steps in this Brownian
            motion
        D :  float, diffusion coefficient
        dt :  float, time interval for each step

    """
    def __init__(self, N, D=1.0, dt=0.5):
        self.N = N 
        self.u = np.zeros(N, dtype='float64')
        self.D = D 
        self.dt = dt 
        self.sig2 = 2 * D * dt 
        self.C = self.sig2 * np.minimum(*np.indices((N, N))) + 1

    def get_time(self):
        return np.arange(self.N) * self.dt 

class FractionalBrownianMotion(MultivariateNormal):
    """
    A fractional Brownian motion under the Riemann-
    Liouville fractional integral (rather than 
    Mandelbrot's Weyl integral representation).

    init
    ----
        N :  int, the number of steps in the fractional
                Brownian motion

        hurst :  float between 0 and 1, the Hurst
                parameter

        D :  float, diffusion coefficient, scalar
                for the step size

        dt :  float, the time interval for each step

        D_kind : int, the type of diffusion coefficient
            (either 1 or 2). If *1*, D has units of 
            space^2 time^{-2H}. If *2*, D has units of 
            space^2 time^-1. See `definitions.ipynb` for
            more details.

    """
    def __init__(self, N, hurst, D=1.0, dt=1.0, D_kind=1):
        self.N = N 
        self.hurst = hurst 
        self.D = D 
        self.dt = dt 
        self.D_kind = D_kind
        assert D_kind in [1, 2]

        # Build the FBM covariance matrix
        if D_kind == 1:
            T, S = (np.indices((N, N))+1) * dt 
            self.C = D * (np.power(T, 2*hurst) + \
                np.power(S, 2*hurst) - \
                np.power(np.abs(T-S), 2*hurst))
        elif D_kind == 2:
            T, S = (np.indices((N, N))+1) * dt * D
            self.C = (np.power(T, 2*hurst) + \
                np.power(S, 2*hurst) - \
                np.power(np.abs(T-S), 2*hurst))

        # Set zero mean
        self.u = np.zeros(N, dtype='float64')

    def get_time(self):
        """
        Return the set of times on which the 
        Brownian motion is defined.

        """
        return np.arange(self.N) * self.dt 

# Convenience functions
def align_to_zero(S):
    return S - S[0,:]

def show_sim(S):
    fig, ax = plt.subplots(figsize = (4, 2))
    t = np.arange(S.shape[0])
    for i in range(S.shape[1]):
        ax.plot(t, S[:,i], color='k')
    plt.show(); plt.close()

def kymograph(S, n_bins=200, vmax_mod=2.0, plot=True):
    S = align_to_zero(S)
    T, n_trajs = S.shape
    abs_max = max([abs(S.max()), abs(S.min())])
    bin_edges = np.linspace(-abs_max, abs_max, n_bins+1)

    grid = np.zeros((n_bins, T), dtype='float64')
    for t in range(T):
        counts, edges = np.histogram(S[t,:], bins=bin_edges)
        grid[:,t] = counts.copy()

    if plot:
        plt.imshow(
            grid,
            cmap='magma',
            vmax = grid[1:,:].mean() + vmax_mod*grid[1:,:].std(),
        )
        plt.show(); plt.close()

    return grid 

def kymograph_3d(S, n_bins=200, vmax_mod=4.0):
    grid = kymograph(S, n_bins=n_bins, vmax_mod=vmax_mod,
        plot=False)

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm 

    T, M = np.indices(grid.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(M, T, grid, linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Space')
    ax.set_zlabel('Trajectory density')
    ax.set_zlim((0, grid[1:,:].mean()+grid[1:,:].std()*vmax_mod))
    plt.show(); plt.close()





