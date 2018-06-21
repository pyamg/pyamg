import memory_profiler

@profile
def MGsetup(nx):
    import numpy as np
    import scipy as sp
    import scipy.sparse
    import pyamg
    import scipy.io
    # scipy.io.savemat('A.mat', {'A': A})
    A = scipy.io.loadmat('A.mat')['A'].tocsr()
    ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
    b = np.random.rand(A.shape[0])


if __name__ == '__main__':
    MGsetup(100)
