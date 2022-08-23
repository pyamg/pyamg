from memory_profiler import profile
import numpy as np
import scipy.sparse
import scipy.io
import pyamg


@profile
def MGsetup():
    # scipy.io.savemat('A.mat', {'A': A})
    A = scipy.io.loadmat('A.mat')['A'].tocsr()
    _ = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
    _ = np.random.rand(A.shape[0])


if __name__ == '__main__':
    MGsetup()
