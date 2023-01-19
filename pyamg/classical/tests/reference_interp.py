"""Reference AMG interpolation."""

import numpy as np

def reference_classical_interpolation_9(A, S, splitting):
    """Implement (9) with dense arrays.

    splitting[i] == 0 F-point
    splitting[i] == 1 C-point
    S[i, j] != 0      --->  j strongly influences i, i strongly depends on j

    wij = - 1
          ------------(aij + sum2)
          aii + sum1

          sum1 = sum_{k in Nwi or Fsistar} aik

          sum2 = sum_{k in Fsi not in Fsistar} aik * aakj / sum_{m in Csi} aakm

          aaij = 0 if sign(aij) == sign(aii)

    """
    F = np.where(splitting == 0)[0]
    C = np.where(splitting == 1)[0]
    P = np.zeros((A.shape[0], A.shape[0]))

    def aa(i, j):
        if np.sign(A[i, j]) == np.sign(A[i, i]):
            return 0
        return A[i, j]

    for i in F:
        # define some sets
        Ni = np.where(A[i, :])[0]
        Si = np.where((S[i, :] != 0))[0]
        Fsi = np.intersect1d(F, Si)
        Csi = np.intersect1d(C, Si)
        Nwi = np.setdiff1d(Ni, np.intersect1d(Fsi, Csi))
        Fsistar = []
        for k in Fsi:                                 # points k in F
            Csk = np.intersect1d(C, np.where((S[k, :] != 0))[0])
            if len(np.intersect1d(Csi, Csk)) == 0:    # that "do not have a common C-point"
                Fsistar.append(k)

        Fsistar = np.array(Fsistar, dtype=int)

        for j in Csi:
            sum1 = np.sum(A[i, np.union1d(Nwi, Fsistar)])

            sum2 = np.sum([A[i, k] * aa(k, j) / np.sum([aa(k, m) for m in Csi])
                           for k in np.setdiff1d(Fsi, Fsistar)])

            P[i, j] = - 1 / (A[i, i] + sum1) * (A[i, j] * sum2)

    return P[:, C]


def reference_direct_interpolation_6(A, S, splitting):
    """Implement (9) with dense arrays.

    splitting[i] == 0 F-point
    splitting[i] == 1 C-point
    S[i, j] != 0      --->  j strongly influences i, i strongly depends on j

    wij = - aij sum_{k in Ni} aik
            ---------------------
            aii sum_{k in Csi} Aik
    """
    F = np.where(splitting == 0)[0]
    C = np.where(splitting == 1)[0]
    P = np.zeros((A.shape[0], A.shape[0]))

    for i in F:
        # define some sets
        Ni = np.where(A[i, :] != 0)[0]
        Si = np.where(S[i, :] != 0)[0]
        Csi = np.intersect1d(C, Si)

        for j in Csi:
            P[i, j]  = - A[i, j] * np.sum(A[i, Ni])
            P[i, j] /=   A[i, i] * np.sum(A[i, Csi])

    for i in C:
        P[i, i] = 1.0

    return P[:, C]
