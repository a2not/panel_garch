import numpy as np

def spectralRadiusOfKroneckers(iN, vLambda):
    assert vLambda.shape == (4, 1), "vLambda not size (4, 1)"
    gam, rho, varphi, eta = vLambda

    # Definition (13)
    mC = np.full((iN, iN), rho) + \
        (gam - rho) * np.identity(iN)
    mD = np.full((iN, iN), eta) + \
        (varphi - eta) * np.identity(iN)

    # Assumption 5
    mM = np.kron(mC, mC) + np.kron(mD, mD)
    vL = np.linalg.eigvals(mM)

    radius = max(abs(vL))

    return radius
