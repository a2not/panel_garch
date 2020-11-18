import numpy as np

def initialPositiveDefiniteness(iN, vLambda, mSig):
    assert vLambda.shape == (4,), "vLambda not size (4,)"
    assert mSig.shape == (iN, iN), "mSig not size (iN, iN)"
    gam, rho, varphi, eta = vLambda

    # Definition (13)
    mC = np.full((iN, iN), rho) + \
        (gam - rho) * np.identity(iN)
    mD = np.full((iN, iN), eta) + \
        (varphi - eta) * np.identity(iN)

    # Definition (14)
    mK = mSig - np.dot(np.dot(mC, mSig), mC) - np.dot(np.dot(mD, mSig), mD)

    # H_0 := mSig
    # H_t is positive definite if K and H_0 are
    return abs(np.linalg.det(mK)) * abs(np.linalg.det(mSig))
