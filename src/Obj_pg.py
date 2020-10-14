import numpy as np
import math

# function obj = Obj_pg(vLambda,mU,mSig)


def Obj_pg(iN, iT, vLambda, mU, mSig):
    assert mU.shape == (iT, iN), "mU not size (iT, iN)"
    gam, rho, varphi, eta = vLambda

    # Definition (13)
    mC = np.full((iN, iN), rho) + \
        (gam - rho) * np.identity(iN)
    mD = np.full((iN, iN), eta) + \
        (varphi - eta) * np.identity(iN)

    # Definition (14)
    mK = mSig - np.dot(np.dot(mC, mSig), mC) - np.dot(np.dot(mD, mSig), mD)

    # H_0 := mSig
    mH = mSig

    # H_t is positive definite if K and H_0 are
    vL = np.linalg.eigvals(mK)
    if (abs(vL) > 1).sum() > 0:
        return 1e+16
    vL = np.linalg.eigvals(mH)
    if (abs(vL) > 1).sum() > 0:
        return 1e+16

    ll = 0
    for t in range(iT):
        # Equation (22)
        ll -= math.log(np.linalg.det(mH)) - \
            np.inner(mU[t], np.linalg.solve(mH, mU[t].T))

        # Equation (17)
        mH = mK + \
            np.dot(np.dot(mC, np.outer(mU[t], mU[t])), mC) + \
            np.dot(np.dot(mD, mH), mD)

        # check if H_t is not positive definite
        vLam = np.linalg.eigvals(mH)
        if min(vLam) <= 0:
            return 1e+16

        # check if det(mH) == 0 (linearly dependent)
        if np.linalg.det(mH) < 1e-12:
            # the next computation of log(det(mH)) will cause ValueError
            # since log(0) is undefined
            return 1e+16

    ll -= iN * iT * math.log(2 * math.pi)
    ll *= 0.5
    if abs(np.imag(ll)) > 0:
        return 1e+16

    # maximizing f(x) <=> minimizing -f(x)
    print("obj func runs successfully")
    return -ll
