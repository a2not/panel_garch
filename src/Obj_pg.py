import numpy as np
import math

# function obj = Obj_pg(vLambda,mU,mSig)


def Obj_pg(iN, iT, vLambda, mU, mSig):
    assert vLambda.shape == (4,), "vLambda not size (4,)"
    assert mU.shape == (iT, iN), "mU not size (iT, iN)"
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
    mH = mSig

    ll = 0
    for t in range(iT):
        # Equation (22)
        # print(np.linalg.det(mH))
        ll -= math.log(np.linalg.det(mH)) + \
            np.inner(mU[t], np.linalg.solve(mH, mU[t].T))

        # Equation (17)
        mH = mK + \
            np.dot(np.dot(mC, np.outer(mU[t], mU[t])), mC) + \
            np.dot(np.dot(mD, mH), mD)

    ll -= iN * iT * math.log(2 * math.pi)
    ll *= 0.5
    if abs(np.imag(ll)) > 0:
        return 1e+5 - ll

    # maximizing f(x) <=> minimizing -f(x)
    print("obj func runs successfully: f() = ", -ll)
    return -ll
