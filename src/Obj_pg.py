import numpy as np
import math

# function obj = Obj_pg(vLambda,mU,mSig)


def Obj_pg(vLambda, mU, mSig):
    iT, iN = mU.shape
    gam, rho, varphi, eta = vLambda

    mC = np.full((iN, iN), rho) + (gam - rho) * np.identity(iN)
    mD = np.full((iN, iN), eta) + (varphi - eta) * np.identity(iN)

    mK = mSig - np.dot(np.dot(mC, mSig), mC) - np.dot(np.dot(mD, mSig), mD)
    mH = mSig
    mCs = np.kron(mC, mC)
    mDs = np.kron(mD, mD)

    _, mL = np.linalg.eig(mCs + mDs)
    vL1 = np.diag(mL)
    _, mL = np.linalg.eig(mK)
    vL2 = np.diag(mL)

    iL = (abs(vL1) > 1).sum() + (abs(vL2) > 1).sum()
    if iL > 0:
        ll = -1e+16
    else:
        ll = -0.5 * math.log(np.linalg.det(mH)) - 0.5 * \
            np.inner(mU[0], np.linalg.solve(mH, mU[0].T))
        for t in range(1, iT):
            mH = mK + \
                np.dot(
                    np.dot(mC, np.outer(mU[t-1], mU[t-1])), mC) + np.dot(np.dot(mD, mH), mD)
            _, mLam = np.linalg.eig(mH)
            if min(np.diag(mLam)) < 0:
                ll = ll - 1e+16
                break

            ll = ll - 0.5 * math.log(np.linalg.det(mH)) - 0.5 * \
                np.inner(mU[t], np.linalg.solve(mH, mU[t].T))

    iC = -0.5 * iN * math.log(2 * math.pi)
    ll += iT * iC
    obj = -ll / iT
    if abs(np.imag(obj)) > 0:
        obj = 1e+16

    return obj
