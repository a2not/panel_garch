import numpy as np
import random
from scipy.linalg import sqrtm
from .unvech import unvech


def DGP(vTheta, vAlpha, vSigma, vLambda, iT, iN, iI, mX):
    # > "Python uses the Mersenne Twister as the core generator."
    # > https://docs.python.org/2/library/random.html
    random.seed(123 + 2 * iI)

    # Matrix with size (iT * iN), elements are generated with Uniform dist over [0, 1)
    if mX is None:
        mX = np.random.rand(iT, iN)

    assert vTheta.shape == (2,), "vTheta not size (2,)"
    phi, beta = vTheta
    assert vLambda.shape == (4,), "vLambda not size (4,)"
    gam, rho, varphi, eta = vLambda
    assert vSigma.shape == (iN * (iN + 1) // 2,), "vSigma not size (iN * (iN + 1) // 2,)"
    mSig = unvech(vSigma)
    assert mSig.shape == (iN, iN), "mSig not size (iN, iN)"

    # Definition (13)
    mC = np.full((iN, iN), rho) + (gam - rho) * np.identity(iN)
    mD = np.full((iN, iN), eta) + (varphi - eta) * np.identity(iN)

    # Equation (14)
    mK = mSig - np.dot(np.dot(mC, mSig), mC) - np.dot(np.dot(mD, mSig), mD)

    # Equation (15)
    vMy = (1 / (1 - phi)) * (0.5 + vAlpha)
    mSy = (1 / (1 - phi ** 2))*((1/12) + mSig)

    # Definition (4)
    mY = np.zeros((iT, iN))

    # Definition (16)
    # Column vector with dimension of iN, elements are generated with Norm(mu = 0, sd = 1)
    vU = np.dot(sqrtm(mSy), np.random.normal(0, 1, iN))
    mY[0] = vU + vMy
    mH = mSig

    for t in range(1, iT):
        # Definition (12)
        mH = mK + np.dot(np.dot(mC, np.outer(vU, vU)), mC) + \
            np.dot(np.dot(mD, mH), mD)

        # Definition (16)
        vU = np.dot(sqrtm(mH), np.random.normal(0, 1, iN))
        mY[t] = vAlpha + (phi * mY[t-1]) + (beta * mX[t]) + vU

    return mY, mX
