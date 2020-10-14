import numpy as np
import random
import scipy
from .unvech import unvech


def DGP(vTheta, vAlpha, vSigma, vLambda, iT, iN, iI):
    # > "Python uses the Mersenne Twister as the core generator."
    # > https://docs.python.org/2/library/random.html
    random.seed(123 + 2 * iI)

    phi, beta = vTheta
    gam, rho, varphi, eta = vLambda
    mSig = unvech(vSigma)

    # Definition (13)
    mC = np.full((iN, iN), rho) + \
        (gam - rho) * np.identity(iN)
    mD = np.full((iN, iN), eta) + \
        (varphi - eta) * np.identity(iN)

    # Equation (14)
    mK = mSig - np.dot(np.dot(mC, mSig), mC) - np.dot(np.dot(mD, mSig), mD)

    # Matrix with size (iT * iN), elements are generated with Uniform dist over [0, 1)
    mX = np.random.rand(iT, iN)

    # Equation (15)
    vMy = (1 / (1 - phi)) * (0.5 + vAlpha)
    mSy = (1 / (1 - phi ** 2))*((1/12) + mSig)

    # Definition (4)
    mY = np.zeros((iT, iN))

    # Definition (16)
    vU = np.dot(np.random.normal(0, 1, iN), scipy.linalg.sqrtm(mSy))
    mY[0] = vU + vMy.T
    mH = mSig

    for t in range(1, iT):
        # Definition (12)
        mH = mK + np.dot(np.dot(mC, np.outer(vU, vU)), mC) + \
            np.dot(np.dot(mD, mH), mD)

        # Definition (16)
        vU = np.dot(np.random.normal(0, 1, iN),
                    scipy.linalg.sqrtm(mH))
        mY[t] = vAlpha.T + (phi * mY[t-1]) + (beta * mX[t]) + vU

    return mY, mX
