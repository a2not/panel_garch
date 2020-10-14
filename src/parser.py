import numpy as np
from scipy.linalg import sqrtm
from .unvech import unvech


def parser(vTheta, vAlpha, vSigma, vLambda, iT, iN, dataframe):

    assert len(vTheta) == 2, "vTheta not length 2"
    phi, beta = vTheta

    mSig = unvech(vSigma)

    assert len(vLambda) == 4, "vLambda not length 4"
    gam, rho, varphi, eta = vLambda

    # Definition (7)
    mC = np.full((iN, iN), rho) + (gam - rho) * np.identity(iN)
    mD = np.full((iN, iN), eta) + (varphi - eta) * np.identity(iN)

    # Equation (11)
    mK = mSig - np.dot(np.dot(mC, mSig), mC) - np.dot(np.dot(mD, mSig), mD)

    # Data Matrix with size (iT * iN)
    mX = dataframe
    vMy = (1 / (1 - phi)) * (0.5 + vAlpha)
    mSy = (1 / (1 - phi ** 2))*((1/12) + mSig)

    mY = np.zeros((iT, iN))

    # Row vector with dimension of iN, elements are generated with Norm(mu = 0, sd = 1)
    vU = np.dot(np.random.normal(0, 1, iN), sqrtm(mSy))
    mY[0] = vU + vMy.T
    mH = mSig

    for t in range(1, iT):
        # Equation (18)
        mH = mK + np.dot(np.dot(mC, np.outer(vU, vU)), mC) + \
            np.dot(np.dot(mD, mH), mD)

        # Equation (5)
        vU = np.dot(np.random.normal(0, 1, iN), sqrtm(mH))

        # Equation (4)
        mY[t] = vAlpha.T + (phi * mY[t-1]) + (beta * mX[t]) + vU

    return mY, mX
