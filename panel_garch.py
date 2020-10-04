import numpy as np
import pandas
import time
import math
import random
import scipy.optimize


class panel_garch:

    def __init__(self, dataframe, initial_vTheta=None, initial_vAlpha=None, initial_vS=None, initial_vSigma=None, initial_vLambda=None, iR=None):
        self.df = dataframe
        self.iT, self.iN = self.df.shape

        self.eps = 1e-12

        if initial_vTheta == None:
            self.vTheta = np.array([[0.6, 1]]).T
        else:
            self.vTheta = initial_vTheta
        assert len(self.vTheta) == 2, "vTheta not length 2"

        if initial_vAlpha == None:
            self.vAlpha = np.array(
                [[(1 / self.iN) * num for num in range(1, self.iN + 1)]]).T
        else:
            self.vAlpha = initial_vAlpha

        if initial_vS == None:
            self.vS = np.array([[(-0.4)**i for i in range(self.iN)]]).T
        else:
            self.vS = initial_vS

        if initial_vSigma == None:
            self.vSigma = np.zeros((self.iN * (self.iN + 1) // 2, 1))
            for i in range(self.iN):
                # reducing scope of the variable
                rep, subtract, writer = i, self.iN, 0

                while rep > 0:
                    writer += subtract
                    subtract -= 1
                    rep -= 1

                for j in range(self.iN - i):
                    self.vSigma[writer + j] = self.vS[j, 0]
        else:
            self.vSigma = initial_vSigma

        if initial_vLambda == None:
            self.vLambda = np.array([[0.4, -0.1, 0.6, -0.2]]).T
        else:
            self.vLambda = initial_vLambda
        assert len(self.vLambda) == 4, "vLambda not length 4"

        self.vPsi = np.concatenate(
            (self.vTheta, self.vAlpha, self.vSigma, self.vLambda), axis=0)
        self.iP = len(self.vPsi)

        self.iC = 1 - 1e-6
        self.lb = np.full((4, 1), -self.iC)
        self.ub = np.full((4, 1), self.iC)

        if iR == None:
            self.iR = 20
        else:
            self.iR = iR

    def vech(self, mX):
        # take vech operation on d*d symmetric matrix and return (d*(d+1) / 2) dimension column vecto
        n, check = mX.shape
        assert n == check, "vech operation on an non-symmetric matrix"

        vH = np.zeros(((n * (n + 1)) // 2, 1))
        writer = 0
        for i in range(n):
            for j in range(i, n):
                vH[writer] = mX[j][i]
                writer += 1

        return vH

    def unvech(self, vX):
        # unvech column vector to a Toeplitz matrix
        if len(vX) < len(vX.T):
            vX = vX.T

        iN = round(0.5 * ((-1 + (1 + 8 * len(vX)) ** 0.5)))

        mX = np.zeros((iN, iN))

        for i in range(iN):
            for j in range(iN - i):
                mX[j][j + i] = vX[i]
                mX[j + i][j] = vX[i]

        return mX

    def DataGeneratingProcess(self, iI=0):
        # > "Python uses the Mersenne Twister as the core generator."
        # > https://docs.python.org/2/library/random.html
        random.seed(123 + 2 * iI)

        phi, beta = self.vTheta
        gam, rho, varphi, eta = self.vLambda
        mSig = self.unvech(self.vSigma)

        # Definition (7)
        mC = np.full((self.iN, self.iN), rho) + \
            (gam - rho) * np.identity(self.iN)
        mD = np.full((self.iN, self.iN), eta) + \
            (varphi - eta) * np.identity(self.iN)

        # Equation (11)
        mK = mSig - np.dot(np.dot(mC, mSig), mC) - np.dot(np.dot(mD, mSig), mD)

        # Matrix with size (iT * iN), elements are generated with Uniform dist over [0, 1)
        mX = np.random.rand(self.iT, self.iN)

        # Equation (16) ??
        vMy = (1 / (1 - phi)) * (0.5 + self.vAlpha)
        mSy = (1 / (1 - phi ** 2))*((1/12) + mSig)

        # Definition (4)
        mY = np.zeros((self.iT, self.iN))
        # Definition (5)
        vU = np.dot(np.random.normal(0, 1, self.iN), scipy.linalg.sqrtm(mSy))
        mY[0] = vU + vMy.T
        mH = mSig

        for t in range(1, self.iT):
            mH = mK + np.dot(np.dot(mC, np.outer(vU, vU)), mC) + \
                np.dot(np.dot(mD, mH), mD)
            vU = np.dot(np.random.normal(0, 1, self.iN),
                        scipy.linalg.sqrtm(mH))
            mY[t] = self.vAlpha.T + (phi * mY[t-1]) + (beta * mX[t]) + vU

        return mY, mX

    def parser(self):

        phi, beta = self.vTheta
        gam, rho, varphi, eta = self.vLambda
        mSig = self.unvech(self.vSigma)

        # Definition (7)
        mC = np.full((self.iN, self.iN), rho) + \
            (gam - rho) * np.identity(self.iN)
        mD = np.full((self.iN, self.iN), eta) + \
            (varphi - eta) * np.identity(self.iN)

        # Equation (11)
        mK = mSig - np.dot(np.dot(mC, mSig), mC) - np.dot(np.dot(mD, mSig), mD)

        # Data Matrix with size (iT * iN)
        mX = self.df
        vMy = (1 / (1 - phi)) * (0.5 + self.vAlpha)
        mSy = (1 / (1 - phi ** 2))*((1/12) + mSig)

        mY = np.zeros((self.iT, self.iN))

        # Row vector with dimension of iN, elements are generated with Norm(mu = 0, sd = 1)
        vU = np.dot(np.random.normal(0, 1, self.iN), scipy.linalg.sqrtm(mSy))
        mY[0] = vU + vMy.T
        mH = mSig

        for t in range(1, self.iT):
            # Equation (18)
            mH = mK + np.dot(np.dot(mC, np.outer(vU, vU)), mC) + \
                np.dot(np.dot(mD, mH), mD)

            # Equation (5)
            vU = np.dot(np.random.normal(0, 1, self.iN),
                        scipy.linalg.sqrtm(mH))

            # Equation (4)
            mY[t] = self.vAlpha.T + (phi * mY[t-1]) + (beta * mX[t]) + vU

        return mY, mX

    def Obj_pg(self, vLambda, mU, mSig):
        self.iT, self.iN = mU.shape
        gam, rho, varphi, eta = vLambda

        mC = np.full((self.iN, self.iN), rho) + \
            (gam - rho) * np.identity(self.iN)
        mD = np.full((self.iN, self.iN), eta) + \
            (varphi - eta) * np.identity(self.iN)

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
            for t in range(1, self.iT):
                mH = mK + \
                    np.dot(np.dot(mC, np.outer(mU[t-1], mU[t-1])), mC) + \
                    np.dot(np.dot(mD, mH), mD)
                vLam, _ = np.linalg.eig(mH)
                if min(vLam) < 0:
                    ll = ll - 1e+16
                    print("min(vLam) < 0")
                    break

                if np.linalg.det(mH) < self.eps:
                    ll = ll - 1e+16
                    print("det(H) < eps")
                    break

                ll = ll - 0.5 * math.log(np.linalg.det(mH)) - 0.5 * \
                    np.inner(mU[t], np.linalg.solve(mH, mU[t].T))

        ll += self.iT * (-0.5 * self.iN * math.log(2 * math.pi))
        obj = -ll / self.iT
        if abs(np.imag(obj)) > 0:
            obj = 1e+16

        return obj

    def run(self, debug_print=False, DGP=True):
        mR = np.zeros((self.iR, self.iP))
        vLambda_ini = self.vLambda

        start = time.time()

        for j in range(self.iR):
            if debug_print:
                print(j + 1, "th iteration")

            if DGP:
                mY0, mX0 = self.DataGeneratingProcess(iI=j)
            else:
                mY0, mX0 = self.parser()

            # mY = mY0#[1:]
            # mX = mX0#[1:]
            mZ = np.zeros((2, self.iT, self.iN))

            mZ[0] = mY0  # [:len(mY0) - 1]
            mZ[1] = mX0

            mZb = np.mean(mZ, axis=1)
            mZt = np.zeros((2, self.iT, self.iN))

            mZt[0] = mZ[0] - np.outer(np.ones(self.iT), mZb[0])
            mZt[1] = mZ[1] - np.outer(np.ones(self.iT), mZb[1])

            vYb = np.mean(mY0, axis=0)
            mQ = np.zeros((2, 2))
            vQ = np.zeros(2)
            mYt = mY0 - np.outer(np.ones(self.iT), vYb)

            for i in range(self.iN):
                mQ1 = np.reshape(mZt[:, :, i], (self.iT, 2))
                mQ += np.dot(mQ1.T, mQ1)
                vQ += np.dot(mQ1.T, mYt[:, i])

            vTheta_h = np.linalg.solve(mQ, vQ)
            mZbb = np.reshape(mZb, (self.iN, 2))

            vAlpha_h = vYb.T - np.dot(mZbb, vTheta_h)
            mU = np.zeros((self.iT, self.iN))

            for i in range(self.iN):
                mZi = np.reshape(mZ[:, :, i], (self.iT, 2))
                mU[:, i] = mY0[:, i] - vAlpha_h[i] - np.dot(mZi, vTheta_h)

            mSig_h = (1 / self.iT) * np.dot(mU.T, mU)
            vSig_h = self.vech(mSig_h)

            result = scipy.optimize.minimize(
                fun=self.Obj_pg,
                x0=vLambda_ini,
                args=(mU, mSig_h),
                bounds=scipy.optimize.Bounds(self.lb, self.ub)
            )

            if debug_print:
                print(result)

                if -self.iT * result.fun < -1e05:
                    print("-self.iT * result.fun == ", -self.iT * result.fun, " < -1e05;  xyzxyzxyzxyzxyzxyzxyzxyzxyz")
                print("______________________________________________________")

            vLambda_h = result.x
            mR[j] = np.concatenate(
                (vTheta_h.T, vAlpha_h.T, vSig_h.T, vLambda_h.T),
                axis=None
            )

        print("Took {:.2f} s to complete".format(time.time() - start))

        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        with open('res.out', 'w') as f:
            f.write(np.array2string(mR, separator=',', formatter={
                    'float_kind': lambda x: "\t%.2f" % x}))
        # np.savetxt('test.out', mR, fmt='%.2f\t')
