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
        assert self.vTheta.shape == (2, 1), "vTheta not size (2, 1)"

        if initial_vAlpha == None:
            self.vAlpha = np.array(
                [[(1 / self.iN) * num for num in range(1, self.iN + 1)]]).T
        else:
            self.vAlpha = initial_vAlpha
        assert self.vAlpha.shape == (self.iN, 1), "vAlpha not size (iN, 1)"

        if initial_vS == None:
            self.vS = np.array([[(-0.4)**i for i in range(self.iN)]]).T
        else:
            self.vS = initial_vS
        assert self.vS.shape == (self.iN, 1), "vS not size (iN, 1)"

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
        assert self.vSigma.shape == (self.iN * (self.iN + 1) // 2, 1), "vSigma not size (iN * (iN + 1) // 2, 1)"

        if initial_vLambda == None:
            self.vLambda = np.array([[0.4, -0.1, 0.6, -0.2]]).T
        else:
            self.vLambda = initial_vLambda
        assert self.vLambda.shape == (4, 1), "vLambda not size (4, 1)"

        self.vPsi = np.concatenate(
            (self.vTheta, self.vAlpha, self.vSigma, self.vLambda), axis=0)
        self.iP = len(self.vPsi)

        if iR == None:
            self.iR = 20
        else:
            self.iR = iR

    def vech(self, mX):
        # take vech operation on d*d symmetric matrix and return (d*(d+1) / 2) dimension column vecto
        n, check = mX.shape
        assert n == check, "vech operation on a non-symmetric matrix"

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

    def spectralRadiusOfKroneckers(self, vLambda):
        assert vLambda.shape == (4, 1), "vLambda not size (4, 1)"
        gam, rho, varphi, eta = vLambda

        # Definition (13)
        mC = np.full((self.iN, self.iN), rho) + \
            (gam - rho) * np.identity(self.iN)
        mD = np.full((self.iN, self.iN), eta) + \
            (varphi - eta) * np.identity(self.iN)
        
        # Assumption 5
        mM = np.kron(mC, mC) + np.kron(mD, mD)
        vL = np.linalg.eigvals(mM)

        radius = max(abs(vL))
        print(radius)

        return radius

    def DataGeneratingProcess(self, iI=0):
        # > "Python uses the Mersenne Twister as the core generator."
        # > https://docs.python.org/2/library/random.html
        random.seed(123 + 2 * iI)

        phi, beta = self.vTheta
        gam, rho, varphi, eta = self.vLambda
        mSig = self.unvech(self.vSigma)

        # Definition (13)
        mC = np.full((self.iN, self.iN), rho) + \
            (gam - rho) * np.identity(self.iN)
        mD = np.full((self.iN, self.iN), eta) + \
            (varphi - eta) * np.identity(self.iN)

        # Equation (14)
        mK = mSig - np.dot(np.dot(mC, mSig), mC) - np.dot(np.dot(mD, mSig), mD)

        # Matrix with size (iT * iN), elements are generated with Uniform dist over [0, 1)
        mX = np.random.rand(self.iT, self.iN)

        # Equation (15)
        vMy = (1 / (1 - phi)) * (0.5 + self.vAlpha)
        mSy = (1 / (1 - phi ** 2))*((1/12) + mSig)

        # Definition (4)
        mY = np.zeros((self.iT, self.iN))

        # Definition (16)
        vU = np.dot(np.random.normal(0, 1, self.iN), scipy.linalg.sqrtm(mSy))
        mY[0] = vU + vMy.T
        mH = mSig

        for t in range(1, self.iT):
            # Definition (12)
            mH = mK + np.dot(np.dot(mC, np.outer(vU, vU)), mC) + \
                np.dot(np.dot(mD, mH), mD)

            # Definition (16)
            vU = np.dot(np.random.normal(0, 1, self.iN),
                        scipy.linalg.sqrtm(mH))
            mY[t] = self.vAlpha.T + (phi * mY[t-1]) + (beta * mX[t]) + vU

        return mY, mX

    def parser(self):

        phi, beta = self.vTheta
        gam, rho, varphi, eta = self.vLambda
        mSig = self.unvech(self.vSigma)

        # Definition (13)
        mC = np.full((self.iN, self.iN), rho) + \
            (gam - rho) * np.identity(self.iN)
        mD = np.full((self.iN, self.iN), eta) + \
            (varphi - eta) * np.identity(self.iN)

        # Equation (14)
        mK = mSig - np.dot(np.dot(mC, mSig), mC) - np.dot(np.dot(mD, mSig), mD)

        # Data Matrix with size (iT * iN)
        mX = self.df
        vMy = (1 / (1 - phi)) * (0.5 + self.vAlpha)
        mSy = (1 / (1 - phi ** 2))*((1/12) + mSig)

        mY = np.zeros((self.iT, self.iN))

        # Equation (5)
        # epsilon_t is row vector with dimension of iN,
        # elements are generated with Norm(mu = 0, sd = 1)
        vU = np.dot(scipy.linalg.sqrtm(mSy),
                np.random.normal(0, 1, (self.iN, 1)))
        mY[0] = vU.T + vMy.T
        # H_0 := mSig
        mH = mSig

        for t in range(1, self.iT):
            # Equation (18)
            mH = mK + np.dot(np.dot(mC, np.outer(vU, vU)), mC) + \
                np.dot(np.dot(mD, mH), mD)

            # Equation (5)
            vU = np.dot(scipy.linalg.sqrtm(mH),
                    np.random.normal(0, 1, (self.iN, 1)))
            # Equation (4)
            mY[t] = self.vAlpha.T + (phi * mY[t-1]) + (beta * mX[t]) + vU.T

        return mY, mX

    def Obj_pg(self, vLambda, mU, mSig):
        assert mU.shape == (self.iT, self.iN), "mU not size (iT, iN)"
        gam, rho, varphi, eta = vLambda

        # Definition (13)
        mC = np.full((self.iN, self.iN), rho) + \
            (gam - rho) * np.identity(self.iN)
        mD = np.full((self.iN, self.iN), eta) + \
            (varphi - eta) * np.identity(self.iN)

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
        for t in range(self.iT):
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
            if np.linalg.det(mH) < self.eps:
                # the next computation of log(det(mH)) will cause ValueError
                # since log(0) is undefined
                return 1e+16


        ll -= self.iN * self.iT * math.log(2 * math.pi)
        ll *= 0.5
        if abs(np.imag(ll)) > 0:
            return 1e+16

        # maximizing f(x) <=> minimizing -f(x)
        print("obj func runs successfully")
        return -ll

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

            # bound on parameters x
            iC = 1 - 1e-6
            lb = np.full((4, 1), -iC)
            ub = np.full((4, 1), iC)

            # constraints: Assumption 5
            # the spectrum radius of kron(C, C) + kron(D, D) <= 1
            assumptions = scipy.optimize.NonlinearConstraint(
                fun=self.spectralRadiusOfKroneckers,
                lb=-np.inf,
                ub=1
            )

            result = scipy.optimize.minimize(
                fun=self.Obj_pg,
                x0=vLambda_ini,
                args=(mU, mSig_h),
                bounds=scipy.optimize.Bounds(lb, ub),
                constraints=assumptions
            )

            if debug_print:
                print(result)

                if -self.iT * result.fun < -1e05:
                    print("-self.iT * result.fun == ", -self.iT *
                          result.fun, " < -1e05;  xyzxyzxyzxyzxyzxyzxyzxyzxyz")
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
