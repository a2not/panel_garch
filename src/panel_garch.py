import numpy as np
import pandas
import time
import math
import random
import scipy.optimize
from . import unvech, vech, Obj_pg, DGP, radius


class panel_garch:

    def __init__(self, dataframe, initial_vTheta=None, initial_vAlpha=None, initial_vS=None, initial_vSigma=None, initial_vLambda=None, iR=None):
        self.df = dataframe
        self.iT, self.iN = self.df.shape
        self.df = np.resize(self.df, (self.iT, self.iN, 1))

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
        assert self.vSigma.shape == (
            self.iN * (self.iN + 1) // 2, 1), "vSigma not size (iN * (iN + 1) // 2, 1)"

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
        return vech.vech(mX)

    def unvech(self, vX):
        return unvech.unvech(vX)

    def spectralRadiusOfKroneckers(self, vLambda):
        return radius.spectralRadiusOfKroneckers(self.iN, vLambda)

    def DataGeneratingProcess(self, iI=0, df=None):
        return DGP.DGP(self.vTheta, self.vAlpha, self.vSigma, self.vLambda, self.iT, self.iN, iI, df)

    def Obj_pg(self, vLambda, mU, mSig):
        Obj_pg.Obj_pg(self.iN, self.iT, vLambda, mU, mSig)

    def run(self, debug_print=False, DGP=True):
        mR = np.zeros((self.iR, self.iP))
        vLambda_ini = self.vLambda

        start = time.time()

        for j in range(self.iR):
            if debug_print:
                print(j + 1, "th iteration")

            if DGP:
                # The Date Matrix X with size (iT * iN), elements are generated with Uniform dist over [0, 1)
                mY0, mX0 = self.DataGeneratingProcess(iI=j)
            else:
                # X is obtained from the dataframe
                mY0, mX0 = self.DataGeneratingProcess(iI=j, df=self.df)

            # mZ = np.dstack((mY0, mX0))
            # Equation (6); z_{i, t} := (y_{i, t-1}, x_{i, t})
            mZ = np.zeros((self.iT, self.iN, 2))
            for i in range(self.iT):
                for j in range(self.iN):
                    if i - 1 >= 0:
                        mZ[i][j][0] = mY0[i-1][j]
                    mZ[i][j][1] = mX0[i][j]

            mZb = np.mean(mZ, axis=0)

            # Z^{tilde} := Z - Z^{bar}
            mZt = np.zeros((self.iT, self.iN, 2))
            for i in range(self.iT):
                mZt[i] = mZ[i] - mZb

            # Y^{bar} := E[ Y_t ]
            vYb = np.mean(mY0, axis=0)
            mYt = np.resize(mY0, (self.iT, self.iN)) - np.outer(np.ones(self.iT), vYb)

            # Assumption 1 (c); plim Q^{bar}_
            mQ = np.zeros((2, 2))
            vQ = np.zeros(2)

            for i in range(self.iN):
                mQ1 = np.reshape(mZt[:, i, :], (self.iT, 2))
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
            # assumptions = scipy.optimize.NonlinearConstraint(
            #     fun=self.spectralRadiusOfKroneckers,
            #     lb=np.array(-np.inf),
            #     ub=np.array(1)
            # )

            result = scipy.optimize.minimize(
                fun=self.Obj_pg,
                x0=vLambda_ini,
                args=(mU, mSig_h),
                bounds=scipy.optimize.Bounds(lb, ub)
                # constraints=assumptions
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
