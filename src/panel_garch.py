import numpy as np
import pandas
import time
import math
import random
import scipy.optimize
from . import unvech, vech, Obj_pg, DGP, radius, posdef


class panel_garch:

    def __init__(self, dataframe, initial_vTheta=None, initial_vAlpha=None, initial_vS=None, initial_vSigma=None, initial_vLambda=None, iR=None):
        self.df = dataframe
        self.iT, self.iN = self.df.shape
        self.df = np.resize(self.df, (self.iT, self.iN))

        if initial_vTheta == None:
            self.vTheta = np.array([0.6, 1])
        else:
            self.vTheta = initial_vTheta
        assert self.vTheta.shape == (2,), "vTheta not size (2,)"

        if initial_vAlpha == None:
            self.vAlpha = np.array(
                [(1 / self.iN) * num for num in range(1, self.iN + 1)])
        else:
            self.vAlpha = initial_vAlpha
        assert self.vAlpha.shape == (self.iN,), "vAlpha not size (iN,)"

        if initial_vS == None:
            self.vS = np.array([(-0.4)**i for i in range(self.iN)])
        else:
            self.vS = initial_vS
        assert self.vS.shape == (self.iN,), "vS not size (iN,)"

        if initial_vSigma == None:
            self.vSigma = np.zeros(self.iN * (self.iN + 1) // 2)
            for i in range(self.iN):
                # reducing scope of the variable
                rep, subtract, writer = i, self.iN, 0

                while rep > 0:
                    writer += subtract
                    subtract -= 1
                    rep -= 1

                for j in range(self.iN - i):
                    self.vSigma[writer + j] = self.vS[j]
        else:
            self.vSigma = initial_vSigma
        assert self.vSigma.shape == (
            self.iN * (self.iN + 1) // 2,), "vSigma not size (iN * (iN + 1) // 2,)"

        if initial_vLambda == None:
            self.vLambda = np.array([0.4, -0.1, 0.6, -0.2])
        else:
            self.vLambda = initial_vLambda
        assert self.vLambda.shape == (4,), "vLambda not size (4,)"

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

    def initialPositiveDefiniteness(self, vLambda):
        return posdef.initialPositiveDefiniteness(self.iN, vLambda, self.mSig_h)

    def DataGeneratingProcess(self, iI=0, df=None):
        return DGP.DGP(self.vTheta, self.vAlpha, self.vSigma, self.vLambda, self.iT, self.iN, iI, df)

    def Obj_pg(self, vLambda, mU, mSig):
        return Obj_pg.Obj_pg(self.iN, self.iT, vLambda, mU, mSig)
    
    def objfunc(self, mU, mSig):
        def objective_function(x):
            return Obj_pg.Obj_pg(self.iN, self.iT, x, mU, mSig)
        return objective_function

    # https://stackoverflow.com/questions/52208363/scipy-minimize-violates-given-bounds
    def gradient_respecting_bounds(self, bounds, fun, eps=1e-8):
        """bounds: list of tuples (lower, upper)"""
        def gradient(x):
            fx = fun(x)
            grad = np.zeros(len(x))
            for k in range(len(x)):
                d = np.zeros(len(x))
                d[k] = eps if x[k] + eps <= bounds[k][1] else -eps
                grad[k] = (fun(x + d) - fx) / d[k]
            return grad
        return gradient

    def run(self, debug_print=False, DGP=True):
        mR = []
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
            mYt = np.resize(mY0, (self.iT, self.iN)) - \
                np.outer(np.ones(self.iT), vYb)

            # Assumption 1 (c); plim Q_x^{bar} = Q_x
            mQ = np.zeros((2, 2))
            vQ = np.zeros(2)

            for i in range(self.iN):
                mQ1 = np.reshape(mZt[:, i, :], (self.iT, 2))
                mQ += np.dot(mQ1.T, mQ1)
                vQ += np.dot(mQ1.T, mYt[:, i])

            self.vTheta = np.linalg.solve(mQ, vQ)
            mZbb = np.reshape(mZb, (self.iN, 2))

            self.vAlpha = vYb - np.dot(mZbb, self.vTheta)
            mU = np.zeros((self.iT, self.iN))

            for i in range(self.iT):
                mU[i] = mY0[i] - self.vAlpha - np.dot(mZ[i], self.vTheta)

            self.mSig_h = (1 / self.iT) * np.dot(mU.T, mU)
            vSig_h = self.vech(self.mSig_h)

            # bound on parameters x
            iC = 1 - 1e-6
            # lb = np.full(4, -iC)
            # ub = np.full(4, iC)
            # bounds = scipy.optimize.Bounds(lb, ub)
            bounds = tuple([(-iC, iC) for _ in range(4)])

            # constraints: Assumption 5
            # the spectrum radius of kron(C, C) + kron(D, D) <= 1
            assumptions = []
            assumptions.append(scipy.optimize.NonlinearConstraint(
                fun=self.spectralRadiusOfKroneckers,
                lb=-np.inf,
                ub=1
            ))
            # In the BEKK model, H_t is positive definite if K and H_0 are
            assumptions.append(scipy.optimize.NonlinearConstraint(
                fun=self.initialPositiveDefiniteness,
                lb=1e-10,
                ub=np.inf
            ))
            result = scipy.optimize.minimize(
                # fun=self.Obj_pg,
                fun=self.objfunc(mU, self.mSig_h),
                x0=self.vLambda,
                # args=(mU, self.mSig_h),
                method='SLSQP',
                # jac=self.gradient_respecting_bounds(bounds=bounds, fun=self.objfunc(mU, self.mSig_h)),
                bounds=bounds,
                constraints=assumptions
            )

            if debug_print:
                print(result)
                print("______________________________________________________")

            self.vLambda = result.x
            mR.append(np.concatenate(
                (self.vTheta.T, self.vAlpha.T, vSig_h.T, self.vLambda.T),
                axis=None
            ))

        mR = np.array(mR)

        # writing mR to ./res.out
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        with open('res.out', 'w') as f:
            f.write(np.array2string(mR, separator=',', formatter={
                    'float_kind': lambda x: "\t%.2f" % x}))

        self.vPsi = np.concatenate(
            (self.vTheta, self.vAlpha, self.vSigma, self.vLambda), axis=0)

        # writing stats to ./stats.out
        # parameters, sample mean, sample sd, MSE (mean square error)
        # [vPsi mean(mR)' sqrt(var(mR))' sqrt(mean((mR-ones(iR,1)*vPsi').^2))']
        sampleMean = np.mean(mR, axis=0)
        sampleSd = np.sqrt(np.var(mR, axis=0))
        meanDiff = np.mean(mR - np.outer(np.ones((self.iR, 1)), self.vPsi), axis=0)
        MSE = np.array([np.sqrt(np.dot(meanDiff.T, meanDiff))])
        stats = np.concatenate(
            (self.vPsi, sampleMean, sampleSd, MSE)
        )
        print("stats.out shape: ", stats.shape)
        with open('stats.out', 'w') as f:
            f.write(np.array2string(stats, separator=',', formatter={
                    'float_kind': lambda x: "\t%.2f" % x}))

        print("Took {:.2f} s to complete".format(time.time() - start))
