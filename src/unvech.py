import numpy as np

# unvech column vector to a Toeplitz matrix


def unvech(vX):
    if len(vX) < len(vX.T):
        vX = vX.T

    iN = round(0.5 * ((-1 + (1 + 8 * len(vX)) ** 0.5)))

    mX = np.zeros((iN, iN))

    for i in range(iN):
        for j in range(iN - i):
            mX[j][j + i] = vX[i]
            mX[j + i][j] = vX[i]

    return mX


def test_unvech(vX):
    print("before: ")
    print(vX)
    res = unvech(vX)
    print("after: ")
    print(res)
    print()


# test
test_unvech(np.array([[1, 2, 3, 1, 2, 1]]).T)
test_unvech(np.array([[1, 2, 3, 1, 2, 1]]))
test_unvech(np.array([[1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 1, 2, 1]]).T)
