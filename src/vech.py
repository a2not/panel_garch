import numpy as np

# take vech operation on d*d symmetric matrix and return (d*(d+1) / 2) dimension column vector


def vech(mX):
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


def test_vech(mX):
    print("before: ")
    print(mX)
    res = vech(mX)
    print("after: ")
    print(res)
    print()


if __name__ == "__main__":
    # test
    test_vech(np.ones((3, 3)))
    test_vech(np.array([[1, 2, 3],
                        [2, 4, 2],
                        [3, 21, 8]]))
    # fail case; non-symetric
    # test_vech(np.ones((2, 3)))
