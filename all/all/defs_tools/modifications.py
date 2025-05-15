import numpy as np


def col_flatten(
    arr,
):  # for matrices---- to give input as  r+v+a+..(higher derivatives) , both are vectors
    rsize = len(arr)
    csize = len(arr[0])
    return np.array([arr[r][c] for c in range(csize) for r in range(rsize)])
