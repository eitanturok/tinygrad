import numpy as np
from tinygrad import Tensor, dtypes
from icecream import install
install()

def max_off_diagonal(A:Tensor):
    off_diag_mask = Tensor.eye(A.shape[0], dtype=dtypes.bool).logical_not()
    max_val = A.clone().masked_select(off_diag_mask).max()
    max_mask = max_val == A

    # Get flat index of maximum
    flat_idx = max_mask.flatten().argmax()
    row_idx = flat_idx // A.shape[1]
    col_idx = flat_idx % A.shape[1]
    return row_idx.item(), col_idx.item()

def compute(A, i, j):
    tau = (A[i,i] - A[j,j]) / (2 * A[i, j])
    # choose the root with the smallest absolute value
    t = -tau - (1 + tau * tau).sqrt()
    c = 1.0 / (1 + t * t).sqrt()
    s = c * t
    return s, c

def rotate(A, i, j, s, c):
    a_ii = c * (c * A[i,i] - s * A[j,i]) - s * (c * A[i,j] - s * A[j,j])
    a_jj = s * (s * A[i,i] + c * A[j,i]) + c * (s * A[i,j] + c * A[j,j])
    A[i,i], A[j,j], A[i,j], A[j,i] = a_ii, a_jj, 0, 0
    return A

def eigenvalues(A:Tensor, max_iterations:int=5):
    A = A.clone()
    ic(A.numpy())
    for i_iter in range(max_iterations):
        i, j = max_off_diagonal(A)
        s, c = compute(A, i, j)
        A = rotate(A, i, j, s, c)
        ic(i_iter, A.numpy())

    # Extract eigenvalues from diagonal
    mask = Tensor.eye(A.shape[0], dtype=dtypes.bool)
    eigenvalues = A.masked_select(mask)
    return eigenvalues

def main():
    # x = Tensor([[1,2,3],[4,5,-6],[-7,8,9]])
    # out = max_off_diagonal(x)
    # ic(out)
    # ic(out.numpy())

    t = np.array([[2,1,1],[1,-3,1],[1,1,16]], dtype=np.float32) # symmetric matrix
    eig_np = np.linalg.eigvals(t)
    ic(eig_np)
    eig_tg = eigenvalues(Tensor(t))
    ic(eig_tg.numpy())

if __name__ == '__main__':
    main()
