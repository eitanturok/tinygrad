from tinygrad import Tensor, dtypes
import torch
import numpy as np
from icecream import install
install()

def nonzero(t: Tensor) -> Tensor:
    mask = t != 0
    ic(mask.numpy())
    num_non_zeros = mask.sum().item()

    sorted_values, indices = mask.sort(descending=True)
    ic(sorted_values.numpy(), indices.numpy())
    idx = indices[:num_non_zeros]
    ic(idx.numpy())
    ret = t[idx]
    ic(ret.numpy())
    return ret

    # dim = 0
    # s = t.shape[dim]
    # idxs = Tensor.arange(s).expand(*t.shape)
    # nonzero_idxs = mask.where(idxs, Tensor.full_like(t, -1))
    # ic(nonzero_idxs.numpy())
    # nonzero_idxs2 = nonzero_idxs.sort(descending=True)[0][:n_non_zeros]
    # ic(nonzero_idxs2.numpy())
    # return nonzero_idxs


def diag(x):
    pass

if __name__ == "__main__":

    # Test cases
    t = np.array([2, 0, -1])
    t_tiny = Tensor(t)
    t_torch = torch.Tensor(t)
    ic(t)

    ret_tiny = nonzero(t_tiny)
    ic(ret_tiny.numpy())

    ret_torch = t_torch.nonzero(as_tuple=False)
    ic(ret_torch.numpy())
