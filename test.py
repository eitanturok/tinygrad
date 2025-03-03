from tinygrad import Tensor
from icecream import install
install()

t = Tensor([[1,2,3],[4,5,6]])
t.max_pool2d(return_indices=False)
t.max_pool2d(return_indices=True)
