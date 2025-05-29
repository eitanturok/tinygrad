from tinygrad import Tensor, dtypes
from tinygrad.shape.shapetracker import ShapeTracker, View
from icecream import install
install()

mask_size, idx_size = 50_257, 324
mask = (Tensor.rand(1, mask_size) < 0.5).contiguous()
ic(mask)
# mask = Tensor.zeros(1, mask_size, dtype=dtypes.bool)
mask.lazydata.st = ShapeTracker([View.create(shape=(1, 50257), strides=(0, 1), offset=0, mask=None)])
mask.contiguous()


# mask = Tensor.ones(1, tensor_size, dtype=dtypes.bool).contiguous()
idx1 = Tensor.zeros(idx_size, dtype=dtypes.int)
idx2 = Tensor(Tensor.randint(idx_size, low=0, high=mask_size, dtype=dtypes.int).numpy())
ic(mask, idx1, idx1.numpy(), idx2, idx2.numpy())
ic(mask)
