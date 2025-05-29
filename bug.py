from tinygrad import Tensor, dtypes

mask_size, idx_size = 256, 256
mask = (Tensor.rand(1, mask_size) < 0.5).contiguous()
idx1 = Tensor.zeros(idx_size, dtype=dtypes.int)
idx2 = Tensor.arange(0, idx_size, dtype=dtypes.int)
mask[idx1, idx2] = False
