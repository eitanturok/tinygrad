from tinygrad import Tensor

# t1 = Tensor([[1, 2, 3] for _ in range(256)])
# t2 = Tensor([[4, 5, 6] for _ in range(256)])
t1 = Tensor([[1, 2, 3]])
t2 = Tensor([[4, 5, 6]])
t3 = Tensor.cat(t1, t2)
print(t3.tolist())
