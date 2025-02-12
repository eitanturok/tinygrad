import unittest
from tinygrad import Tensor, dtypes
# from tinygrad.helpers import Context, getenv, DEVECTORIZE
from tinygrad.ops import UOp, Ops
from tinygrad.codegen.rewriter import full_graph_rewrite
from icecream import install, ic
install()

t1 = Tensor([3, 4, 5, 6, 7]).log2()
print(f'{t1.tolist()=}')

t2 = Tensor([3, 4, 5, 6, 7]).sum()
print(f'{t2.tolist()=}')

t3 = Tensor([3, 4, 5, 6, 7]).reciprocal()
print(f'{t3.tolist()=}')



# # test/unit/test_graph_rewrite.py::test_graph_rewrite_div_folding_bug

# def apply_rewrite(expr):
#   return full_graph_rewrite(expr.sink()).src[0]

# vconst = UOp(Ops.VCONST, dtypes.int.vec(4), arg=(0, 256, 512, 768), src=())
# print(f'{vconst=}')

# print('*****ADD*****')
# lhs = UOp(Ops.ADD, dtypes.int.vec(4), src=(
#     UOp(Ops.VECTORIZE, dtypes.int.vec(4), arg=None, src=(UOp(Ops.SPECIAL, dtypes.int, arg=('lidx0', 32), src=()),)*4),
#     UOp(Ops.VCONST, dtypes.int.vec(4), arg=(0, 256, 512, 768), src=())))
# rhs = UOp.const(dtypes.int.vec(4), 2)
# unopt = lhs<rhs
# opt = apply_rewrite(unopt)
# print(f'{unopt=}')
# print(f'{opt=}\n')


# print('*****LOG2*****')
# lhs = UOp(Ops.LOG2, dtypes.int.vec(4), src=(
#     UOp(Ops.VECTORIZE, dtypes.int.vec(4), arg=None, src=(UOp(Ops.SPECIAL, dtypes.int, arg=('lidx0', 32), src=()),)*4),
#     UOp(Ops.VCONST, dtypes.int.vec(4), arg=(0, 256, 512, 768), src=())))
# rhs = UOp.const(dtypes.int.vec(4), 2)
# unopt = lhs<rhs
# opt = apply_rewrite(unopt)
# print(f'{unopt=}')
# print(f'{opt=}')








class TestVectorizedLog2(unittest.TestCase):
  def test_vectorized_log2_combines(self):
    # Create 4 log2 operations that should be vectorized
    c = UOp.const(dtypes.float32.vec(4), (1.0, 2.0, 4.0, 8.0))
    log_ops = [c.gep(i).log2() for i in range(4)]
    vec_log = UOp(Ops.VECTORIZE, dtypes.float32.vec(4), tuple(log_ops))

    sink = vec_log.sink()
    opt_sink = full_graph_rewrite(sink)

    # Verify we only have one LOG2 operation after optimization
    log2_ops = [x for x in opt_sink.toposort if x.op is Ops.LOG2]
    self.assertEqual(len(log2_ops), 1, "Expected single vectorized LOG2 operation")
    self.assertEqual(log2_ops[0].dtype.count, 4, "Expected LOG2 operation to be vectorized with size 4")

  def test_vectorized_log2_correctness(self):
    # Test that vectorized log2 computes correctly
    inputs = (1.0, 2.0, 4.0, 8.0)
    expected = (0.0, 1.0, 2.0, 3.0)

    c = UOp.const(dtypes.float32.vec(4), inputs)
    log_ops = [c.gep(i).log2() for i in range(4)]
    vec_log = UOp(Ops.VECTORIZE, dtypes.float32.vec(4), tuple(log_ops))

    opt = full_graph_rewrite(vec_log.sink()).src[0]
    self.assertEqual(opt.op, Ops.LOG2)
    self.assertEqual(opt.dtype.count, 4)

    # Extract results and verify
    results = [full_graph_rewrite(opt.gep(i).sink()).src[0].arg for i in range(4)]
    for exp, res in zip(expected, results):
      self.assertAlmostEqual(exp, res, places=5)

# if __name__ == '__main__':
#   unittest.main(verbosity=2)


# c = UOp.const(dtypes.float32.vec(4), (1.0, 2.0, 4.0, 8.0))
# log_ops = [c.gep(i).log2() for i in range(4)]
# vec_log = UOp(Ops.VECTORIZE, dtypes.float32.vec(4), tuple(log_ops))

# sink = vec_log.sink()
# opt_sink = full_graph_rewrite(sink)
# ic(sink, opt_sink)


# with Context(DEVECTORIZE=0):
#   print(getenv("DEVECTORIZE"))
#   t1 = Tensor([3, 4, 5, 6, 7]).log2()
#   print(t1.tolist())

# with Context(DEVECTORIZE=1):
#   print(getenv("DEVECTORIZE"))
#   t2 = Tensor([3, 4, 5, 6, 7]).log2()
#   print(t2.tolist())
