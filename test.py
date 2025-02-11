from tinygrad import Tensor, dtypes
from tinygrad.ops import UOp, Ops
from tinygrad.codegen.rewriter import full_graph_rewrite
from icecream import install
install()

t1 = Tensor([3, 4, 5]).sum()
print(t1.tolist())

t2 = Tensor([3, 4, 5]).log2()
print(t2.tolist())











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
