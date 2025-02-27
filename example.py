from dtype import dtypes
from ops import UOp, Ops, KernelInfo
from tinygrad.codegen.expander import expand_rewrite

sink1 = UOp(Ops.SINK, dtypes.void, arg=KernelInfo(local_dims=0, upcasted=1, dont_use_locals=False), src=(
            UOp(Ops.STORE, dtypes.void, arg=None, src=(
              UOp(Ops.INDEX, dtypes.float.ptr(1), arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1), arg=0, src=()),
                x3:=UOp(Ops.CONST, dtypes.int, arg=0, src=()),)),
              UOp(Ops.ASSIGN, dtypes.float, arg=None, src=(
                x5:=UOp(Ops.DEFINE_ACC, dtypes.float, arg=(0,), src=(
                  UOp(Ops.CONST, dtypes.float, arg=0.0, src=()),
                  x7:=UOp(Ops.RANGE, dtypes.int, arg=0, src=(
                     x3,
                    UOp(Ops.CONST, dtypes.int, arg=4194304, src=()),)),)),
                UOp(Ops.ADD, dtypes.float, arg=None, src=(
                   x5,
                  UOp(Ops.ADD, dtypes.float, arg=None, src=(
                    UOp(Ops.ADD, dtypes.float, arg=None, src=(
                      UOp(Ops.ADD, dtypes.float, arg=None, src=(
                        UOp(Ops.GEP, dtypes.float, arg=(0,), src=(
                          x14:=UOp(Ops.CONTRACT, dtypes.float.vec(4), arg=((1, 4),), src=(
                            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                              UOp(Ops.INDEX, dtypes.float.ptr(16777216), arg=None, src=(
                                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(16777216), arg=1, src=()),
                                UOp(Ops.ADD, dtypes.int, arg=None, src=(
                                  UOp(Ops.UNROLL, dtypes.int, arg=((1, 4),), src=(
                                    UOp(Ops.VCONST, dtypes.int.vec(4), arg=(0, 1, 2, 3), src=()),)),
                                  UOp(Ops.MUL, dtypes.int, arg=None, src=(
                                     x7,
                                    UOp(Ops.CONST, dtypes.int, arg=4, src=()),)),)),)),)),)),)),
                        UOp(Ops.GEP, dtypes.float, arg=(1,), src=(
                           x14,)),)),
                      UOp(Ops.GEP, dtypes.float, arg=(2,), src=(
                         x14,)),)),
                    UOp(Ops.GEP, dtypes.float, arg=(3,), src=(
                       x14,)),)),)),)),)),))
