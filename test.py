from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.opt.kernel import Kernel
from icecream import install
install()

def test1():
    t = Tensor(1)
    ic(t.numpy())

def test2():
    """
    VIEW_CONST=0
    ic| modified_ast:
                  UOp(Ops.SINK, dtypes.void, arg=KernelInfo(name='E_\x1b[90m\x1b[0m', local_dims=0, upcasted=0, dont_use_locals=False, applied_opts=(), opts_to_apply=None), src=(
                    UOp(Ops.STORE, dtypes.void, arg=None, src=(
                      UOp(Ops.VIEW, dtypes.int.ptr(1), arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(1), arg=0, src=()),)),
                      UOp(Ops.CONST, dtypes.int, arg=1, src=(
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=()),)),)),))

    VIEW_CONST=1
    ic| modified_ast:
                  UOp(Ops.SINK, dtypes.void, arg=KernelInfo(name='E_\x1b[90m\x1b[0m', local_dims=0, upcasted=0, dont_use_locals=False, applied_opts=(), opts_to_apply=None), src=(
                    UOp(Ops.STORE, dtypes.void, arg=None, src=(
                      UOp(Ops.VIEW, dtypes.void.ptr(1), arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.void.ptr(1), arg=0, src=()),)),
                      UOp(Ops.CONST, dtypes.int, arg=1, src=(
                        UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),)),)),))


    VIEW_CONST=0
    ic| "kernel.__init__": 'kernel.__init__'
    self.ast: UOp(Ops.SINK, dtypes.void, arg=None, src=(
                UOp(Ops.STORE, dtypes.void, arg=None, src=(
                  UOp(Ops.VIEW, dtypes.int.ptr(1), arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(1), arg=0, src=()),)),
                  UOp(Ops.CONST, dtypes.int, arg=1, src=(
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=()),)),)),))

    VIEW_CONST=1
    ic| "kernel.__init__": 'kernel.__init__'
    self.ast: UOp(Ops.SINK, dtypes.void, arg=None, src=(
                UOp(Ops.STORE, dtypes.void, arg=None, src=(
                  UOp(Ops.VIEW, dtypes.void.ptr(1), arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.void.ptr(1), arg=0, src=()),)),
                  UOp(Ops.CONST, dtypes.int, arg=1, src=()),)),))

    CHANGE: now Ops.VIEW has the correct dtype
    VIEW_CONST=1
    ic| "kernel.__init__": 'kernel.__init__'
    self.ast: UOp(Ops.SINK, dtypes.void, arg=None, src=(
                UOp(Ops.STORE, dtypes.void, arg=None, src=(
                  UOp(Ops.VIEW, dtypes.int.ptr(1), arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(1), arg=0, src=()),)),
                  UOp(Ops.CONST, dtypes.int, arg=1, src=()),)),))


    VIEW_CONST=0
    ic| "before kernelize map": 'before kernelize map'
    sink: UOp(Ops.SINK, dtypes.void, arg=None, src=(
            UOp(Ops.COPY, dtypes.int, arg=None, src=(
              UOp(Ops.CONTIGUOUS, dtypes.int, arg=None, src=(
                UOp(Ops.CONST, dtypes.int, arg=1, src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
                    UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),)),)),)),
              UOp(Ops.DEVICE, dtypes.void, arg='CPU', src=()),)),))

    VIEW_CONST=1
    ic| "before kernelize map": 'before kernelize map'
    sink: UOp(Ops.SINK, dtypes.void, arg=None, src=(
            UOp(Ops.COPY, dtypes.int, arg=None, src=(
              UOp(Ops.CONTIGUOUS, dtypes.int, arg=None, src=(
                UOp(Ops.VIEW, dtypes.int, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
                  UOp(Ops.CONST, dtypes.int, arg=1, src=(
                    UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),)),)),)),
              UOp(Ops.DEVICE, dtypes.void, arg='CPU', src=()),)),))
    """


    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
           UOp(Ops.STORE, dtypes.void, arg=None, src=(
             UOp(Ops.VIEW, dtypes.void.ptr(1), arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
               UOp(Ops.DEFINE_GLOBAL, dtypes.void.ptr(1), arg=0, src=()),)),
             UOp(Ops.CONST, dtypes.int, arg=1, src=(
               UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),)),)),))

    k = Kernel(ast)
    modified_ast = k.get_optimized_ast()
    ic(ast, modified_ast)

if __name__ == '__main__':
    test1()
