import unittest, math
import numpy as np
from tinygrad import dtypes
from tinygrad.ops import UOp, Ops
from tinygrad.helpers import Context
from tinygrad.codegen.transcendental import payne_hanek_reduction, cody_waite_reduction, frexp, rintk, pow2if, TRANSCENDENTAL_SUPPORTED_DTYPES
from hypothesis import given, strategies as strat
from test.helpers import eval_uop

class TestTranscendentalFunctions(unittest.TestCase):
  def test_payne_hanek_reduction(self):
    # TODO: Test constant input when constant folding is fixed (or maybe test both variants)
    # Load input value from a buffer to prevent constant folding
    input_buf = UOp(Ops.DEFINE_GLOBAL, dtypes.double.ptr(), arg=1, src=())
    loaded_value = UOp.load(input_buf.index(UOp.const(dtypes.int, 0)), dtype=dtypes.double)
    def eval_payne_hanek_reduction(v:float) -> tuple[float, int]:
      return tuple(eval_uop(u, [(dtypes.float64, [v])]) for u in payne_hanek_reduction(loaded_value))

    r, q = eval_payne_hanek_reduction(12 * math.pi + 0.1)
    np.testing.assert_allclose(r, 0.1 - math.pi / 2)
    np.testing.assert_equal(q, 1)

    r, q = eval_payne_hanek_reduction(12 * math.pi)
    np.testing.assert_allclose(r, 0.0, atol=1e-8)
    np.testing.assert_equal(q, 4)

    r, q = eval_payne_hanek_reduction(12 * math.pi - 0.1)
    np.testing.assert_allclose(r, -0.1)
    np.testing.assert_equal(q, 4)

  def test_cody_waite_reduction(self):
    r, q = (eval_uop(u) for u in cody_waite_reduction(UOp.const(dtypes.float64, 12 * math.pi + 0.1)))
    np.testing.assert_allclose(r, 0.1)
    np.testing.assert_equal(q, 12)

  def test_frexp(self):
    for x in (1, -1):
      mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, x)))
      np.testing.assert_equal(mantissa, 0.5)
      np.testing.assert_equal(exponent, 1)

    for x in (2, -2):
      mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, 2.0)))
      np.testing.assert_equal(mantissa, 0.5)
      np.testing.assert_equal(exponent, 2)

    mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, 5.0)))
    np.testing.assert_equal(mantissa, 0.625)
    np.testing.assert_equal(exponent, 3)

    mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, 1000.0)))
    np.testing.assert_allclose(mantissa, 0.9765625)
    np.testing.assert_equal(exponent, 10)

  def test_rintk(self):
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 0.0))), 0)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 5.0))), 5)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 5.5))), 6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 5.999))), 6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, -5.0))), -5)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, -5.5))), -6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, -5.999))), -6)

  def test_pow2if(self):
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 0), dtypes.float)), 1.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 1), dtypes.float)), 2.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 2), dtypes.float)), 4.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 10), dtypes.float)), 1024.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 63), dtypes.float)), 2**63)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, -1), dtypes.float)), 0.5)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, -2), dtypes.float)), 0.25)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, -10), dtypes.float)), 2**-10)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, -63), dtypes.float)), 2**-63)

class TestVectorizedTranscendetalFunctions(unittest.TestCase):

  def _check_all_vectorized(self, u: tuple|UOp, vcount: int):
    # check all UOps in u are vectorized
    if isinstance(u, UOp): assert u.dtype.vcount == vcount, f'expected {vcount=} but got {u.dtype.vcount=} for UOp {u=}'
    [self._check_all_vectorized(x, vcount) for x in (u if isinstance(u, tuple) else u.src)]

  @given(strat.floats(width=16, allow_subnormal=False), strat.integers(1, 255), strat.sampled_from(TRANSCENDENTAL_SUPPORTED_DTYPES))
  def test_vectorized_payne_hanek_reduction(self, val, vcount, dtype):
    # test payne_hanek_reduction preserves vectorization
    with Context(DEVECTORIZE=0):
      d = UOp.const(dtype.vec(vcount), val)
      out = payne_hanek_reduction(d)
      self._check_all_vectorized(out, vcount)

  @given(strat.floats(width=16, allow_subnormal=False), strat.integers(1, 255), strat.sampled_from(TRANSCENDENTAL_SUPPORTED_DTYPES))
  def test_vectorized_cody_waite_reduction(self, val, vcount, dtype):
    # test cody_waite_reduction preserves vectorization
    with Context(DEVECTORIZE=0):
      d = UOp.const(dtype.vec(vcount), val)
      out = cody_waite_reduction(d)
      self._check_all_vectorized(out, vcount)

if __name__ == '__main__':
  unittest.main()
