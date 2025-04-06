import unittest, math
import numpy as np
from tinygrad import dtypes
from tinygrad.ops import UOp, Ops
from tinygrad.helpers import Context
from tinygrad.dtype import DType
from tinygrad.codegen.transcendental import TRANSCENDENTAL_SUPPORTED_DTYPES, payne_hanek_reduction, cody_waite_reduction, rintk, pow2if, xlog2, ilogb2k, frexp, sin_poly, xsin, xexp2, ldexp2k, ldexp3k,  sin_poly_large, sin_poly_small, xpow, shl, shr, trig_poly, _lazy_map_numbers, _ifand
from test.helpers import eval_uop
from icecream import install, ic
install()

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
  def _check_all_uops_vectorized(self, u:tuple|UOp, vcount:int):
    # check all UOps in u are vectorized with vcount
    if isinstance(u, UOp): assert u.dtype.vcount == vcount, f'expected {vcount=} but got {u.dtype.vcount=} for UOp {u=}'
    [self._check_all_uops_vectorized(x, vcount) for x in (u if isinstance(u, tuple) else u.src)]

  def _get_inputs(self, dtype_mode:str, vcounts:list[int]=[1,2,16,19], vals:list[float|int]=[1.3, -2, 194]) -> tuple[UOp, DType]:
    _dtypes: tuple(dtypes) = TRANSCENDENTAL_SUPPORTED_DTYPES if dtype_mode == 'floats' else (dtypes.int64, dtypes.int32, dtypes.int16)
    for vcount in vcounts:
      for val in vals:
        for _dtype in _dtypes:
          dtype: DType = _dtype.vec(vcount)
          d = UOp.const(dtype, val)
          yield d, dtype


  def test_preserves_vec(self):
    for d, dtype in self._get_inputs(dtype_mode='floats'):
      self._check_all_uops_vectorized(payne_hanek_reduction(d), dtype.vcount)
      self._check_all_uops_vectorized(cody_waite_reduction(d), dtype.vcount)
      self._check_all_uops_vectorized(xpow(d, d), dtype.vcount)

  def test_preserves_vectorization(self):
    # verify that when given a vectorized (or scalar) input, the function returns a vectorized (or scalar) output
    for (d, dtype), (e, _) in zip(self._get_inputs(dtype_mode='floats'), self._get_inputs(dtype_mode='ints')):
      self._check_all_uops_vectorized(rintk(d), dtype.vcount)
      self._check_all_uops_vectorized(ilogb2k(d), dtype.vcount)
      self._check_all_uops_vectorized(frexp(d), dtype.vcount)
      self._check_all_uops_vectorized(sin_poly(d), dtype.vcount)
      self._check_all_uops_vectorized(xsin(d), dtype.vcount)
      self._check_all_uops_vectorized(xexp2(d), dtype.vcount)
      self._check_all_uops_vectorized(xlog2(d), dtype.vcount)
      self._check_all_uops_vectorized(cody_waite_reduction(d), dtype.vcount)
      self._check_all_uops_vectorized(payne_hanek_reduction(d), dtype.vcount)

      self._check_all_uops_vectorized(ldexp3k(d, d), dtype.vcount)
      self._check_all_uops_vectorized(sin_poly_large(d, d), dtype.vcount)
      self._check_all_uops_vectorized(sin_poly_small(d, d), dtype.vcount)
      self._check_all_uops_vectorized(xpow(d, d), dtype.vcount)

      self._check_all_uops_vectorized(ldexp2k(d, e), dtype.vcount)
      self._check_all_uops_vectorized(pow2if(e, d.dtype), dtype.vcount)
      self._check_all_uops_vectorized(trig_poly(d, [0.1], [0.1]), dtype.vcount)
      self._check_all_uops_vectorized(_lazy_map_numbers(d, d.const_like(0.0), d.const_like(0.0), d.const_like(0.0), d), dtype.vcount)
      # to get integer value from e, use e._eval((e.dtype,), int) instead of int(e) because e is vectorized
      self._check_all_uops_vectorized(shl(d, e._eval((e.dtype,), int)), dtype.vcount)
      self._check_all_uops_vectorized(shr(d, e._eval((e.dtype,), int)), dtype.vcount)
      self._check_all_uops_vectorized(_ifand(d, e._eval((e.dtype,), int)), dtype.vcount)

  # def test_dtype_cases(self):
  #   in16 = UOp.const(dtypes.float16.vec(5), 0.1)
  #   in32 = UOp.const(dtypes.float32.vec(5), 0.1)
  #   in64 = UOp.const(dtypes.float64.vec(5), 0.1)

  #   # test trig_poly
  #   coeff32, coeff64 = [0.1], [0.2]
  #   out32, out64 = trig_poly(in32, coeff32, coeff64), trig_poly(in64, coeff32, coeff64)
  #   assert out32 != out64
  #   out16, out32, out64 = cody_waite_reduction(in16), cody_waite_reduction(in32), cody_waite_reduction(in64)
  #   ic(out16, out32, out64)

  def test_dtype_cases(self):
    in_scalar, in_vec = UOp.const(dtypes.float64, 0.1), UOp.const(dtypes.float64.vec(2), 0.1)

    coeff32, coeff64 = [0.1], [0.2]
    assert eval_uop(trig_poly(in_scalar, coeff32, coeff64)), eval_uop(trig_poly(in_vec, coeff32, coeff64))



if __name__ == '__main__':
  unittest.main()
