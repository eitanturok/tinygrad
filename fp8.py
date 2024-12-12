from tinygrad import Tensor, dtypes
from tinygrad.device import Device
from icecream import ic, install

"""

PTX=1 python fp8.py
t = Tensor([1, 2, 3], dtype=dtypes.fp8_e4m3, device="CUDA")
KeyError: dtypes.fp8_e4m3

CUDA=1 python fp8.py
t = Tensor([1, 2, 3], dtype=dtypes.fp8_e4m3, device="CUDA")
AttributeError: module 'tinygrad.runtime.autogen.nvrtc' has no attribute 'nvrtcVersion'
"""
install()

devices = [d for d in Device.get_available_devices()]
t = Tensor([1, 2, 3], dtype=dtypes.fp8_e4m3, device="CUDA")
ic(t.realize())
ic(t.numpy())