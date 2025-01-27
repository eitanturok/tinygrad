from tinygrad import Tensor, dtypes
from tinygrad.dtype import DType
from tinygrad.device import Device
from test.helpers import rand_for_dtype
from test.test_dtype import get_available_cast_dtypes, _test_bitcast
from icecream import ic, install
import modal
install()

image = modal.Image.debian_slim(python_version="3.11.9").pip_install("icecream", "tinygrad[testing]", gpu="H100").add_local_python_source("tinygrad")
app = modal.App("fp8", image=image)

@app.function(gpu='h100')
def test():
    ic([d for d in Device.get_available_devices()])
    t = Tensor([2, 3, 4], dtype=dtypes.fp8_e4m3)
    ic(t)
    ic(t.cast(dtypes.float))
    ic(t.cast(dtypes.float).realize())
    ic(t.cast(dtypes.float).numpy())
    ic(t.realize())
    ic(t.numpy())

