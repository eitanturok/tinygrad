# https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#torchdynamo-and-fx-graphs
import torch
import os, time, unittest
import torch._dynamo
from extra.torch_backend.backend import unwrap, wrap

from torch._dynamo.backends.registry import register_backend
from torch._functorch.aot_autograd import aot_module_simplified

from tinygrad import Tensor, TinyJit

from icecream import install
install()

@register_backend
def tiny(gm:torch.fx.GraphModule, sample_inputs):
  def my_compiler(gm:torch.fx.GraphModule, sample_inputs):
    # TODO: the jit should capture the graph directly, not need three runs. this is a planned tinygrad refactor after becomes_map
    @TinyJit
    def tiny_function(*args:Tensor):
      outs = gm(*[wrap(x) for x in args])
      for x in outs: unwrap(x).realize()
      return outs
    # TODO: this should be able to pass in .tiny() Tensors, not need to convert them. it tries to access Storage if you pass in.
    def torch_function(*args:torch.Tensor):
      return tiny_function(*[unwrap(x.tiny()) for x in args])
    return torch_function
  gm.graph.print_tabular()
  return aot_module_simplified(gm, sample_inputs, decompositions={}, fw_compiler=my_compiler)

class TestTorchCompile(unittest.TestCase):
  def setUp(self):
    import tinygrad.frontend.torch
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCH_DEBUG"] = "1"

  def _run_fxn(self, f, *args):
    torch._dynamo.reset() # remove?
    st = time.perf_counter_ns()
    ret = f(*args)
    et = time.perf_counter_ns() - st
    return ret, et

  def test_correctness(self):
    def fxn(x): return x.sum()
    t = torch.randn(4, 4)

    ret1, _ = self._run_fxn(torch.compile(fxn), t)
    ret2, _ = self._run_fxn(torch.compile(fxn), t.to("tiny"))
    ret3,  = self._run_fxn(torch.compile(fxn, backend="tiny"), t)
    ret4, _ = self._run_fxn(torch.compile(fxn, backend="tiny"), t.to("tiny"))

    expected = fxn(t.numpy())
    assert ret1.numpy() == ret2.numpy() == ret3.numpy() == expected

  def test_correctness_old(self):
    def fxn(x:torch.Tensor): return x.sum()
    x = torch.randn(4, 4)

    # tensor backend = torch, compiler backend = torch
    torch_outputs, _ = self.run_fxn(torch.compile(fxn), x)

    # tensor backend = torch, compiler backend = tiny
    torch._dynamo.reset() # remove?
    tiny_outputs, _ = self.run_fxn(torch.compile(fxn, backend=tiny), x)

    # tensor backend = tiny, compiler backend = tiny
    torch._dynamo.reset() # remove?
    tiny_outputs, _ = self.run_fxn(torch.compile(fxn, backend="tiny"), x.to("tiny"))

    # assert
    assert [out.item() for out in torch_outputs] == [out.item() for out in tiny_outputs]

  def test_speed(self):
    def fxn(x:torch.Tensor): return x.sum()
    x = torch.randn(10, 10)

    _, torch_times = self.run_fxn(torch.compile(fxn), x)
    torch._dynamo.reset() # reset since we are using a different backend
    _, tiny_times = self.run_fxn(torch.compile(fxn, backend="tiny"), x.to("tiny"))
    assert torch_times[0] == max(torch_times)
    assert tiny_times[0] == max(tiny_times)

if __name__ == "__main__":
  unittest.main()
