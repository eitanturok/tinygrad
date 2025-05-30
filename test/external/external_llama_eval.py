# from lm_eval.base import BaseLM
from lm_eval.api.model import TemplateLM, LM
from lm_eval import evaluator, tasks
import torch, json, argparse
from pathlib import Path

from tinygrad import Device, nn, Tensor
from tinygrad.helpers import tqdm, _ensure_downloads_dir
from examples.llama import LLaMa
from examples.llama3 import download_weights
# from examples.gpt2 import GPT2

from icecream import install
install()

class LLaMaAdaptor(LM):
  def __init__(
    self,
    model_size="7B",
    model_gen="1",
    device=None,
    quantize=False,
    batch_size=1,
    max_batch_size=1,
    do_sample=False,
    temperature=1.0,
    checkpoint_path="",
    tokenizer_path="",
    max_length=128,
  ):
    super().__init__()
    ic(quantize)

    if batch_size is None: batch_size = 1
    if quantize is False: quantize = None
    self.do_sample = do_sample
    self.temperature = temperature
    self._device = device

    assert isinstance(model_gen, str)
    assert isinstance(model_size, str)
    assert isinstance(batch_size, int)

    # self.llama = GPT2.build()
    self.llama = LLaMa.build(checkpoint_path, tokenizer_path, model_gen, model_size, quantize, device)

  @classmethod
  def create_from_arg_string(cls, arg_string, additional_config=None):
    kwargs = {el.split("=")[0]: el.split("=")[1] for el in arg_string.split(",")}
    return cls(**kwargs, **additional_config)

  @property
  def eot_token_id(self):
    # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
    return self.llama.tokenizer.eos_id()

  @property
  def max_length(self):
    return 1024

  @property
  def max_gen_toks(self):
    return 256

  @property
  def batch_size(self):
    return 1

  @property
  def device(self):
    return self._device

  def tok_encode(self, string: str):
    return [self.llama.tokenizer.bos_id()] + self.llama.tokenizer.encode(string)

  def tok_decode(self, tokens):
    return self.llama.tokenizer.decode(tokens)

  def _model_call(self, inps):
    return torch.Tensor(self.llama.model(Tensor(inps.numpy()), 0).numpy())

  def greedy_until(self, prompt:str, until, max_length, temperature):
    # only used in old eval script
    import numpy as np
    toks = [self.llama.tokenizer.bos_id()] + self.llama.tokenizer.encode(prompt)
    start_pos = 0
    for i in range(max_length):
      output = self.llama.generate(prompt, max_length, temperature)
      probs = self.llama.model(Tensor([toks[start_pos:]]), start_pos, temperature).realize()
      probs_np = probs.numpy()
      tok = int(np.random.choice(len(probs_np), p=probs_np))
      start_pos = len(toks)
      toks.append(tok)

      if tok == self.tokenizer.eos_id(): break
      output = self.tokenizer.decode(toks)
      for s in until:
        if output.endswith(s): return output[0:-len(s)]
    return output

  def _model_generate(self, context, max_length, eos_token_id):
    raise NotImplementedError()

  def loglikelihood(self, requests, disable_tqdm: bool = False):
    raise NotImplementedError("No support for logits.")

  def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
    raise NotImplementedError("No support for logits.")

  def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
    results = []
    for request in tqdm(requests, disable=disable_tqdm):
      prompt, until, temperature = request.arguments[0].strip(), request.arguments[1]['until'],  request.arguments[1]['temperature']
      result = self.greedy_until(prompt, until, self.max_length, temperature)
      results.append(result)
    return results

if __name__ == '__main__':
  print(f"using {Device.DEFAULT} backend")

  parser = argparse.ArgumentParser(description='Run LLaMA evals in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--size', type=str, default="7B", help="Size of model to use [7B, 13B, 30B, 65B] for Gen 1, [7B, 13B] for Gen 2")
  parser.add_argument('--gen', default="1", help="Generation of the model to use [1, 2]")
  parser.add_argument('--quantize', action='store_true', help="Quantize the weights to int8 in memory")
  parser.add_argument('--eval', type=str, default="gsm8k", help="Run in evaluation mode")
  parser.add_argument('--limit', type=int, default=None, help="Limit tests in eval")
  parser.add_argument('--weights', type=str, default="./weights/LLaMa/", help="Location of the weights")
  parser.add_argument('--tokenizer', type=str, default="./weights/LLaMa/tokenizer.model", help="Location of the tokenizer")
  args = parser.parse_args()

  if args.gen == "3":
    index_path = download_weights(args.size, args.weights)
    args.weights = index_path
    args.tokenizer = index_path.parent / "tokenizer.model"
  adaptor = LLaMaAdaptor(model_gen=args.gen, model_size=args.size, quantize=args.quantize,
                         checkpoint_path=args.weights, tokenizer_path=args.tokenizer, device="cpu")
  results = evaluator.evaluate(adaptor, tasks.get_task_dict(args.eval.split(",")), cache_requests=False, limit=args.limit)
  print(json.dumps(results, indent=2))
