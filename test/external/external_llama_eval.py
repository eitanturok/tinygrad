# from lm_eval.base import BaseLM
from lm_eval.api.model import TemplateLM, LM
from lm_eval import evaluator, tasks
import torch, json, argparse
from pathlib import Path

from tinygrad import Device, nn, Tensor
from examples.llama import LLaMa
from examples.gpt2 import GPT2

class LLaMaAdaptor(LM):
  def __init__(
    self,
    model_size="7B",
    model_gen=1,
    device="",
    quantize=False,
    batch_size=1,
    max_batch_size=1,
    do_sample=False,
    temperature=1.0,
    checkpoint_path="",
    tokenizer_path="",
  ):
    super().__init__()

    if batch_size is None:
      batch_size = 1
    self.do_sample = do_sample
    self.temperature = temperature
    self._device = device

    assert isinstance(model_gen, int) or isinstance(model_gen, str)
    assert isinstance(model_size, str)
    assert isinstance(batch_size, int)
    assert isinstance(checkpoint_path, str)
    assert isinstance(tokenizer_path, str)

    self.llama = GPT2.build()
    # self.llama = LLaMa.build(checkpoint_path, tokenizer_path, model_gen, model_size, quantize)

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
    Tensor.no_grad = True
    return torch.Tensor(self.llama.model(Tensor(inps.numpy()), 0).numpy())

  def greedy_until(self, requests):
    continuations = []
    for request in requests:
      prompt, until = request[0], request[1]['until']
      output = self.llama.generate(self, prompt, max_length=128, temperature=0.0)
      # output = self.llama.greedy_until(prompt, until, max_length=128, temperature=0.0)
      continuations.append(output[len(prompt):])
    return continuations

  def _model_generate(self, context, max_length, eos_token_id):
    raise NotImplementedError()

  def loglikelihood(self, requests, disable_tqdm: bool = False):
    raise NotImplementedError("No support for logits.")

  def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
    raise NotImplementedError("No support for logits.")

  def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
    raise NotImplementedError("No support for logits.")

def fetch_weights(total_num_weights:int=4) -> dict[str, Tensor]:
  weights: dict[str, Tensor] = {}
  subdir = Path("weights/LLaMA")
  for i in range(1, total_num_weights+1):
    filename = Path(f"model-{i:05d}-of-{total_num_weights:05d}.safetensors")
    weight = Tensor.from_url(f"https://huggingface.co/NousResearch/Meta-Llama-3.1-8B/resolve/main/{filename}", name=filename, subdir=subdir).to(Device.default)
    weights |= nn.state.safe_load(weight)
  return weights

if __name__ == '__main__':
  print(f"using {Device.DEFAULT} backend")

  parser = argparse.ArgumentParser(description='Run LLaMA evals in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--size', type=str, default="7B", help="Size of model to use [7B, 13B, 30B, 65B] for Gen 1, [7B, 13B] for Gen 2")
  parser.add_argument('--gen', type=int, default="1", help="Generation of the model to use [1, 2]")
  parser.add_argument('--quantize', action='store_true', help="Quantize the weights to int8 in memory")
  parser.add_argument('--eval', type=str, default="arc_easy", help="Run in evaluation mode")
  parser.add_argument('--limit', type=int, default=None, help="Limit tests in eval")
  parser.add_argument('--weights', type=str, default="./weights/LLaMa/", help="Location of the weights")
  parser.add_argument('--tokenizer', type=str, default="./weights/LLaMa/tokenizer.model", help="Location of the tokenizer")
  args = parser.parse_args()

  # # run eval and exit
  # fetch_weights(4)
  # args.size = "tiny"
  # args.gen = "1B"
  # LLAMA_SUFFIX = {"1": "", "2": "-2", "3": "-3", "code": "-code", "tiny": "-tiny"}[args.gen]
  # MODEL_PATH = args.model or Path(__file__).parents[1] / f"weights/LLaMA{LLAMA_SUFFIX}/{args.size}"
  # TOKENIZER_PATH = (MODEL_PATH if MODEL_PATH.is_dir() else MODEL_PATH.parent) / "tokenizer.model"


  adaptor = LLaMaAdaptor(model_gen=args.gen, model_size=args.size, quantize=args.quantize,
                         checkpoint_path=args.weights, tokenizer_path=args.tokenizer, device="cpu")
  results = evaluator.evaluate(adaptor, tasks.get_task_dict(args.eval.split(",")), False, 0, args.limit)
  print(json.dumps(results, indent=2))
