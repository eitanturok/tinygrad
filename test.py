import sys
from typing import Callable, Optional
from tinygrad import Tensor, Device, TinyJit
from tinygrad.helpers import GlobalCounters, Timing, Profiling, tqdm, PROFILE
from tinygrad.nn.state import get_parameters
from examples.gpt2 import GPT2
from examples.llama3 import build_transformer, Tokenizer, fetch_weights
from structured_generation import RegexLogitsProcessor, JSONLogitsProcessor
from icecream import install
install()

from pydantic import BaseModel

"""
todo:
1. JIT=1
1. skip - multiple batches
2. done - llama model
3. done - recursion depth error
4. done - print stats
"""

"""
len(next_tokens_mask) == 1003
the problem is this line:
all_indices = Tensor.cat(*[Tensor.full((len(tokens),), idx, dtype=dtypes.int32) for tokens, idx in data]).to(logits.device)
we have len(n.toposort())==7026
"""


def encode_role(tokenizer, role: str):
    return [tokenizer.special_tokens["<|start_header_id|>"]] + tokenizer.encode(role) + [tokenizer.special_tokens["<|end_header_id|>"]] + tokenizer.encode("\n\n")
def encode_message(tokenizer, role: str, content: str):
    return encode_role(tokenizer, role) + tokenizer.encode(content.strip()) + [tokenizer.special_tokens["<|eot_id|>"]]

last_seen_toks = []
@TinyJit
def prefill(model, toks, temperature, start_pos:int=0, device:Optional[str]=None):
    global last_seen_toks

    # we can skip part of the prompt if it is the same as last and start_pos=0
    if start_pos == 0:
        for i, (a, b) in enumerate(zip(toks, last_seen_toks)):
            if a != b: break
        else: i = min(len(toks), len(last_seen_toks))
        start_pos += i
        last_seen_toks = toks
        toks = toks[i:]

    # prefill the model
    for tok in tqdm(toks, desc="Prefill"):
        GlobalCounters.reset()
        model(Tensor([[tok]], device=device), start_pos, temperature).realize()
        start_pos += 1
    return start_pos

def print_stats(et): return f", {GlobalCounters.mem_used/1e9:5.2f} GB ram, {GlobalCounters.global_mem/1e9:5.2f} GB global mem"

@Timing(f"Total Generation:\t", on_exit=print_stats)
def generate(model, tokenizer, prompt, device=None, temperature=0.0, max_length=30, batch_size:int=1, logits_processor:Callable=lambda x,y:y, profile=False, timing=True):
    param_bytes = sum(x.lazydata.size * x.dtype.itemsize for x in get_parameters(model))
    toks = [tokenizer.bos_id] + encode_message(tokenizer, "user", prompt) + encode_role(tokenizer, "assistant")
    start_pos = prefill(model, toks[:-1], temperature)
    last_tok = toks[-1]
    generated = ""
    print(f"\nPrompt:\n{prompt}\n", flush=True)

    max_gen_toks =  max_length - len(toks)
    for i in range(max_gen_toks):
        GlobalCounters.reset()
        with Profiling(enabled=PROFILE):
            with Timing(f"Generate token {i:03d}:\t", enabled=timing, on_exit=print_stats):
                tok_tensor = model(Tensor([[last_tok]], device=device), start_pos, temperature, logits_processor=logits_processor)
                tok = tok_tensor.item()
        start_pos += 1
        last_tok = tok
        if tok in tokenizer.stop_tokens: break
        generated += tokenizer.decode([tok])
        print(prompt+generated, end="\n\n", flush=True)
    print(prompt+generated, end="\n\n", flush=True)
    return generated

class OutlinesTokenizer(Tokenizer):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.vocabulary = self.model._mergeable_ranks
        self.eos_token = "<|end_of_text|>"
        self.eos_token_id = self.special_tokens[self.eos_token]
    def convert_token_to_string(self, token:str) -> str: return token

def main():

    seed = 42
    Tensor.manual_seed(seed)
    print(f"{seed=}")

    device = Device.DEFAULT
    print(f"{device=}")

    # model_size = "gpt2"
    # gpt2 = GPT2.build(model_size)
    # model, tokenizer = gpt2.model, gpt2.tokenizer

    model_size = "1B"
    weights_path = fetch_weights(model_size)
    model = build_transformer(weights_path, model_size, device=device)
    tokenizer = OutlinesTokenizer(str(weights_path.parent / "tokenizer.model"))
    print(f'loaded llama-{model_size} weights + tokenizer from {weights_path.parent}')

    ip_address_regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    logits_processor = RegexLogitsProcessor(ip_address_regex, tokenizer, device)

    # class User(BaseModel):
    #     # name: str
    #     # last_name: str
    #     id: int
    # logits_processor = JSONLogitsProcessor(User, tokenizer, device=device)

    prompt = "The secret to the universe is "
    output = generate(model, tokenizer, prompt, device=device, logits_processor=logits_processor)
    print(output)

if __name__ == '__main__':
    main()
