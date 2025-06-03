from typing import Callable
from tinygrad import Tensor, Device
from tinygrad.helpers import getenv, fetch
from examples.gpt2 import GPT2
from examples.llama3 import build_transformer, Tokenizer, fetch_weights
from structured_generation import RegexLogitsProcessor, JSONLogitsProcessor
from icecream import install
install()

from pydantic import BaseModel

class User(BaseModel):
    name: str
    last_name: str
    id: int

"""
todo:
1. JIT=1
1. multiple batches
2. llama model
3. recursion depth error
"""

def generate(model, tokenizer, prompt, temperature=0.0, max_length=30, batch_size:int=1, logits_processor:Callable=lambda x:x, verbose=False):
    # prompt_tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    prompt_tokens = tokenizer.encode(prompt, allow_special=True)
    toks = [prompt_tokens[:] for _ in range(batch_size)]
    assert (max_new_tokens :=  max_length - len(toks[0])) < getenv("MAX_CONTEXT", 1024), f"{max_new_tokens=} must not exceed the context"
    start_pos = 0
    for _ in range(max_new_tokens):
        new_toks = model(Tensor([x[start_pos:] for x in toks]), start_pos, temperature, logits_processor=logits_processor)
        # if all([tokenizer.eos_token_id == x.item() for x in [new_toks]]): break
        if all([x.item() in tokenizer.stop_tokens for x in [new_toks]]): break
        for i,x in enumerate([new_toks]): toks[i].append(x.item())
        if verbose: print([tokenizer.decode(x) for x in toks])
        start_pos = len(toks[0]) - 1
    return [tokenizer.decode(x) for x in toks]

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
    print(f"using {seed=}")

    device = Device.DEFAULT
    print(f"using {device=}")

    # model_size = "gpt2"
    # gpt2 = GPT2.build(model_size)
    # model, tokenizer = gpt2.model, gpt2.tokenizer

    model_size = "1B"
    weights_path = fetch_weights(model_size)
    print(f'Llama {model_size} weights + tokenizer saved in {weights_path.parent}')
    model = build_transformer(weights_path, model_size, device=device)
    tokenizer = OutlinesTokenizer(str(weights_path.parent / "tokenizer.model"))

    prompt = "The secret to the universe is "
    # ip_address_regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    # logits_processor = RegexLogitsProcessor(ip_address_regex, tokenizer, device)
    logits_processor = JSONLogitsProcessor(User, tokenizer, device=device)
    output = generate(model, tokenizer, prompt, logits_processor=logits_processor, verbose=True)
    ic(output)

if __name__ == '__main__':
    main()
