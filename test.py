from typing import Callable
from tinygrad import Tensor, Device
from tinygrad.helpers import trange, getenv
from examples.gpt2 import GPT2
from structured_generation import RegexLogitsProcessor
from icecream import install
install()

def generate(model, tokenizer, prompt, temperature=0.0, max_length=10, batch_size:int=1, logits_processor:Callable=lambda x:x):
    prompt_tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    toks = [prompt_tokens[:] for _ in range(batch_size)]
    assert (max_new_tokens :=  max_length - len(toks[0])) < getenv("MAX_CONTEXT", 1024), f"{max_new_tokens=} must not exceed the context"
    start_pos = 0
    for _ in trange(max_new_tokens):
        new_toks = model(Tensor([x[start_pos:] for x in toks]), start_pos, temperature, logits_processor=logits_processor)
        for i,x in enumerate(new_toks): toks[i].append(x.item())
        start_pos = len(toks[0]) - 1
    return [tokenizer.decode(x) for x in toks]

def main():
    prompt = "The secret to the universe is"
    model_size = "gpt2"
    device = Device.DEFAULT
    seed = 42

    print(f"using {device} backend")
    Tensor.no_grad = True
    Tensor.manual_seed(seed)
    gpt2 = GPT2.build(model_size)
    model, tokenizer = gpt2.model, gpt2.tokenizer

    logits_processor = RegexLogitsProcessor("\d{2}", tokenizer, device)
    output = generate(model, tokenizer, prompt, logits_processor=logits_processor)
    ic(output)

if __name__ == '__main__':
    main()
