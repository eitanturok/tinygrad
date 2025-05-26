from tinygrad import Tensor, Device
from tinygrad.helpers import trange, getenv
from examples.gpt2 import GPT2
from icecream import install
install()

def generate(model, tokenizer, prompt, temperature=0.0, max_length=10, batch_size:int=1):
    prompt_tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    toks = [prompt_tokens[:] for _ in range(batch_size)]
    assert (max_new_tokens :=  max_length - len(toks[0])) < getenv("MAX_CONTEXT", 1024), f"{max_new_tokens=} must not exceed the context"
    start_pos = 0
    for _ in trange(max_new_tokens+1):
        new_toks = model(Tensor([x[start_pos:] for x in toks]), start_pos, temperature).tolist()
        for i,x in enumerate([new_toks]): toks[i].append(x)
        start_pos = len(toks[0]) - 1
    return [tokenizer.decode(x) for x in toks]

def main():
    prompt = "The secret to the universe is"
    model_size = "gpt2"
    seed = 42

    print(f"using {Device.DEFAULT} backend")
    Tensor.no_grad = True
    Tensor.manual_seed(seed)
    gpt2 = GPT2.build(model_size)
    model, tokenizer = gpt2.model, gpt2.tokenizer

    output = generate(model, tokenizer, prompt)
    ic(output)

if __name__ == '__main__':
    main()
