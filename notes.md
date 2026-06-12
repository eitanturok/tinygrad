What do we need to support multiple sequences?

A single sequence needs:
1. `v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)`
2. `v_toks = UOp.variable("toks", 1, chunk_size)`
3. `temperature`
4. `tokens`
5. `_cached_tokens`

We want to support multiple sequences of varying lengths by using padding. how does that change things?
1. We need different attn mask patterns to indicate sequence
2. `assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))` if each seq has it's own kv cache then we need to do a for loop over the `self.cache_kv` batch dim and use a different start_pos for each one

Questions:
1. two support BS>1 with varying seq len we can 1) pad or 2) flatten the tokens across all sequences to 1D and track the cum length to indicate where each seq ends. which should we do? padding cause it is simpler and works with the current Tensor.scaled_dot_product_attention() implementation via setting a custom attention mask
2. so each seq has a few lengths:
    a. `start_pos` the length of tokens already processed so far
    b. `T` the length the new tokens we still have left to process (in decode = 1)
    c. `pad_length` - we have to pad all sequences to the same length
    d. `sp` -
    how does padding effect T or start_pos?
    In prefill, `start_pos` is 0 and can the T varies or be the same? If we want T's to be the same, then we pad before passing into the model. And then we unpad at the end of generate to send to client.
    But then for decode, we really want different start_positions based on the different prompt lengths. if we have different `start_pos`s, then what do we do? Where do all the problems lie?
    a. rope
    b. kv cache
    c. mask
    can we do something as dumb and simple for now as using a list of `start_pos` and just creating the different ropes, kv cache and masks that way? we can unpad the `out` tensor
    `out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()`

3. For the JIT to work we need to run model on the same exact input and output tensors `output_ids = model(input_ids)` and when we have different iterations or seqs or batches, we need to simply write the new tokens into the `input_ids` and `output_ids` tensors. That's what this line does
`out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()`

But I think we don't have clear division between these concepts. During prefill we pass in `t` and during decode we pass in `out`. However, for both of them `out` is the output tensor. I guess this makes sense because the output of prefill becomes the input for decode. This is a more condensed way of writing this.

However, I think we might need to explicitly separate the prefill and decode buffers if we have sequences of different lengths where prefill and decode might be different between them.

```py
# chunked prefill
# this should run math.ceil(max([len(seq) for seq in seqs]) / chunk_size)
prefill_input = tensor.zeros(max_seqs, chunk_size)
out = tensor.zeros(max_seqs)

for n_chunk in range(math.ceil(max([len(seq) for seq in seqs]) / chunk_size))
    # for the JIT to work, we must assign new values to the SAME tensor. use assign?
    prefill_input.zero() # add padding
    for i, seq, is_prefill in enumerate(seqs):
        if n_chunk*chunk_size < len(seq):
            prefill_input[i, :min(len(seq), n_chunk*chunk_size)] = seq[:min(len(seq), n_chunk*chunk_size)]
    # forward pass
    out = model(prefill_input).realize()

    # save chunk to tokens
    for i, seq in enumerate(seqs):
        if n_chunk*chunk_size < len(seq):
        tokens[i] += out[i, :min(len(seq), n_chunk*chunk_size)].tolist() # remove padding

# decode
out = model(out).realize()
for i in range(seqs): tokens[i].append(out[i])
```

The big problem here is that for the JIT to work in tinygrad, we cannot use regular ints, we must idnex with symbolic ints.

I think we might need a single symbolic `start_pos` that keeps on going until the longest sequence finished prefill
And then we need int start_pos that allows us to add and undo the padding


Okay this currently seems to owrk which doesn't make sense because I didn't change ROPE, atten mask, kv cache accessing, and the starT_pos is only based on the cache_tokens from seq 0.
If we make the different prompt lengths so different that one takes up two things of chunked preill and the other takes only one iteration then I think it'll break.
If we swithc so first seq is longest instead of second, then it will break too.
How doe sthis even work now?


I want to print out the attn mask to make sure it handles multiple batches correctly.
But
```py
print(mask.numpy())
```
gives
```py
  File "/Users/eitanturok/tinygrad/tinygrad/tensor.py", line 351, in numpy
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
AssertionError: no data if shape is symbolic, self.shape=(1, 1, UOp(Ops.BIND, dtypes.weakint, arg=None, src=(
  UOp(Ops.DEFINE_VAR, dtypes.weakint, arg=('toks', 1, 32), src=()),
  UOp(Ops.CONST, dtypes.weakint, arg=17, src=()),)), UOp(Ops.ADD, dtypes.weakint, arg=None, src=(
  UOp(Ops.BIND, dtypes.weakint, arg=None, src=(
    UOp(Ops.DEFINE_VAR, dtypes.weakint, arg=('start_pos', 0, 4095), src=()),
    UOp(Ops.CONST, dtypes.weakint, arg=0, src=()),)),
  UOp(Ops.BIND, dtypes.weakint, arg=None, src=(
    UOp(Ops.DEFINE_VAR, dtypes.weakint, arg=('toks', 1, 32), src=()),
    UOp(Ops.CONST, dtypes.weakint, arg=17, src=()),)),)))
```
because `mask` is defined as
```py
mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
    if resolve(T != 1) else None
```
How do I resolve the dimensions of mask before calling numpy?
