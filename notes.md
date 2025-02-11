When we call `UOp.broadcast()` this creates a `Ops.VECTORIZE`.

In `rewriter.py`'s `expander` PatternMatcher we have:
```py
(UPat(Ops.VECTORIZE, src=UPat(Ops.DEFINE_ACC, name="acc"), name="v"),
lambda acc,v: acc.replace(dtype=v.dtype, src=(acc.src[0].broadcast(v.dtype.count),)+acc.src[1:]))
```
which says we vectorize all `DEFINE_ACC` Ops.

Where does `Ops.DEFINE_ACC` come from?

In `lowerer.py`'s `lower_reduce_axis` we have
```py
# create ACC and assign
acc = UOp(Ops.DEFINE_ACC, x.dtype, (x.const_like(identity_element(alu_op, x.dtype.scalar())),) + tuple(reduce_range), (ctx.acc_num,))
```
which is part of the `pm_lowerer` `PatternMatcher`
```py
(UPat(Ops.REDUCE_AXIS, name="x"), lower_reduce_axis),
```
So we need to have an operation with a `REDUCE_AXIS`, something like summing along a particular axis.

Using `VIZ=1`, the code
```py
t1 = Tensor([3, 4, 5]).sum()
print(t1.tolist())
```
result in `Ops.VECTORIZE`! Nice!

So it seems like `Ops.VECTORIZE` is only applied when we are reducing an axis...

But if we try using `log2` as a reduce op instead of sum:
```py
t2 = Tensor([3, 4, 5]).log2()
print(t2.tolist())
```
then we don't see `Ops.VECTORIZE` when we use `VIZ=1`.

This is precisely the issue we have to fix!

Let's take a closer look. After all the rewrite rules, with `sum` as the reduce op we get
![alt text](/images/viz_add_final.png)
but with `log2` as the reduce op we get
![alt text](/images/viz_log2_final.png)

The difference is that `log2` has a `SPECIAL ('lidx0', 3) dtypes.int` operation and `sum` does not. Is this important? What does this mean?


Part of `rewriter.py` includes
```py
devectorize_load_store = PatternMatcher([
  # TODO: add vectorized support to transcendental
  (UPat((Ops.INDEX, Ops.EXP2, Ops.LOG2, Ops.SIN), name="alu"), no_vectorized_alu),
  (UPat((Ops.LOAD, Ops.STORE), name="ls"), no_vectorized_load_store),
])
```
let's try running this instead with allowing transcendental ops to be vectorized
```py
devectorize_load_store = PatternMatcher([
  # TODO: add vectorized support to transcendental
  (UPat((Ops.INDEX), name="alu"), no_vectorized_alu),
  (UPat((Ops.LOAD, Ops.STORE), name="ls"), no_vectorized_load_store),
])
```
After all of the rewriting, this looks the same as without vectorization
![alt text](/images/viz_log2_enable_trans_final.png)
We still have the extra `SPECIAL ('lidx0', 3) dtypes.int` operation.

This makes sense because we have not added the vectorization yet.

The difference between using `log2` and `sum` before the first rewrite is that `sum` counts as a reduce op and `log2` does not.
Sum:
![alt text](/images/viz_sum_yes_reduce.png)
Log2:
![alt text](/images/viz_log2_no_reduce.png)

So the first order of buisness is to make `log2` a reduce op. Let's find where that is done in the code:

In `tensor.py` `sum` is defined as
```py
def sum(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False, acc_dtype:Optional[DTypeLike]=None):
    ret = self.cast(sum_acc_dtype(self.dtype) if acc_dtype is None else acc_dtype)._reduce(Ops.ADD, axis, keepdim)
    return ret.cast(self.dtype) if acc_dtype is None and self.dtype in (dtypes.float16, dtypes.bfloat16) else ret
```
Notice that we have a reduce in the `_reduce(Ops.ADD, axis, keepdim)` part.

What about log2? In `tensor.py` it is defined as
```py
def log2(self):
    return self.cast(least_upper_float(self.dtype))._apply_uop(UOp.log2)
```
Let's make this a reduce op:



The reason why it is tricky to ectorize transencdsal ops like `log2` vs `sum` is that for `sum` you can repeatedly call a[i]+a[i+1] for all i. But we cannot do that for log.

In fact, log is not really a reduce op...
If we have a tensor of 5 numbers and take it's log then we cannot reduce that to a single number. If we take the log we get five different numbers. So we do NOT want to make log a reduce op.

So how do we make a vectorized log2 without it being a reduce op? What are we missing?
