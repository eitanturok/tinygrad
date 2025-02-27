In `expander.py`, we have
```py
(UPat(Ops.VECTORIZE, src=UPat(Ops.DEFINE_ACC, name="acc"), name="v"),
    lambda acc,v: acc.replace(dtype=v.dtype, src=(acc.src[0].broadcast(v.dtype.count),)+acc.src[1:])),
```
So if I can make there be a `Ops.VECTORIZE` in front of `Ops.DEFINE_ACC`, then I will have everything vectorized, right?

I should manually create an example to check this!
