from tinygrad import Tensor, dtypes
from icecream import install
install()

def make_square_mask(shape, mask_size) -> Tensor:
    BS, _, H, W = shape
    low_x = Tensor.randint(BS, low=0, high=W-mask_size).reshape(BS,1,1,1)
    low_y = Tensor.randint(BS, low=0, high=H-mask_size).reshape(BS,1,1,1)
    idx_x = Tensor.arange(W, dtype=dtypes.int32).reshape((1,1,1,W))
    idx_y = Tensor.arange(H, dtype=dtypes.int32).reshape((1,1,H,1))
    return (idx_x >= low_x) * (idx_x < (low_x + mask_size)) * (idx_y >= low_y) * (idx_y < (low_y + mask_size))

def random_crop(X:Tensor, crop_size=32):
    mask = make_square_mask(X.shape, crop_size)
    mask = mask.expand((-1,3,-1,-1))
    X_cropped = Tensor(X.numpy()[mask.numpy()])
    X_cropped = Tensor(mask).reshape((-1, 3, crop_size, crop_size))
    return X_cropped

# def random_crop2(X:Tensor, crop_size:int=16) -> Tensor:
#     BS, c, H, W = X.shape
#     low_x = Tensor.randint(BS, low=0, high=W-crop_size).reshape(BS,1,1,1)
#     low_y = Tensor.randint(BS, low=0, high=H-crop_size).reshape(BS,1,1,1)
#     return X.shrink((None, None, None, (low_x, min(W, low_x+crop_size))))

def cutmix(X:Tensor, Y:Tensor, mask_size=3):
    # fill the square with randomly selected images from the same batch
    mask = make_square_mask(X.shape, mask_size)
    order = list(range(0, X.shape[0]))
    random.shuffle(order)
    X_patch = Tensor(X.numpy()[order], device=X.device, dtype=X.dtype)
    Y_patch = Tensor(Y.numpy()[order], device=Y.device, dtype=Y.dtype)
    X_cutmix = mask.where(X_patch, X)
    mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
    Y_cutmix = mix_portion * Y_patch + (1. - mix_portion) * Y
    return X_cutmix, Y_cutmix

def cutmix_2(X:Tensor, Y:Tensor, mask_size=3):
    # fill the square with randomly selected images from the same batch
    mask = make_square_mask(X.shape, mask_size)
    order = list(range(0, X.shape[0]))
    random.shuffle(order)
    X_patch = X.numpy()[order], device=X.device, dtype=X.dtype)
    Y_patch = Tensor(Y.numpy()[order], device=Y.device, dtype=Y.dtype)
    X_cutmix = X.masked_fill(mask, X_patch)
    mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
    Y_cutmix = mix_portion * Y_patch + (1. - mix_portion) * Y
    return X_cutmix, Y_cutmix


def random_crop(X:Tensor, crop_size:int=16) -> Tensor:
    c = x.shape[1]
    mask = make_square_mask(X.shape, crop_size)
    mask = mask.expand((-1, c, -1, -1))
    X_cropped = X.masked_select(mask).reshape((-1, c, crop_size, crop_size))
    return X_cropped

def main():
    bs, c, h, w = 1, 3, 28, 28
    X = Tensor.randint(bs, c, h, w)
    out = random_crop2(X)
    ic(X.shape, out.shape, out.numpy())

if __name__ == '__main__':
    main()
