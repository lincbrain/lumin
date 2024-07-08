import math
import numcodecs


def center_affine(affine, shape):
    """
    perform a centered-affine crop on a volume
    """
    if len(shape) == 2:
        shape = [*shape, 1]
    shape = np.asarray(shape)
    affine[:3, -1] = -0.5 * affine[:3, :3] @ (shape - 1)
    return affine


def ceildiv(x, y):
    """
    ceiling division
    """
    return int(math.ceil(x / y))


def floordiv(x, y):
    """
    flooring division
    """
    return int(math.floor(x, y))


def compute_new_shape(prev_shape):
    """
    compute new shape for downsampling
    """
    return [max(1, x // 2) for x in prev_shape]


def make_compressor(name, **args):
    """
    return an image compressor object
    """
    if not isinstance(name, str):
        return name
    name = name.lower()
    if name == "blosc":
        Compressor = numcodecs.Blosc
    elif name == "zlib":
        Compressor = numcodecs.Zlib
    else:
        raise ValueError(f"Unknown compressor: {name}")
    return Compressor(**args)
