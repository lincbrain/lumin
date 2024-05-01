import os
import glob


def glob_imgs(path: str):
    imgs = []
    for ext in ["*.png"]:
        imgs.extend(glob.glob(os.path.join(path, ext)))
    return imgs
