import glob
import os


def exists(filepath: str, exts: list[str]):
    for ext in exts:
        e = filepath.split(".")[-1]
        imagefile = os.path.join(filepath.replace(e, ext))
        if os.path.exists(imagefile):
            return True
    return False


def clear(
    dir: str, recursive=True, ext: list[str] = ["png", "jpg", "jpeg", "webp", "avif"]
):
    glob_str = f"{dir}/**/*.json" if recursive else f"{dir}/*.json"
    for file in glob.glob(glob_str, recursive=recursive):
        if not exists(file, ext):
            try:
                os.remove(file)
                os.remove(file.replace(".json", ".txt"))
            except:
                None
