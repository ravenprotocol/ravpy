import os
import tenseal as ts
from ravop.config import BASE_DIR


def dump_context(context, cid):
    filename = "context_{}.txt".format(cid)
    fpath = os.path.join(BASE_DIR, filename)
    with open(fpath, "wb") as f:
        f.write(context.serialize())

    return filename, fpath


def load_context(file_path):
    with open(file_path, "rb") as f:
        return ts.context_from(f.read())
