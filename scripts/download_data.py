"""Download PneumoniaMNIST from the official Zenodo record and verify its MD5.

Usage: python -m scripts.download_data --size 28 [--root data]
"""

import argparse
import os
import sys
import urllib.request

from medxai.data import OFFICIAL_MD5, ZENODO_URL, md5sum, npz_filename


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=28, choices=sorted(OFFICIAL_MD5))
    ap.add_argument("--root", default="data")
    a = ap.parse_args(argv)
    os.makedirs(a.root, exist_ok=True)
    dest = os.path.join(a.root, npz_filename(a.size))
    if not os.path.exists(dest):
        url = ZENODO_URL.format(flag="" if a.size == 28 else f"_{a.size}")
        print(f"downloading {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest + ".part")
        except Exception as e:
            print(f"download failed: {e}\nDownload manually from {url} and place it at {dest}.",
                  file=sys.stderr)
            return 1
        os.replace(dest + ".part", dest)
    got = md5sum(dest)
    if got != OFFICIAL_MD5[a.size]:
        print(f"MD5 mismatch for {dest}: {got} != {OFFICIAL_MD5[a.size]}", file=sys.stderr)
        return 2
    print(f"OK {dest} md5={got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
