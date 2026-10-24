"""Download a registered MedMNIST dataset from the official Zenodo record and verify its MD5.

Usage: python -m scripts.download_data --size 28 [--dataset pneumoniamnist] [--root data]
"""

import argparse
import os
import sys
import urllib.request

from medxai.data import DATASETS, dataset_url, md5sum, npz_filename


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224])
    ap.add_argument("--dataset", default="pneumoniamnist", choices=sorted(DATASETS))
    ap.add_argument("--root", default="data")
    a = ap.parse_args(argv)
    expected = DATASETS[a.dataset]["md5"][a.size]
    os.makedirs(a.root, exist_ok=True)
    dest = os.path.join(a.root, npz_filename(a.size, a.dataset))
    if not os.path.exists(dest):
        url = dataset_url(a.dataset, a.size)
        print(f"downloading {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest + ".part")
        except Exception as e:
            print(f"download failed: {e}\nDownload manually from {url} and place it at {dest}.",
                  file=sys.stderr)
            return 1
        os.replace(dest + ".part", dest)
    got = md5sum(dest)
    if got != expected:
        print(f"MD5 mismatch for {dest}: {got} != {expected}", file=sys.stderr)
        return 2
    print(f"OK {dest} md5={got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
