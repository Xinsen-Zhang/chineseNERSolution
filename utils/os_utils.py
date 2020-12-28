# encoding:utf-8

import os


def make_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)
