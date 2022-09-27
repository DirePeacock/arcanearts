from .main import main

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("prompt", nargs="+", action="store", default="Cute shiba inu dog", type=str)
args = parser.parse_args(sys.argv[1:])

main(args)
