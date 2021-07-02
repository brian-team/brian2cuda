from __future__ import print_function
import argparse
parser = argparse.ArgumentParser(description=("Extract benchmark data from pickled SpeedTest "
                                              "obejct into csv files from pickle file. Overwrites "
                                              "already existing csv files of same name."))

parser.add_argument('files', nargs='+', type=str,
                    help=(".pkl files to translate. * wildcard supported."))

args = parser.parse_args()

import glob
from helpers import translate_pkl_to_csv

for pattern in args.files:
    for pkl in glob.glob(pattern):
        print("Extracting csv files from {}".format(pkl))
        translate_pkl_to_csv(pkl)
