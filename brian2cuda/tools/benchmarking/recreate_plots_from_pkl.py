
import argparse
parser = argparse.ArgumentParser(description=("Recreate plots from pickled SpeedTest objects. "
                                              "Overwrites already existing plots of same name."))

parser.add_argument('files', nargs='+', type=str,
                    help=(".pkl files to translate. * wildcard supported."))

parser.add_argument('plot_dir', type=str,
                    help=("Where to create the plots."))


args = parser.parse_args()

import matplotlib
matplotlib.use('agg')
import glob
from helpers import plot_from_pkl

for pattern in args.files:
    for pkl in glob.glob(pattern):
        print(f"Plotting from {pkl} ...")
        plot_from_pkl(pkl, args.plot_dir)

