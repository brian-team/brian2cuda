#!/usr/bin/env python
import os
from glob import glob


def update_benchmark_readme(directory=None):
    if directory is None:
        directory = os.path.dirname(os.path.realpath(__file__))

    lines = []
    for readme in sorted(glob(directory + '/*/README.md'), reverse=True):
        d = os.path.dirname(readme)
        lines.append("[{d}]({d})\n".format(d=os.path.basename(os.path.normpath(d))))

    readme_md = '\n'.join(lines)

    readme_filename = os.path.join(directory,  "README.md")
    with open(readme_filename, "w") as readme_file:
        readme_file.write(readme_md)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description='Update README in results dir with links to benchmark runs'
    )
    parser.add_argument('-d', '--directory', default=None,
                        help="Directory where results will be stored")
    args = parser.parse_args()

    update_benchmark_readme(directory=args.directory)
