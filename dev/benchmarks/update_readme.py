import os
from glob import glob

def update_benchmark_readme():
    filedir = os.path.dirname(os.path.realpath(__file__))

    lines = []
    for readme in sorted(glob(filedir + '/*/README.md'), reverse=True):
        d = os.path.split(readme)[0]
        lines.append("[{d}]({d})\n".format(d=os.path.basename(os.path.normpath(d))))

    readme_md = '\n'.join(lines)

    with open(filedir + "/README.md", "w") as readme_file:
        readme_file.write(readme_md)

if __name__ == '__main__':
    update_benchmark_readme()
