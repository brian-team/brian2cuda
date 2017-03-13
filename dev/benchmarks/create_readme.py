import sys
import os
import glob
from subprocess import check_output

readme_tmpl = '''
# Benchmark results from {date}
## Description:
{description}


## Last git log:
```
{git_log}
```
There is also a `git diff` saved in the current directory.

## Results
{results}
'''

result_tmpl = '''
### {name}
{plots}
{nvprof}
'''

nvprof_tmpl = '''
<details><summary>Examplary `nvprof` results for **{config}**</summary><p>
Profile summary for `N = {N}`:

```
{log}
```

</p></details>
'''

def create_readme(result_dir, description=''):
    test_names = set()
    for plot_file in sorted(glob.glob(result_dir + "/plots/*")):
        name = os.path.splitext(os.path.basename(plot_file))[0].split('_')[2]
        test_names.add(name)
    test_names = sorted(test_names)
    
    result_md = []
    for name in test_names:
        plots_md = []
        for plot_file in sorted(glob.glob(result_dir+ "/plots/speed_test_{}_*".format(name))):
            plot_path = os.path.join(*(plot_file.split(os.path.sep)[1:]))
            md = "![]({plot})".format(plot=plot_path)
            plots_md.append(md)
        profile_md = []
        for nvprof_file in sorted(glob.glob(result_dir + "/nvprof/nvprof_{}_*".format(name))):
            nvprof_filename = os.path.splitext(os.path.basename(nvprof_file))[0]
            config = nvprof_filename.split('_')[2]
            N = nvprof_filename.split('_')[3].split('.')[0]
            with open(nvprof_file, 'r') as infile:
                log = infile.read()
            md = nvprof_tmpl.format(config=config, N=N, log=log)
            profile_md.append(md)
        md = result_tmpl.format(name=name,
                                plots='\n'.join(plots_md),
                                nvprof='\n'.join(profile_md))
        result_md.append(md)
    
    date_md = '.'.join(os.path.basename(result_dir).split('_')[1:4][::-1])
    git_log = check_output('git log -1'.split()).decode()
    readme_md = readme_tmpl.format(date=date_md, description=description, git_log=git_log,
                                   results='\n***\n'.join(result_md))
    
    
    with open(result_dir + "/README.md", "w") as readme_file:
        readme_file.write(readme_md)

if __name__ == '__main__':

    assert len(sys.argv)== 2, 'Provide result directory as only command line argument! Got {}'.format(len(sys.argv)-1)
    result_dir = sys.argv[1]
    create_readme(result_dir)
