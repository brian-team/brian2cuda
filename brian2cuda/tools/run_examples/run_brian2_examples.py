import os, nose, sys, subprocess, warnings, unittest
import tempfile, pickle
from nose.plugins import Plugin
from nose.plugins.capture import Capture
from nose.plugins.xunit import Xunit
warnings.simplefilter('ignore')


class RunTestCase(unittest.TestCase):
    '''
    A test case that simply executes a python script and notes the execution of
    the script in a file `examples_completed.txt`.
    '''
    def __init__(self, filename, codegen_target, image_dir, prefs_dict):
        unittest.TestCase.__init__(self)
        self.filename = filename
        self.codegen_target = codegen_target
        self.image_dir = image_dir
        for k, v in prefs_dict.items():
            if k == 'core.default_float_dtype':
                try:
                    prefs_dict[k] = v.__name__
                except AttributeError:
                    pass
        self.prefs_dict = prefs_dict

    def id(self):
        # Remove the .py and pretend the dirname is a package and the filename
        # is a class.
        name = os.path.splitext(os.path.split(self.filename)[1])[0]
        pkgname = os.path.split(os.path.split(self.filename)[0])[1]
        return pkgname + '.' + name.replace('.', '_')

    def shortDescription(self):
        return str(self)

    def runTest(self):
        # a simpler version of what the nosepipe plugin achieves:
        # isolate test execution in a subprocess:
        tempfilename = tempfile.mktemp('exception')

        # Catch any exception and save it to a temporary file
        code_string = """
# needed for some scripts that load data
__file__ = '{fname}'
import matplotlib as _mpl
_mpl.use('Agg')
import warnings, traceback, pickle, sys, os, re
warnings.simplefilter('ignore')
try:
    import brian2
    from brian2 import prefs
    from brian2.utils.filetools import ensure_directory_of_file
    # Move to the file's directory for the run, so that it can do relative
    # imports and load files (e.g. figure style files)
    curdir = os.getcwd()
    os.chdir(os.path.dirname(r'{fname}'))
    rel_fname = os.path.basename(r'{fname}')
    sub_str = "from brian2 import *\\n"
    if '{target}' == 'cpp_standalone':
        sub_str += "set_device('cpp_standalone', directory=None)"
        print("Running with cpp_standalone")
    elif '{target}' == 'cuda_standalone':
        sub_str += ("import brian2cuda\\n"
                    "set_device('cuda_standalone', directory=None)\\n")
        print("Running with cuda_standalone")
    for k, v in {prefs_dict}.items():
        sub_str += "prefs['{{k}}'] = {{v}}\\n".format(k=k, v=v)
    with open(rel_fname, "rb") as f:
        file_str = f.read()
    # remove any set device
    file_str = re.sub("set_device\\(.*?\\)", '', file_str)
    # import brian2cuda and set device
    file_str = re.sub("from brian2 import \*", sub_str, file_str)
    #print(file_str)
    exec(compile(file_str, rel_fname, 'exec'))
    os.chdir(curdir)
    brian_dir = os.path.dirname(brian2.__file__)
    example_dir = os.path.join(brian_dir, '../examples')
    for fignum in _mpl.pyplot.get_fignums():
        fname = r'{fname}'
        fname = os.path.relpath(fname, example_dir)
        fname = fname.replace('/', '.').replace('\\\\', '.')
        fname = fname.replace('.py', '.%d.png' % fignum)
        fname = os.path.join('{image_dir}', fname)
        print(fname)
        ensure_directory_of_file(fname)
        _mpl.pyplot.figure(fignum).savefig(fname)
except Exception as ex:
    traceback.print_exc(file=sys.stdout)
    f = open(r'{tempfname}', 'wb')
    pickle.dump(ex, f, -1)
    f.close()
""".format(fname=self.filename,
           tempfname=tempfilename,
           target=self.codegen_target,
           image_dir=self.image_dir,
           prefs_dict=self.prefs_dict)

        #print(code_string)

        args = [sys.executable, '-c',
                code_string]
        # Run the example in a new process and make sure that stdout gets
        # redirected into the capture plugin
        p = subprocess.Popen(args, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        # Write both stdout and stderr to stdout so it gets captured by the
        # Capture plugin
        sys.stdout.write(stdout)
        sys.stdout.write(stderr)
        sys.stdout.flush()

        # Re-raise any exception that occured
        if os.path.exists(tempfilename):
            with open(tempfilename, 'rb') as f:
                ex = pickle.load(f)
            self.successful = False
            raise ex
        else:
            self.successful = True

    def __str__(self):
        return f'Example: {self.filename} ({self.codegen_target})'


class SelectFilesPlugin(Plugin):
    '''
    This plugin makes nose descend into all directories and exectue all files.
    '''
    # no command line arg needed to activate plugin
    enabled = True
    name = "select-files"

    def __init__(self, targets, image_dir, prefs_dict={}):
        self.targets = targets
        self.image_dir = image_dir
        self.prefs_dict = prefs_dict

    def configure(self, options, conf):
        pass # always on

    def wantDirectory(self, dirname):
        # we want all directories
        return True

    def find_examples(self, name):
        examples = []
        if os.path.isdir(name):
            for subname in os.listdir(name):
                examples.extend(self.find_examples(os.path.join(name, subname)))
            return examples
        elif name.endswith('.py'):  # only execute Python scripts
            return [name]
        else:
            return []

    def loadTestsFromName(self, name, module=None, discovered=False):
        all_examples = self.find_examples(name)
        all_tests = []
        for target in self.targets:
            for example in all_examples:
                all_tests.append(RunTestCase(example, target, self.image_dir,
                                             self.prefs_dict))
        return all_tests


if __name__ == '__main__':

    import argparse
    import utils

    parser = argparse.ArgumentParser(description='Run brian2 examples')

    parser.add_argument('--plot-dir', default=['brian2_examples/'], type=str,
                        nargs=1, help=("Where to save the created figures "
                                       "[default: brian2_examples]"))

    args = utils.parse_arguments(parser)

    from io import StringIO
    import brian2
    from brian2 import prefs

    all_prefs_combinations = utils.set_preferences(args, prefs)

    brian_dir = os.path.dirname(brian2.__file__)
    example_dir = os.path.abspath(os.path.join(brian_dir, '../examples'))
    argv = [__file__, '-v', '--with-xunit', '--verbose', '--exe', example_dir]

    all_successes = []
    for target in args.targets:
        print("Running examples with target", target)
        target_list = [target]
        target_image_dir = os.path.join(args.plot_dir[0], target.split('_')[0])

        if target == 'cuda_standalone':
            preference_dictionaries = all_prefs_combinations
        else:
            preference_dictionaries = [None]

        successes = []
        for n, prefs_dict in enumerate(preference_dictionaries):

            if prefs_dict is not None:
                print(f"{n + 1}. RUN: running on {target} with prefs:")
                # print preferences (setting happens in RunTestCase.runTest())
                utils.print_single_prefs(prefs_dict, set_prefs=prefs)
            else:  # None
                print(f"Running {target} with default preferences")
                # RunTestCase.runTest() needs a dictionary
                prefs_dict = {}

            image_dir = os.path.join(target_image_dir,
                                     utils.dict_to_name(prefs_dict))

            success = nose.run(argv=argv,
                               plugins=[SelectFilesPlugin(target_list,
                                                          image_dir,
                                                          prefs_dict),
                                        Capture(), Xunit()])
            successes.append(success)

        print(f"\nTARGET: {target.upper()}")
        all_success = utils.check_success(successes, all_prefs_combinations)

        all_successes.append(all_success)

    if len(args.targets) > 1:
        print("\nFINISHED ALL TARGETS")

        if all(all_successes):
            print("\nALL TARGETS PASSED")
        else:
            print("\n{}/{} TARGETS FAILED:".format(sum(all_successes) -
                                                   len(all_successes),
                                                   len(all_successes)))
            for n, target in enumerate(args.targets):
                if not all_successes[n]:
                    print(f"\t{target} failed.")
            sys.exit(1)

    elif not all_successes[0]:
        sys.exit(1)
