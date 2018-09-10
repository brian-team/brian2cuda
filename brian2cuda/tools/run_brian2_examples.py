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
    def __init__(self, filename, codegen_target, image_dir):
        unittest.TestCase.__init__(self)
        self.filename = filename
        self.codegen_target = codegen_target
        self.image_dir = image_dir

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
        fname_prefix = 'CPP.'
        print("Running with cpp_standalone")
    elif '{target}' == 'cuda_standalone':
        sub_str += ("import brian2cuda\\n"
                    "set_device('cuda_standalone', directory=None)")
        fname_prefix = 'CUDA.'
        print("Running with cuda_standalone")
    with open(rel_fname, "rb") as f:
        file_str = f.read()
    # remove any set device
    file_str = re.sub("set_device\\\(.*?\\)", '', file_str)
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
        fname = fname_prefix + fname
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
           image_dir=self.image_dir)

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
        return 'Example: %s (%s)' % (self.filename, self.codegen_target)


class SelectFilesPlugin(Plugin):
    '''
    This plugin makes nose descend into all directories and exectue all files.
    '''
    # no command line arg needed to activate plugin
    enabled = True
    name = "select-files"

    def __init__(self, targets, image_dir):
        self.targets = targets
        self.image_dir = image_dir

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
                all_tests.append(RunTestCase(example, target, self.image_dir))
        return all_tests


if __name__ == '__main__':
    import sys
    # run as `python run_brian2_examples.py image_dir`
    assert len(sys.argv) == 2, \
            'Need image direcotyr as argumenty, got {}'.format(sys.argv)
    image_dir = sys.argv[1]

    import brian2

    brian_dir = os.path.dirname(brian2.__file__)
    example_dir = os.path.join(brian_dir, '../examples')
    argv = [__file__, '-v', '--with-xunit', '--verbose', '--exe', example_dir]

    targets = ['cuda_standalone']#, 'cpp_standalone']

    nose.main(argv=argv, plugins=[SelectFilesPlugin(targets, image_dir),
                                  Capture(), Xunit()])
