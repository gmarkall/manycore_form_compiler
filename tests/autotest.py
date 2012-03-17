"""
Autotest framework for MCFC.

Runs the MCFc on each of the ufl files in inputs, stores the output in outputs,
and compares them with those in expected. Anomalies are highlighted, and the
user may replace the testcase with the new output. In addition the PDF
visualiser is tested, but no diffs are presented. A test for the PDF object
visualiser is optionally available.

usage: autotest.py [OPTIONS] [INPUT-FILE]
    where OPTIONS can be one of
        -h, --help             Print this usage message
        -n, --non-interactive  Run in non-interactive (batch) mode
        --no-cuda              Do not test the CUDA backend
        --no-op2               Do not test the OP2 backend
        --no-optionfile        Do not test the option file parser
        --no-visualiser        Do not test the PDF visualiser
        --with-objvis[ualiser] Also test the PDF object visualiser
        -r, --replace-all      Replace all changed expected results
                               (Implies --non-interactive)
"""

# Python modules
import sys, os, shutil, getopt
from subprocess import Popen, PIPE

# A nicer traceback from IPython
# Try IPython.core.ultratb first (recommended from 0.11)
try:
    from IPython.core import ultratb
# If that fails fall back to ultraTB (deprecated as of 0.11)
except ImportError:
    from IPython import ultraTB as ultratb

# For colouring diffs
from pygments import highlight
from pygments.lexers import DiffLexer
from pygments.formatters import TerminalFormatter

# MCFC modules
from mcfc import frontend, optionfileparser

def main():

    sys.excepthook = ultratb.FormattedTB(mode='Context')

    opts, args = get_options()
    keys = opts.keys()

    if 'help' in keys or 'h' in keys:
        print __doc__
        sys.exit(0)

    singleTester = SingleFileTester()
    multiTester = MultiFileTester()

    # Check for non-interactive execution (e.g. on the
    # buildbot).
    if 'non-interactive' in keys or 'n' in keys:
        print "Running in non-interactive mode."
        singleTester.interactive = False
        multiTester.interactive = False
    else:
        sys.excepthook = ultratb.FormattedTB(mode='Context')
    if 'replace-all' in keys or 'r' in keys:
        print "Replacing all changed expected results."
        singleTester.interactive = False
        singleTester.replaceall = True
        multiTester.interactive = False
        multiTester.replaceall = True
    
    check_cuda =  'no-cuda' not in keys
    check_op2 =  'no-op2' not in keys
    check_optionfile = 'no-optionfile' not in keys
    check_visualiser = 'no-visualiser' not in keys

    flml_sources = ['identity', 'helmholtz', 'helmholtz-p2', 'euler-advection',
                    'diffusion-3', 'simple-advection-diffusion', 'noop']
    ufl_sources = ['diffusion-1', 'diffusion-2', 'laplacian',
                   'identity-vector', 'swapped-advection']
    optionfile_sources = ['test', 'cdisk_adv_diff']

    # Check a single file if specified. Otherwise check
    # all default input files.
    if (len(args) > 0):
        if args[0] in flml_sources:
            flml_sources = [ args[0] ]
            check_optionfile = False
        elif args[0] in optionfile_sources:
            optionfile_sources = [ args[0] ]
            check_cuda = False
            check_op2 = False
            check_visualiser = False
        else:
            print "Unsupported source file:",args[0]
            print "Needs to be in",flml_sources,"or",optionfile_sources
            sys.exit(-1)

    # Delete the outputs folder if it exists, to avoid
    # any stale files.
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')

    # Create the outputs folder
    os.mkdir('outputs', 0755)

    if check_cuda:

        # Create the cuda outputs folder
        os.mkdir('outputs/cuda', 0755)

        multiTester.test(frontend.testHook,
            lambda name: "inputs/flml/" + name + ".flml",
            lambda name: "outputs/cuda/" + name,
            lambda name: "expected/cuda/" + name,
            flml_sources,
            'Running form compiler tests (CUDA backend, flml input)...')

        multiTester.test(frontend.testHook,
            lambda name: "inputs/ufl/" + name + ".ufl",
            lambda name: "outputs/cuda/" + name,
            lambda name: "expected/cuda/" + name,
            ufl_sources,
            'Running form compiler tests (CUDA backend, ufl input)...')

    if check_op2:

        # Create the cuda outputs folder
        os.mkdir('outputs/op2', 0755)

        multiTester.test(lambda infile, outfile: frontend.testHook(infile, outfile, 'op2'),
            lambda name: "inputs/flml/" + name + ".flml",
            lambda name: "outputs/op2/" + name,
            lambda name: "expected/op2/" + name,
            flml_sources,
            'Running form compiler tests (OP2 backend, flml input)...')

        multiTester.test(lambda infile, outfile: frontend.testHook(infile, outfile, 'op2'),
            lambda name: "inputs/ufl/" + name + ".ufl",
            lambda name: "outputs/op2/" + name,
            lambda name: "expected/op2/" + name,
            ufl_sources,
            'Running form compiler tests (OP2 backend, ufl input)...')

    if check_optionfile:

        # Create the option file outputs folder
        os.mkdir('outputs/optionfile', 0755)

        singleTester.test(optionfileparser.testHook,
            lambda name: "inputs/flml/" + name + ".flml",
            lambda name: "outputs/optionfile/" + name + ".dat",
            lambda name: "expected/optionfile/" + name + ".dat",
            optionfile_sources,
            'Running option file parser tests...')

    if check_visualiser:

        # Create the visualiser outputs folder
        os.mkdir('outputs/visualiser', 0755)

        multiTester.test(frontend.testHookVisualiser,
            lambda name: "inputs/flml/" + name + ".flml",
            lambda name: "outputs/visualiser/" + name,
            lambda name: None,
            flml_sources,
            'Running PDF visualiser tests (flml input)...')


        multiTester.test(frontend.testHookVisualiser,
            lambda name: "inputs/ufl/" + name + ".ufl",
            lambda name: "outputs/visualiser/" + name,
            lambda name: None,
            ufl_sources,
            'Running PDF visualiser tests (ufl input)...')

        # If object visualiser tests have been requested
        if 'with-objvis' in keys:

            # Create the visualiser outputs folder
            os.mkdir('outputs/objvisualiser', 0755)

            multiTester.test(lambda infile, outfile: frontend.testHookVisualiser(infile, outfile, True),
                lambda name: "inputs/ufl/" + name + ".ufl",
                lambda name: "outputs/objvisualiser/" + name,
                lambda name: None,
                ufl_sources,
                'Running PDF object visualiser tests...')
    
    # Exit code is 0 if no tests failed.
    sys.exit(multiTester.failed or singleTester.failed)

def get_options():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hnr", [ "help",
                                                          "non-interactive",
                                                          "no-cuda",
                                                          "no-op2",
                                                          "no-optionfile",
                                                          "no-visualiser",
                                                          "with-objvis",
                                                          "with-objvisualise",
                                                          "replace-all"])
    except getopt.error, msg:
        print msg
        print __doc__
        sys.exit(-1)

    opts_dict = {}
    for opt in opts:
        key = opt[0].lstrip('-')
        value = opt[1]
        opts_dict[key] = value

    return opts_dict, args

class AutoTester:

    def __init__(self, interactive = True, replaceall = False):
        self.interactive = interactive
        self.replaceall = replaceall
        self.failed = 0

    def check(self, sourcefile):
        raise NotImplementedError("The check method must be implemented.")

    def test(self, testhook, infile, outfile, expectfile, sources, message = None):
        self.testhook = testhook
        self.infile = infile
        self.outfile = outfile
        self.expectfile = expectfile

        if message:
            print message

        for sourcefile in sources:
            print "  Testing " + sourcefile
            inputfile = self.infile(sourcefile)
            outputfile = self.outfile(sourcefile)
            expectedfile = self.expectfile(sourcefile)

            # Test hook returns 0 if successful, 1 if failed
            hookfailed = self.testhook(inputfile, outputfile)

            # Print a message if the test hook failed
            if hookfailed:
                print "    test hook failed."
                self.failed = 1
            # Otherwise, if we have an expected output, diff against it
            elif expectedfile:
                self.check(inputfile, outputfile, expectedfile)

    def diffmenu(self, sourcefile, destfile, diffout):
        print "    [Continue, Abort, View, Replace, Show IR?] ",
        response = sys.stdin.readline()
        rchar = response[0].upper()

        if rchar=='C':
            return
        elif rchar=='A':
            sys.exit(-1)
        elif rchar=='V':
            print highlight(diffout, DiffLexer(), TerminalFormatter(bg="dark"))
            self.diffmenu(sourcefile, destfile, diffout)
        elif rchar=='R':
            self.replace(sourcefile, destfile)
        elif rchar=='S':
            frontend.showGraph()
            self.diffmenu(sourcefile, destfile, diffout)
        else:
            print "    Please enter a valid option: ",
            self.diffmenu(sourcefile, destfile, diffout)
    
    def check_diff(self, sourcefile, destfile, diff):

        if diff:
            print "    Difference detected in %s." % sourcefile
            self.failed = 1

            if self.replaceall:
                self.replace(sourcefile, destfile)
            elif self.interactive:
                self.diffmenu(sourcefile, destfile, diff)

    def replace(self, src, dst):
        print "    Replacing '%s' with '%s'." % (dst, src)
        shutil.copy(src, dst)

class SingleFileTester(AutoTester):

    def check(self, inputfile, outputfile, expectedfile):
        cmd = "diff -u " + expectedfile + " " + outputfile
        diff = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        diffout, differr = diff.communicate()
        if differr:
            print differr
            self.failed = 1
        else:
            self.check_diff(outputfile, expectedfile, diffout)

class MultiFileTester(AutoTester):

    def check(self, inputfile, outputfile, expectedfile):
        # Did we generate the right files?
        expectedfiles = os.listdir(expectedfile)
        outputfiles = os.listdir(outputfile)
        if expectedfiles != outputfiles:
            self.failed = 1
            print "    Expected list of files does not match generated list of files."
            return

        # Are those files what we expect?
        for currfile in os.listdir(expectedfile):
            currentoutput = outputfile + "/" + currfile
            currentexpect = expectedfile + "/" + currfile
            cmd = "diff -u " + currentexpect + " " + currentoutput
            diff = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
            diffout, differr = diff.communicate()
            if differr:
                print differr
                self.failed = 1
            else:
                self.check_diff(currentoutput, currentexpect, diffout)

# Execute main

if __name__ == "__main__":
    main()

# vim:sw=4:ts=4:sts=4:et
