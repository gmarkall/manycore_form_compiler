"""
Autotest framework for MCFC.

Runs the MCFc on each of the ufl files in inputs, stores the output in outputs,
and compares them with those in expected. Anomalies are highlighted, and the
user may replace the testcase with the new output. 

usage: autotest.py [OPTIONS] [INPUT-FILE]
    where OPTIONS can be one of
        -n, --non-interactive  Run in non-interactive (batch) mode
        --no-cuda              Do not test the CUDA backend
        --no-op2               Do not test the OP2 backend
        --no-optionfile        Do not test the option file parser
"""

# Python modules
import sys, os, shutil, getopt
from subprocess import Popen, PIPE

# A nicer traceback from IPython
from IPython import ultraTB

# For colouring diffs
from pygments import highlight
from pygments.lexers import DiffLexer
from pygments.formatters import TerminalFormatter

# MCFC modules
from mcfc import frontend, optionfileparser

def main():

    sys.excepthook = ultraTB.FormattedTB(mode='Context')

    opts, args = get_options()
    keys = opts.keys()
 
    # Check for non-interactive execution (e.g. on the
    # buildbot).
    if 'non-interactive' in keys or 'n' in keys:
        print "Running in non-interactive mode."
        tester = AutoTester(False)
    else:
        tester = AutoTester()

    check_cuda =  'no-cuda' not in keys
    check_op2 =  'no-op2' not in keys
    check_optionfile = 'no-optionfile' not in keys

    ufl_sources = ['noop', 'diffusion-1', 'diffusion-2', 'diffusion-3', 'identity', \
            'laplacian', 'helmholtz', 'euler-advection', 'identity-vector', \
            'simple-advection-diffusion' ]
    optionfile_sources = ['test', 'cdisk_adv_diff']

    # Check a single file if specified. Otherwise check
    # all default input files.
    if (len(args) > 0):
        if args[0] in ufl_sources:
            ufl_sources = [ args[0] ]
            check_optionfile = False
        elif args[0] in optionfile_sources:
            optionfile_sources = [ args[0] ]
            check_cuda = False
            check_op2 = False
        else:
            print "Unsupported source file:",args[0]
            print "Needs to be in",ufl_sources,"or",optionfile_sources
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

        tester.test(frontend.testHook, 
            lambda name: "inputs/ufl/" + name + ".ufl",
            lambda name: "outputs/cuda/" + name + ".cu",
            lambda name: "expected/cuda/" + name + ".cu",
            ufl_sources,
            'Running form compiler tests (CUDA backend)...')

    if check_op2:

        # Create the cuda outputs folder
        os.mkdir('outputs/op2', 0755)

        tester.test(lambda infile, outfile: frontend.testHook(infile, outfile, 'op2'), 
            lambda name: "inputs/ufl/" + name + ".ufl",
            lambda name: "outputs/op2/" + name + ".cpp",
            lambda name: "expected/op2/" + name + ".cpp",
            ufl_sources,
            'Running form compiler tests (OP2 backend)...')

    if check_optionfile:

        # Create the option file outputs folder
        os.mkdir('outputs/optionfile', 0755)

        tester.test(optionfileparser.testHook, 
            lambda name: "inputs/flml/" + name + ".flml",
            lambda name: "outputs/optionfile/" + name + ".dat",
            lambda name: "expected/optionfile/" + name + ".dat",
            optionfile_sources,
            'Running option file parser tests...')

    # Exit code is 0 if no tests failed.
    sys.exit(tester.failed)

def get_options():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n", ["non-interactive", "no-cuda", "no-op2", "no-optionfile"])
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

    def __init__(self, interactive = True):
        self.interactive = interactive
        self.failed = 0

    def test(self, testhook, infile, outfile, expectfile, sources, message = None):
        self.testhook = testhook
        self.infile = infile
        self.outfile = outfile
        self.expectfile = expectfile

        if message:
            print message

        for sourcefile in sources:
            self.check(sourcefile)

    def check(self, sourcefile):

        print "  Testing " + sourcefile

        inputfile = self.infile(sourcefile)
        outputfile = self.outfile(sourcefile)
        expectedfile = self.expectfile(sourcefile)

        self.testhook(inputfile, outputfile)

        cmd = "diff -u " + expectedfile + " " + outputfile
        diff = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        diffout, differr = diff.communicate()
        if differr:
            print differr
            self.failed = 1
        else:
            self.check_diff(sourcefile, diffout)

    def check_diff(self, sourcefile, diff):

        if diff:
            print "    Difference detected in %s." % sourcefile
            self.failed = 1
            
            if self.interactive:
                self.diffmenu(sourcefile, diff)

    def diffmenu(self, sourcefile, diffout):
        
        print "    [Continue, Abort, View, Replace, Show IR?] ",
        response = sys.stdin.readline()
        rchar = response[0].upper()

        if rchar=='C':
            return
        elif rchar=='A':
            sys.exit(-1)
        elif rchar=='V':
            print highlight(diffout, DiffLexer(), TerminalFormatter(bg="dark"))
            self.diffmenu(sourcefile, diffout)
        elif rchar=='R':
            src = self.outfile(sourcefile)
            dest = self.expectfile(sourcefile)
            shutil.copy(src, dest)
        elif rchar=='S':
            frontend.showGraph()
            self.diffmenu(sourcefile, diffout)
        else:
            print "    Please enter a valid option: ", 
            self.diffmenu(sourcefile, diffout)

# Execute main

if __name__ == "__main__":
    main()

# vim:sw=4:ts=4:sts=4:et
