"""
Autotest framework for mcfc.

Runs the mcfc on each of the ufl files in inputs, stores the output in outputs,
and compares them with those in expected. Anomalies are highlighted, and the
user may replace the testcase with the new output. 
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

# Global status
failed = 0
interactive = True

def main():

    sys.excepthook = ultraTB.FormattedTB(mode='Context')

    opts, args = get_options()
    keys = opts.keys()

    # Check a single file if specified. Otherwise check
    # all those specified above.
    if (len(args) > 0):
        sources = [ args[0] ]
 
    # Check for non-interactive execution (e.g. on the
    # buildbot.
    if 'noninteractive' in keys or 'n' in keys:
        print "Running in non-interactive mode."
        global interactive
        interactive = False

    # Delete the outputs folder if it exists, to avoid
    # any stale files.
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')

    # Create the outputs folder
    os.mkdir('outputs', 0755)

    test_formcompiler()
    test_optionfileparser()

    # Exit code is 0 if no tests failed.
    sys.exit(failed)

def test_formcompiler():

    print 'Running form compiler tests...'

    sources = ['diffusion-1', 'diffusion-2', 'diffusion-3', 'identity', \
               'laplacian', 'helmholtz', 'euler-advection', 'identity-vector' ]

    tester = AutoTester(frontend.testHook, 
        lambda name: "inputs/ufl/" + name + ".ufl",
        lambda name: "outputs/cuda/" + name + ".cu",
        lambda name: "expected/cuda/" + name + ".cu")

    # Create the outputs folder
    os.mkdir('outputs/cuda', 0755)

    for sourcefile in sources:
        tester.check(sourcefile)

def test_optionfileparser():

    print 'Running option file parser tests...'

    sources = ['test', 'cdisk_adv_diff']

    tester = AutoTester(optionfileparser.testHook, 
        lambda name: "inputs/flml/" + name + ".flml",
        lambda name: "outputs/optionfile/" + name + ".dat",
        lambda name: "expected/optionfile/" + name + ".dat")

    # Create the outputs folder
    os.mkdir('outputs/optionfile', 0755)

    for sourcefile in sources:
        tester.check(sourcefile)

def get_options():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n", ["noninteractive"])
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

    def __init__(self, testhook, infile, outfile, expectfile):
        self.testhook = testhook
        self.infile = infile
        self.outfile = outfile
        self.expectfile = expectfile

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
            global failed
            failed = 1
        else:
            self.check_diff(sourcefile, diffout)

    def check_diff(self, sourcefile, diff):

        if diff:
            print "    Difference detected in %s." % sourcefile
            global failed
            failed = 1
            
            if interactive:
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
