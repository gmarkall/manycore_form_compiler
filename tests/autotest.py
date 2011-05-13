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
from mcfc import frontend

# Global status
failed = 0
interactive = True

def main():

    sys.excepthook = ultraTB.FormattedTB(mode='Context')

    opts, args = get_options()
    keys = opts.keys()

    sources = ['diffusion-1', 'diffusion-2', 'diffusion-3', 'identity', \
               'laplacian', 'helmholtz', 'euler-advection' ]

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

    for sourcefile in sources:
        check(sourcefile)

    # Exit code is 0 if no tests failed.
    sys.exit(failed)

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

def check(sourcefile):

    print "Testing " + sourcefile

    inputfile = infile(sourcefile)
    outputfile = outfile(sourcefile)
    expectedfile = expectfile(sourcefile)

    frontend.testHook(inputfile, outputfile)

    cmd = "diff -u " + expectedfile + " " + outputfile
    diff = Popen(cmd, shell=True, stdout=PIPE)
    diffout, differr = diff.communicate()
    
    check_diff(sourcefile, diffout)

def check_diff(sourcefile, diff):

    if diff:
        print "Difference detected in %s." % sourcefile
	global failed
	failed = 1
        
	if interactive:
            diffmenu(sourcefile, diff)

def diffmenu(sourcefile, diffout):
    
    print "[Continue, Abort, View, Replace, Show IR?] ",
    response = sys.stdin.readline()
    rchar = response[0].upper()

    if rchar=='C':
        return
    elif rchar=='A':
        sys.exit(-1)
    elif rchar=='V':
        print highlight(diffout, DiffLexer(), TerminalFormatter(bg="dark"))
        diffmenu(sourcefile, diffout)
    elif rchar=='R':
        src = outfile(sourcefile)
        dest = expectfile(sourcefile)
        shutil.copy(src, dest)
    elif rchar=='S':
        frontend.showGraph()
        diffmenu(sourcefile, diffout)
    else:
        print "Please enter a valid option: ", 
        diffmenu(sourcefile, diffout)

def infile(name):
    return "inputs/" + name + '.ufl'

def outfile(name):
    return "outputs/" + name + ".cu"

def expectfile(name):
    return "expected/" + name + ".cu"

# Execute main

if __name__ == "__main__":
    main()

# vim:sw=4:ts=4:sts=4:et
