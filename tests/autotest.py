"""
Autotest framework for mcfc.

Runs the mcfc on each of the ufl files in inputs, stores the output in outputs,
and compares them with those in expected. Anomalies are highlighted, and the
user may replace the testcase with the new output. 
"""

# Python modules
import sys
import os
from subprocess import Popen, PIPE
from shutil import copy

# A nicer traceback from IPython
from IPython import ultraTB

# For colouring diffs
from pygments import highlight
from pygments.lexers import DiffLexer
from pygments.formatters import TerminalFormatter

# MCFC modules
from mcfc import frontend

def main():

    sys.excepthook = ultraTB.FormattedTB(mode='Context')

    sources = ['diffusion-1']

    if(len(sys.argv) > 1):
      sources = [sys.argv[1]]
    
    for sourcefile in sources:
	check(sourcefile)

    sys.exit(0)

def check(sourcefile):

    print "Testing " + sourcefile

    inputfile = infile(sourcefile)
    outputfile = outfile(sourcefile)
    expectedfile = expectfile(sourcefile)

    frontend.testHook(inputfile, outputfile)

    cmd = "diff -u " + expectedfile + " " + outputfile
    diff = Popen(cmd, shell=True, stdout=PIPE)
    diffout, differr = diff.communicate()
    
    if diffout:
        print "Difference detected in ", sourcefile, ", ",
        diffmenu(sourcefile, diffout)

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
	copy(src, dest)
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

