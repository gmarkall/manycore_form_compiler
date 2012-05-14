# This file is part of the Manycore Form Compiler.
#
# The Manycore Form Compiler is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
# 
# The Manycore Form Compiler is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
# 
# You should have received a copy of the GNU General Public License along with
# the Manycore Form Compiler.  If not, see <http://www.gnu.org/licenses>
#
# Copyright (c) 2011, Graham Markall <grm08@doc.ic.ac.uk> and others. Please see
# the AUTHORS file in the main source directory for a full list of copyright
# holders.


"""    Usage: frontend.py [OPTIONS] FILE

    Options: 
    
      --visualise, -v to output a visualisation of the AST
      --objvisualise  to output a deep visualisation of every ufl object
                      (warning: creates very big PDFs), implies --visualise
      -o:<filename>   to specify the output filename (- to print to stdout)
      -g, --debug     enable debugging mode (drop into pdb on error)
      -b:<backend>    to specify the backend (defaults to CUDA)"""

# Python libs
import os, sys, getopt, pprint, ast
# MCFC libs
from inputparser import inputparsers
from driverfactory import drivers

extensions = {'cuda': '.cu', 'op2': '.cpp'}

def run(inputFile, opts = None):

    # Parse options

    # Debugging mode
    debug = False
    if 'g' in opts or 'debug' in opts:
        # A nicer traceback from IPython
        # Try IPython.core.ultratb first (recommended from 0.11)
        try:
            from IPython.core import ultratb
        # If that fails fall back to ultraTB (deprecated as of 0.11)
        except ImportError:
            try:
                from IPython import ultraTB as ultratb
            except ImportError:
                print "Warning: IPython not available, not enabling debug mode."
        else:
            # Print a verbose backtrace and drop into pdb on error
            # Note: will automatically choose ipdb if available
            sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                    color_scheme='Linux', call_pdb=1)
            debug = True

    # Backend
    if 'b' in opts:
        backend = opts['b']
    else:
        backend = "cuda"

    # Output file name
    screen = None
    outputFileBase = os.path.splitext(inputFile)[0]
    if 'o' in opts:
        # Output to stdout
        if opts['o'] == '-' and debug:
            print "Warning: Cannot print to stdout when debugging is enabled."
        elif opts['o'] == '-':
            screen = sys.stdout
            outputFileBase = '.'
        else:
            outputFileBase = os.path.splitext(opts['o'])[0]

    # PDF visualiser
    vis = False
    if 'visualise' in opts or 'v' in opts:
        vis = True
    objvis = False
    if 'objvisualise' in opts or 'objvis' in opts:
        vis = True
        objvis = True

    # Create the output directory if it does not yet exist
    if not os.path.exists(outputFileBase):
        os.mkdir(outputFileBase, 0755)
    
    # Parse input and trigger the pipeline for each UFL equation
    parser = inputparsers[os.path.splitext(inputFile)[1]]
    for equation in parser.parse(inputFile):
        
        outputFile = outputFileBase + "/" + equation.name

        if vis:
            equation.execVisualisePipeline(outputFile, objvis)
        else:
            with screen or open(outputFile+extensions[backend], 'w') as fd:
                equation.execPipeline(fd, drivers[backend])

def _get_options():
    try: 
        opts_list, args = getopt.getopt(sys.argv[1:], "b:ghvo:",
            ["debug", "visualise", "objvisualise"])
    except getopt.error, msg:
        print msg
        print __doc__
        sys.exit(-1)

    opts = {}
    for opt in opts_list:
        key = opt[0].lstrip('-')
        value = opt[1]
        opts[key] = value

    if len(args) > 0: 
        inputFile = args[0]
    else:
        print "No input file given."
        print __doc__
        sys.exit(-1)

    return inputFile, opts

def main():
    inputFile, opts = _get_options()
    run(inputFile, opts)
    return 0

def testHook(inputFile, outputFile, debug=True, backend="cuda"):

    opts = {'o': outputFile, 'b': backend, 'g': debug}
    run(inputFile, opts)
    return 0

def testHookVisualiser(inputFile, outputFile, objvis = False):

    opts = {'o': outputFile, 'v': None}
    if objvis:
        opts['objvisualise'] = None
    run(inputFile, opts)
    return 0

if __name__ == "__main__":
    sys.exit(main())

# vim:sw=4:ts=4:sts=4:et
