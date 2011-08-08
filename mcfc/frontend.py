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
      -o:<filename>   to specify the output filename
      -p, --print     to print code to screen
      -b:<backend>    to specify the backend (defaults to CUDA)"""

# Python libs
import os, sys, getopt, pprint, ast
# MCFC libs
from inputparser import inputparsers
from driverfactory import drivers

extensions = {'cuda': '.cu', 'op2': '.cpp'}

def run(inputFile, opts = None):

    # Parse options

    # Backend
    if 'b' in opts:
        backend = opts['b']
    else:
        backend = "cuda"

    # Output file name
    if 'o' in opts:
        outputFileBase = os.path.splitext(opts['o'])[0]
    else:
        outputFileBase = os.path.splitext(inputFile)[0]

    # Output to stdout
    if 'p' in opts:
        screen = sys.stdout
    else:
        screen = None

    # PDF visualiser
    vis = False
    if 'visualise' in opts or 'v' in opts:
        vis = True
    objvis = False
    if 'objvisualise' in opts or 'objvis' in opts:
        vis = True
        objvis = True

    # Parse input and trigger the pipeline for each UFL equation
    parser = inputparsers[os.path.splitext(inputFile)[1]]
    for equation in parser.parse(inputFile):
        outputFile = outputFileBase + equation.name

        if vis:
            equation.execVisualisePipeline(outputFile, objvis)
        else:
            with screen or open(outputFile+extensions[backend], 'w') as fd:
                equation.execPipeline(fd, drivers[backend])

def _get_options():
    try: 
        opts_list, args = getopt.getopt(sys.argv[1:], "b:hvo:p", ["visualise", "objvisualise", "print"])
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

def testHook(inputFile, outputFile, backend = "cuda"):

    opts = {'o': outputFile, 'b': backend}
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
