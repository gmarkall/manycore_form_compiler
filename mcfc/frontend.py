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
    
      --visualise, -v Output a visualisation of the AST as a PDF file
      -o:<filename>   Specify the output file basename
      -p, --print     Output to stdout instead of file
      -b:<backend>    Specify the backend (defaults to CUDA)"""

# Python libs
import os, sys, getopt, pprint
# ANTLR runtime and generated code
import antlr3
from uflLexer import uflLexer
from uflParser import uflParser
# MCFC libs
from optionfileparser import OptionFileParser
import visualiser
import canonicaliser
from driverfactory import drivers
from uflstate import UflState

extensions = {'cuda': '.cu', 'op2': '.cpp'}

def run(inputFile, opts = {}):

    # Parse options

    if 'b' in opts:
        backend = opts['b']
    else:
        backend = "cuda"

    if 'o' in opts:
        outputFileBase = os.path.splitext(opts['o'])[0]
    else:
        outputFileBase = os.path.splitext(inputFile)[0]

    if 'print' in opts or 'p' in opts:
        screen = True
        fd = sys.stdout
    else:
        screen = False

    if 'visualise' in opts or 'v' in opts:
        vis = True
    else:
        vis = False

    # Parse input

    states, uflinput = parse_input(inputFile)

    for key in uflinput:

        outputFile = outputFileBase + key
        ufl = uflinput[key][0].splitlines()
        state = uflinput[key][1]

        ast, uflObjects = readSource(ufl, state, states)

        if vis:
            visualise(ast, outputFile+".pdf")

        if not screen:
            fd = open(outputFile+extensions[backend], 'w')

        driver = drivers[backend]()
        driver.drive(ast, uflObjects, fd)

        if not screen:
            fd.close()

def parse_input(inputFile):

    # FIXME this is for backwards compatibility, remove when not needed anymore
    if inputFile.endswith('ufl'):
        # read ufl input file
        with open(inputFile, 'r') as fd:
            ufl_input = fd.read()
        # Build a fake state
        phase = ""
        state = UflState()
        state.insert_field('Tracer',0)
        state.insert_field('Height',0)
        state.insert_field('Velocity',1)
        state.insert_field('NewVelocity',1)
        state.insert_field('TracerDiffusivity',2)
        return {phase: state}, {phase: (ufl_input, state)}
    else:
        p = OptionFileParser(inputFile)
        return p.states, p.uflinput


def visualise(ast, filename):

    v = visualiser.Visualiser(filename)
    v.visualise(ast)

def readSource(ufl, state, states):

    canned, uflObjects = canonicaliser.canonicalise(ufl, state, states)
    charStream = antlr3.ANTLRStringStream(canned)
    lexer = uflLexer(charStream)
    tokens = antlr3.CommonTokenStream(lexer)
    tokens.discardOffChannelTokens = True
    parser = uflParser(tokens)
    r = parser.file_input()
    root = r.tree
    
    return root, uflObjects

def _get_options():
    try: 
        opts_list, args = getopt.getopt(sys.argv[1:], "b:hvpo:", ["visualise", "print"])
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

if __name__ == "__main__":
    sys.exit(main())

# vim:sw=4:ts=4:sts=4:et
