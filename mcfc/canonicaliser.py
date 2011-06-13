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

"""
MCFC Canonicaliser. Uses the FEniCS UFL implementation to canonicalise
the input and write it out.
"""

# The UFL packages are required so that the sources execute correctly
# when they are read in
import ufl
from ufl.algorithms.tuplenotation import as_form
# Regular python modules
import getopt, sys, ast
# The remaining modules are part of the form compiler
import state
from symbolicvalue import SymbolicValue

def canonicalise(filename):

    dt = SymbolicValue("dt")
    n = state.TemporalIndex()
    c = state.ConstantTemporalIndex()
    namespace = { "dt": dt, "n": n, "c": c, "solve": solve }

    fd = open(filename, 'r')
    code = fd.read()
    fd.close

    st = ast.parse(code)

    code = "from ufl import *\n" + \
           "from mcfc import state\n" + \
	   "" + code
    exec code in namespace
    
    uflObjects = {}

    for name, value in namespace.iteritems():
        if isinstance(value, (ufl.form.Form, tuple)):
	    uflObjects[name] = ufl.algorithms.preprocess(as_form(value))
	elif isinstance(value, (ufl.coefficient.Coefficient, ufl.argument.Argument)):
	    uflObjects[name] = value

    return st, uflObjects

# Solve needs to return an appropriate function in order for the interpretation
# to continue

def solve(M,b):
    vector = ufl.algorithms.preprocess(b)
    form_data = vector.form_data()
    element = form_data.arguments[0].element()
    return ufl.coefficient.Coefficient(element)

def main():
    
    # Get options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:", ["help"])
    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)
    
    # process options
    if len(args)>0:
        inputfile = args[0]
    outputfile = None

    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
    
    # Run canonicaliser
    print "Canonicalising " + inputfile
    canonical, uflObjects = canonicalise(inputfile);
    print canonical

    return 0

if __name__ == "__main__":
    sys.exit(main())

# vim:sw=4:ts=4:sts=4:et
