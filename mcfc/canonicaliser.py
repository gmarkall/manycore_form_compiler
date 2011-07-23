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
from ufl.algorithms import extract_arguments
from ufl.algorithms.tuplenotation import as_form
# Regular python modules
import getopt, sys, ast
# The remaining modules are part of the form compiler
from symbolicvalue import SymbolicValue

# Solve needs to return an appropriate function in order for the interpretation
# to continue

class solveFunctor:

    def __init__(self):
        # Dictionary to remember all solves with result coefficient count as
        # the index and the operarands (forms for matrix and rhs) as data
        self._solves = {}

    def __call__(self,M,b):
        # FIXME we currently lose the variable names of the forms
        # FIXME what if we have multiple integrals?
        # FIXME is that the proper way of getting the element? (issue #23)
        element = extract_arguments(b)[0].element()
        coeff = ufl.coefficient.Coefficient(element)
        self._solves[coeff.count()] = (M,b)

        return coeff

# Intended as the front-end interface to the parser. e.g. to use,
# call canonicalise(filename).

def canonicalise(code, _state, _states):

    for key in _states:
        _states[key].readyToRun()

    dt = SymbolicValue("dt")
    solve = solveFunctor()
    namespace = { "dt": dt, "solve": solve, "state": _state, "states": _states }

    st = ast.parse(code)

    code = "from ufl import *\n" + \
           "" + code
    exec code in namespace

    # Pre-populate with state, states and solve
    uflObjects = {"state": _state, "states": _states, "solve": solve}

    for name, value in namespace.iteritems():
        # UFL Forms
        if isinstance(value, (ufl.form.Form, tuple)):
            # Swap form for its preprocessed equivalent and re-attach form data
            form = as_form(value)
            form_data = form.compute_form_data()
            # We keep the original (not preprocessed form for name lookup in
            # the uflObjects dictionary
            form_data.original_form = form
            form = form_data.preprocessed_form
            form._form_data = form_data
            uflObjects[name] = form
        # UFL Coefficients and Arguments
        elif isinstance(value, (ufl.coefficient.Coefficient, ufl.argument.Argument)):
            uflObjects[name] = value

    return st, uflObjects

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
