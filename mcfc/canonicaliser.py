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
# For indices, measures, spaces and cells pretty syntax (dx, i, j, etc...)
from ufl.objects import *
# Regular python modules
import getopt, sys, ast
# Python unparser
from unparse import Unparser
# For getting unparsed stuff to execute it
import StringIO
# If we need debugging
import pdb
# The remaining modules are part of the form compiler
import state
from symbolicvalue import SymbolicValue


def init():

    # Symbolic values that we need during the interpretation of UFL
    global dt
    # List of the temporary fields, and their elements
    global _uflObjects

    _uflObjects = {}
    dt = SymbolicValue("dt")

# Intended as the front-end interface to the parser. e.g. to use,
# call canonicalise(filename).

def canonicalise(filename):
   
    init()

    # Read in the AST
    fd = open(filename, 'r')
    chars = fd.read()
    lines = charstolines(chars)
    fd.close

    canonical = StringIO.StringIO()

    # Interpret the UFL file line by line
    for line in lines:
        UFLInterpreter(line, canonical)

    return canonical.getvalue(), _uflObjects

# Solve needs to return an appropriate function in order for the interpretation
# to continue

def solve(M,b):
    vector = ufl.algorithms.preprocess(b)
    form_data = vector.form_data()
    element = form_data.arguments[0].element()
    return Coefficient(element)

# Action needs to do the same trick

def action(M,v):
    return ufl.formoperators.action(M,v)

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


def Coefficient(arg):
    return ufl.coefficient.Coefficient(arg)

def TestFunction(arg):
    return ufl.argument.TestFunction(arg)

def TestFunctions(*args):
    uflArgs = args[0]
    return ufl.argument.TestFunctions(uflArgs)

def TrialFunctions(*args):
    uflArgs = args[0]
    return ufl.argument.TrialFunctions(uflArgs)

def TrialFunction(arg):
    return ufl.argument.TrialFunction(arg)

def dot(arg1, arg2):
    return ufl.operators.dot(arg1,arg2)

def inner(arg1, arg2):
    return ufl.operators.inner(arg1, arg2)

def grad(arg):
    return ufl.operators.grad(arg)

def div(arg):
    return ufl.operators.div(arg)

def adjoint(arg):
    return ufl.formoperators.adjoint(arg)

def as_vector(*args):
    uflArgs = args[0]
    return ufl.tensors.as_vector(uflArgs)

def split(arg):
    return ufl.split_functions.split(arg)

def charstolines(chars):

    lines = []
    line = ''

    for char in chars:
        if not char == '\n':
            line = line + char
        else:
            lines.append(line)
            line = ''
    
    return lines

class UFLInterpreter:

    def __init__(self, line, f=sys.stdout):
        self._file = f
        st = ast.parse(line)
        print >>self._file, '# ' + line
        self.dispatch(st)
        self._file.flush

    def dispatch(self, tree):
        if isinstance(tree, list):
            for s in tree:
                self.dispatch(s)
        name = "_"+tree.__class__.__name__
        
        try:
            meth = getattr(self, name)
        except AttributeError:
            return
        meth(tree)

    def _Module(self, tree):
        for stmt in tree.body:
            self.dispatch(stmt)
            
    def _Assign(self, tree):

        global _uflObjects
        
        # Since we execute code from the source file in this function,
        # all locals are prefixed with _ to prevent name collision.
 
        # Get the left-hand side of the assignment
        _lhs = tree.targets[0]
        _target = unparse(_lhs)

        # Get the AST of the right-hand side of the expression
        _rhs = tree.value
        # Evaluate the RHS of the expression
        _statement =  _target + ' = ' + unparse(_rhs)
        exec(_statement,globals())
        
        # Get hold of the result of the execution
        _result = eval(_target)
        # If the result of executing the RHS is a form, we need to
        # preprocess it
        if isinstance(_result, ufl.form.Form):
            _result = ufl.algorithms.preprocess(_result)

        # Stash the resulting object. If it's a tuple of objects,
	# we need to stash each individual one separately so they
	# can be retrieved individually later on.
        if isinstance(_lhs, ast.Tuple):
	    count = len(_lhs.elts)
	    for i in range(count):
	        key = _lhs.elts[i].id
	        _uflObjects[key] = _result[i]
	else:
            _uflObjects[_target] = _result
            
        if isToBeExecuted(_lhs,_rhs):
            # Create an assignment statement that assigns the result of 
            # executing the statement to the LHS.
           
            # Construct the representation that we are going to print
            _newstatement = _target + ' = ' + repr(_result)
            
            # If the RHS uses values from a field, we need to remember
            # which field it is from.
            if usesValuesFromField(_rhs):
                _source = getSourceField(_rhs)
                _newstatement = _newstatement + ' & source(' + _source + ')'

            print >>self._file, _newstatement
        # This is the case where we print the original statement, not the
        # result of executing it
        else:
            print >>self._file, _statement

def isToBeExecuted(lhs,rhs):
    # We don't want to put the result of executing certain functions
    # in our output source code. Instead, we want to print them out
    # verbatim. These include solve, and action.
    
    # If the LHS is a subscript, then it is returning something into
    # state. Either the RHS is a variable, or it is a solve or
    # action, or a sum of functions. In any of these cases, we don't
    # want to print the result of executing it.
    if isinstance(lhs, ast.Subscript):
        return False

    # Anything that isn't a function is to be executed, except subscripts
    # (Because subscripts provide access to the syntax for state)
    if not isinstance(rhs, ast.Call):
        if isinstance(rhs, ast.Subscript):
            return False
        else:
            return True
    
    # Check if the function is one to avoid executing
    name = rhs.func.id
    if name in ['solve']:
        return False
    else:
        return True

def usesValuesFromField(st):
    # TestFunctions, TrialFunctions and Coefficients use values that
    # we need to pull from fields.
    
    # Things that aren't functions don't use values from a field
    # (well, not directly anyway, so we don't care)
    if not isinstance(st, ast.Call):
        return False
    
    name = st.func.id
    if name in ['TestFunction', 'TrialFunction', 'Coefficient']:
        return True
    else:
        return False

def getSourceField(st):
    # Get us the name of the field from which the argument's value
    # is derived.
    return st.args[0].id

def unparse(st):
    value = StringIO.StringIO()
    Unparser(st,value)
    return value.getvalue().rstrip('\n')


if __name__ == "__main__":
    sys.exit(main())

# vim:sw=4:ts=4:sts=4:et
