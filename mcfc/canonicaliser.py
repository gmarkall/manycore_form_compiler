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
the input and return a dictionary of UFL objects.
"""

# The UFL packages are required so that the sources execute correctly
# when they are read in
import ufl
from ufl.algorithms.tuplenotation import as_form

# Intended as the front-end interface to the parser. e.g. to use,
# call canonicalise(filename).

def canonicalise(equation):
    """Execute code in namespace and return an AST represenation of the code
       and a collection of UFL objects (preprocessed forms, coefficients,
       arguments)"""

    for name, value in equation.namespace.iteritems():
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
            equation.uflObjects[name] = form
        # UFL Coefficients and Arguments
        elif isinstance(value, (ufl.coefficient.Coefficient, ufl.argument.Argument)):
            equation.uflObjects[name] = value

    return equation

# vim:sw=4:ts=4:sts=4:et
