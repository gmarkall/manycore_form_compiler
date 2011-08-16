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

import ufl

def TestFunction(initialiser):
    """UFL value: Create a test function argument to a form."""
    if isinstance(initialiser, ufl.Coefficient):
        initialiser = initialiser.element()
    return ufl.TestFunction(initialiser)

def TrialFunction(initialiser):
    """UFL value: Create a trial function argument to a form."""
    if isinstance(initialiser, ufl.Coefficient):
        initialiser = initialiser.element()
    return ufl.TrialFunction(initialiser)

def Jacobian(element):
    """UFL value: Create a Jacobian matrix argument to a form."""
    return ufl.Coefficient(element, -3)

def JacobianInverse(element):
    """UFL value: Create a Jacobian inverse argument to a form."""
    return ufl.Coefficient(element, -4)

def JacobianDeterminant(element):
    """UFL value: Create a Jacobian determinant argument to a form."""
    return ufl.Coefficient(element, -5)

# vim:sw=4:ts=4:sts=4:et
