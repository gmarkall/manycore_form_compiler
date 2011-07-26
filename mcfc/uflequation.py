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

# UFL modules
from ufl.algorithms import extract_arguments
from ufl.coefficient import Coefficient
# MCFC modules
from symbolicvalue import SymbolicValue
from uflfunction import *

class solveFunctor:
    """A functor class to be called as 'solve' in the execution namespace of
       the UFL code. Tracks forms solved for and coefficients returned for all
       solves."""

    def __init__(self):
        # Dictionary to remember all solves with result coefficient count as
        # the index and the operarands (forms for matrix and rhs) as data
        self._solves = {}
        self.__iter__ = self._solves.__iter__
        self.items = self._solves.items

    def __call__(self,M,b):
        # FIXME we currently lose the variable names of the forms (this can be
        # looked up in the uflObjects)

        # b as the rhs should always only have a single argument
        # Extract its element
        element = extract_arguments(b)[0].element()
        coeff = Coefficient(element)
        self._solves[coeff.count()] = (M,b)
        return coeff

    def __getitem__(self, count):
        return self._solves[count]

class UflEquation:
    """Base class representing an equation in UFL with basic attributes.
       Objects of this type are passed from stage to stage in the MCFC
       pipeline.
       
       Attributes:
         name:        equation name
         code:        UFL code
         namespace:   namespace the UFL is executed in
         frontendAST: Python AST corresponding to the UFL code
         uflObjects:  UFL forms, coefficients, arguments
         dag:         intermediate DAG representation of the equation
         backendAst:  custom backend specific AST for unparsing"""

    def __init__(self):
        self.name = ""
        self.code = ""
        self.namespace = {}
        self.frontendAst = None
        self.uflObjects = {}
        self.dag = None
        self.backendAst = None

class FluidityEquation(UflEquation):
    """Container class representing a Fluidity UFL equation with the additional
       attributes state, states and solves."""
    # FIXME we could even do without those attributes since they're already in
    # namespace

    def __init__(self, name, code, state, states):
        UflEquation.__init__(self)
        dt = SymbolicValue("dt")
        solve = solveFunctor()
        self.name = name
        self.code = code
        self.namespace = { "dt": dt, "solve": solve, "state": state, "states": states }
        # Import UFL modules into namespace
        exec "from ufl import *" in self.namespace
        # Import MCFC UFL overrides into namespace
        exec "from mcfc.uflnamespace import *" in self.namespace
        self.solves = solve
        self.state = state
        self.states = states

# vim:sw=4:ts=4:sts=4:et
