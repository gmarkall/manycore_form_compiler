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
from optionfileparser import OptionFileParser
from symbolicvalue import SymbolicValue
from uflstate import UflState

class solveFunctor:
    """A functor class to be called as 'solve' in the execution namespace of
       the UFL code. Tracks forms solved for and coefficients returned for all
       solves."""

    def __init__(self):
        # Dictionary to remember all solves with result coefficient count as
        # the index and the operarands (forms for matrix and rhs) as data
        self._solves = {}

    def __call__(self,M,b):
        # FIXME we currently lose the variable names of the forms
        # FIXME what if we have multiple integrals?
        # FIXME is that the proper way of getting the element? (issue #23)
        element = extract_arguments(b)[0].element()
        coeff = Coefficient(element)
        self._solves[coeff.count()] = (M,b)
        return coeff

def buildNamespace(state, states):
    "Return a namespace populated with dt, solve, state and states"
    dt = SymbolicValue("dt")
    solve = solveFunctor()
    return { "dt": dt, "solve": solve, "state": state, "states": states }

class InputParser:
    "Abstract base class for MCFC input parsers"

    def parse(self, filename):
        "Parse function taking a filename and returning a list of tuples"
        raise NotImplementedError("You're supposed to implement parse()!")

# FIXME this is for backwards compatibility, remove when not needed anymore
class FluflParser(InputParser):
    "A legacy input parser for flufl files using a 'fake' state"
    
    def parse(self, filename):
        # read ufl input file
        with open(filename, 'r') as fd:
            ufl_input = fd.read()
        phase = ""
        # Build a fake state dict
        state = UflState()
        state.insert_field('Tracer',0)
        state.insert_field('Height',0)
        state.insert_field('Velocity',1)
        state.insert_field('NewVelocity',1)
        state.insert_field('TracerDiffusivity',2)
        # Build a fake states dict
        states = {phase: state}
        return {phase: (ufl_input, buildNamespace(state, states))}

class FlmlParser(InputParser):
    "An input parser for Fluidity flml files"
    
    def parse(self, filename):
        p = OptionFileParser(inputFile)
        flmlinput = {}
        for equation, data in p.uflinput.items():
            ufl, state = data
            flmlinput[equation] = ufl, buildNamespace(state, p.states)
        return flmlinput

# FIXME flufl files should have extension .flufl
inputparsers = {'.ufl': FluflParser, '.flml': FlmlParser}

# vim:sw=4:ts=4:sts=4:et
