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

# MCFC modules
from optionfileparser import OptionFileParser
from uflstate import UflState
from fluidityequation import FluidityEquation

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
        return [FluidityEquation(phase, ufl_input, state, states)]

class FlmlParser(InputParser):
    "An input parser for Fluidity flml files"
    
    def parse(self, filename):
        "Return a list of one FluidityEquation for each equation in the flml input file"
        p = OptionFileParser(inputFile)
        return [FluidityEquation(equationname, data[0], data[1], p.states) for equationname, data in p.uflinput.items()]

# FIXME flufl files should have extension .flufl
inputparsers = {'.ufl': FluflParser, '.flml': FlmlParser}

# vim:sw=4:ts=4:sts=4:et
