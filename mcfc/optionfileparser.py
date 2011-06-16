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

from uflstate import UflState
from optionfile import OptionFile

class OptionFileParser:

    def __init__(self, filename):
        self.element_types = {}
        self.states = {}
        self.uflinput = {}

        optionfile = OptionFile(filename)

        # Build dictionary of element types for meshes
        self.element_types = {}
        # Get shape and degree for each mesh
        for mesh in optionfile.meshes():
            self.element_types[mesh.name] = mesh.shape, mesh.degree

        # Build dictionary of material phases
        aliased_fields = []
        for phase in optionfile.material_phases():
            # Build state (dictionary of fields)
            state = UflState()
            for field in optionfile.fields(phase.path):
                # For an aliased field, store material phase and field it is
                # aliased to, come back later to assign element of the target
                # field
                if field.field_type == 'aliased':
                    field.phase = phase.name
                    aliased_fields.append(field)
                else:
                    shape, degree = self.element_types[field.mesh]
                    state.insert_field(field.name, field.rank, shape, degree)
                    # Store the UFL input if present
                    if hasattr(field, 'ufl_equation'):
                        self.uflinput[phase.name+field.name] = (phase.name, field.name, field.ufl_equation)

            self.states[phase.name] = state

        # Resolve aliased fields
        for alias in aliased_fields:
            self.states[alias.phase][alias.rank][alias.name] = self.states[alias.to_phase][alias.rank][alias.to_field]

        # Build list of UFL equations with associated state
        for key in self.uflinput:
            ufl = self.uflinput[key][2]
            phase = self.uflinput[key][0]
            self.uflinput[key] = ufl, self.states[phase]

def testHook(inputFile, outputFile = None):
    p = OptionFileParser(inputFile)
    # Print to output file if given, stdout otherwise
    with open(outputFile, 'w') if outputFile else sys.stdout as fd:
        print >>fd, 'element types: ', p.element_types
        print >>fd, 'states: ', p.states
        print >>fd, 'ufl input: ', p.uflinput

if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    testHook(filename, sys.argv[2] if len(sys.argv)>2 else None)
        
# vim:sw=4:ts=4:sts=4:et
