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
        for mesh in optionfile.mesh_iterator():
            self.element_types[mesh.name] = (mesh.shape, mesh.degree)

        # Build dictionary of material phases
        aliased_fields = []
        for phase in optionfile.material_phase_iterator():
            # Build state (dictionary of fields)
            state = UflState()
            for field in optionfile.field_iterator(phase.path):
                # For an aliased field, store material phase and field it is
                # aliased to, come back later to assign element of the target
                # field
                if field.field_type == 'aliased':
                    field.phase = phase.name
                    aliased_fields.append(field)
                else:
                    state.insert_field(field.name, field.rank, self.element_types[field.mesh])
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

if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    p = OptionFileParser(filename)
    print 'element types: ', p.element_types
    print 'states: ', p.states
    print 'ufl input: ', p.uflinput
        
# vim:sw=4:ts=4:sts=4:et
