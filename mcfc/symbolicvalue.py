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


from ufl.constantvalue import ConstantValue, IndexAnnotated

# We need to be able to specify certain values that are constant, but we do not
# know the value at compile time - for example, dt, the timestep size, which is
# specified using Diamond. So we extend the UFL ConstantValue class to represent
# a symbolic value, and provide methods to allow it to work with the rest of the
# UFL code.

class SymbolicValue(IndexAnnotated, ConstantValue):
    def __init__(self, value, shape=(), free_indices=(), index_dimensions=None):
        ConstantValue.__init__(self)
        IndexAnnotated.__init__(self, shape, free_indices, index_dimensions)
        self._value = value
        self._repr = "%s(%s, %s, %s, %s)" % (type(self).__name__, repr(self._value), repr(self._shape), repr(self._free_indices), repr(self._index_dimensions))

    def shape(self):
        return self._shape

    def free_indices(self):
        return self._free_indices

    def index_dimensions(self):
        return self._index_dimensions

    def value(self):
        return self._value

    def __repr__(self):
        return self._repr

    def __str__(self):
        return self._repr

# vim:sw=4:ts=4:sts=4:et
