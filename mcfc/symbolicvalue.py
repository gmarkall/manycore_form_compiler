import ufl.constantvalue

# We need to be able to specify certain values that are constant, but we do not
# know the value at compile time - for example, dt, the timestep size, which is
# specified using Diamond. So we extend the UFL ConstantValue class to represent
# a symbolic value, and provide methods to allow it to work with the rest of the
# UFL code.

class SymbolicValue(ufl.constantvalue.ConstantValue, ufl.constantvalue.IndexAnnotated):
    def __init__(self, value, shape=(), free_indices=(), index_dimensions=None):
        ufl.constantvalue.ConstantValue.__init__(self)
        ufl.constantvalue.IndexAnnotated.__init__(self, shape, free_indices, index_dimensions)
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
