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


"""codegeneration.py - provides tools to build a C++/CUDA/OpenCL ASTs and
unparse it to a file."""

import numpy

class ModifierMixin:

    def __init__(self):
        # FIXME need the order of modifiers be enforced / preserved?
        self._modifier = set()

    def _setModifier(self, setModifier, modifier):
        if setModifier:
            self._modifier.add(modifier)
        else:
            self._modifier.discard(modifier)

    def unparse_modifier(self):
        return ''.join([m + ' ' for m in self._modifier])

class BackendASTNode:
    pass

class Bracketed(BackendASTNode):

    def __init__(self, op):
        self._op = op

    def unparse(self):
        # Only bracketing binary ops makes sense
        if isinstance(self._op, BinaryOp):
            return '(' + self._op.unparse() + ')'
        return self._op.unparse()

class Subscript(BackendASTNode):

    def __init__(self, base, offset):
        self._base = base
        self._offset = offset

    def unparse(self):
        base = self._base.unparse()
        offset = self._offset.unparse()
        return '%s[%s]' % (base, offset)

    __str__ = unparse

class Dereference(BackendASTNode):

    def __init__(self, expr):
        self._expr = expr

    def unparse(self):
        expr = self._expr.unparse()
        return '*%s' % (expr)

    __str__ = unparse

class NullExpression(BackendASTNode):

    def unparse(self):
        return ''

class Variable(BackendASTNode):

    def __init__(self, name, t=None):
        self._name = name
        self._t = t or Type()

    def __hash__(self):
        return self._name.__hash__()

    def __eq__(self, other):
        return self.name() == other.name()

    def setCudaShared(self, isCudaShared):
        self._t.setCudaShared(isCudaShared)

    def type(self):
        return self._t

    def name(self):
        return self._name

    def unparse(self):
        return self._name

    def unparse_declaration(self):
        name = self._name
        t = self._t.unparse()
        t_post = self._t.unparse_post()
        return '%s %s%s' % (t, name, t_post)

    __str__ = unparse_declaration

class Literal(BackendASTNode):

    def __init__(self, value):
        if isinstance(value, str):
            self._value = '"' + value + '"'
        else:
            self._value = str(value)

    def unparse(self):
        return self._value

    __str__ = unparse

class InitialiserList(BackendASTNode):

    def __init__(self, array):
        self._array = array

    def unparse(self):
        arrStr = ", ".join(map(lambda x: x.unparse(), self._array))
        return "{ %s }" % arrStr

class ArrayInitialiserList(BackendASTNode):

    def __init__(self, array, newlines = False, indentation = ''):
        # Make input a NumPy array (and fail it it doesn't work)
        self._array = numpy.asarray(array, numpy.float)
        self._newlines = newlines
        self._indentation = indentation

    def unparse(self):
        self.arrStr = numpy.array2string(self._array, separator=',',
                prefix=self._indentation + ' =  ' + getIndent())
        if not self._newlines:
           self.arrStr = self.arrStr.replace('\n','')
        # Replace all [ delimiters by { in string representation of the array
        return self.arrStr.replace('[','{ ').replace(']',' }')

class ForLoop(BackendASTNode):

    def __init__(self, init, test, inc, body=None):
        self._init = init
        self._test = test
        self._inc = inc
        self._body = body or Scope()

    def append(self, statement):
        self._body.append(statement)

    def prepend(self, statement):
        self._body.prepend(statement)

    def body(self):
        return self._body

    def unparse(self):
        init = self._init.unparse()
        test = self._test.unparse()
        inc = self._inc.unparse()
        header = 'for(%s; %s; %s)\n' % (init, test, inc)
        body = self._body.unparse()
        return header + body

    __str__ = unparse

class ParameterList(BackendASTNode):

    def __init__(self, params=None):
        self._params = as_list(params)

    def unparse(self):
        return '(' + ", ".join([p.unparse_declaration() for p in self._params]) + ')'

    __str__ = unparse

class ExpressionList(BackendASTNode):

    def __init__(self, expressions=None):
        self._expressions = as_list(expressions)

    def append(self, expression):
         self._expressions.append(expression)

    def prepend(self, expression):
         self._expressions.insert(0, expression)

    def unparse(self):
        return '(' + ', '.join([e.unparse() for e in self._expressions]) + ')'

    __str__ = unparse

class FunctionDefinition(BackendASTNode, ModifierMixin):

    def __init__(self, t, name, params=None, body=None):
        ModifierMixin.__init__(self)
        self._t = t
        self._name = name
        self._params = ParameterList(params)
        self._body = body or Scope()

    def setCudaKernel(self, isCudaKernel):
        self._setModifier(isCudaKernel, '__global__')

    def setExternC(self, isExternC):
        self._setModifier(isExternC, 'extern "C"')

    def append(self, statement):
        self._body.append(statement)

    def prepend(self, statement):
        self._body.prepend(statement)

    def unparse(self):
        mod = self.unparse_modifier()
        t = self._t.unparse()
        params = self._params.unparse()
        body = self._body.unparse()
        return '%s%s %s%s\n%s' % (mod, t, self._name, params, body)

    __str__ = unparse

class FunctionCall(BackendASTNode):

    def __init__(self, name, params=None):
        self._name = name
        self._params = ExpressionList(params)

    def unparse(self):
        name = self._name
        params = self._params.unparse()
        return '%s%s' % (name, params)

    __str__ = unparse

class CudaKernelCall(FunctionCall):

    def __init__(self, name, params, gridDim, blockDim, shMemSize=None, stream=None):
        FunctionCall.__init__(self, name, params)
        self._gridDim = gridDim
        self._blockDim = blockDim
        self._shMemSize = shMemSize
        self._stream = stream

    def unparse(self):
        name = self._name
        params = self._params.unparse()

        gridDim = self._gridDim.unparse()
        blockDim = self._blockDim.unparse()
        config = '%s,%s' % (gridDim, blockDim)

        if self._shMemSize is not None:
            config += ',' + self._shMemSize.unparse()
            if self._stream is not None:
                config += ',' + self._stream.unparse()

        return '%s<<<%s>>>%s' % (name, config, params)

    __str__ = unparse

class Scope(BackendASTNode):

    def __init__(self, statements=None):
        self._statements = as_list(statements)

    def append(self, statement):
        self._statements.append(statement)

    def prepend(self, statement):
        self._statements.insert(0, statement)

    def find(self, matches):
        for s in self._statements:
            if matches(s):
                return s

    def unparse(self):
        indent = getIndent()
        code = indent + '{\n'
        indent = incIndent()
        code += '\n'.join([indent + s.unparse() + ';' for s in self._statements])
        indent = decIndent()
        code += '\n' + indent + '}'
        return code

    __str__ = unparse

class GlobalScope(Scope):

    def unparse(self):
        code = ''
        for s in self._statements:
            if isinstance(s, Include):
                code += s.unparse() + '\n'
            elif isinstance(s, FunctionDefinition):
                code += s.unparse() + '\n\n'
            else:
                code += s.unparse() + ';\n'
        return code

    __str__ = unparse

class New(BackendASTNode):

    def __init__(self, t, params=None):
        self._t = t
        self._params = ExpressionList(params)

    def unparse(self):
        t = self._t.unparse_internal()
        params = self._params.unparse()
        return 'new %s%s' % (t, params)

    __str__ = unparse

class Delete(BackendASTNode):

    def __init__(self, var):
        self._var = var

    def unparse(self):
        return 'delete %s' % (self._var.unparse())

    __str__ = unparse

class BinaryOp(BackendASTNode):

    def __init__(self, lhs, rhs, op):
        self._lhs = lhs
        self._rhs = rhs
        self._op = op

    def unparse(self):
        lhs = self._lhs.unparse()
        rhs = self._rhs.unparse()
        return '%s%s%s' % (lhs, self._op, rhs)

    __str__ = unparse

def _fixup(op):
    if isinstance(op, (AddOp, DivideOp)):
        return Bracketed(op)
    return op

class MultiplyOp(BinaryOp):

    def __init__(self, lhs, rhs):
        BinaryOp.__init__(self, _fixup(lhs), _fixup(rhs), ' * ')

class DivideOp(BinaryOp):

    def __init__(self, lhs, rhs):
        BinaryOp.__init__(self, _fixup(lhs), _fixup(rhs), ' / ')

class AddOp(BinaryOp):

    def __init__(self, lhs, rhs):
        BinaryOp.__init__(self, lhs, rhs, ' + ')

class AssignmentOp(BinaryOp):

    def __init__(self, lhs, rhs):
        BinaryOp.__init__(self, lhs, rhs, ' = ')

class PlusAssignmentOp(BinaryOp):

    def __init__(self, lhs, rhs):
        BinaryOp.__init__(self, lhs, rhs, ' += ')

class InitialisationOp(AssignmentOp):

    def unparse(self):
        lhs = self._lhs.unparse_declaration()
        rhs = self._rhs.unparse()
        return '%s%s%s' % (lhs, self._op, rhs)

    __str__ = unparse

class LessThanOp(BinaryOp):

    def __init__(self, lhs, rhs):
        BinaryOp.__init__(self, lhs, rhs, ' < ')

class ArrowOp(BinaryOp):

    def __init__(self, lhs, rhs):
        BinaryOp.__init__(self, lhs, rhs, '->')

class PlusPlusOp(BackendASTNode):

    def __init__(self, expr):
        self._expr = expr

    def unparse(self):
        return '%s++' % (self._expr.unparse())

    __str__ = unparse

class Declaration(BackendASTNode):

    def __init__(self, var):
        self._var = var

    def setCudaShared(self, isCudaShared):
        self._var.setCudaShared(isCudaShared)

    def unparse(self):
        return self._var.unparse_declaration()

    __str__ = unparse

class Cast(BackendASTNode):

    def __init__(self, t, var):
        self._t = t
        self._var = var

    def unparse(self):
        t = '(%s)' % (self._t.unparse())
        var = '(%s)' % (self._var.unparse())
        return '%s%s' % (t, var)

    __str__ = unparse

class AddressOfOp(BackendASTNode):

    def __init__(self, var):
        self._var = var

    def unparse(self):
        return '&%s' % (self._var.unparse())

    __str__ = unparse

class SizeOf(BackendASTNode):

    def __init__(self, t):
        self._t = t

    def unparse(self):
        return 'sizeof(%s)' % (self._t.unparse())

    __str__ = unparse

class Include(BackendASTNode):

    def __init__(self, header, isSystem=False):
        self._header = header
        self._isSystem = isSystem

    def unparse(self):
        if self._isSystem:
            marks = ['<', '>']
        else:
            marks = ['"', '"']

        return '#include %s%s%s' % (marks[0], self._header, marks[1])

    __str__ = unparse

class ArbitraryString(BackendASTNode):

    def __init__(self, s):
        self._s = s

    def unparse(self):
        return self._s

# Types

class Type(ModifierMixin):

    def __init__(self, isConst = False, isCudaShared = False):
        ModifierMixin.__init__(self)
        self.setConst(isConst)
        self.setCudaShared(isCudaShared)

    def unparse(self):
        modifier = self.unparse_modifier()
        internal = self.unparse_internal()
        code = '%s%s' % (modifier, internal)
        return code

    def setConst(self, isConst):
        self._setModifier(isConst, 'const')

    def setCudaShared(self, isCudaShared):
        self._setModifier(isCudaShared, '__shared__')

    def unparse_post(self):
        return ''

    __str__ = unparse

class Void(Type):

    def unparse_internal(self):
        return 'void'

class Real(Type):

    def unparse_internal(self):
        return 'double'

class Integer(Type):

    def unparse_internal(self):
        return 'int'

class Pointer(Type):

    def __init__(self, base):
        Type.__init__(self)
        self._base = base

    def getBaseType(self):
        return self._base

    def unparse_internal(self):
        return '%s*' % (self._base.unparse())

class Array(Type):

    def __init__(self, base, extents):
        Type.__init__(self)
        self._base = base
        # Convert ints to literals
        self._extents = [Literal(x) if isinstance(x,int) else x for x in as_list(extents)]

    def unparse_internal(self):
        return self._base.unparse()

    def unparse_post(self):
        code = ''
        for extent in self._extents:
            code = '%s[%s]' % (code, extent.unparse())
        return code

class Class(Type):

    def __init__(self, name):
        Type.__init__(self)
        self._name = name

    def unparse_internal(self):
        return self._name

# Utility functions

def as_list(item):
    # Empty list if we get passed None
    if item is None:
        return []
    # Convert iterable to list...
    try:
        return list(item)
    # ... or create a list of a single item
    except TypeError:
        return [item]

def buildIndexForLoop(index):
    return buildSimpleForLoop(index.name(), index.extent())

def buildSimpleForLoop(indVarName, upperBound):
    var = Variable(indVarName, Integer())
    init = InitialisationOp(var, Literal(0))
    test = LessThanOp(var, Literal(upperBound))
    inc = PlusPlusOp(var)
    return ForLoop(init, test, inc)

def getScopeFromNest(nest, depth):
    body = nest.body()
    # Descend through the bodies until we reach the correct one
    for i in range(1,depth):
        loop = body.find(lambda x: isinstance(x, ForLoop))
        body = loop.body()
    return body

def buildConstArrayInitializer(arrayName, values):
    "Build an initializer for a constant array from a given NumPy array."

    # Make input a NumPy array (and fail it it doesn't work)
    values = numpy.asarray(values, numpy.float)

    # Create a const array of appropriate shape
    var = Variable(arrayName, Array(Real(isConst=True), [Literal(x) for x in values.shape]))

    array = ArrayInitialiserList(values, newlines=True, indentation=var.unparse_declaration())
    return InitialisationOp(var, array)

# Unparser-specific functions

indentLevel = 0
indentSize = 2

def getIndent():
    return ' ' * indentLevel

def incIndent():
    global indentLevel
    indentLevel = indentLevel + indentSize
    return getIndent()

def decIndent():
    global indentLevel
    indentLevel = indentLevel - indentSize
    return getIndent()

# vim:sw=4:ts=4:sts=4:et
