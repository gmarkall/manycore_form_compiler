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

class BackendASTNode:
    pass

class Subscript(BackendASTNode):

    def __init__(self, base, offset):
        self._base = base
        self._offset = offset

    def unparse(self):
        base = self._base.unparse()
        offset = self._offset.unparse()
        code = '%s[%s]' % (base, offset)
        return code

    __str__ = unparse

class Dereference(BackendASTNode):

    def __init__(self, expr):
        self._expr = expr

    def unparse(self):
        expr = self._expr.unparse()
        code = '*%s' % (expr)
        return code

    __str__ = unparse

class NullExpression(BackendASTNode):

    def unparse(self):
        return ''

class Variable(BackendASTNode):

    def __init__(self, name, t=None):
        self._name = name

        if t is None:
            self._t = Type()
        else:
            self._t = t

    def __hash__(self):
        return self._name.__hash__()

    def __eq__(self, other):
        return self._name == other._name

    def setCudaShared(self, isCudaShared):
        self._t.setCudaShared(isCudaShared)

    def getType(self):
        return self._t

    def getName(self):
        return self._name

    def unparse(self):
        code = self._name
        return code

    def unparse_declaration(self):
        name = self._name
        t = self._t.unparse()
        t_post = self._t.unparse_post()
        code = '%s %s%s' % (t, name, t_post)
        return code

    __str__ = unparse_declaration

class Literal(BackendASTNode):

    def __init__(self, value):
        if isinstance(value, str):
            self._value = '"' + value + '"'
        else:
            self._value = str(value)

    def unparse(self):
        code = self._value
        return code

    __str__ = unparse

class ForLoop(BackendASTNode):

    def __init__(self, init, test, inc, body=None):
        self._init = init
        self._test = test
        self._inc = inc
        if body is None:
            self._body = Scope()
        else:
            self._body = body

    def append(self, statement):
        self._body.append(statement)

    def prepend(self, statement):
        self._body.prepend(statement)

    def body(self):
        return self._body

    def unparse(self):
        init = self._init.unparse()
        test = self._test.unparse(False)
        inc = self._inc.unparse()
        body = self._body.unparse()
        code = 'for(%s; %s; %s)\n' % (init, test, inc)
        code = code + body
        return code

    __str__ = unparse

class ParameterList(BackendASTNode):

    def __init__(self, params=None):
        if params is None:
            self._params = []
        else:
            self._params = params

    def unparse(self):
        code = "("
        if len(self._params) > 0:
            code = code + self._params[0].unparse_declaration()
            for p in self._params[1:]:
                code = code + ", " + p.unparse_declaration()
        code = code + ")"
        return code

    __str__ = unparse

class ExpressionList(BackendASTNode):

    def __init__(self, expressions=None):
        if expressions is None:
            self._expressions = []
        else:
            self._expressions = expressions

    def append(self, expression):
         self._expressions.append(expression)

    def prepend(self, expression):
         self._expressions.insert(0, expression)

    def unparse(self):
        code = '('
        if len(self._expressions) > 0:
            code = code + self._expressions[0].unparse()
            for e in self._expressions[1:]:
                code = code + ', ' + e.unparse()
        code = code + ')'
        return code

    __str__ = unparse

class FunctionDefinition(BackendASTNode):

    def __init__(self, t, name, params=None, body=None):
        self._t = t
        self._name = name
        self._modifier = ''
        self._params = ParameterList(params)

        if body is None:
            self._body = Scope()
        else:
            self._body = body

    def setCudaKernel(self, isCudaKernel):
        if isCudaKernel:
            self._modifier = "__global__ "
        else:
            self._modifier = ""

    def setExternC(self, isExternC):
        if isExternC:
            self._modifier = 'extern "C" '
        else:
            self._modifier = ''

    def append(self, statement):
        self._body.append(statement)

    def prepend(self, statement):
        self._body.prepend(statement)

    def unparse(self):
        mod = self._modifier
        t = self._t.unparse()
        params = self._params.unparse()
        body = self._body.unparse()
        code = '%s%s %s%s\n%s' % (mod, t, self._name, params, body)
        return code

    __str__ = unparse

class FunctionCall(BackendASTNode):

    def __init__(self, name, params=None):
        self._name = name
        if params is None:
            self._params = ExpressionList()
        elif isinstance(params, list):
            self._params = ExpressionList(params)
        else:
            self._params = params

    def unparse(self):
        name = self._name
        params = self._params.unparse()
        code = '%s%s' % (name, params)
        return code

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
            shMemSize = self._shMemSize.unparse()
            config = config + ',' + shMemSize
            if self._stream is not None:
                stream = self._stream.unparse()
                config = config + ',' + stream

        code = '%s<<<%s>>>%s' % (name, config, params)
        return code

    __str__ = unparse

class Scope(BackendASTNode):

    def __init__(self, statements=None):
        if statements is None:
            self._statements = []
        else:
            self._statements = statements

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
        code = '%s{' % (indent)
        indent = incIndent()
        for s in self._statements:
            if isinstance(s, BinaryOp):
                unparsed = s.unparse(False)
            else:
                unparsed = s.unparse()
            code = code + '\n' + indent + unparsed + ';'
        indent = decIndent()
        code = code + '\n' + indent + '}'
        return code

    __str__ = unparse

class GlobalScope(Scope):

    def __init__(self, statements=None):
        Scope.__init__(self, statements)

    def unparse(self):
        code = ''
        for s in self._statements:
            if isinstance(s, Include):
                code = code + s.unparse() + '\n'
            elif isinstance(s, FunctionDefinition):
                code = code + s.unparse() + '\n\n'
            else:
                code = code + s.unparse() + ';\n'
        return code

    __str__ = unparse

class New(BackendASTNode):

    def __init__(self, t, params=None):
        self._t = t
        if params is None:
            self._params = ExpressionList()
        else:
            self._params = params

    def unparse(self):
        t = self._t.unparse_internal()
        params = self._params.unparse()
        code = 'new %s%s' % (t, params)
        return code

    __str__ = unparse

class Delete(BackendASTNode):

    def __init__(self, var):
        self._var = var

    def unparse(self):
        var = self._var.unparse()
        code = 'delete %s' % (var)
        return code

    __str__ = unparse

class BinaryOp(BackendASTNode):

    def __init__(self, lhs, rhs, op):
        self._lhs = lhs
        self._rhs = rhs
        self._op = op

    def unparse(self, bracketed=True):
        lhs = self._lhs.unparse()
        rhs = self._rhs.unparse()
        code = '%s %s %s' % (lhs, self._op, rhs)
        if bracketed:
            code = '(' + code + ')'
        return code

    __str__ = unparse

class MultiplyOp(BinaryOp):

    def __init__(self, lhs, rhs):
        BinaryOp.__init__(self, lhs, rhs, '*')

class AddOp(BinaryOp):

    def __init__(self, lhs, rhs):
        BinaryOp.__init__(self, lhs, rhs, '+')

class AssignmentOp(BinaryOp):

    def __init__(self, lhs, rhs):
        BinaryOp.__init__(self, lhs, rhs, '=')

class PlusAssignmentOp(BinaryOp):

    def __init__(self, lhs, rhs):
        BinaryOp.__init__(self, lhs, rhs, '+=')

class InitialisationOp(AssignmentOp):

    def unparse(self):
        lhs = self._lhs.unparse_declaration()
        rhs = self._rhs.unparse()
        return '%s %s %s' % (lhs, self._op, rhs)

    __str__ = unparse

class LessThanOp(BinaryOp):

    def __init__(self, lhs, rhs):
        BinaryOp.__init__(self, lhs, rhs, '<')

class ArrowOp(BinaryOp):

    def __init__(self, lhs, rhs):
        BinaryOp.__init__(self, lhs, rhs, '->')

class PlusPlusOp(BackendASTNode):

    def __init__(self, expr):
        self._expr = expr

    def unparse(self):
        expr = self._expr.unparse()
        code = '%s++' % (expr)
        return code

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
        code = '%s%s' % (t, var)
        return code

    __str__ = unparse

class AddressOfOp(BackendASTNode):

    def __init__(self, var):
        self._var = var

    def unparse(self):
        var = self._var.unparse()
        code = '&%s' % (var)
        return code

    __str__ = unparse

class SizeOf(BackendASTNode):

    def __init__(self, t):
        self._t = t

    def unparse(self):
        t = self._t.unparse()
        code = 'sizeof(%s)' % (t)
        return code

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

        code = '#include %s%s%s' % (marks[0], self._header, marks[1])
        return code

    __str__ = unparse

# Types

class Type:

    def __init__(self, isConst = False, isCudaShared = False):
        # FIXME need the order of modifiers be enforced / preserved?
        self._modifier = set()
        self.setConst(isConst)
        self.setCudaShared(isCudaShared)

    def unparse(self):
        modifier = self.unparse_modifier()
        internal = self.unparse_internal()
        code = '%s%s' % (modifier, internal)
        return code

    def _setModifier(self, setModifier, modifier):
        if setModifier:
            self._modifier.add(modifier)
        else:
            self._modifier.discard(modifier)

    def setConst(self, isConst):
        self._setModifier(isConst, 'const ')

    def setCudaShared(self, isCudaShared):
        self._setModifier(isCudaShared, '__shared__ ')

    def unparse_modifier(self):
        return ''.join(self._modifier)

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
        base = self._base.unparse()
        code = '%s*' % (base)
        return code

class Array(Type):

    def __init__(self, base, extents):
        Type.__init__(self)
        self._base = base
        self._extents = extents if isinstance(extents,list) else [extents]

    def unparse_internal(self):
        return self._base.unparse_internal()

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

def buildSimpleForLoop(indVarName, upperBound):
    var = Variable(indVarName, Integer())
    init = InitialisationOp(var, Literal(0))
    test = LessThanOp(var, Literal(upperBound))
    inc = PlusPlusOp(var)
    ast = ForLoop(init, test, inc)
    return ast

def getScopeFromNest(nest, depth):
    body = nest.body()
    # Descend through the bodies until we reach the correct one
    for i in range(1,depth):
        loop = body.find(lambda x: isinstance(x, ForLoop))
        body = loop.body()
    return body

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
