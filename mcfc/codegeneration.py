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

    def unparse(self):
        code = self._name
	return code

    def unparse_declaration(self):
        name = self._name
	t = self._t.unparse()
	t_post = self._t.unparse_post()
	code = '%s %s%s' % (t, name, t_post)
	return code

class Literal(BackendASTNode):

    def __init__(self, value):
        self._value = str(value)

    def unparse(self):
        code = self._value
	return code

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

class ExpressionList(BackendASTNode):

    def __init__(self, expressions=None):
        if expressions is None:
	    self._expressions = []
	else:
	    self._expressions = expressions

    def unparse(self):
        code = '('
	if len(self._expressions) > 0:
	    code = code + self._expressions[0].unparse()
	    for e in self._expressions[1:]:
	        code = code + ', ' + e.unparse()
	code = code + ')'
	return code

class FunctionDefinition(BackendASTNode):

    def __init__(self, t, name, params=None, body=None):
        self._t = t
	self._name = name

	if params is None:
	    self._params = ParameterList()
	else:
	    self._params = params

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

class FunctionCall(BackendASTNode):

    def __init__(self, name, params=None):
        self._name = name
	if params is None:
	    self._params = ExpressionList()
	else:
	    self._params = params

    def unparse(self):
        name = self._name
	params = self._params.unparse()
        code = '%s%s' % (name, params)
	return code

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
	    code = code + '\n' + indent + s.unparse() + ';'
	indent = decIndent()
	code = code + '\n' + indent + '}'
	return code

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

    def __init__(self, lhs, rhs):
        AssignmentOp.__init__(self, lhs, rhs)

    def unparse(self):
        t = self._lhs._t.unparse()
        assignment = AssignmentOp.unparse(self, False)
	code = '%s %s' % (t, assignment)
	return code

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

class Declaration(BackendASTNode):
    
    def __init__(self, var):
        self._var = var

    def setCudaShared(self, isCudaShared):
        self._var.setCudaShared(isCudaShared)

    def unparse(self):
        return self._var.unparse_declaration()

# Types

class Type:
    
    def __init__(self):
        self._modifier = ''

    def unparse(self):
        modifier = self.unparse_modifier()
	internal = self.unparse_internal()
        code = '%s%s' % (modifier, internal)
	return code

    def setCudaShared(self, isCudaShared):
        if isCudaShared:
	    self._modifier = '__shared__ '
	else:
	    self._modifier = ''

    def unparse_modifier(self):
	return self._modifier

    def unparse_post(self):
        return ''

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

    def unparse_internal(self):
        base = self._base.unparse()
	code = '%s*' % (base)
	return code

class Array(Type):

    def __init__(self, base, length):
        Type.__init__(self)
        self._base = base
	self._length = length

    def unparse_internal(self):
        return self._base.unparse_internal()

    def unparse_post(self):
        length = self._length.unparse()
	code = '[%s]' % (length)
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
    
