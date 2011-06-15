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

import sys
import visitor
import pydot
import subprocess
import string
from ast import NodeVisitor


class ASTVisualiser:

    def __init__(self, outputFile="tmpvis.pdf"):
        self._pdfFile = outputFile
	self._count = 0
	self._history = []

    def _getFreshID(self):
        nodeID = self._count
	self._count = self._count + 1
	return str(nodeID)

    def visualise(self, tree):
        self._graph = pydot.Dot(graph_type='digraph')

	# We need one node to be the root of all others
	beginID = self._getFreshID()
	self._graph.add_node(pydot.Node(beginID, label='begin'))
	self._history.append(beginID)

	# Traverse the rest of the tree
	self.dispatch(tree)

	# The root node is left over, and needs popping
	self._history.pop()

	# Something went wrong if there's anything left
	if not len(self._history) == 0:
	    raise RuntimeError("History stack not empty.")

	# Create and write out pdf
	pdf = self._graph.create_pdf(prog='dot')
	fd = open(self._pdfFile, 'w')
	fd.write(pdf)
	fd.close()

    def dispatch(self, tree):
        if isinstance(tree, list):
	    # FIXME: Create a list node to act as the root
	    for t in tree:
		self.dispatch(t)
	    return
	meth = getattr(self, "_"+tree.__class__.__name__)
	meth(tree)
	self._history.pop()

    def _build_node(self, nodeLabel, edgeLabel=""):
        # Identifiers for the new and previous node
        nodeID = self._getFreshID()
        prevNodeID = self._history[-1]
                
	# Construct new node and edge
        node = pydot.Node(nodeID, label=nodeLabel)
        edge = pydot.Edge(prevNodeID,nodeID,label=edgeLabel)
        
        # Add node and edge to graph
        self._graph.add_node(node)
        self._graph.add_edge(edge)
        
        # Add the current node to the history stack
        self._history.append(nodeID)

    def _Module(self, tree):
        self._build_node("Module")
	for stmt in tree.body:
	    self.dispatch(stmt)

    def _Expr(self, tree):
        self._build_node("Expr")
	self.dispatch(tree.value)

    def _Import(self, t):
        self._build_node("Import")
	for a in t.names:
	    self._build_node(a.name)
	    self._history.pop()

    def _ImportFrom(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Assign(self, t):
        self._build_node("Assign")
	self._build_node("targets")
	for target in t.targets:
	    self.dispatch(target)
	self._history.pop()
	self.dispatch(t.value)

    def _AugAssign(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Return(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Pass(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Break(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Continue(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Delete(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Assert(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Exec(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Print(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Global(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Yield(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Raise(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _TryExcept(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _TryFinally(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _ExceptHandler(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _ClassDef(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _FunctionDef(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _For(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _If(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _While(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _With(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Str(self, t):
        self._build_node(repr(t.s))

    def _Num(self, t):
        self._build_node(repr(t.n))

    def _Name(self, t):
        self._build_node(repr(t.id))

    def _Repr(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _List(self, t):
        self._build_node("List")
	for el in t.elts:
	    self.dispatch(el)
	
    def _ListComp(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _GeneratorExp(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _comprehension(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _IfExp(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Dict(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Tuple(self, t):
        self._build_node("Tuple")
	for el in t.elts:
	    self.dispatch(el)

    unop = {"Invert":"~", "Not": "not", "UAdd":"+", "USub":"-"}

    def _UnaryOp(self, t):
        self._build_node(self.unop[t.op.__class__.__name__])
	self.dispatch(t.operand)
    
    binop = { "Add":"+", "Sub":"-", "Mult":"*", "Div":"/", "Mod":"%",
              "LShift":">>", "RShift":"<<", "BitOr":"|", "BitXor":"^", "BitAnd":"&",
	      "FloorDiv":"//", "Pow": "**"}

    def _BinOp(self, t):
        self._build_node(self.binop[t.op.__class__.__name__])
	self.dispatch(t.left)
	self.dispatch(t.right)

    def _Compare(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Attribute(self, t):
        self.dispatch(t.value)
	self._build_node("Attr")
	self._build_node(t.attr)
	self._history.pop()

    def _Call(self, t):
        self._build_node("Call")
	self.dispatch(t.func)
	self._build_node("Arguments")
	for arg in t.args:
	    self.dispatch(arg)
	self._history.pop()
	
    def _Subscript(self, t):
        self._build_node("Subscript")
	self.dispatch(t.value)
	self.dispatch(t.slice)

    def _Ellipsis(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Index(self, t):
        self._build_node("Index")
	self.dispatch(t.value)

    def _Slice(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _ExtSlice(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _arguments(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _keyword(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _Lambda(self, t):
        raise NotImplementedError("Not implemented yet.")

    def _alias(self, t):
        raise NotImplementedError("Not implemented yet.")

class ObjectVisualiser:

    def __init__(self, outputFile="tmpvis.pdf"):
        self._pdfFile = outputFile
	self._count = 0
	self._history = []

    def _getFreshID(self):
        nodeID = self._count
	self._count = self._count + 1
	return str(nodeID)

    def visualise(self, tree):
        self._graph = pydot.Dot(graph_type='digraph')
        self._seen = {}
	self._edgeLabel = ""

	# We need one node to be the root of all others
	beginID = self._getFreshID()
	self._graph.add_node(pydot.Node(beginID, label='begin'))
	self._history.append(beginID)

	# Traverse the rest of the tree
	self._dispatch(tree)

	# The root node is left over, and needs popping
	self._history.pop()

	# Something went wrong if there's anything left
	if not len(self._history) == 0:
	    raise RuntimeError("History stack not empty.")

	# Create and write out pdf
	pdf = self._graph.create_pdf(prog='dot')
	fd = open(self._pdfFile, 'w')
	fd.write(pdf)
	fd.close()
    
    def _build_node(self, nodeLabel, edgeLabel=""):
        # Identifiers for the new and previous node
        nodeID = self._getFreshID()
        prevNodeID = self._history[-1]
                
	# Construct new node and edge
        node = pydot.Node(nodeID, label=nodeLabel)
        edge = pydot.Edge(prevNodeID,nodeID,label=edgeLabel)
        
        # Add node and edge to graph
        self._graph.add_node(node)
        self._graph.add_edge(edge)
        
        # Add the current node to the history stack
        self._history.append(nodeID)

	# In case we want to do anything with the new node
	return nodeID

    def _build_edge_to_existing(self, existingID, edgeLabel=""):
        prevNodeID = self._history[-1]
	edge = pydot.Edge(prevNodeID, existingID, label=edgeLabel)
	self._graph.add_edge(edge)

    def _dispatch(self, obj):
        # Don't redraw an object we've already seen; just link to it.
	OID = id(obj)
	if OID in self._seen.keys():
	    nodeID = self._seen[OID]
	    self._build_edge_to_existing(nodeID, self._edgeLabel)
	    return
	
	try:
	    meth = getattr(self, "_visit_" + obj.__class__.__name__)
	except AttributeError:
	    meth = self._generic_visit
	meth(obj)
	self._history.pop()

    def _printable(self, s):
        printable = True
	for c in s:
	    if c not in string.printable:
	        printable = False
	if printable:
	    return s
	else:
	    return "Unprintable"

    def _generic_visit(self, obj):
        label = obj.__class__.__name__
	nodeID = self._build_node(label, self._edgeLabel)
	OID = id(obj)
	self._seen[OID] = nodeID
	savedLabel = self._edgeLabel
	attrs = dir(obj)
	visible = [ s for s in attrs if s[:2] != "__" ]
	for a in visible:
	    self._edgeLabel = a
	    try:
	        objattr = getattr(obj, a)
	    except:
	        self._handle_exception(sys.exc_info()[1])
		continue
	    attrtype = objattr.__class__.__name__
	    if attrtype not in [ "builtin_function_or_method", "instancemethod" ]:
	        self._dispatch(objattr)
	self._edgeLabel = savedLabel

    def _visit_int(self, obj):
        self._build_node(str(obj), self._edgeLabel)

    def _visit_str(self, obj):
        oneline = obj.replace("\n", "/").replace("\r","")
	printable = self._printable(oneline)
	if len(printable) > 30:
	    short = printable[:30] + "..."
	else:
	    short = printable
        self._build_node(short, self._edgeLabel)

    def _visit_list(self, obj):
        self._visit_array(obj, "list")

    def _visit_tuple(self, obj):
        self._visit_array(obj, "tuple")

    def _visit_array(self, arr, rootLabel):
        nodeID = self._build_node(rootLabel, self._edgeLabel)
	OID = id(arr)
	self._seen[OID] = nodeID
	savedLabel = self._edgeLabel
	count = 0
	for i in arr:
	    self._edgeLabel = str(count)
	    self._dispatch(i)
	    count = count + 1
	self._edgeLabel = savedLabel

    def _handle_exception(self, ex):
        print "getattr() failed. Not fatal. Exception follows:"
	print str(ex)

# vim:sw=4:ts=4:sts=4:et
