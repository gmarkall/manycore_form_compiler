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
import pydot
import subprocess
import string
from ast import NodeVisitor


class DOTVisualiser:
    
    def __init__(self, tree, outputFile="tmpvis.pdf"):
        self._count = 0
	self._history = []
	self._seen = {}
	self._edgeLabel = ""

        self._graph = pydot.Dot(graph_type='digraph')
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
	fd = open(outputFile, 'w')
	fd.write(pdf)
	fd.close()

    def _dispatch(self, tree):
        raise NotImplementedError("You're supposed to implement _dispatch!")
 
    def _getFreshID(self):
        nodeID = self._count
	self._count = self._count + 1
	return str(nodeID)

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


class ASTVisualiser(DOTVisualiser):

    def _dispatch(self, tree):
        if isinstance(tree, list):
	    for t in tree:
		self._dispatch(t)
	    return
	try:
	    meth = getattr(self, "_"+tree.__class__.__name__)
	except AttributeError:
            raise NotImplementedError("Not implemented yet.")
	meth(tree)
	self._history.pop()

    def _Module(self, tree):
        self._build_node("Module")
	for stmt in tree.body:
	    self._dispatch(stmt)

    def _Expr(self, tree):
        self._build_node("Expr")
	self._dispatch(tree.value)

    def _Import(self, t):
        self._build_node("Import")
	for a in t.names:
	    self._build_node(a.name)
	    self._history.pop()

    def _Assign(self, t):
        self._build_node("Assign")
	self._build_node("targets")
	for target in t.targets:
	    self._dispatch(target)
	self._history.pop()
	self._dispatch(t.value)

    def _Str(self, t):
        self._build_node(repr(t.s))

    def _Num(self, t):
        self._build_node(repr(t.n))

    def _Name(self, t):
        self._build_node(repr(t.id))

    def _List(self, t):
        self._build_node("List")
	for el in t.elts:
	    self._dispatch(el)
	
    def _Tuple(self, t):
        self._build_node("Tuple")
	for el in t.elts:
	    self._dispatch(el)

    unop = {"Invert":"~", "Not": "not", "UAdd":"+", "USub":"-"}

    def _UnaryOp(self, t):
        self._build_node(self.unop[t.op.__class__.__name__])
	self._dispatch(t.operand)
    
    binop = { "Add":"+", "Sub":"-", "Mult":"*", "Div":"/", "Mod":"%",
              "LShift":">>", "RShift":"<<", "BitOr":"|", "BitXor":"^", "BitAnd":"&",
	      "FloorDiv":"//", "Pow": "**"}

    def _BinOp(self, t):
        self._build_node(self.binop[t.op.__class__.__name__])
	self._dispatch(t.left)
	self._dispatch(t.right)

    def _Attribute(self, t):
        self._dispatch(t.value)
	self._build_node("Attr")
	self._build_node(t.attr)
	self._history.pop()

    def _Call(self, t):
        self._build_node("Call")
	self._dispatch(t.func)
	self._build_node("Arguments")
	for arg in t.args:
	    self._dispatch(arg)
	self._history.pop()
	
    def _Subscript(self, t):
        self._build_node("Subscript")
	self._dispatch(t.value)
	self._dispatch(t.slice)

    def _Index(self, t):
        self._build_node("Index")
	self._dispatch(t.value)


class ObjectVisualiser(DOTVisualiser):

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

    def _generic_visit_number(self, obj):
        self._build_node(str(obj), self._edgeLabel)

    _visit_int = _generic_visit_number
    _visit_float = _generic_visit_number

    def _visit_str(self, obj):
        oneline = obj.replace("\n", "/").replace("\r","")
	printable = self._printable(oneline)
	if len(printable) > 30:
	    short = printable[:30] + "..."
	else:
	    short = printable
        self._build_node(short, self._edgeLabel)

    def _visit_list(self, obj):
        self._generic_visit_array(obj, "list")

    def _visit_tuple(self, obj):
        self._generic_visit_array(obj, "tuple")

    def _generic_visit_array(self, arr, rootLabel):
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

class ReprVisualiser(DOTVisualiser):

    def _dispatch(self, obj):
        if isinstance(obj, list):
	    for o in obj:
		self._dispatch(o)
	    return
	try:
	    meth = getattr(self, "_visit_"+obj.__class__.__name__)
	except AttributeError:
            print "Class:", obj.__class__.__name__
	    raise NotImplementedError("Not implemented yet.")
	meth(obj)
        
    def _visit_Module(self, obj):
        for stmt in obj.body:
	    self._dispatch(stmt)

    def _visit_Expr(self, obj):
        self._dispatch(obj.value)

    def _visit_Call(self, obj):
        self._build_node(obj.func.id)
        for arg in obj.args:
	    self._dispatch(arg)
	self._history.pop()

    def _visit_List(self, obj):
        self._generic_visit_array(obj, "list")

    def _visit_Tuple(self, obj):
        self._generic_visit_array(obj, "tuple")

    def _generic_visit_array(self, obj, label):
        self._build_node(label)
	for item in obj.elts:
	    self._dispatch(item)
	self._history.pop()

    def _visit_Num(self, obj):
        self._build_node(str(obj.n))
	self._history.pop()

    def _visit_Dict(self, obj):
        self._build_node("dict")
        for key, value in zip(obj.keys, obj.values):
	    self._build_node("item")
	    self._dispatch(key)
	    self._dispatch(value)
	    self._history.pop()
	self._history.pop()

    def _visit_Str(self, obj):
        s = "'%s'" % obj.s
	self._build_node(s)
	self._history.pop()

    def _visit_Name(self, obj):
        self._build_node(obj.id)
	self._history.pop()

# vim:sw=4:ts=4:sts=4:et
