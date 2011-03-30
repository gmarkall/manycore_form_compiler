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


import visitor
import pydot
import subprocess

class Visualiser(visitor.AntlrVisitor):

    def __init__(self, outputFile="tmpvis.pdf"):
        visitor.AntlrVisitor.__init__(self, visitor.preOrder)
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
        self.traverse(tree)
        
        # The root node is left over, and needs popping
        self._history.pop()
        
        # Something went wrong if there's anything left
        if not len(self._history) == 0:
            raise RuntimeError("History stack not empty.")

        # Create and write out pdf
        pdf = self._graph.create_pdf(prog='dot')
        fd = open(self._pdfFile, 'w')
        fd.write(pdf)
        fd.close

    def visit(self, tree):
        # Identifiers for the new and previous node
        nodeID = self._getFreshID()
        prevNodeID = self._history[-1]
        nodeLabel = str(tree)
        
        # Construct new node and edge
        node = pydot.Node(nodeID, label=nodeLabel)
        edge = pydot.Edge(prevNodeID,nodeID)
        
        # Add node and edge to graph
        self._graph.add_node(node)
        self._graph.add_edge(edge)
        
        # Add the current node to the history stack
        self._history.append(nodeID)

    def pop(self):
        self._history.pop()

# vim:sw=4:ts=4:sts=4:et
