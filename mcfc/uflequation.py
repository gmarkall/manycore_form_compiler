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

# MCFC libs
from uflvisualiser import ASTVisualiser, ObjectVisualiser, ReprVisualiser

class UflEquation:
    """Base class representing an equation in UFL with basic attributes.
       Objects of this type are passed from stage to stage in the MCFC
       pipeline.
       
       Attributes:
         name:        equation name
         code:        UFL code
         namespace:   namespace the UFL is executed in
         frontendAST: Python AST corresponding to the UFL code
         uflObjects:  UFL forms, coefficients, arguments
         dag:         intermediate DAG representation of the equation
         backendAst:  custom backend specific AST for unparsing"""

    def __init__(self):
        self.name = ""
        self.code = ""
        self.namespace = {}
        self.frontendAst = None
        self.uflObjects = {}
        self.dag = None
        self.backendAst = None

    def preExecute(self):
        """Called immediately prior to execution of the UFL code. Override this
           method in a derived class to do any necessary initialisation."""
        pass

    def execPipeline(self, fd, driver):
        "Invokes the code generation pipeline"
        raise NotImplementedError("You're supposed to implement execPipeline()")

    def execVisualisePipeline(self, outputFile, objvis):
        "Invokes the visualisation pipeline"
        raise NotImplementedError("You're supposed to implement execVisualisePipeline()")

    def visualise(self, outputFile, obj=False):
        ASTVisualiser(self.frontendAst, outputFile + ".pdf")
        for name, obj in self.uflObjects.items():
            objectfile = "%s_%s.pdf" % (outputFile, name)
            if obj:
                ObjectVisualiser(obj, objectfile)
            else:
                rep = ast.parse(repr(obj))
                ReprVisualiser(rep, objectfile)

# vim:sw=4:ts=4:sts=4:et
