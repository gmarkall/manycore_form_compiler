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

"Module for all pipeline stages dealing with the frontend AST."

# Regular python modules
import ast

def generateFrontendAst(equation):
    "Generate the frontend AST from the UFL code"
    equation.frontendAst = ast.parse(equation.code)
    return equation

def preprocessFrontendAst(equation):
    "Preprocess the frontend AST (currently a no-op)"
    return equation

def executeFrontendAst(equation):
    "Exectute the frontend AST in the given namespace"

    # Do any initialisation the UFL equation might require
    equation.preExecute()

    # Compile the frontend AST and execute it in the UFL equation namespace
    exec compile(equation.frontendAst, equation.name, 'exec') in equation.namespace
    return equation

# vim:sw=4:ts=4:sts=4:et
