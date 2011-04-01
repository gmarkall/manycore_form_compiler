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
import form
from codegeneration import Variable, Integer, Pointer, Real
from utilities import uniqify
# FEniCS UFL libs
import ufl
from ufl.algorithms.transformations import Transformer

numElements = Variable("n_ele", Integer() )
statutoryParameters = [ form.localTensor, numElements, form.timestep, form.detwei ]

class KernelParameterGenerator(Transformer):
    """Mirrors the functionality of the kernelparametercomputer
    in cudaform.py - maybe merge at some point?"""

    def generate(self, tree, rootForm):
        # Temporary storage for each type
        self._coefficients = []
        self._arguments = []
        self._spatialDerivatives = []

        # Traversal records instances of each type
        self.visit(tree)

        # Initialise with parameters common to all kernels
        formalParameters = list(statutoryParameters)
        actualParameters = []
        
        # We need to map the coefficient and argument counts that
        # we see in the form back to the ones with their original
        # counts that are used earlier in the AST
        form_data = rootForm.form_data()
        formCoefficients = form_data.coefficients
        originalCoefficients = form_data.original_coefficients
        formArguments = form_data.arguments
        originalArguments = form_data.original_arguments

        # We need to process the lists we have created during
        # the traversal. These results are stored in these new lists.
        coefficients = self._coefficients
        arguments = self._arguments
        argumentDerivatives = []

        # SpatialDerivatives are tricky: if its a derivative of a basis function,
        # then we just need to pass the derivative of the basis function. If its a
        # derivative of a coefficient, we need the values of the coefficients and
        # the derivatives of the trial functions. 
        for d in self._spatialDerivatives:
            subject = d.operands()[0]
            indices = d.operands()[1]

            if isinstance(subject, ufl.argument.Argument):
                argumentDerivatives.append(d)
            elif isinstance(subject, ufl.coefficient.Coefficient):
                coefficients.append(subject)
                element = subject.element()
                # The reasoning behind giving the argument a count of 0 is that
                # there will always be at least one basis function, which will
                # be numbered 0. Need to check if this is correct for the case
                # where there is no basis function with the same basis as the 
                # coefficient though.
                argument = ufl.argument.Argument(element, 0)
                argumentDeriv = ufl.differentiation.SpatialDerivative(argument, indices)
                argumentDerivatives.append(argumentDeriv)

        # We don't want the same argument to appear twice, so we need to
        # uniqify the lists. 
        coefficients = uniqify(coefficients)
        arguments = uniqify(arguments, lambda x: x.element())
        argumentDerivatives = uniqify(argumentDerivatives, lambda x: x.operands()[0].element())
        
        # Build the lists of parameters based on what we have observed
        # in the form.
        for coeff in coefficients:
            # Actual parameter
            i = formCoefficients.index(coeff)
            originalCoefficient = originalCoefficients[i]
            actualParameters.append(originalCoefficient)

            # Formal parameter
            name = form.buildCoefficientName(coeff)
            parameter = Variable(name, Pointer(Real()))
            formalParameters.append(parameter)

        for arg in arguments:
            # Actual parameter
            i = formArguments.index(arg)
            originalArgument = originalArguments[i]
            actualParameters.append(originalArgument)
        
            # Formal parameter
            name = form.buildArgumentName(arg)
            parameter = Variable(name, Pointer(Real()))
            formalParameters.append(parameter)
        
        for argDeriv in argumentDerivatives:
            subject = argDeriv.operands()[0]
            indices = argDeriv.operands()[1]
            
            # Actual parameter
            i = formArguments.index(subject)
            originalArgument = originalArguments[i]
            actualParameters.append(ufl.differentiation.SpatialDerivative(originalArgument,indices))

            # Formal parameter
            name = form.buildSpatialDerivativeName(argDeriv)
            parameter = Variable(name, Pointer(Real()))
            formalParameters.append(parameter)

        return formalParameters, actualParameters

    def expr(self, tree, *ops):
        pass

    def spatial_derivative(self, tree):
        self._spatialDerivatives.append(tree)

    def argument(self, tree):
        self._arguments.append(tree)

    def coefficient(self, tree):
        self._coefficients.append(tree)

def generateKernelParameters(tree, form):
    KPG = KernelParameterGenerator()
    return KPG.generate(tree, form)
