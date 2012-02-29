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

# UFL modules
from ufl.algorithms import extract_arguments
from ufl.coefficient import Coefficient
from ufl.form import Form
# MCFC modules
from uflequation import UflEquation
from symbolicvalue import SymbolicValue
import canonicaliser, frontendast, formutils

class solveFunctor:
    """A functor class to be called as 'solve' in the execution namespace of
       the UFL code. Tracks forms solved for and coefficients returned for all
       solves."""

    def __init__(self):
        # Dictionary to remember all solves with result coefficient count as
        # the index and the operarands (forms for matrix and rhs) as data
        self._solves = {}
        self.__iter__ = self._solves.__iter__
        self.__len__ = self._solves.__len__
        self.items = self._solves.items

    def __call__(self,M,b):
        # FIXME we currently lose the variable names of the forms (this can be
        # looked up in the uflObjects)

        # b as the rhs should always only have a single argument
        # Extract its element
        element = extract_arguments(b)[0].element()
        coeff = Coefficient(element)
        self._solves[coeff.count()] = (M,b)
        return coeff

    def __getitem__(self, count):
        return self._solves[count]

class FluidityEquation(UflEquation):
    """Container class representing a Fluidity UFL equation with the additional
       attributes state, states and solves."""
    # FIXME we could even do without those attributes since they're already in
    # namespace

    def __init__(self, name, code, state, states):
        UflEquation.__init__(self)
        dt = SymbolicValue("dt")
        solve = solveFunctor()
        self.name = name
        self.code = code
        self.namespace = { "dt": dt, "solve": solve, "state": state, "states": states }
        # Import UFL modules into namespace
        exec "from ufl import *" in self.namespace
        # Import MCFC UFL overrides into namespace
        exec "from mcfc.uflnamespace import *" in self.namespace
        self.solves = solve
        self.state = state
        self.states = states

    def preExecute(self):
        for state in self.states.values():
            state.readyToRun()

    def _execCommonPipeline(self):
        self.preprocessCode()
        self = frontendast.generateFrontendAst(self)
        self = frontendast.preprocessFrontendAst(self)
        self = frontendast.executeFrontendAst(self)
        self = canonicaliser.canonicalise(self)
        self = formutils.partition(self)
        return self

    def execPipeline(self, fd, driver):
        self = self._execCommonPipeline()

        driver.drive(self, fd)

    def execVisualisePipeline(self, outputFile, objvis):
        self = self._execCommonPipeline()

        self.visualise(outputFile, objvis)

    def preprocessCode(self):
        # Pre-multiply dx by the Jacobian determinant (FIXME: hack!)
        import re
        self.code = re.sub(r'\bdx\b', 'detJ*dx', self.code)
        self.code = """
x = state.vector_fields['Coordinate']
J, invJ, detJ = transform(x)
def grad(u):
    return ufl.dot(invJ, ufl.grad(u))
def div(u):
    ufl.div(ufl.inner(invJ, u))
""" + self.code

    def getInputCoeffName(self, count):
        "Get the name of an input coefficient from the coefficient count"
        return self._getCoeffName(count, self.state.accessedFields(), self.getInputFieldName)

    def getResultCoeffName(self, count):
        "Get the name of a returned coefficient from the coefficient count"
        return self._getCoeffName(count, self.state.returnedFields(), self.getReturnedFieldName)

    def _getCoeffName(self, count, fields, fieldName):
        # If the coefficient is returned to state, use the field name
        if count in fields:
            return fieldName(count)
        # Otherwise (if it is temporary) use its variable name in the UFL
        for name, obj in self.uflObjects.items():
            if isinstance(obj, Coefficient) and obj.count() == count:
                return name
        # If we reach this, the coefficient wasn't found
        raise RuntimeError("Coefficient with count %s was not found." % count)

    def getResultCoeffNames(self):
        "Get a list of field names of all the fields solved for"
        return [self.getResultCoeffName(count) for count in self.solves]

    def getTmpCoeffNames(self):
        "Get a list of coefficient names solved for but not written back to state"
        return [name for name, obj in self.uflObjects.iteritems() \
                if isinstance(obj, Coefficient) \
                    and obj.count() in self.solves \
                    and obj.count() not in self.state.returnedFields()]

    def getReturnedFieldName(self, count):
        "Get the field name of a returned field from the coefficient count"
        try:
            return self.state.returnedFields()[count]
        except:
            # Raise a more informative exception
            raise RuntimeError("Coefficient with count %s is not returned to state." % count)

    def getReturnedFieldNames(self):
        return self.state.returnedFields().values()

    def getInputFieldName(self, count):
        "Get the field name of an extracted field from the coefficient count"
        return self.state.accessedFields()[count][1]

    def getFormName(self, form):
        "Look up the name of a given form in uflObjects"
        # Sanity check: we only accept forms
        assert isinstance(form, Form)
        for name, obj in self.uflObjects.items():
            if isinstance(obj, Form) and obj.form_data().original_form == form:
                return name
        # We went through all UFL objects and found nothing
        raise RuntimeError("Given form was not found:\n%s" % form)

    def getFieldFromCoeff(self, coeff):
        "Returns the first accessed field found defined over the same element as coeff"
        elem = self.uflObjects[coeff].element()
        rank = self.uflObjects[coeff].rank()
        accessed =  [name for count, name in self.state.accessedFields().values()]
        return [name for name, coeff in self.state[rank].items() \
                    if coeff.element() == elem and name in accessed][0]

# vim:sw=4:ts=4:sts=4:et
