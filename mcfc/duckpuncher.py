import asttools
import types

#additionalMethods = [ ("Form", getIntegrals), \
#                      ("scalar_fields", getField, getTimestep) \
#		      ("Argument", getElement, getCount) \
#		      ("FiniteElement", getBasis, getCell, getOrder) \
#		      ("Cell", getDomain, getDegree, getSpace) ]

def transform(tree):
    forms = asttools.findForms(tree)
    
    for form in forms:
        punch(form, getIntegral)

# Generic function for adding a bound method to an object

def punch(obj, func):
    setattr(obj, func.__name__, types.MethodType(func, obj))

# Methods to punch objects with

def getIntegrals(self):
    return self.getChildren()
