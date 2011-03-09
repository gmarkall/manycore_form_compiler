"Algorithms for tuple notation a = (v, u) + (grad(v), grad(u))."

__authors__ = "Anders Logg (logg@simula.no)"
__date__    = "2009-04-05 -- 2009-06-19"

from ufl.log import error
from ufl.form import Form
from ufl.integral import Measure, Integral
from ufl.operators import inner
from ufl.objects import dx

def tuple2form(objects):
    "Convert given tuple (or list) to a UFL form."

    # Make sure we get a list or tuple
    if not isinstance(objects, (list, tuple)):
        error("Unable to extract UFL form, expecting a tuple: %s" % repr(objects))

    # Operands
    v = w = None

    # Iterate over objects and extract integrals
    integrals = []
    for object in objects:

        # Found plain integral, just append
        if isinstance(object, Integral):
            integrals.append(object)

        # Found measure, append inner(v, w)*dm
        elif isinstance(object, Measure):
            dm = object
            if v is None or w is None:
                error("Unable to extract UFL form, found measure without matching integrands: " + str(dm))
            else:
                form = inner(v, w)*dm
                integrals += form.integrals()
                v = w = None

        # Found first operand, store v
        elif v is None and w is None:
            v = object

        # Found second operand, store w
        elif w is None:
            w = object

        # Found new operand, assume measure is dx
        elif not v is None and not w is None:
            form = inner(v, w)*dx
            integrals += form.integrals()
            v = object
            w = None

        # Default case, should not get here
        else:
            error("Unable to extract UFL form, expression does not make sense: %s" % repr(objects))

    # Add last inner product if any
    if not v is None and not w is None:
        form = inner(v, w)*dx
        integrals += form.integrals()

    # Create Form from integrals
    form = Form(integrals)

    return form

# TODO: This might fit better in form.py but I wasn't able
# TODO: to place it there because of a recursive import.

def as_form(form):
    "Convert to form if not a form, otherwise return form."

    # Check form Form
    if isinstance(form, Form):
        return form

    # Check for tuple
    if isinstance(form, tuple):
        return tuple2form(form)

    error("Unable to convert object to a UFL form: %s" % repr(form))
