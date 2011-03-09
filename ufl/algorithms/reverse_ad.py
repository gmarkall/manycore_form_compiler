"""Reverse mode AD implementation."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-12-28 -- 2009-01-07"

# TODO: Imports!

from ufl.algorithms.pdiffs import PartialDerivativeComputer
from ufl.differentiation import SpatialDerivative, VariableDerivative, CoefficientDerivative

def reverse_ad(expr, G): # TODO: Finish this!
    # --- Forward sweep expressions have already been recorded as vertices in the DAG

    # TODO: Can't we just build the graph from expr in here? Need special treatement if a VariableDerivative!
    #G = build_graph(expr)
    V, E = G
    m = len(V)

    # We're computing df/fv:
    f, v = expr.operands()

    if isinstance(expr, SpatialDerivative):
        # Need to define dx_i/dx_j = delta_ij?
        pass

    if isinstance(expr, VariableDerivative):
        # Avoid putting contents of the differentiation Variable in graph, since it's not a Terminal anymore... TODO
        pass

    if isinstance(expr, CoefficientDerivative):
        # Define dw/dw := v (what we really mean by d/dw is d/dw_j where w = sum_j w_j phi_j, and what we really mean by v is phi_j for any j)
        pass

    # Initialize graph
    x = [0]*m

    # Size of v, could be larger if v is multiple thingys
    n = 1
    x[:n] = TODO
    # Actually, we don't have these variables...
    # v is a MultiIndex instead of SpatialCoordinate,
    # or a Coefficient instead of a dof,
    # or a Variable (in which case the Variable shouldn't be traversed when building the graph)
    # ... but then again, we have many expressions that doesn't depend directly on v, but implicitly by definition...

    x[n:] = V[:] # = f_i( <x[j]> ) for j in dependencies of f_i
    #for i in range(n, m+1):
    #    x[i] = V[i-n] # = f_i( <x[j]> ) for j in dependencies of f_i

    # Initialize xd
    gamma = 1
    g = [0]*n
    xd = [0]*m
    xd[m-1] = gamma
    xd[:n] = g

    # Compute c[i,j] = df_i/dx_j TODO
    pdc = PartialDerivativeComputer()
    c = {}
    for i, v in enumerate(V):
        pdiffs = pdc(v)
        vi_edges = TODO
        for (j, dvidvj) in zip(vi_edges, pdiffs):
            c[(i,j)] = dvidvj

    # Reverse accumulation
    for i in range(m-1, n-1, -1):
        xdi = xd[i]
        for j in Eout[i-n]: # TODO: Correct edges, j should be the x indices of the operands of x[i]
            xd[j] += xdi*c[i,j]
    result = xd[:n]

    return result
