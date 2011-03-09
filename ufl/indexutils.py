
from collections import defaultdict
from ufl.indexing import Index

#--- Utility functions ---

def complete_shape(shape, default_dim):
    "Complete shape tuple by replacing non-integers with a default dimension."
    return tuple((s if isinstance(s, int) else default_dim) for s in shape)

def unique_indices(indices):
    """Return a tuple of all indices from input, with
    no single index occuring more than once in output."""
    s = set()
    newindices = []
    for i in indices:
        if isinstance(i, Index):
            if not i in s:
                s.add(i)
                newindices.append(i)
    return tuple(newindices)

def repeated_indices(indices):
    "Return tuple of indices occuring more than once in input."
    ri = []
    s = set()
    for i in indices:
        if isinstance(i, Index):
            if i in s:
                ri.append(i)
            else:
                s.add(i)
    return tuple(ri)

def shared_indices(ai, bi):
    "Return a tuple of indices occuring in both index tuples ai and bi."
    bis = set(bi)
    return tuple(i for i in unique_indices(ai) if i in bis)

def single_indices(indices):
    "Return a tuple of all indices occuring exactly once in input."
    count = defaultdict(int)
    for i in indices:
        count[i] += 1
    return tuple(i for i in unique_indices(indices) if count[i] == 1)

