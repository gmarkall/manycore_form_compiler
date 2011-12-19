from ufl.finiteelement import FiniteElement, VectorElement, TensorElement

# From http://www.peterbe.com/plog/uniqifiers-benchmark. This is version f5.
def uniqify_list(seq, idfun=None):
    idfun = idfun or (lambda x: x)
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

# From http://code.activestate.com/recipes/52560-remove-duplicates-from-a-sequence/
def uniqify_unordered(seq):
    seen = {}
    for item in seq:
        seen[item] = 1
    return seen.keys()

def uniqify(seq, idfun=None, seen=None):
    idfun = idfun or (lambda x: x)
    seen = seen or {}
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        yield item

def uniqify_rec(seq, idfun=None, seen=None):
    idfun = idfun or (lambda x: x)
    seen = seen or {}
    for item in seq:
        try:
            for s in uniqify_rec(item, idfun, seen):
                yield s
        except TypeError:
            marker = idfun(item)
            if marker in seen: continue
            seen[marker] = 1
            yield item

# Used by both the assembler generator and the form generator.

def formElementRank(form):
    # Use the element from the first argument, which should be the TestFunction
    arg = form.form_data().arguments[0]
    e = arg.element()

    if isinstance(e, FiniteElement):
        return 0
    elif isinstance(e, VectorElement):
        return 1
    elif isinstance(e, TensorElement):
        return 2
    else:
        raise RuntimeError("Not a recognised element.")

def formElementSpaceDim(form):
    # Use the element from the first argument, which should be the TestFunction
    arg = form.form_data().arguments[0]
    e = arg.element()
    return e.cell().geometric_dimension()

# From http://ipython.scipy.org/doc/manual/html/interactive/reference.html#embedding-ipython
#
# paste these 2 lines at the point you want to drop into the IPython shell:
# from utilities import embedded_ipython_shell as ipshell
# ipshell()()
def embedded_ipython_shell(msg=None):
    "Return an embedded IPython shell. Call to drop into the shell."
    from IPython.Shell import IPShellEmbed
    ipshell = IPShellEmbed(argv=[], banner=msg or "Dropping into IPython shell...")
    return ipshell

# vim:sw=4:ts=4:sts=4:et
