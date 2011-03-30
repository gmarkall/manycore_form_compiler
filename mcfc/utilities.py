# From http://www.peterbe.com/plog/uniqifiers-benchmark. This is version f5.
def uniqify(seq, idfun=None):
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

# vim:sw=4:ts=4:sts=4:et
