# From http://www.peterbe.com/plog/uniqifiers-benchmark. This is version f5.
def uniqify_list(seq, idfun=None):
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

def uniqify(seq, idfun=None):
    if idfun is None:
        def idfun(x): return x
    seen = {}
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        yield item

def embedded_ipython_shell(msg=None):
    "Return an embedded IPython shell. Call to drop into the shell."
    from IPython.Shell import IPShellEmbed
    ipshell = IPShellEmbed(banner=msg or "Dropping into IPython shell...")
    return ipshell

# vim:sw=4:ts=4:sts=4:et
