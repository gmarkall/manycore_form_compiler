import cudaform

import ufl.form

def drive(ast, uflObjects):

    formBackend = cudaform.CudaFormBackend()

    for k in uflObjects.keys():
        o = uflObjects[k]
	if isinstance(o, ufl.form.Form):
	    ast = formBackend.compile(o)
	    print ast.unparse()
	    print
