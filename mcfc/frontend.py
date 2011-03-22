"""    Frontend. Usage:
 
    frontend.py [options] input

    Options can be: 
    
      --visualise, -v (to output a visualisation of the AST)
      -o:<filename>   (to specify the output filename)"""

# Python libs
import sys, getopt
# ANTLR runtime and generated code
import antlr3
from uflLexer import uflLexer
from uflParser import uflParser
# MCFC libs
import visualiser
import canonicaliser

def main():

    opts,args = get_options()
    keys = opts.keys()

    if len(args) > 0: 
        inputFile = args[0]
    else:
        print "No input."
	print __doc__
	sys.exit(-1)

    root = readSource(inputFile)

    if 'visualise' in keys or 'v' in keys:
        if 'o' in keys:
	    outputFile = opts['o']
	else:
	    outputFile = inputFile[:-3] + "pdf"
        visualise(root, outputFile)
	return 0
	

    return 0

def get_options():
    try: 
        opts, args = getopt.getopt(sys.argv[1:], "hvo:", ["visualise"])
    except getopt.error, msg:
        print msg
	print __doc__
	sys.exit(-1)

    opts_dict = {}
    for opt in opts:
        key = opt[0].lstrip('-')
	value = opt[1]
	opts_dict[key] = value
    
    return opts_dict, args

def visualise(ast, filename):

    v = visualiser.Visualiser(filename)
    v.visualise(ast)

def readSource(inputFile):

    canned, tempFields = canonicaliser.canonicalise(inputFile)
    print tempFields
    charStream = antlr3.ANTLRStringStream(canned)
    lexer = uflLexer(charStream)
    tokens = antlr3.CommonTokenStream(lexer)
    tokens.discardOffChannelTokens = True
    parser = uflParser(tokens)
    r = parser.file_input()
    root = r.tree

    return root

if __name__ == "__main__":
    sys.exit(main())
