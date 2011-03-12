# The usual suspects
import sys, getopt
# ANTLR runtime and generated code
import antlr3
from uflLexer import uflLexer
from uflParser import uflParser
# Visualiser, handy for debugging
import visualiser
# AST Utilities
import asttools

def main():

    try: 
        opts, args = getopt.getopt(sys.argv[1:], "ho:", ["help"])
    except getopt.error, msg:
        print msg
	print __doc__
	sys.exit(-1)

    if len(args) > 0: 
        inputFile = args[0]
    else:
        print "No input."
	sys.exit(-1)

    charStream = antlr3.ANTLRFileStream(inputFile)
    lexer = uflLexer(charStream)
    tokens = antlr3.CommonTokenStream(lexer)
    tokens.discardOffChannelTokens = True
    parser = uflParser(tokens)
    r = parser.file_input()
    root = r.tree

    outputFile = inputFile[:-4] + "pdf"
    v = visualiser.Visualiser(outputFile)
    v.visualise(root)

    return 0

if __name__ == "__main__":
    sys.exit(main())
