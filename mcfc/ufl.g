// Tree Grammar for UFL.
//
// Based on a Python grammar by Ales Teska.
// For license and authorship see originalheader.txt
//
// Modified to tree parse UFL by G. Markall.

grammar ufl;

options
{
	language=Python;
	backtrack=true;

	output=AST;
	ASTLabelType=pANTLR3_BASE_TREE;
}

///////////////////////// PARSER ///////////////////////////////////////////////

file_input	: ( NEWLINE! | stmt )* ;

stmt            : lhs=expr eq=ASSIGN rhs=expr 
                      -> ^($eq $lhs $rhs);

expr		: arith_expr
		| source_expr
		| ufl_expr
		| form_expr
		| compute_expr
		;

arith_expr	: term ( ( PLUS | MINUS )^ term )* ;


//////////////////////////////////////////////
// UFL Expressions

form_expr	: (FORM LPAREN LBRACK integral=integral_expr RBRACK RPAREN)
                      -> ^(FORM $integral);

integral_expr	: (INTEGRAL LPAREN integrand=ufl_expr COMMA measure=measure_expr RPAREN) 
                      -> ^(INTEGRAL $integrand $measure);

measure_expr	: (MEASURE LPAREN dom_type=string COMMA dom_id=number COMMA 
                   metadata=atom RPAREN) 
                      -> ^(MEASURE $dom_type $dom_id $metadata);

ufl_expr	: binary_ufl_expr
		| multiindex_expr
		| coeff_expr
		| arg_expr
		| ele_expr
		| value_expr
		| state_expr
		;

state_expr      : STATE DOT type=fields_expr LBRACK LPAREN 
                  field=string COMMA LPAREN? timestep=arith_expr RPAREN? RPAREN RBRACK
	              -> ^($type $field $timestep);

fields_expr     : (SCALAR_FIELDS | VECTOR_FIELDS | TENSOR_FIELDS)^;

source_expr     : ufl_object=ufl_expr AMPERSAND SOURCE LPAREN field=atom RPAREN 
                      -> ^($field $ufl_object);

value_expr      : (op=value_op LPAREN value=constant_value COMMA LPAREN RPAREN 
                   COMMA LPAREN RPAREN COMMA LCURLY RCURLY RPAREN) 
                      -> ^($op $value);

value_op        : (SYMVALUE | FLOATVALUE | INTVALUE)^ ;

constant_value  : (number | string)^ ;

binary_ufl_expr : (op=binary_op LPAREN arg1=ufl_expr COMMA arg2=ufl_expr RPAREN) 
                      -> ^($op $arg1 $arg2);

binary_op       : (SPATDERIV | COMPTENSOR | SUM | INDEXED | PRODUCT | INDEXSUM)^ ;

arg_expr	: (ARGUMENT LPAREN element=ufl_expr COMMA id=number RPAREN) 
                      -> ^(ARGUMENT $element $id);

ele_expr	: (type=element_op LPAREN family=string COMMA cell=cell_expr COMMA degree=number 
                   (COMMA LPAREN? shape1=number 
		    (COMMA shape2=number RPAREN COMMA symmetry=atom
		   )? 
		  )? RPAREN) 
                      -> ^($type $family $cell $degree $shape1? $shape2? $symmetry?);

element_op      : (FINELE | VECELE | TENELE)^ ;

cell_expr	: (CELL LPAREN domain=string COMMA degree=number COMMA 
                   space=space_expr RPAREN) 
                      -> ^(CELL $domain $degree $space);

space_expr	: (SPACE LPAREN dim=number RPAREN) 
                      -> ^(SPACE $dim);

multiindex_expr	: (MULTIINDEX LPAREN LPAREN idx=index* RPAREN COMMA LCURLY 
                   idx_dim=index_dim* RCURLY RPAREN) 
                      -> ^(MULTIINDEX $idx $idx_dim);

index		: (UINDEX LPAREN id=number RPAREN COMMA?) 
                      -> ^(UINDEX $id);

index_dim	: (UINDEX LPAREN idx=number RPAREN COLON dim=number COMMA?) 
                      -> ^(UINDEX $idx $dim);

coeff_expr	: (COEFF LPAREN element=ufl_expr COMMA id=number RPAREN) 
                      -> ^(COEFF $element $id);

compute_expr    : (SOLVE LPAREN lhs=atom COMMA rhs=atom RPAREN) 
                      -> ^(SOLVE $lhs $rhs);

//////////////// terms, factors, atoms, strings, numbers etc ///////////////////

term		: factor ( ( STAR | SLASH  )^ factor )* ;
factor		: PLUS^ factor
		| MINUS^ factor
		| atom
		;

atom		: IDENTIFIER
		| number
		| string
		| NONE
		| TRUE
		| FALSE
		;

string		: STRINGLITERAL+
		| BYTESLITERAL +
		;

number		: INTEGER
		| FLOATNUMBER
		;


//////////////////////////////////// LEXER /////////////////////////////////////

// UFL keywords
FORM		: 'Form';
INTEGRAL	: 'Integral';
INDEXSUM	: 'IndexSum';
PRODUCT		: 'Product';
INDEXED		: 'Indexed';
COMPTENSOR	: 'ComponentTensor';
SPATDERIV	: 'SpatialDerivative';
ARGUMENT	: 'Argument';
FINELE		: 'FiniteElement';
VECELE          : 'VectorElement';
TENELE          : 'TensorElement';
CELL		: 'Cell';
SPACE		: 'Space';
MULTIINDEX	: 'MultiIndex';
COEFF		: 'Coefficient';
MEASURE		: 'Measure';
UINDEX		: 'Index';
SYMVALUE        : 'SymbolicValue';
INTVALUE        : 'IntValue';
FLOATVALUE      : 'FloatValue';
SUM             : 'Sum';

// Not part of UFL, but appear in the canonicalised code
SOURCE          : 'source';
STATE           : 'state';
SCALAR_FIELDS   : 'scalar_fields';
VECTOR_FIELDS   : 'vector_fields';
TENSOR_FIELDS   : 'tensor_fields';
SOLVE           : 'solve';

////////////////////////////////////////////////////////////////////////////////
// $<String and Bytes literals

STRINGLITERAL	: STRINGPREFIX? ( SHORTSTRING | LONGSTRING ) ;

fragment STRINGPREFIX
		: ( 'r' | 'R' ) ;

fragment SHORTSTRING
		: '"' ( ESCAPESEQ | ~( '\\'|'\n'|'"' ) )* '"'
		| '\'' ( ESCAPESEQ | ~( '\\'|'\n'|'\'' ) )* '\''
		;

fragment LONGSTRING
		: '\'\'\'' ( options {greedy=false;}:TRIAPOS )* '\'\'\''
		| '"""' ( options {greedy=false;}:TRIQUOTE )* '"""'
		;

BYTESLITERAL	: BYTESPREFIX ( SHORTBYTES | LONGBYTES ) ;

fragment BYTESPREFIX
		: ( 'b' | 'B' ) ( 'r' | 'R' )? ;

fragment SHORTBYTES
		: '"' ( ESCAPESEQ | ~( '\\' | '\n' | '"' ) )* '"' 
		| '\'' ( ESCAPESEQ | ~( '\\' | '\n' | '\'' ) )* '\'' 
		;

fragment LONGBYTES 
		: '\'\'\'' ( options {greedy=false;}:TRIAPOS )* '\'\'\''
		| '"""' ( options {greedy=false;}:TRIQUOTE )* '"""'
		;

fragment TRIAPOS
		: ( '\'' '\'' | '\''? ) ( ESCAPESEQ | ~( '\\' | '\'' ) )+ ;

fragment TRIQUOTE
		: ( '"' '"' | '"'? ) ( ESCAPESEQ | ~( '\\' | '"' ) )+ ;
	
fragment ESCAPESEQ
		: '\\' . ;

// $>

////////////////////////////////////////////////////////////////////////////////
// $<Keywords

FALSE		: 'False' ;
NONE		: 'None' ;
TRUE		: 'True' ;

////////////////////////////////////////////////////////////////////////////////
// $<Integer literals

INTEGER		: NEGINTEGER | DECIMALINTEGER | OCTINTEGER | HEXINTEGER | BININTEGER ;

fragment NEGINTEGER
                : MINUS NONZERODIGIT DIGIT* ;

fragment DECIMALINTEGER
		: NONZERODIGIT DIGIT* | '0'+ ;

fragment NONZERODIGIT
		: '1' .. '9' ;

fragment DIGIT
		: '0' .. '9' ;

fragment OCTINTEGER
		: '0' ( 'o' | 'O' ) OCTDIGIT+ ;

fragment HEXINTEGER
		: '0' ( 'x' | 'X' ) HEXDIGIT+ ;

fragment BININTEGER
		: '0' ( 'b' | 'B' ) BINDIGIT+ ;

fragment OCTDIGIT
		: '0' .. '7' ;

fragment HEXDIGIT
		: DIGIT | 'a' .. 'f' | 'A' .. 'F' ;

fragment BINDIGIT
		: '0' | '1' ;

////////////////////////////////////////////////////////////////////////////////
// $<Floating point literals

FLOATNUMBER	: POINTFLOAT | EXPONENTFLOAT ;

fragment POINTFLOAT
		: ( INTPART? FRACTION )
		| ( INTPART '.' )
		;

fragment EXPONENTFLOAT
		: ( INTPART | POINTFLOAT ) EXPONENT ;

fragment INTPART
		: DIGIT+ ;

fragment FRACTION
		: '.' DIGIT+ ;

fragment EXPONENT
		: ( 'e' | 'E' ) ( '+' | '-' )? DIGIT+ ;

////////////////////////////////////////////////////////////////////////////////
// $<Identifiers

IDENTIFIER	: ID_START ID_CONTINUE* ;

fragment ID_START
		: '_'
		| 'A'.. 'Z'
		| 'a' .. 'z'
		;

fragment ID_CONTINUE
		: '_'
		| 'A'.. 'Z'
		| 'a' .. 'z'
		| '0' .. '9'
		;
 
// $>

////////////////////////////////////////////////////////////////////////////////
// $<Operators

PLUS		: '+' ;
MINUS		: '-' ;
STAR		: '*' ;
DOUBLESTAR	: '**' ;
SLASH		: '/' ;
AMPERSAND	: '&' ;

// $>

//////////////////////////////////////////////
// $<Delimiters

LPAREN		: '(' ;
RPAREN		: ')' ;
LBRACK		: '[' ;
RBRACK		: ']' ;
LCURLY		: '{' ;
RCURLY		: '}' ;

COMMA		: ',' ;
COLON		: ':' ;
DOT		: '.' ;
ASSIGN		: '=' ;

// $>

NEWLINE         : (  '\r'? '\n' )+ ;

WS		: ( ' ' | '\t' )+ {$channel=HIDDEN;};

COMMENT
	@init
	{
		$channel=HIDDEN;
	}
	: ( ' ' | '\t' )* '#' ( ~'\n' )* '\n'+ 
	|  '#' ( ~'\n' )* // let NEWLINE handle \n unless char pos==0 for '#'
	;

