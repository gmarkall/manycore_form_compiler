// [The 'BSD licence']
// Copyright (c) 2009 Ales Teska
// Copyright (c) 2011 Graham Markall
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

// Tree Grammar for UFL.
// Graham Markall, 2011.
//
// Based on a Python3 grammar by Ales Teska.  Parts (mainly lexer and approach
// to parser) are based on Python 2.5 Grammar created by Terence Parr, Loring
// Craymer and Aaron Maxwell.
// Copyright (c) 2004 Terence Parr and Loring Craymer

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

tuple_expr	: LPAREN exprs=expr_list RPAREN
                      -> ^(TUPLE $exprs);

expr_list	: expr COMMA expr_list
		| expr;

expr		: arith_expr
		| source_expr
		| ufl_expr
		| form_expr
		| compute_expr
		| tuple_expr
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
		| mix_ele_expr
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

binary_op       : (SPATDERIV | LISTTENSOR | COMPTENSOR | SUM | INDEXED | PRODUCT | INDEXSUM)^ ;

arg_expr	: (ARGUMENT LPAREN element=ufl_expr COMMA id=number RPAREN) 
                      -> ^(ARGUMENT $element $id);

ele_expr	: (type=element_op LPAREN family=string COMMA cell=cell_expr COMMA degree=number 
                   (COMMA LPAREN? shape1=number 
		    (COMMA shape2=number RPAREN COMMA symmetry=atom
		   )? 
		  )? RPAREN) 
                      -> ^($type $family $cell $degree $shape1? $shape2? $symmetry?);

element_op      : (FINELE | VECELE | TENELE)^ ;

// For two elements only now. 
// Long term, needs a rethink.
mix_ele_expr	: (MIXELE LPAREN STAR LBRACK ele1=ele_expr COMMA ele2=ele_expr RBRACK COMMA attrs=mix_ele_dict RPAREN  )
                      -> ^(MIXELE $ele1 $ele2 $attrs) ;

mix_ele_dict	: (DOUBLESTAR LCURLY key=string COLON LPAREN shape=number COMMA RPAREN RCURLY)
		      -> ^($key $shape);

cell_expr	: (CELL LPAREN domain=string COMMA degree=number COMMA 
                   space=space_expr RPAREN) 
                      -> ^(CELL $domain $degree $space);

space_expr	: (SPACE LPAREN dim=number RPAREN) 
                      -> ^(SPACE $dim);

multiindex_expr	: (MULTIINDEX LPAREN LPAREN idx=index* RPAREN COMMA LCURLY 
                   idx_dim=index_dim* RCURLY RPAREN) 
                      -> ^(MULTIINDEX $idx* $idx_dim*);

index		: (index_type=index_obj LPAREN id=number RPAREN COMMA?) 
                      -> ^($index_type $id);

index_dim	: (index_type=index_obj LPAREN idx=number RPAREN COLON dim=number COMMA?) 
                      -> ^($index_type $idx $dim);

index_obj	: (UINDEX | FIXEDINDEX)^ ;

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
LISTTENSOR	: 'ListTensor';
COMPTENSOR	: 'ComponentTensor';
SPATDERIV	: 'SpatialDerivative';
ARGUMENT	: 'Argument';
FINELE		: 'FiniteElement';
VECELE          : 'VectorElement';
TENELE          : 'TensorElement';
MIXELE		: 'MixedElement';
CELL		: 'Cell';
SPACE		: 'Space';
MULTIINDEX	: 'MultiIndex';
COEFF		: 'Coefficient';
MEASURE		: 'Measure';
UINDEX		: 'Index';
FIXEDINDEX	: 'FixedIndex';
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

// Only in the AST
TUPLE		: 'Tuple';

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
		: MINUS? DIGIT+ ;

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

