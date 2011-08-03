/*
Copyright (c) 1998-2000, Non, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in
  the documentation and/or other materials provided with the
  distribution.

  All advertising materials mentioning features or use of this
  software must display the following acknowledgement:

    This product includes software developed by Non, Inc. and
    its contributors.

  Neither name of the company nor the names of its contributors
  may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COMPANY OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Copyright (c) Non, Inc. 1997 -- All Rights Reserved

PROJECT:        C Compiler
MODULE:         Parser
FILE:           stdc.g

AUTHOR:         John D. Mitchell (john@non.net), Jul 12, 1997

REVISION HISTORY:

        Name    Date            Description
        ----    ----            -----------
        JDM     97.07.12        Initial version.
        JTC     97.11.18        Declaration vs declarator & misc. hacking.
        JDM     97.11.20        Fixed:  declaration vs funcDef,
                                        parenthesized expressions,
                                        declarator iteration,
                                        varargs recognition,
                                        empty source file recognition,
                                        and some typos.


DESCRIPTION:

        This grammar supports the Standard C language.

        Note clearly that this grammar does *NOT* deal with
        preprocessor functionality (including things like trigraphs)
        Nor does this grammar deal with multi-byte characters nor strings
        containing multi-byte characters [these constructs are "exercises
        for the reader" as it were :-)].

        Please refer to the ISO/ANSI C Language Standard if you believe
        this grammar to be in error.  Please cite chapter and verse in any
        correspondence to the author to back up your claim.

TODO:

        - typedefName is commented out, needs a symbol table to resolve
        ambiguity.

        - trees

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
header
{
package cetus.base.grammars;
}

{
import java.io.*;
import antlr.CommonAST;
import antlr.DumpASTVisitor;
import java.util.*;
import cetus.hir.*;
}

class NewCParser extends Parser;

options
{
k = 2;
exportVocab = NEWC;
// Copied following options from java grammar.
codeGenMakeSwitchThreshold = 2;
codeGenBitsetTestThreshold = 3;
}

{
Expression baseEnum = null,curEnum = null;
NewCLexer curLexer=null;
boolean isFuncDef = false;
boolean isExtern = false;
PreprocessorInfoChannel preprocessorInfoChannel = null;
SymbolTable symtab = null;
CompoundStatement curr_cstmt = null;
boolean hastypedef = false;
HashMap typetable = null;
LinkedList currproc = new LinkedList();
Declaration prev_decl = null;
boolean old_style_func = false;
HashMap func_decl_list = new HashMap();

public void getPreprocessorInfoChannel(PreprocessorInfoChannel preprocChannel)
{
  preprocessorInfoChannel = preprocChannel;
}

public void setLexer(NewCLexer lexer)
{
  curLexer=lexer;
  curLexer.setParser(this);
}

public NewCLexer getLexer()
{
  return curLexer;
}

public LinkedList getPragma(int a)
{
  return
      preprocessorInfoChannel.extractLinesPrecedingTokenNumber(new Integer(a));
}

public void putPragma(Token sline, SymbolTable sym)
{
  LinkedList v  = null;
  v = getPragma(((CToken)sline).getTokenNumber());
  Iterator iter = v.iterator();
  Pragma p = null;
  PreAnnotation anote = null;
  while(iter.hasNext()) {
    p = (Pragma)iter.next();
    anote = new PreAnnotation(p.str);
    if (p.type ==Pragma.pragma)
      anote.setPrintMethod(PreAnnotation.print_raw_method);
    else if(p.type ==Pragma.comment)
      anote.setPrintMethod(PreAnnotation.print_raw_method);
    //sym.addStatement(new DeclarationStatement(anote));
    if (sym instanceof CompoundStatement)
      ((CompoundStatement)sym).addStatement(new DeclarationStatement(anote));
    else
      sym.addDeclaration(anote);
  }
}

// Suppport C++-style single-line comments?
public static boolean CPPComments = true;
public Stack symtabstack = new Stack();
public Stack typestack = new Stack();

public void enterSymtab(SymbolTable curr_symtab)
{
  symtabstack.push(symtab);
  typetable = new HashMap();
  typestack.push(typetable);
  symtab = curr_symtab;
}

public void exitSymtab()
{
  Object o = symtabstack.pop();
  if (o != null) {
    typestack.pop();
    typetable = (HashMap)(typestack.peek());
    symtab = (SymbolTable)o;
  }
}

public boolean isTypedefName(String name)
{
  //System.err.println("Typename "+name);
  int n = typestack.size()-1;
  Object d = null;
  while(n>=0) {
    d = ((HashMap)(typestack.get(n))).get(name);
    if (d != null )
      return true;
    n--;
  }
  if (name.equals("__builtin_va_list"))
    return true;

  //System.err.println("Typename "+name+" not found");
  return false;
}

int traceDepth = 0;

public void reportError(RecognitionException ex)
{
  try {
    System.err.println("ANTLR Parsing Error: " + "Exception Type: "
        + ex.getClass().getName());
    System.err.println("Source: " + getLexer().lineObject.getSource()
        + " Line:" + ex.getLine() + " Column: " + ex.getColumn()
        + " token name:" + tokenNames[LA(1)]);
    ex.printStackTrace(System.err);
    System.exit(1);
  } catch (TokenStreamException e) {
    System.err.println("ANTLR Parsing Error: "+ex);
    ex.printStackTrace(System.err);
    System.exit(1);
  }
}

public void reportError(String s)
{
  System.err.println("ANTLR Parsing Error from String: " + s);
}

public void reportWarning(String s)
{
  System.err.println("ANTLR Parsing Warning from String: " + s);
}

public void match(int t) throws MismatchedTokenException
{
  boolean debugging = false;
  if ( debugging ) {
    for (int x=0; x<traceDepth; x++)
      System.out.print(" ");
    try {
      System.out.println("Match(" + tokenNames[t] + ") with LA(1)="
          + tokenNames[LA(1)] + ((inputState.guessing>0)?
          " [inputState.guessing " + inputState.guessing + "]":""));
    } catch (TokenStreamException e) {
      System.out.println("Match("+tokenNames[t]+") "
          + ((inputState.guessing>0)?
          " [inputState.guessing "+ inputState.guessing + "]":""));
    }
  }
  try {
    if ( LA(1)!=t ) {
      if ( debugging ) {
        for (int x=0; x<traceDepth; x++)
          System.out.print(" ");
        System.out.println("token mismatch: "+tokenNames[LA(1)]
            + "!=" + tokenNames[t]);
      }
      throw new MismatchedTokenException
          (tokenNames, LT(1), t, false, getFilename());
    } else {
      // mark token as consumed -- fetch next token deferred until LA/LT
      consume();
    }
  } catch (TokenStreamException e) {
  }
}

public void traceIn(String rname)
{
  traceDepth += 1;
  for (int x=0; x<traceDepth; x++)
    System.out.print(" ");
  try {
    System.out.println("> "+rname+"; LA(1)==("+ tokenNames[LT(1).getType()]
        + ") " + LT(1).getText() + " [inputState.guessing "
        + inputState.guessing + "]");
  } catch (TokenStreamException e) {
  }
}

public void traceOut(String rname)
{
  for (int x=0; x<traceDepth; x++)
    System.out.print(" ");
  try {
    System.out.println("< "+rname+"; LA(1)==("+ tokenNames[LT(1).getType()]
        + ") " + LT(1).getText() + " [inputState.guessing "
        + inputState.guessing + "]");
  } catch (TokenStreamException e) {
  }
  traceDepth -= 1;
}

}

/* TranslationUnit */
translationUnit [TranslationUnit init_tunit] returns [TranslationUnit tunit]
{
/* build a new Translation Unit */
if (init_tunit == null)
  tunit = new TranslationUnit(getLexer().originalSource);
else
  tunit = init_tunit;
enterSymtab(tunit);
}
        :
        ( externalList[tunit] )?  /* Empty source files are allowed.  */
{exitSymtab();}
        ;


externalList [TranslationUnit tunit]
{boolean flag = true;}
        :
        (
        pre_dir:PREPROC_DIRECTIVE
{
String value = (pre_dir.getText()).substring(7).trim();
putPragma(pre_dir,symtab);
/*
if(value.startsWith("endinclude")){
  flag = true;
}
else if(value.startsWith("startinclude")){
  flag = false;
}
*/
PreAnnotation anote = new PreAnnotation(pre_dir.getText());
tunit.addDeclaration(anote);
anote.setPrintMethod(PreAnnotation.print_raw_method);
//elist.add(pre_dir.getText());
}
        |
        externalDef[tunit]
        )+
        ;


/* Declaration */
externalDef [TranslationUnit tunit]
{Declaration decl = null;}
        :
        ( "typedef" | declaration ) => decl=declaration
{
if (decl != null) {
  //PrintTools.printStatus("Adding Declaration: ",3);
  //PrintTools.printlnStatus(decl,3);
  tunit.addDeclaration(decl);
}
}
        |
        ( functionPrefix ) => decl=functionDef
{
//PrintTools.printStatus("Adding Declaration: ",3);
//PrintTools.printlnStatus(decl,3);
tunit.addDeclaration(decl);
}
        |
        decl=typelessDeclaration
{
//PrintTools.printStatus("Adding Declaration: ",3);
//PrintTools.printlnStatus(decl,3);
tunit.addDeclaration(decl);
}
        |
        asm_expr // not going to handle this now
        |
        SEMI // empty declaration - ignore it
        ;


/* these two are here because GCC allows "cat = 13;" as a valid program! */
functionPrefix
{Declarator decl = null;}
        :
        (
        (functionDeclSpecifiers) => functionDeclSpecifiers
        |
        //epsilon
        )
        // Passing "null" could cause a problem
        decl = declarator
        ( declaration )* (VARARGS)? ( SEMI )*
        LCURLY
        ;


/* Type Declaration */
typelessDeclaration returns [Declaration decl]
{decl=null; List idlist=null;}
        :
        idlist=initDeclList SEMI
        /* Proper constructor missing */
{decl = new VariableDeclaration(new LinkedList(),idlist); }
        ;


// going to ignore this
asm_expr
{Expression expr1 = null;}
        :
        "asm"^ ("volatile")? LCURLY expr1=expr RCURLY ( SEMI )+
        ;


/* Declaration */
declaration returns [Declaration bdecl]
{bdecl=null; List dspec=null; List idlist=null;}
        :
        dspec=declSpecifiers
        (
        // Pass specifier to add to Symtab
        idlist=initDeclList
        )?
{
if (idlist != null) {
  if (old_style_func) {
    Iterator iter = idlist.iterator();
    Declarator d  = null;
    Declaration newdecl = null;
    while (iter.hasNext()) {
      d = (Declarator)(iter.next());
      newdecl = new VariableDeclaration(dspec,d);
      func_decl_list.put(d.getID().toString(),newdecl);
    }
  bdecl = null;
  } else
    bdecl = new VariableDeclaration(dspec,idlist);
  prev_decl = null;
} else {
  // Looks like a forward declaration
  if (prev_decl != null) {
    bdecl = prev_decl;
    prev_decl = null;
  }
}
hastypedef = false;
}
        ( dsemi:SEMI )+
{
int sline = 0;
sline = dsemi.getLine();
putPragma(dsemi,symtab);
hastypedef = false;
}
        ;


/* Specifier List */
// The main type information
declSpecifiers returns [List decls]
{decls = new LinkedList(); Specifier spec = null; Specifier temp=null;}
        :
        (
        // this loop properly aborts when it finds a non-typedefName ID MBZ
        options {warnWhenFollowAmbig = false;}
        :
        /* Modifier */
        spec = storageClassSpecifier
{decls.add(spec);}
        |
        /* Modifier */
        spec = typeQualifier
{decls.add(spec);}
        |
        /* SubType */
        ( "struct" | "union" | "enum" | typeSpecifier ) =>
        temp = typeSpecifier
{decls.add(temp);}
        // MinGW specific
        |
        attributeDecl
        )+
        ;


/*********************************
 * Specifiers                    *
 *********************************/


storageClassSpecifier returns [Specifier cspec]
{cspec= null;}
        :
        "auto"
{cspec = Specifier.AUTO;}
        |
        "register"
{cspec = Specifier.REGISTER;}
        |
        "typedef"
{cspec = Specifier.TYPEDEF; hastypedef = true;}
        |
        cspec = functionStorageClassSpecifier
        ;


functionStorageClassSpecifier returns [Specifier type]
{type= null;}
        :
        "extern"
{type= Specifier.EXTERN;}
        |
        "static"
{type= Specifier.STATIC;}
        |
        "inline"
{type= Specifier.INLINE;}
        ;


typeQualifier returns [Specifier tqual]
{tqual=null;}
        :
        "const"
{tqual = Specifier.CONST;}
        |
        "volatile"
{tqual = Specifier.VOLATILE;}
        ;


// A Type Specifier (basic type and user type)
/***************************************************
 * Should a basic type be an int value or a class ? *
 ****************************************************/
typeSpecifier returns [Specifier types]
{
types = null;
String tname = null;
Expression expr1 = null;
List tyname = null;
boolean typedefold = false;
}
        :
{typedefold = hastypedef; hastypedef = false;}
        (
        "void"
{types = Specifier.VOID;}
        |
        "char"
{types = Specifier.CHAR;}
        |
        "short"
{types = Specifier.SHORT;}
        |
        "int"
{types = Specifier.INT;}
        |
        "long"
{types = Specifier.LONG;}
        |
        "float"
{types = Specifier.FLOAT;}
        |
        "double"
{types = Specifier.DOUBLE;}
        |
        "signed"
{types = Specifier.SIGNED;}
        |
        "unsigned"
{types = Specifier.UNSIGNED;}
        /* C99 built-in type support */
        |
        "_Bool"
{types = Specifier.CBOOL;}
        |
        "_Complex"
{types = Specifier.CCOMPLEX;}
        |
        "_Imaginary"
{types = Specifier.CIMAGINARY;}
        |
        types = structOrUnionSpecifier
        ( options{warnWhenFollowAmbig=false;}: attributeDecl )*
        |
        types = enumSpecifier
        |
        types = typedefName
        |
        /* Maybe unused */
        "typeof"^ LPAREN
        ( ( typeName ) => tyname=typeName | expr1=expr )
        RPAREN
        |
        "__complex"
{types = Specifier.DOUBLE;}
        )
{hastypedef = typedefold;}
        ;


typedefName returns[Specifier name]
{name = null;}
        :
{isTypedefName ( LT(1).getText() )}?
        i:ID
        //{ ## = #(#[NTypedefName], #i); }
{name = new UserSpecifier(new NameID(i.getText()));}
        ;


structOrUnion returns [int type]
{type=0;}
        :
        "struct"
{type = 1;}
        |
        "union"
{type = 2;}
        ;


/* A User Type */
structOrUnionSpecifier returns [Specifier spec]
{
ClassDeclaration decls = null;
String name=null;
int type=0;
spec = null;
int linenum = 0;
}
        :
        type=structOrUnion!
        (
        //Named stucture with body
        ( ID LCURLY ) => i:ID l:LCURLY
{
name = i.getText();linenum = i.getLine(); putPragma(i,symtab);
String sname = null;
if (type == 1) {
  decls = new ClassDeclaration(ClassDeclaration.STRUCT, new NameID(name));
  spec = new UserSpecifier(new NameID("struct "+name));
} else {
  decls = new ClassDeclaration(ClassDeclaration.UNION, new NameID(name));
  spec = new UserSpecifier(new NameID("union "+name));
}
}
        ( structDeclarationList[decls] )?
{
if (symtab instanceof ClassDeclaration) {
  int si = symtabstack.size()-1;
  for (;si>=0;si--) {
    if (!(symtabstack.get(si) instanceof ClassDeclaration)) {
      ((SymbolTable)symtabstack.get(si)).addDeclaration(decls);
      break;
    }
  }
} else
  symtab.addDeclaration(decls);
}
        RCURLY
        |
        // unnamed structure with body
        // This is for one time use
        // Added "named_" to prevent illegal identifiers.
        l1:LCURLY
{
name = "named_"+getLexer().originalSource +"_"+ ((CToken)l1).getTokenNumber();
name = name.replaceAll("[.]","_");
name = name.replaceAll("-","_");
linenum = l1.getLine(); putPragma(l1,symtab);
if (type == 1) {
  decls = new ClassDeclaration(ClassDeclaration.STRUCT, new NameID(name));
  spec = new UserSpecifier(new NameID("struct "+name));
} else {
  decls = new ClassDeclaration(ClassDeclaration.UNION, new NameID(name));
  spec = new UserSpecifier(new NameID("union "+name));
}
}
        ( structDeclarationList[decls] )?
{
if (symtab instanceof ClassDeclaration) {
  int si = symtabstack.size()-1;
  for (;si>=0;si--) {
    if (!(symtabstack.get(si) instanceof ClassDeclaration)) {
      ((SymbolTable)symtabstack.get(si)).addDeclaration(decls);
      break;
    }
  }
} else
  symtab.addDeclaration(decls);
}
        RCURLY
        | // named structure without body
        sou3:ID
{
name = sou3.getText();linenum = sou3.getLine(); putPragma(sou3,symtab);
if(type == 1) {
  spec = new UserSpecifier(new NameID("struct "+name));
  decls = new ClassDeclaration(ClassDeclaration.STRUCT,new NameID(name),true);
} else {
  spec = new UserSpecifier(new NameID("union "+name));
  decls = new ClassDeclaration(ClassDeclaration.UNION,new NameID(name),true);
}
prev_decl = decls;
}
        )
        ;


/* Declarations are added to ClassDeclaration */
structDeclarationList [ClassDeclaration cdecl]
{Declaration sdecl= null;/*SymbolTable prev_symtab = symtab;*/}
        :
{enterSymtab(cdecl);}
        (
        sdecl=structDeclaration
{if(sdecl != null ) cdecl.addDeclaration(sdecl);}
        )+
{exitSymtab(); /*symtab = prev_symtab;*/}
        ;


/* A declaration */
structDeclaration returns [Declaration sdecl]
{
List bsqlist=null;
List bsdlist=null;
sdecl=null;
}
        :
        bsqlist = specifierQualifierList
        // passes specifier to put in symtab
        bsdlist = structDeclaratorList
        ( COMMA! )? ( SEMI! )+
{sdecl = new VariableDeclaration(bsqlist,bsdlist); hastypedef = false;}
        ;


/* List of Specifiers */
specifierQualifierList returns [List sqlist]
{
sqlist=new LinkedList();
Specifier tspec=null;
Specifier tqual=null;
}
        :
        (
        // this loop properly aborts when it finds a non-typedefName ID MBZ
        options {warnWhenFollowAmbig = false;}
        :
        /* A type : BaseType */
        ( "struct" | "union" | "enum" | typeSpecifier ) =>
        tspec = typeSpecifier
{sqlist.add(tspec);}
        |
        /* A Modifier : int value */
        tqual=typeQualifier
{sqlist.add(tqual);}
        )+
        ;


/* List of Declarators */
structDeclaratorList returns [List sdlist]
{
sdlist = new LinkedList();
Declarator sdecl=null;
}
        :
        sdecl = structDeclarator
{
// why am I getting a null value here ?
if (sdecl != null)
  sdlist.add(sdecl);
}
        (
        options{warnWhenFollowAmbig=false;}
        :
        COMMA! sdecl=structDeclarator
{sdlist.add(sdecl);}
        )*
        ;


/* Declarator */
structDeclarator returns [Declarator sdecl]
{
sdecl=null;
Expression expr1=null;
}
        :
        ( sdecl = declarator )?
        //( COLON expr1=constExpr )?
        /* bit-field recognition */
        ( COLON expr1=expr )?
{
if (sdecl != null && expr1 != null) {
  expr1 = Symbolic.simplify(expr1);
  if (expr1 instanceof IntegerLiteral)
    sdecl.addTrailingSpecifier(new BitfieldSpecifier(expr1));
  else
    ; // need to throw parse error
}
}
/* This needs to be fixed */
//{sdecl.addExpr(expr1);}
// Ignore this GCC dialect
/*
{if(sdecl == null && expr1 == null){
System.err.println("Errorororororo");
}
}*/
        ( attributeDecl )*
        ;


/* UserSpecifier (Enumuration) */
enumSpecifier returns[Specifier spec]
{
cetus.hir.Enumeration espec = null;
String enumN = null;
List elist=null;
spec = null;
}
        :
        "enum"^
        (
        ( ID LCURLY ) => i:ID
        LCURLY elist=enumList RCURLY
{enumN =i.getText();}
        |
        el1:LCURLY elist=enumList RCURLY
{
enumN = getLexer().originalSource +"_"+ ((CToken)el1).getTokenNumber();
enumN =enumN.replaceAll("[.]","_");
enumN =enumN.replaceAll("-","_");
}
        |
        espec2:ID
{enumN =espec2.getText();}
        )
        // has name and list of members
{
if (elist != null) {
  espec = new cetus.hir.Enumeration(new NameID(enumN),elist);
  if (symtab instanceof ClassDeclaration) {
    int si = symtabstack.size()-1;
    for (;si>=0;si--) {
      if (!(symtabstack.get(si) instanceof ClassDeclaration)) {
        ((SymbolTable)symtabstack.get(si)).addDeclaration(espec);
        break;
      }
    }
  } else
    symtab.addDeclaration(espec);
}
spec = new UserSpecifier(new NameID("enum "+enumN));
}
        ;


/* List of Declarator */
enumList returns [List elist]
{
Declarator enum1=null;
elist = new LinkedList();
}
        :
        enum1=enumerator
{elist.add(enum1);}
        (
        options{warnWhenFollowAmbig=false;}
        :
        COMMA! enum1=enumerator {elist.add(enum1);}
        )*
        ( COMMA! )?
        ;


/* Declarator */

// Complicated due to setting values for each enum value
enumerator returns[Declarator decl]
{decl=null;Expression expr2=null; String val = null;}
        :
        /* Variable Declarator */
        i:ID
{
val = i.getText();
decl = new VariableDeclarator(new NameID(val));
}
        /* Initializer */
        (
        ASSIGN expr2=constExpr
{decl.setInitializer(new Initializer(expr2));}
        )?
        ;

// Not handling this as of now (sort of GCC stuff)
attributeDecl
        :
        "__attribute"^ 
        LPAREN LPAREN attributeList RPAREN RPAREN
        (attributeDecl)*
        |
        "__asm"^
        LPAREN stringConst RPAREN
        ;


attributeList
        :
        attribute
        (
        options{warnWhenFollowAmbig=false;}
        :
        COMMA attribute
        )*
        //( COMMA )?
        ;


attribute
        :
        (
        // Word
        (
            ID
            //| declSpecifiers
            |
            storageClassSpecifier
            |
            typeQualifier
        )
        (
            LPAREN 
            (
            ID
            //|
            //assignExpr
            |
            //epsilon
            )
            (
            //(COMMA assignExpr)*
            expr
            |
            //epsilon
            )
            RPAREN
        )?
        //~(LPAREN | RPAREN | COMMA)
        //|  LPAREN attributeList RPAREN
        )?
        ;


/* List of Declarator */
initDeclList returns [List dlist]
{
Declarator decl=null;
dlist = new LinkedList();
}
        :
        decl = initDecl
{dlist.add(decl);}
        (
        options{warnWhenFollowAmbig=false;}
        :
        COMMA!
        decl = initDecl
{dlist.add(decl);}
        )*
        ( COMMA! )?
        ;


/* Declarator */
initDecl returns [Declarator decl]
{
decl = null;
//Initializer binit=null;
Object binit = null;
Expression expr1=null;
}
        :
        // casting could cause a problem
        decl = declarator
        ( attributeDecl )* // Not Handled
        (
        ASSIGN binit=initializer
        |
        COLON expr1=expr // What is this guy ?
        )?
{
if (binit instanceof Expression)
  binit = new Initializer((Expression)binit);
if (binit != null) {
  decl.setInitializer((Initializer)binit);
/*
System.out.println("Initializer " + decl.getClass());
decl.print(System.out);
System.out.print(" ");
((Initializer)binit).print(System.out);
System.out.println("");
*/
}
}
        ;


// add a pointer to the type list
pointerGroup returns [List bp]
{
bp = new LinkedList();
Specifier temp = null;
boolean b_const = false;
boolean b_volatile = false;
}
        :
        (
        STAR
        // add the modifer
        (
        temp = typeQualifier
{
if (temp == Specifier.CONST)
  b_const = true;
else if (temp == Specifier.VOLATILE)
  b_volatile = true;
}
        )*
{
if (b_const && b_volatile)
  bp.add(PointerSpecifier.CONST_VOLATILE);
else if (b_const)
  bp.add(PointerSpecifier.CONST);
else if (b_volatile)
  bp.add(PointerSpecifier.VOLATILE);
else
  bp.add(PointerSpecifier.UNQUALIFIED);
b_const = false;
b_volatile = false;
}
        )+
        ;


// need to decide what to add
idList returns [List ilist]
{
int i = 1;
String name;
Specifier temp = null;
ilist = new LinkedList();
}
        :
        idl1:ID
{
name = idl1.getText();
ilist.add(
    new VariableDeclaration(new VariableDeclarator(new NameID(name))));
}
        (
        options{warnWhenFollowAmbig=false;}
        :
        COMMA
        idl2:ID
{
name = idl2.getText();
ilist.add(
    new VariableDeclaration(new VariableDeclarator(new NameID(name))));
}
        )*
        ;


initializer returns [Object binit]
{binit = null; Expression expr1 = null; List ilist = null;}
        :
        (
        (
            ( (initializerElementLabel)=> initializerElementLabel )?
            (
            expr1=assignExpr
{binit = expr1;}
            |
            ilist=lcurlyInitializer
{binit = new Initializer(ilist);}
            )
        )
        |
        ilist=lcurlyInitializer
{binit = new Initializer(ilist);}
        )
        ;


// GCC allows more specific initializers
initializerElementLabel
{Expression expr1 = null,expr2=null;}
        :
        (
        (
            LBRACKET
            (
            (expr1=constExpr VARARGS) => expr1=rangeExpr
            |
            expr2=constExpr
            )
            RBRACKET (ASSIGN)?
        )
        |
        ID COLON
        |
        DOT ID ASSIGN
        )
        ;


// GCC allows empty initializer lists
lcurlyInitializer returns [List ilist]
{ilist = new LinkedList();}
        :
        LCURLY^
        (ilist=initializerList ( COMMA! )? )?
        RCURLY
        ;


initializerList returns [List ilist]
{Object init = null; ilist = new LinkedList();}
        :
        (
        init = initializer
{ilist.add(init);}
        )
        (
        options{warnWhenFollowAmbig=false;}
        :
        COMMA! init = initializer
{ilist.add(init);}
        )*
        ;


/* Declarator */
declarator returns [Declarator decl]
{
Expression expr1=null;
String declName = null;
decl = null;
Declarator tdecl = null;
IDExpression idex = null;
List plist = null;
List bp = null;
Specifier aspec = null;
boolean isArraySpec = false;
boolean isNested = false;
List llist = new LinkedList();
List tlist = null;
}
        :
        // Pass "Type" to add pointer Type
        ( bp=pointerGroup )?
{/* if(bp == null) bp = new LinkedList(); */}
        (attributeDecl)? // For cygwin support
        (
        id:ID
        // Add the name of the Var
{
declName = id.getText();
idex = new NameID(declName);
if(hastypedef) {
  typetable.put(declName,"typedef");
}
}
        |
        /* Nested Declarator */
        LPAREN
        tdecl = declarator
        RPAREN
        )
        // Attribute Specifier List Possible
        (attributeDecl)?
        // I give up this part !!!
        (
        /* Parameter List */
        plist = declaratorParamaterList
        |
        /* ArraySpecifier */
        LBRACKET ( expr1=expr )? RBRACKET
{
isArraySpec = true;
llist.add(expr1);
}
        )*
{
/* Possible combinations []+, () */
if (plist != null) {
  /* () */
  ;
} else {
  /* []+ */
  if (isArraySpec) {
    aspec = new ArraySpecifier(llist);
    tlist = new LinkedList();
    tlist.add(aspec);
  }
}
if (bp == null)
  bp = new LinkedList();
if (tdecl != null) { // assume tlist == null
  //assert tlist == null : "Assertion (tlist == null) failed 2";
  if (tlist == null)
    tlist = new LinkedList();
    decl = new NestedDeclarator(bp,tdecl,plist,tlist);
} else {
  if (plist != null) // assume tlist == null
    decl = new ProcedureDeclarator(bp,idex,plist);
  else {
    if (tlist != null)
      decl = new VariableDeclarator(bp,idex,tlist);
    else
      decl = new VariableDeclarator(bp,idex);
  }
}
}
        ;


/* List */
declaratorParamaterList returns [List plist]
{plist = new LinkedList();}
        :
        LPAREN^
        (
        (declSpecifiers) => plist=parameterTypeList
        |
        (plist=idList)?
        )
        ( COMMA! )?
        RPAREN
        ;


/* List of (?) */
parameterTypeList returns [List ptlist]
{ptlist = new LinkedList(); Declaration pdecl = null;}
        :
        pdecl=parameterDeclaration
{ptlist.add(pdecl);}
        (
        options {warnWhenFollowAmbig = false;}
        :
        ( COMMA | SEMI )
        pdecl = parameterDeclaration
{ptlist.add(pdecl);}
        )*
        /* What about "..." ? */
        (
        ( COMMA | SEMI )
        VARARGS
{
ptlist.add(
    new VariableDeclaration(new VariableDeclarator(new NameID("..."))));
}
        )?
        ;


/* Declaration (?) */
parameterDeclaration returns [Declaration pdecl]
{
pdecl =null;
List dspec = null;
Declarator decl = null;
boolean prevhastypedef = hastypedef;
hastypedef = false;
}
        :
        dspec=declSpecifiers
        (
        ( declarator )=> decl = declarator
        |
        decl = nonemptyAbstractDeclarator
        )?
{
if (decl != null) {
  pdecl = new VariableDeclaration(dspec,decl);
if (isFuncDef) {
  currproc.add(pdecl);
}
} else
  pdecl = new VariableDeclaration(
      dspec,new VariableDeclarator(new NameID("")));
hastypedef = prevhastypedef;
}
        ;


/* JTC:
* This handles both new and old style functions.
* see declarator rule to see differences in parameters
* and here (declaration SEMI)* is the param type decls for the
* old style.  may want to do some checking to check for illegal
* combinations (but I assume all parsed code will be legal?)
*/

functionDef returns [Procedure curFunc]
{
CompoundStatement stmt=null;
Declaration decl=null;
Declarator bdecl=null;
List dspec=null;
curFunc = null;
String declName=null;
int dcount = 0;
SymbolTable prev_symtab =null;
SymbolTable temp_symtab = new CompoundStatement();
}
        :
        (
{isFuncDef = true;}
        (functionDeclSpecifiers) => dspec=functionDeclSpecifiers
        |
        //epsilon
        )
{if (dspec == null) dspec = new LinkedList();}
        bdecl = declarator
        /* This type of declaration is a problem */
{enterSymtab(temp_symtab); old_style_func = true; func_decl_list.clear();}
        ( declaration {dcount++;})* (VARARGS)? ( SEMI! )*
{
old_style_func = false;
exitSymtab();
isFuncDef = false;
if (dcount > 0) {
  HashMap hm = null;
  NameID name = null;
  Declaration tdecl = null;
/**
 *  This implementation is not so good since it relies on
 * the fact that function parameter starts from the second
 *  children and getChildren returns a reference to the
 * actual internal list
 */
ListIterator i = ((List)(bdecl.getChildren())).listIterator();
i.next();
while (i.hasNext()) {
  VariableDeclaration vdec = (VariableDeclaration)i.next();
  Iterator j = vdec.getDeclaredIDs().iterator();
  while (j.hasNext()) {
    // declarator name
    name = (NameID)(j.next());
    // find matching Declaration
    tdecl = (Declaration)(func_decl_list.get(name.toString()));
    if (tdecl == null) {
      PrintTools.printlnStatus("cannot find symbol " + name
          + "in old style function declaration, now assuming an int",1);
      tdecl = new VariableDeclaration(
          Specifier.INT, new VariableDeclarator(name.clone()));
      i.set(tdecl);
    } else // replace declaration
      i.set(tdecl);
    tdecl.setParent(bdecl);
  }
}
Iterator diter = temp_symtab.getDeclarations().iterator();
Object tobject = null;
while (diter.hasNext()) {
  tobject = diter.next();
  if (tobject instanceof PreAnnotation)
    symtab.addDeclaration((Declaration)tobject);
}
}

}
        stmt=compoundStatement
{
// support for K&R style declaration: "dcount" is counting the number of
// declaration in old style.
curFunc = new Procedure(dspec, bdecl, stmt, dcount>0);
PrintTools.printStatus("Creating Procedure: ",1);
PrintTools.printlnStatus(bdecl,1);
// already handled in constructor
currproc.clear();
}
        ;


functionDeclSpecifiers returns [List dspec]
{
dspec = new LinkedList();
Specifier type=null;
Specifier tqual=null;
Specifier tspec=null;
}
        :
        (
        // this loop properly aborts when it finds a non-typedefName ID MBZ
        options {warnWhenFollowAmbig = false;}
        :
        type=functionStorageClassSpecifier
{dspec.add(type);}
        |
        tqual=typeQualifier
{dspec.add(tqual);}
        |
        ( "struct" | "union" | "enum" | tspec=typeSpecifier)=>
        tspec=typeSpecifier
{dspec.add(tspec);}
        )+
        ;


declarationList
{Declaration decl=null;LinkedList tlist = new LinkedList();}
        :
        (
        // this loop properly aborts when it finds a non-typedefName ID MBZ
        options {warnWhenFollowAmbig = false;}
        :
        localLabelDeclaration
        |
        ( declarationPredictor )=>
        decl=declaration
{if(decl != null ) curr_cstmt.addDeclaration(decl);}
        )+
        ;


declarationPredictor
{Declaration decl=null;}
        :
        (
        //only want to look at declaration if I don't see typedef
        options {warnWhenFollowAmbig = false;}
        :
        "typedef"
        |
        decl=declaration
        )
        ;


localLabelDeclaration
        :
        (
        // GNU note:  any __label__ declarations must come before regular
        // declarations.
        "__label__"^ ID
        (
        options{warnWhenFollowAmbig=false;}
        : COMMA! ID
        )*
        ( COMMA! )? ( SEMI! )+
        )
        ;


compoundStatement returns [CompoundStatement stmt]
{
stmt = null;
int linenum = 0;
SymbolTable prev_symtab = null;
CompoundStatement prev_cstmt = null;
}
        :
        lcur:LCURLY^
{
linenum = lcur.getLine();
prev_symtab = symtab;
prev_cstmt = curr_cstmt;
stmt = new CompoundStatement();
enterSymtab(stmt);
stmt.setLineNumber(linenum);
putPragma(lcur,prev_symtab);
curr_cstmt = stmt;
}
        (
        // this ambiguity is ok, declarationList and nestedFunctionDef end
        // properly
        options {warnWhenFollowAmbig = false;}
        :
        ( "typedef" | "__label__" | declaration ) => declarationList
        |
        (nestedFunctionDef) => nestedFunctionDef // not going to handle this
        )*
        ( statementList )?
        rcur:RCURLY
{
linenum = rcur.getLine();
putPragma(rcur,symtab);
curr_cstmt = prev_cstmt;
exitSymtab();
}
        ;


// Not handled now
nestedFunctionDef
{Declarator decl=null;}
        :
        ( "auto" )? //only for nested functions
        ( (functionDeclSpecifiers)=> functionDeclSpecifiers )?
        // "null" could cause a problem
        decl = declarator
        ( declaration )*
        compoundStatement
        ;


statementList
{Statement statb = null;}
        :
        (
        statb = statement
{curr_cstmt.addStatement(statb);}
        )+
        ;


statement returns [Statement statb]
{
Expression stmtb_expr;
statb = null;
Expression expr1=null, expr2=null, expr3=null;
Statement stmt1=null,stmt2=null;
int a=0;
int sline = 0;
}
        :
        /* NullStatement */
        tsemi:SEMI
{
sline = tsemi.getLine();
statb = new NullStatement();
putPragma(tsemi,symtab);
}
        |
        /* CompoundStatement */
        statb=compoundStatement
        |
        /* ExpressionStatement */
        stmtb_expr=expr exprsemi:SEMI!
{
sline = exprsemi.getLine();
putPragma(exprsemi,symtab);
/* I really shouldn't do this test */
statb = new ExpressionStatement(stmtb_expr);
}
        /* Iteration statements */
        |
        /* WhileLoop */
        twhile:"while"^ LPAREN!
{
sline = twhile.getLine();
putPragma(twhile,symtab);
}
        expr1=expr RPAREN! stmt1=statement
{
statb = new WhileLoop(expr1, stmt1);
statb.setLineNumber(sline);
}
        |
        /* DoLoop */
        tdo:"do"^
{
sline = tdo.getLine();
putPragma(tdo,symtab);
}
        stmt1=statement "while"! LPAREN!
        expr1=expr RPAREN! SEMI!
{
statb = new DoLoop(stmt1, expr1);
statb.setLineNumber(sline);
}
        |
        /* ForLoop */

        !tfor:"for"
{
sline = tfor.getLine();
putPragma(tfor,symtab);
}
        LPAREN ( expr1=expr )?
        SEMI ( expr2=expr )? 
        SEMI ( expr3=expr )?
        RPAREN
        stmt1=statement
{
if(expr1 != null)
  statb = new ForLoop(new ExpressionStatement(expr1),expr2,expr3,stmt1);
else
  statb = new ForLoop(new NullStatement(),expr2,expr3,stmt1);
statb.setLineNumber(sline);
}
        /* Jump statements */
        |
        /* GotoStatement */
        tgoto:"goto"^
{
sline = tgoto.getLine();
putPragma(tgoto,symtab);
}
        gotoTarget:ID SEMI!
{
statb = new GotoStatement(new NameID(gotoTarget.getText()));
statb.setLineNumber(sline);
}
        |
        /* ContinueStatement */
        tcontinue:"continue" SEMI!
{
sline = tcontinue.getLine();
statb = new ContinueStatement();
statb.setLineNumber(sline);
putPragma(tcontinue,symtab);
}
        |
        /* BreakStatement */
        tbreak:"break" SEMI!
{
sline = tbreak.getLine();
statb = new BreakStatement();
statb.setLineNumber(sline);
putPragma(tbreak,symtab);
}
        |
        /* ReturnStatement */
        treturn:"return"^
{
sline = treturn.getLine();
}
        ( expr1=expr )? SEMI!
{
if (expr1 != null)
  statb=new ReturnStatement(expr1);
else
  statb=new ReturnStatement();
statb.setLineNumber(sline);
putPragma(treturn,symtab);
}
        |
        /* Label */
        lid:ID COLON!
{
sline = lid.getLine();
Object o = null;
Declaration target = null;
statb = new Label(new NameID(lid.getText()));
statb.setLineNumber(sline);
putPragma(lid,symtab);
}
        // Attribute Specifier List Possible
        (attributeDecl)?
        (
        options {warnWhenFollowAmbig=false;}
        :
        stmt1=statement
{
CompoundStatement cstmt = new CompoundStatement();
cstmt.addStatement(statb);
statb = cstmt;
cstmt.addStatement(stmt1);
}
        )?
        // GNU allows range expressions in case statements
        |
        /* Case */
        tcase:"case"^
{
sline = tcase.getLine();
}
        (
        (constExpr VARARGS)=> expr1=rangeExpr
        |
        expr1=constExpr
        )
{
statb = new Case(expr1);
statb.setLineNumber(sline);
putPragma(tcase,symtab);
}
        COLON!
        (
        options{warnWhenFollowAmbig=false;}
        :
        stmt1=statement
{
CompoundStatement cstmt = new CompoundStatement();
cstmt.addStatement(statb);
statb = cstmt;
cstmt.addStatement(stmt1);
}
        )?
        |
        /* Default */
        tdefault:"default"^
{
sline = tdefault.getLine();
statb = new Default();
statb.setLineNumber(sline);
putPragma(tdefault,symtab);
}
        COLON!
        (
        options{warnWhenFollowAmbig=false;}
        :
        stmt1=statement
{
CompoundStatement cstmt = new CompoundStatement();
cstmt.addStatement(statb);
statb = cstmt;
cstmt.addStatement(stmt1);
}
        )?
        /* Selection statements */
        |
        /* IfStatement  */
        tif:"if"^
{
sline = tif.getLine();
putPragma(tif,symtab);
}
        LPAREN! expr1=expr RPAREN! stmt1=statement
        //standard if-else ambiguity
        (
        options {warnWhenFollowAmbig = false;}
        :
        "else" stmt2=statement
        )?
{
if (stmt2 != null)
  statb = new IfStatement(expr1,stmt1,stmt2);
else
  statb = new IfStatement(expr1,stmt1);
statb.setLineNumber(sline);
}
        |
        /* SwitchStatement */
        tswitch:"switch"^ LPAREN!
{
sline = tswitch.getLine();
}
        expr1=expr RPAREN!
{
statb = new SwitchStatement(expr1);
statb.setLineNumber(sline);
putPragma(tswitch,symtab);
}
        stmt1=statement
{
((SwitchStatement)statb).setBody((CompoundStatement)stmt1);
}
        ;


/* Expression */
expr returns [Expression ret_expr]
{
ret_expr = null;
Expression expr1=null,expr2=null;
List elist = new LinkedList();
}
        :
        ret_expr=assignExpr
{elist.add(ret_expr);}
        (
        options {warnWhenFollowAmbig = false;}
        :
        /* MBZ:
        COMMA is ambiguous between comma expressions and
        argument lists.  argExprList should get priority,
        and it does by being deeper in the expr rule tree
        and using (COMMA assignExpr)*
        */
        /* CommaExpression is not handled now */
        c:COMMA^
        expr1=assignExpr
{elist.add(expr1);}
        )*
{
if (elist.size() > 1) {
  ret_expr = new CommaExpression(elist);
}
}
        ;


assignExpr returns [Expression ret_expr]
{ret_expr = null; Expression expr1=null; AssignmentOperator code=null;}
        :
        ret_expr=conditionalExpr
        (
        code = assignOperator!
        expr1=assignExpr
{ret_expr = new AssignmentExpression(ret_expr,code,expr1); }
        )?
        ;


assignOperator returns [AssignmentOperator code]
{code = null;}
        :
        ASSIGN
{code = AssignmentOperator.NORMAL;}
        |
        DIV_ASSIGN
{code = AssignmentOperator.DIVIDE;}
        |
        PLUS_ASSIGN
{code = AssignmentOperator.ADD;}
        |
        MINUS_ASSIGN
{code = AssignmentOperator.SUBTRACT;}
        |
        STAR_ASSIGN
{code = AssignmentOperator.MULTIPLY;}
        |
        MOD_ASSIGN
{code = AssignmentOperator.MODULUS;}
        |
        RSHIFT_ASSIGN
{code = AssignmentOperator.SHIFT_RIGHT;}
        |
        LSHIFT_ASSIGN
{code = AssignmentOperator.SHIFT_LEFT;}
        |
        BAND_ASSIGN
{code = AssignmentOperator.BITWISE_AND;}
        |
        BOR_ASSIGN
{code = AssignmentOperator.BITWISE_INCLUSIVE_OR;}
        |
        BXOR_ASSIGN
{code = AssignmentOperator.BITWISE_EXCLUSIVE_OR;}
        ;


constExpr returns [Expression ret_expr]
{ret_expr = null;}
        :
        ret_expr=conditionalExpr
        ;


logicalOrExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=logicalAndExpr
        (
        LOR^ expr1=logicalAndExpr
{ret_expr = new BinaryExpression(ret_expr,BinaryOperator.LOGICAL_OR,expr1);}
        )*
        ;


logicalAndExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=inclusiveOrExpr
        (
        LAND^ expr1=inclusiveOrExpr
{ret_expr = new BinaryExpression(ret_expr,BinaryOperator.LOGICAL_AND,expr1);}
        )*
        ;


inclusiveOrExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=exclusiveOrExpr
        (
        BOR^ expr1=exclusiveOrExpr
{
ret_expr = new BinaryExpression
    (ret_expr,BinaryOperator.BITWISE_INCLUSIVE_OR,expr1);
}
        )*
        ;


exclusiveOrExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=bitAndExpr
        (
        BXOR^ expr1=bitAndExpr
{
ret_expr = new BinaryExpression
    (ret_expr,BinaryOperator.BITWISE_EXCLUSIVE_OR,expr1);
}
        )*
        ;


bitAndExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=equalityExpr
        (
        BAND^ expr1=equalityExpr
{ret_expr = new BinaryExpression(ret_expr,BinaryOperator.BITWISE_AND,expr1);}
        )*
        ;


equalityExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=relationalExpr
        (
        (
        EQUAL^
{code = BinaryOperator.COMPARE_EQ;}
        |
        NOT_EQUAL^
{code = BinaryOperator.COMPARE_NE;}
        )
        expr1=relationalExpr
{ret_expr = new BinaryExpression(ret_expr,code,expr1);}
        )*
        ;


relationalExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=shiftExpr
        (
        (
        LT^
{code = BinaryOperator.COMPARE_LT;}
        |
        LTE^
{code = BinaryOperator.COMPARE_LE;}
        |
        GT^
{code = BinaryOperator.COMPARE_GT;}
        |
        GTE^
{code = BinaryOperator.COMPARE_GE;}
        )
        expr1=shiftExpr
{ret_expr = new BinaryExpression(ret_expr,code,expr1);}
        )*
        ;


shiftExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=additiveExpr
        (
        (
        LSHIFT^
{code = BinaryOperator.SHIFT_LEFT;}
        |
        RSHIFT^
{code = BinaryOperator.SHIFT_RIGHT;}
        )
        expr1=additiveExpr
{ret_expr = new BinaryExpression(ret_expr,code,expr1);}
        )*
        ;


additiveExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=multExpr
        (
        (
        PLUS^
{code = BinaryOperator.ADD;}
        |
        MINUS^
{code=BinaryOperator.SUBTRACT;}
        )
        expr1=multExpr
{ret_expr = new BinaryExpression(ret_expr,code,expr1);}
        )*
        ;


multExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=castExpr
        (
        (
        STAR^
{code = BinaryOperator.MULTIPLY;}
        |
        DIV^
{code=BinaryOperator.DIVIDE;}
        |
        MOD^
{code=BinaryOperator.MODULUS;}
        )
        expr1=castExpr
{ret_expr = new BinaryExpression(ret_expr,code,expr1);}
        )*
        ;


typeName returns [List tname]
{
tname=null;
Declarator decl = null;
}
        :
        tname = specifierQualifierList
        /* Need to add this part */
        (decl = nonemptyAbstractDeclarator {tname.add(decl);})?
        ;


postfixExpr returns [Expression ret_expr]
{
ret_expr=null;
Expression expr1=null;
}
        :
        expr1=primaryExpr
{ret_expr = expr1;}
        ( ret_expr=postfixSuffix[expr1] )?
        ;


postfixSuffix [Expression expr1] returns [Expression ret_expr]
{
Expression expr2=null;
SymbolTable saveSymtab = null;
String s;
ret_expr = expr1;
List args = null;
}
        :
        (
        /* POINTER_ACCESS */
        PTR ptr_id:ID
{
ret_expr = new AccessExpression(
    ret_expr, AccessOperator.POINTER_ACCESS, SymbolTools.getOrphanID(ptr_id.getText()));
}
        |
        /* MEMBER_ACCESS */
        DOT dot_id:ID
{
ret_expr = new AccessExpression(
    ret_expr, AccessOperator.MEMBER_ACCESS, SymbolTools.getOrphanID(dot_id.getText()));
}
        /* FunctionCall */
        |
        args=functionCall
{
if (args == null)
  ret_expr = new FunctionCall(ret_expr);
else
  ret_expr = new FunctionCall(ret_expr,args);
}
        /* ArrayAcess - Need a fix for multi-dimension access */
        |
        LBRACKET expr2=expr RBRACKET
{
if (ret_expr instanceof ArrayAccess) {
  ArrayAccess aacc = (ArrayAccess)ret_expr;
  int dim = aacc.getNumIndices();
  int n = 0;
  LinkedList alist = new LinkedList();
  for (n = 0;n < dim; n++) {
    alist.add(aacc.getIndex(n).clone());
  }
  alist.add(expr2);
  aacc.setIndices(alist);
} else
  ret_expr = new ArrayAccess(ret_expr,expr2);
}
        |
        INC
{ret_expr = new UnaryExpression(UnaryOperator.POST_INCREMENT,ret_expr);}
        |
        DEC
{ret_expr = new UnaryExpression(UnaryOperator.POST_DECREMENT,ret_expr);}
        )+
        ;


functionCall returns [List args]
{args=null;}
        :
        LPAREN^  (args=argExprList)? RPAREN
        ;


conditionalExpr returns [Expression ret_expr]
{ret_expr=null; Expression expr1=null,expr2=null,expr3=null;}
        :
        expr1=logicalOrExpr
{ret_expr = expr1;}
        (
        QUESTION^ (expr2=expr)? COLON expr3=conditionalExpr
{ret_expr = new ConditionalExpression(expr1,expr2,expr3);}
        )?
        ;


//used in initializers only
rangeExpr returns [Expression ret_expr] 
{ret_expr = null;}
        :
        constExpr VARARGS constExpr
        ;


castExpr returns [Expression ret_expr]
{
ret_expr = null;
Expression expr1=null;
List tname=null;
}
        :
        ( LPAREN typeName RPAREN )=>
        LPAREN^ tname=typeName RPAREN
        (
        expr1=castExpr
{ret_expr = new Typecast(tname,expr1);}
        |
        lcurlyInitializer // What is this ?
        )
        |
        ret_expr=unaryExpr
        ;


/* This causing problems with type casting */
nonemptyAbstractDeclarator returns [Declarator adecl]
{
Expression expr1=null;
List plist=null;
List bp = null;
Declarator tdecl = null;
Specifier aspec = null;
boolean isArraySpec = false;
boolean isNested = false;
List llist = new LinkedList();
List tlist = null;
boolean empty = true;
adecl = null;
}
        :
        (
        bp = pointerGroup
        (
        (
        LPAREN
        (
        (
        (
        tdecl = nonemptyAbstractDeclarator
        )
        |
        // function proto
        plist=parameterTypeList
        )
{empty = false;}
        )?
        ( COMMA! )?
        RPAREN
        )
{
if(empty)
plist = new LinkedList();
empty = true;
}
        |
        (LBRACKET (expr1=expr)? RBRACKET)
{
isArraySpec = true;
llist.add(expr1);
}
        )*
        |
        (
        (
        LPAREN
        (
        (
        (
        tdecl = nonemptyAbstractDeclarator
        )
        |
        // function proto
        plist=parameterTypeList
        )
{empty = false;}
        )?
        ( COMMA! )?
        RPAREN
        )
{
if (empty)
  plist = new LinkedList();
empty = true;
}
        |
        (LBRACKET (expr1=expr)? RBRACKET)
{
isArraySpec = true;
llist.add(expr1);
}
        )+
        )
{
if (isArraySpec) {
  /* []+ */
  aspec = new ArraySpecifier(llist);
  tlist = new LinkedList();
  tlist.add(aspec);
}
NameID idex = null;
// nested declarator (tlist == null ?)
if (bp == null)
  bp = new LinkedList();
// assume tlist == null
if (tdecl != null) {
  //assert tlist == null : "Assertion (tlist == null) failed 2";
  if (tlist == null)
    tlist = new LinkedList();
  adecl = new NestedDeclarator(bp,tdecl,plist,tlist);
} else {
  idex = new NameID("");
  if (plist != null) // assume tlist == null
    adecl = new ProcedureDeclarator(bp,idex,plist);
  else {
    if (tlist != null)
      adecl = new VariableDeclarator(bp,idex,tlist);
    else
      adecl = new VariableDeclarator(bp,idex);
  }
}
}
        ;


unaryExpr returns [Expression ret_expr]
{
Expression expr1=null;
UnaryOperator code;
ret_expr = null;
List tname = null;
}
        :
        ret_expr=postfixExpr
        |
        INC^ expr1=castExpr
{ret_expr = new UnaryExpression(UnaryOperator.PRE_INCREMENT, expr1);}
        |
        DEC^ expr1=castExpr
{ret_expr = new UnaryExpression(UnaryOperator.PRE_DECREMENT, expr1);}
        |
        code=unaryOperator expr1=castExpr
{ret_expr = new UnaryExpression(code, expr1);}
        /* sizeof is not handled */
        |
        "sizeof"^
        (
        (LPAREN typeName ) => LPAREN tname=typeName RPAREN
{ret_expr = new SizeofExpression(tname);}
        |
        expr1=unaryExpr
{ret_expr = new SizeofExpression(expr1);}
        )
        |
        // Handles __alignof__ operator
        "__alignof__"^
        (
        ( LPAREN typeName ) => LPAREN tname=typeName RPAREN
{ret_expr = new AlignofExpression(tname);}
        |
        expr1=unaryExpr
{ret_expr = new AlignofExpression(expr1);}
        )
        |
        // Handles the builtin GCC function __builtin_va_arg
        // as an intrinsic function (operator)
        "__builtin_va_arg"
        (
        (
        LPAREN
        ( expr1 = unaryExpr )
        COMMA
        ( tname = typeName )
        RPAREN
        )
{ret_expr = new VaArgExpression(expr1, tname);}
        )
        |
        // Handles the builtin GCC function __builtin_offsetof
        // as an intrinsic function (operator)
        "__builtin_offsetof"
        (
        (
        LPAREN
        ( tname = typeName )
        COMMA
        ( expr1 = unaryExpr )
        RPAREN
        )
{ret_expr = new OffsetofExpression(tname, expr1);}
        )
        |
        (ret_expr=gnuAsmExpr)
        ;


unaryOperator returns [UnaryOperator code]
{code = null;}
        :
        BAND
{code = UnaryOperator.ADDRESS_OF;}
        |
        STAR
{code = UnaryOperator.DEREFERENCE;}
        |
        PLUS
{code = UnaryOperator.PLUS;}
        |
        MINUS
{code = UnaryOperator.MINUS;}
        |
        BNOT
{code = UnaryOperator.BITWISE_COMPLEMENT;}
        |
        LNOT
{code = UnaryOperator.LOGICAL_NEGATION;}
        |
        "__real"
{code = null;}
        |
        "__imag"
{code = null;}
        ;


gnuAsmExpr returns [Expression ret]
{
ret = null;
String str = "";
List<Traversable> expr_list = new LinkedList<Traversable>();
int count = 0;
}
        :
{count = mark();} // mark the previous token of __asm__
        "__asm"^ ("volatile")?
        LPAREN stringConst
        (
        options { warnWhenFollowAmbig = false; }
        :
        COLON (strOptExprPair[expr_list] ( COMMA strOptExprPair[expr_list])* )?
        (
        options { warnWhenFollowAmbig = false; }
        :
        COLON (strOptExprPair[expr_list] ( COMMA strOptExprPair[expr_list])* )?
        )?
        )?
        ( COLON stringConst ( COMMA stringConst)* )?
        RPAREN
{
// Recover the original stream and stores it in "SomeExpression" augmented with
// list of evaluated expressions.
for (int i=count-mark()+1; i <= 0; i++)
  str += " " + LT(i).getText();
ret = new SomeExpression(str, expr_list);
}
        ;


// GCC requires the PARENs
strOptExprPair [List<Traversable> expr_list]
{Expression e = null;}
        :
        stringConst
        (LPAREN (e=expr) RPAREN
{expr_list.add(e);}
        )?
        ;


primaryExpr returns [Expression p]
{
Expression expr1=null;
CompoundStatement cstmt = null;
p=null;
String name = null;
}
        :
        /* Identifier */
        prim_id:ID
{
name = prim_id.getText();
p=SymbolTools.getOrphanID(name);
}
        |
        /* Need to handle these correctly */
        prim_num:Number
{
name = prim_num.getText();
boolean handled = false;
int i = 0;
int radix = 10;
double d = 0;
Integer i2 = null;
Long in = null;
name = name.toUpperCase();
String suffix = name.replaceAll("[X0-9\\.]","");
name = name.replaceAll("L","");
name = name.replaceAll("U","");
//name= name.replaceAll("I"," ");
if (name.startsWith("0X") == false) {
  name = name.replaceAll("F","");
  name = name.replaceAll("I","");
  // 1.0IF can be generated from _Complex_I
}
try {
  i2 = Integer.decode(name);
  p=new IntegerLiteral(i2.intValue());
  handled = true;
} catch(NumberFormatException e) {
  ;
}
if (handled == false) {
  try {
    in = Long.decode(name);
    //p=new IntegerLiteral(in.intValue());
    p=new IntegerLiteral(in.longValue());
    handled = true;
  } catch(NumberFormatException e) {
    ;
  }
}
if (handled == false) {
  try {
    d = Double.parseDouble(name);
    if (suffix.matches("F|L|IF"))
      p = new FloatLiteral(d, suffix);
    else
      p = new FloatLiteral(d);
    handled = true;
  } catch(NumberFormatException e) {
    p=new NameID(name);
    PrintTools.printlnStatus("Strange number "+name,0);
  }
}
}
        |
        name=charConst
{
if(name.length()==3)
  p = new CharLiteral(name.charAt(1));
// escape sequence is not handled at this point
else {
  p = new EscapeLiteral(name);
}
}
        |
        /* StringLiteral */
        name=stringConst
{
p=new StringLiteral(name);
((StringLiteral)p).stripQuotes();
}
        // JTC:
        // ID should catch the enumerator
        // leaving it in gives ambiguous err
        //      | enumerator
        |
        /* Compound statement Expression */
        (LPAREN LCURLY) =>
        LPAREN^
        cstmt = compoundStatement
        RPAREN
{
PrintTools.printlnStatus("[DEBUG] Warning: CompoundStatement Expression !",1);
p = new StatementExpression(cstmt);
}
        |
        /* Paren */
        LPAREN^ expr1=expr RPAREN
{
p=expr1;
}
        ;


/* Type of list is unclear */
argExprList returns [List eList]
{
Expression expr1 = null;
eList=new LinkedList();
Declaration pdecl = null;
}
        :
        expr1=assignExpr
{eList.add(expr1);}
        (
        COMMA!
        (
        expr1=assignExpr
{eList.add(expr1);}
        |
        pdecl=parameterDeclaration
{eList.add(pdecl);}
        )
        )*
        ;


protected charConst returns [String name]
{name = null;}
        :
        cl:CharLiteral
{name=cl.getText();}
        ;


protected stringConst returns [String name]
{name = "";}
        :
        (
        sl:StringLiteral
{name += sl.getText();}
        )+
        ;


protected
intConst
        :       IntOctalConst
        |       LongOctalConst
        |       UnsignedOctalConst
        |       IntIntConst
        |       LongIntConst
        |       UnsignedIntConst
        |       IntHexConst
        |       LongHexConst
        |       UnsignedHexConst
        ;


protected
floatConst
        :       FloatDoubleConst
        |       DoubleDoubleConst
        |       LongDoubleConst
        ;


dummy
        :       NTypedefName
        |       NInitDecl
        |       NDeclarator
        |       NStructDeclarator
        |       NDeclaration
        |       NCast
        |       NPointerGroup
        |       NExpressionGroup
        |       NFunctionCallArgs
        |       NNonemptyAbstractDeclarator
        |       NInitializer
        |       NStatementExpr
        |       NEmptyExpression
        |       NParameterTypeList
        |       NFunctionDef
        |       NCompoundStatement
        |       NParameterDeclaration
        |       NCommaExpr
        |       NUnaryExpr
        |       NLabel
        |       NPostfixExpr
        |       NRangeExpr
        |       NStringSeq
        |       NInitializerElementLabel
        |       NLcurlyInitializer
        |       NAsmAttribute
        |       NGnuAsmExpr
        |       NTypeMissing
        ;


{
import java.io.*;
import antlr.*;
}

class NewCLexer extends Lexer;

options
{
k = 3;
exportVocab = NEWC;
testLiterals = false;
}

tokens
{
LITERAL___extension__ = "__extension__";
}

{
public void initialize(String src)
{
  setOriginalSource(src);
  initialize();
}

public void initialize()
{
  literals.put(new ANTLRHashString("__alignof__", this),
      new Integer(LITERAL___alignof__));
  literals.put(new ANTLRHashString("__ALIGNOF__", this),
      new Integer(LITERAL___alignof__));
  literals.put(new ANTLRHashString("__asm", this),
      new Integer(LITERAL___asm));
  literals.put(new ANTLRHashString("__asm__", this),
      new Integer(LITERAL___asm));
  literals.put(new ANTLRHashString("__attribute__", this), 
      new Integer(LITERAL___attribute));
  literals.put(new ANTLRHashString("__complex__", this), 
      new Integer(LITERAL___complex));
  literals.put(new ANTLRHashString("__const", this), 
      new Integer(LITERAL_const));
  literals.put(new ANTLRHashString("__const__", this), 
      new Integer(LITERAL_const));
  literals.put(new ANTLRHashString("__imag__", this), 
      new Integer(LITERAL___imag));
  literals.put(new ANTLRHashString("__inline", this), 
      new Integer(LITERAL_inline));
  literals.put(new ANTLRHashString("__inline__", this), 
      new Integer(LITERAL_inline));
  literals.put(new ANTLRHashString("__real__", this), 
      new Integer(LITERAL___real));
  literals.put(new ANTLRHashString("__restrict", this), 
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("__extension", this), 
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("__signed", this), 
      new Integer(LITERAL_signed));
  literals.put(new ANTLRHashString("__signed__", this), 
      new Integer(LITERAL_signed));
  literals.put(new ANTLRHashString("__typeof", this), 
      new Integer(LITERAL_typeof));
  literals.put(new ANTLRHashString("__typeof__", this), 
      new Integer(LITERAL_typeof));
  literals.put(new ANTLRHashString("__volatile", this), 
      new Integer(LITERAL_volatile));
  literals.put(new ANTLRHashString("__volatile__", this), 
      new Integer(LITERAL_volatile));
  // GCC Builtin function
  literals.put(new ANTLRHashString("__builtin_va_arg", this), 
      new Integer(LITERAL___builtin_va_arg));
  literals.put(new ANTLRHashString("__builtin_offsetof", this), 
      new Integer(LITERAL___builtin_offsetof));
  // MinGW specific
  literals.put(new ANTLRHashString("__MINGW_IMPORT", this), 
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("_CRTIMP", this), 
      new Integer(LITERAL___extension__));
  // Microsoft specific
  literals.put(new ANTLRHashString("__cdecl", this), 
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("__w64", this), 
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("__int64", this), 
      new Integer(LITERAL_int));
  literals.put(new ANTLRHashString("__int32", this), 
      new Integer(LITERAL_int));
  literals.put(new ANTLRHashString("__int16", this), 
      new Integer(LITERAL_int));
  literals.put(new ANTLRHashString("__int8", this), 
      new Integer(LITERAL_int));
}

LineObject lineObject = new LineObject();
String originalSource = "";
PreprocessorInfoChannel preprocessorInfoChannel = new PreprocessorInfoChannel();
int tokenNumber = 0;
boolean countingTokens = true;
int deferredLineCount = 0;
int extraLineCount = 1;
NewCParser parser = null;

public void setCountingTokens(boolean ct)
{
  countingTokens = ct;
  if ( countingTokens ) {
    tokenNumber = 0;
  } else {
    tokenNumber = 1;
  }
}

public void setParser(NewCParser p)
{
  parser = p;
}

public void setOriginalSource(String src)
{
  originalSource = src;
  lineObject.setSource(src);
}

public void setSource(String src)
{
  lineObject.setSource(src);
}

public PreprocessorInfoChannel getPreprocessorInfoChannel()
{
  return preprocessorInfoChannel;
}

public void setPreprocessingDirective(String pre,int t)
{
  preprocessorInfoChannel.addLineForTokenNumber(
      new Pragma(pre,t), new Integer(tokenNumber));
}

protected Token makeToken(int t)
{
  if ( t != Token.SKIP && countingTokens) {
    tokenNumber++;
  }
  CToken tok = (CToken) super.makeToken(t);
  tok.setLine(lineObject.line);
  tok.setSource(lineObject.source);
  tok.setTokenNumber(tokenNumber);

  lineObject.line += deferredLineCount;
  deferredLineCount = 0;
  return tok;
}

public void deferredNewline()
{
  deferredLineCount++;
}

public void newline() 
{
  lineObject.newline();
  setColumn(1);
}

}


protected
Vocabulary
        :       '\3'..'\377'
        ;

/* Operators: */
ASSIGN          : '=' ;
COLON           : ':' ;
COMMA           : ',' ;
QUESTION        : '?' ;
SEMI            : ';' ;
PTR             : "->" ;


// DOT & VARARGS are commented out since they are generated as part of
// the Number rule below due to some bizarre lexical ambiguity shme.

// DOT  :       '.' ;
protected
DOT:;

// VARARGS      : "..." ;
protected
VARARGS:;


LPAREN          : '(' ;
RPAREN          : ')' ;
LBRACKET        : '[' ;
RBRACKET        : ']' ;
LCURLY          : '{' ;
RCURLY          : '}' ;

EQUAL           : "==" ;
NOT_EQUAL       : "!=" ;
LTE             : "<=" ;
LT              : "<" ;
GTE             : ">=" ;
GT              : ">" ;

DIV             : '/' ;
DIV_ASSIGN      : "/=" ;
PLUS            : '+' ;
PLUS_ASSIGN     : "+=" ;
INC             : "++" ;
MINUS           : '-' ;
MINUS_ASSIGN    : "-=" ;
DEC             : "--" ;
STAR            : '*' ;
STAR_ASSIGN     : "*=" ;
MOD             : '%' ;
MOD_ASSIGN      : "%=" ;
RSHIFT          : ">>" ;
RSHIFT_ASSIGN   : ">>=" ;
LSHIFT          : "<<" ;
LSHIFT_ASSIGN   : "<<=" ;

LAND            : "&&" ;
LNOT            : '!' ;
LOR             : "||" ;

BAND            : '&' ;
BAND_ASSIGN     : "&=" ;
BNOT            : '~' ;
BOR             : '|' ;
BOR_ASSIGN      : "|=" ;
BXOR            : '^' ;
BXOR_ASSIGN     : "^=" ;


Whitespace
        :
        (
        ( ' ' | '\t' | '\014')
        | "\r\n" {newline();}
        | ( '\n' | '\r' ) {newline();}
        ) { _ttype = Token.SKIP;  }
        ;


Comment
        :
        (
        "/*"
        (
        { LA(2) != '/' }? '*'
        | "\r\n" { deferredNewline();}
        | ( '\r' | '\n' ) { deferredNewline();}
        | ~( '*'| '\r' | '\n' )
        )*
        "*/"
{setPreprocessingDirective(getText(),Pragma.comment);}
        )
{_ttype = Token.SKIP;}
        ;


CPPComment
        :
        (
        "//" ( ~('\n') )*
{setPreprocessingDirective(getText(),Pragma.comment);}
        )
{_ttype = Token.SKIP;}
        ;


PREPROC_DIRECTIVE
        options {paraphrase = "a line directive";}
        :
        '#'
        (
        ( "line" || ((Space)+ Digit)) => LineDirective
{_ttype = Token.SKIP; }
        |
        (
        "pragma"
        (
        ( ~('\n'))*
{setPreprocessingDirective(getText(),Pragma.pragma);_ttype = Token.SKIP;}
        |
        (Space)+ "startinclude" ( ~('\n'))*
        // {startHeader(getText());}
        |
        (Space)+ "endinclude" ( ~('\n'))*
{extraLineCount +=2; lineObject.setLine(lineObject.getLine() - 2);}
        // {endHeader();}
        )
        )
        |
        ( ~('\n'))*
{_ttype = Token.SKIP; }
        )
        ;


protected  Space
        :
        ( ' ' | '\t' | '\014')
        ;


protected LineDirective
{
boolean oldCountingTokens = countingTokens;
countingTokens = false;
}
        :
{
lineObject = new LineObject();
deferredLineCount = 0;
}
        ("line")?
        //this would be for if the directive started "#line",
        //but not there for GNU directives
        (Space)+
        n:Number
{lineObject.setLine(Integer.parseInt(n.getText()) - extraLineCount);}
        (
        (Space)+
        (
        fn:StringLiteral
{
try {
  lineObject.setSource(fn.getText().substring(1,fn.getText().length()-1));
} catch (StringIndexOutOfBoundsException e) { /*not possible*/
}
}
        |
        fi:ID
{lineObject.setSource(fi.getText());}
        )?
        (Space)*
        ("1" {lineObject.setEnteringFile(true);})?
        (Space)*
        ("2" {lineObject.setReturningToFile(true);})?
        (Space)*
        ("3" {lineObject.setSystemHeader(true);})?
        (Space)*
        ("4" {lineObject.setTreatAsC(true);})?
        (~('\r' | '\n'))*
        //("\r\n" | "\r" | "\n")
        )?
{
/*
preprocessorInfoChannel.addLineForTokenNumber(
    new LineObject(lineObject), new Integer(tokenNumber));
*/
countingTokens = oldCountingTokens;
}
        ;


/* Literals: */

/* Note that we do NOT handle tri-graphs nor multi-byte sequences. */

/*
 * Note that we can't have empty character constants (even though we
 * can have empty strings :-).
 */
CharLiteral
        :
        '\'' ( Escape | ~( '\'' ) ) '\''
        ;


protected BadStringLiteral
        :       // Imaginary token.
        ;


protected Escape
        :
        '\\'
        (
        options{warnWhenFollowAmbig=false;}
        :
        ~('0'..'7' | 'x')
        | ('0'..'3') ( options{warnWhenFollowAmbig=false;}: Digit )*
        | ('4'..'7') ( options{warnWhenFollowAmbig=false;}: Digit )*
        | 'x'
        (
        options{warnWhenFollowAmbig=false;}
        :
        Digit | 'a'..'f' | 'A'..'F'
        )+
        )
        ;


/* Numeric Constants: */
protected IntSuffix
        : 'L'
        | 'l'
        | 'U'
        | 'u'
        | 'I'
        | 'i'
        | 'J'
        | 'j'
        ;


protected NumberSuffix
        :
        IntSuffix
        | 'F'
        | 'f'
        ;


protected Digit
        :
        '0'..'9'
        ;


protected Exponent
        :
        ( 'e' | 'E' ) ( '+' | '-' )? ( Digit )+
        ;


Number
        :
        ( ( Digit )+ ( '.' | 'e' | 'E' ) )=> ( Digit )+
        (
        '.' ( Digit )* ( Exponent )?
        |
        Exponent
        )
        (NumberSuffix)*
        |
        ( "..." )=> "..."
{_ttype = VARARGS;}
        |
        '.'
{_ttype = DOT;}
        (
        ( Digit )+ ( Exponent )?
{ _ttype = Number;}
        (NumberSuffix)*
        )?
        |
        '0' ( '0'..'7' )*
        ( NumberSuffix )*
        |
        '1'..'9' ( Digit )*
        ( NumberSuffix )*
        |
        '0' ( 'x' | 'X' ) ( 'a'..'f' | 'A'..'F' | Digit )+
        ( IntSuffix )*
        ;


IDMEAT
        :
        i:ID
{
if ( i.getType() == LITERAL___extension__ ) {
  $setType(Token.SKIP);
} else {
  $setType(i.getType());
}
}
        ;


protected ID
        options {testLiterals = true;}
        :
        ( 'a'..'z' | 'A'..'Z' | '_' | '$')
        ( 'a'..'z' | 'A'..'Z' | '_' | '$' | '0'..'9' )*
        ;


WideCharLiteral
        :
        'L' CharLiteral
{$setType(CharLiteral);}
        ;


WideStringLiteral
        :
        'L' StringLiteral
{$setType(StringLiteral);}
        ;


StringLiteral
        :
        '"'
        (
        ('\\' ~('\n')) => Escape
        |
        (
        '\r'
{newline();}
        |
        '\n'
{newline();}
        |
        '\\' '\n'
{newline();}
        )
        |
        ~( '"' | '\r' | '\n' | '\\' )
        )*
        '"'
        ;
