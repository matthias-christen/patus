header {
package cetus.base.grammars;
import java.io.*;
}

{
}
class PreCParser extends Parser;

options
        {
        k = 2;
        //exportVocab = PreC;
        //buildAST = true;
        //ASTLabelType = "TNode";

        // Copied following options from java grammar.
        codeGenMakeSwitchThreshold = 2;
        codeGenBitsetTestThreshold = 3;
        }

{
	
}

programUnit [PrintStream out]:
		(
			in:Include 
			{
					String s = null;
					s = in.getText();
					
					if(s.startsWith("internal")){
						out.print(s.substring(8));
					}
					else{
						out.print("#pragma startinclude");
						out.print(" "+in.getText());
						out.print(in.getText());
						out.println("#pragma endinclude");
					}
				
			} 
			
		| 	pre:PreprocDirective
			{
				out.print(pre.getText());	
			}
		
		|	re:Rest 
			{
				out.print(re.getText()); 
			}
		)+
		;




class PreCLexer extends Lexer;

options
        {
        k = 3;
        //exportVocab = PreC;
        //testLiterals = false;
        charVocabulary = '\3'..'\377';
        }
         

	
{
	int openCount = 0;
}

PreprocDirective :
		
        '#' 
        ( 
        	( "include" ) => Include
			| Rest                                 
		)
		;
		
Include
        :
        "#include" Rest	
        { 
        	if(openCount != 0) {
        		String text = getText();
        		setText("internal"+text);
        	}
        }
        	
        ;

Rest 
		: 
			(
				~( '\n' | '\r' | '{' | '}') 
				| Lcurly
				| Rcurly
			)*
		 	Newline
		
		;
		

Newline
        :       ( 
				"\r\n"                
                | '\n'
				| '\r'       
                )                       
        ;

protected  Space:
        ( ' ' | '\t' | '\014') 
        ;

Lcurly
		: '{'	{ openCount ++;}
		;
Rcurly
		: '}'   { openCount --;}
		;
