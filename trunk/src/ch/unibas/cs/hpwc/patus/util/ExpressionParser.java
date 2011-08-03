/**
 * 
 */
package ch.unibas.cs.hpwc.patus.util;

import java.io.ByteArrayInputStream;

import antlr.RecognitionException;
import antlr.TokenStreamException;
import cetus.base.grammars.NewCLexer;
import cetus.base.grammars.NewCParser;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.TranslationUnit;

/**
 * Utility class wrapping the cetus parser to parse a piece of
 * code from a {@link String}.
 * 
 * @author Matthias-M. Christen
 */
public class ExpressionParser
{
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	/**
	 * Parses the expression <code>strExpression</code> and returns the
	 * expression object representing the AST. If a parsing error occurs,
	 * a {@link RecognitionException} or a {@link TokenStreamException}
	 * is thrown.
	 * @param strExpression The expression to parse
	 * @return The AST corresponding to the expression <code>strExpression</code>
	 * @throws RecognitionException
	 * @throws TokenStreamException
	 */
	public static Expression parseExpression (String strExpression) throws RecognitionException, TokenStreamException
	{
		StringBuilder sb = new StringBuilder ("void fnx () { ");
		sb.append (strExpression);
		sb.append ("; }");
		
		TranslationUnit tu = parse (sb.toString ());
		for (DepthFirstIterator it = new DepthFirstIterator (tu); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof Expression)
				return (Expression) obj;
		}
		
		return null;
	}
	
	/**
	 * Parses the code supplied in the {@link String} <code>strCode</code>.
	 * If the parsing is successful, a {@link TranslationUnit} object will be
	 * returned that contains the AST of the parsed code.
	 * 
	 * @param strCode The code to parse
	 * @return A {@link TranslationUnit} object that contains the AST of the
	 * 	code that has been parsed
	 * @throws RecognitionException
	 * @throws TokenStreamException
	 */
	public static TranslationUnit parse (String strCode) throws RecognitionException, TokenStreamException
	{
		TranslationUnit tu = new TranslationUnit ("");

		NewCLexer lexer = new NewCLexer (new ByteArrayInputStream (strCode.getBytes ()));
		lexer.setTokenObjectClass ("cetus.base.grammars.CToken");
		lexer.initialize ();

		NewCParser parser = new NewCParser (lexer);
		parser.getPreprocessorInfoChannel (lexer.getPreprocessorInfoChannel ());
		parser.setLexer (lexer);
		parser.translationUnit (tu);

		return tu;
	}
}
