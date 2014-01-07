/*******************************************************************************
 * Copyright (c) 2011 Matthias-M. Christen, University of Basel, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Matthias-M. Christen, University of Basel, Switzerland - initial API and implementation
 ******************************************************************************/
package ch.unibas.cs.hpwc.patus.symbolic;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class ExpressionParser
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Pattern PATTERN_MAXIMA_WARNING = Pattern.compile ("\\w+:.*");

	private static Matcher m_matcher = null;

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private static Map<String, ExpressionData> m_mapExpressionsCache = new HashMap<>();
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Parses the expression string <code>strExpression</code>.
	 * @param strExpression The expression to parse
	 * @return The internal representation of <code>strExpression</code>
	 */
	public static Expression parse (String strExpression)
	{
		return ExpressionParser.parseEx (strExpression).getExpression ();
	}

	/**
	 * Parses the expression string <code>strExpression</code> and returns a (Cetus) internal representation.
	 * @param strExpression
	 * @param rgExprOrigs Original expressions which contain identifiers occurring in <code>strExpression</code>.
	 * 	The identifiers found in <code>rgExprOrigs</code> will be used when parsing the result expression
	 * @return
	 */
	public static Expression parse (String strExpression, Expression... rgExprOrigs)
	{
		return ExpressionParser.parseEx (strExpression, rgExprOrigs).getExpression ();
	}

	/**
	 * Parses the expression <code>strExpression</code> and returns an internal (Cetus) representation along with
	 * some expression metrics.
	 * @param strExpression The expression to parse
	 * @param rgExprOrigs Original expressions which contain identifiers occurring in <code>strExpression</code>.
	 * 	The identifiers found in <code>rgExprOrigs</code> will be used when parsing the result expression
	 * @return
	 */
	public static ExpressionData parseEx (String strExpression, Expression... rgExprOrigs)
	{
		ExpressionData edResult = m_mapExpressionsCache.get(strExpression);
		if (edResult != null)
			return new ExpressionData (edResult);
		
		if (strExpression.startsWith ("Maxima encountered a Lisp error"))
			return null;

		// convert multi-line Maxima outputs to single lines
		String[] rgLines = strExpression.split ("\\\\\\n|\n");

		// find warnings ("<some string>:...") and discard them
		for (int i = 0; i < rgLines.length; i++)
		{
			if (ExpressionParser.m_matcher == null)
				ExpressionParser.m_matcher = ExpressionParser.PATTERN_MAXIMA_WARNING.matcher (rgLines[i]);
			else
				ExpressionParser.m_matcher.reset (rgLines[i]);

			if (ExpressionParser.m_matcher.matches ())
				rgLines[i] = "";
		}

		// convert to a single line
		String strExpressionToParse = StringUtil.concat ((Object[]) rgLines);

		Parser parser = new Parser (new Scanner (strExpressionToParse));
		if (rgExprOrigs != null)
			for (Expression exprOrig : rgExprOrigs)
				parser.setOriginalExpression (exprOrig);
		parser.Parse ();

		m_mapExpressionsCache.put (strExpression, edResult = new ExpressionData (parser.getExpression (), parser.getFlops (), parser.getExpressionType ()));
		return edResult;
	}
}
