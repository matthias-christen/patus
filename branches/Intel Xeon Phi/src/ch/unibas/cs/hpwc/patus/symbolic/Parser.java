package ch.unibas.cs.hpwc.patus.symbolic;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cetus.hir.AccessExpression;
import cetus.hir.AccessOperator;
import cetus.hir.ArrayAccess;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FloatLiteral;
import cetus.hir.FunctionCall;
import cetus.hir.Identifier;
import cetus.hir.IDExpression;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Typecast;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;

import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;




public class Parser {
	public static final int _EOF = 0;
	public static final int _ident = 1;
	public static final int _integer = 2;
	public static final int _float = 3;
	public static final int maxT = 24;

	static final boolean T = true;
	static final boolean x = false;
	static final int minErrDist = 2;

	static final org.apache.log4j.Logger LOGGER = org.apache.log4j.Logger.getLogger (Parser.class);

	public Token t;    // last recognized token
	public Token la;   // lookahead token
	int errDist = minErrDist;
	
	public Scanner scanner;
	public Errors errors;

	private Expression m_expression;
	
	/**
	 * A map of identifier names to the identifiers in the original
	 * expression before evaluating/simplifying, if the orignal expression
	 * is available
	 */
	private Map<String, Identifier> m_mapIdentifiers;
	
	/**
	 * The number of Flops in the resulting expression
	 */
	private int m_nFlops;
	
	/**
	 * The type of the expression (simple expression, equation, inequality)
	 */
	private Symbolic.EExpressionType m_type;


	/**
	 * Sets the original expression.
	 */
	public void setOriginalExpression (Expression exprOrig)
	{
		if (exprOrig == null)
			return;

		if (m_mapIdentifiers == null)			
			m_mapIdentifiers = new HashMap<String, Identifier> ();
			
		for (DepthFirstIterator it = new DepthFirstIterator (exprOrig); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof StencilNode)
				m_mapIdentifiers.put (((StencilNode) obj).toExpandedString (), (Identifier) obj);
			else if (obj instanceof Identifier)
				m_mapIdentifiers.put (((Identifier) obj).getName (), (Identifier) obj);
		}
	}
	
	/**
	 * Returns a clone of identifier named <code>strIdentifierName</code> in the
	 * original expression, if the original expression and the identifier are available,
	 * or a new {@link NameID}.
	 */
	private IDExpression getIdentifier (String strIdentifierName)
	{
		if (m_mapIdentifiers != null)
		{
			Identifier id = m_mapIdentifiers.get (strIdentifierName);
			if (id != null)
				return id.clone ();
		}
		
		return new NameID (strIdentifierName);
	}

	/**
	 * Returns the expression that has been parsed.
	 */
	public Expression getExpression ()
	{
		return m_expression;
	}
	
	/**
	 * Returns the number of Flops in the expression
	 */
	public int getFlops ()
	{
		return m_nFlops;
	}
	
	/**
	 * Returns the type of the expression
	 */
	public Symbolic.EExpressionType getExpressionType ()
	{
		return m_type;
	}
	
	protected void assertParamCount (int nParamsCount, String strFunctionName, List<Expression> listArgs)
	{
		if (listArgs.size () != nParamsCount)
			errors.SemErr (la.line, la.col, StringUtil.concat (strFunctionName, " expects ", nParamsCount, " parameters, but ", listArgs.size (), " were provided."));
	}

	public Expression processFunctionCall (Expression id, List<Expression> listArgs)
	{
		if (id instanceof NameID)
		{
			String strFunctionName = ((NameID) id).getName ();

			// special function to be replaced with other operations
			if ("mod".equals (strFunctionName))
			{
				assertParamCount (2, strFunctionName, listArgs);
				return new BinaryExpression (listArgs.get (0), BinaryOperator.MODULUS, listArgs.get (1));
			}
			if ("floor".equals (strFunctionName))
			{
				assertParamCount (1, strFunctionName, listArgs);
				return new Typecast (CodeGeneratorUtil.specifiers (Globals.SPECIFIER_INDEX), listArgs.get (0));
			}

			// function names to replace			
			if ("ceiling".equals (strFunctionName))
				id = new NameID ("ceil");

			return new FunctionCall (id, listArgs);
		}

		return id;
	}


///////////////////////////////////////////////////////////////////////////
// Tokens



	public Parser(Scanner scanner) {
		this.scanner = scanner;
		errors = new Errors(scanner.buffer.getText ());
	}

	void SynErr (int n) {
		if (errDist >= minErrDist) errors.SynErr(la.line, la.col, n);
		errDist = 0;
	}

	public void SemErr (String msg) {
		if (errDist >= minErrDist) errors.SemErr(t.line, t.col, msg);
		errDist = 0;
	}
	
	void Get () {
		for (;;) {
			t = la;
			la = scanner.Scan();
			if (la.kind <= maxT) {
				++errDist;
				break;
			}

			la = t;
		}
	}
	
	void Expect (int n) {
		if (la.kind==n) Get(); else { SynErr(n); }
	}
	
	boolean StartOf (int s) {
		return set[s][la.kind];
	}
	
	void ExpectWeak (int n, int follow) {
		if (la.kind == n) Get();
		else {
			SynErr(n);
			while (!StartOf(follow)) Get();
		}
	}
	
	boolean WeakSeparator (int n, int syFol, int repFol) {
		int kind = la.kind;
		if (kind == n) { Get(); return true; }
		else if (StartOf(repFol)) return false;
		else {
			SynErr(n);
			while (!(set[syFol][kind] || set[repFol][kind] || set[0][kind])) {
				Get();
				kind = la.kind;
			}
			return StartOf(syFol);
		}
	}
	
	Expression  MaximaExpression() {
		Expression  expr;
		m_nFlops = 0; m_type = Symbolic.EExpressionType.EXPRESSION; 
		expr = ComparisonExpression();
		m_expression = expr; /* m_expression.setParens (false); */ 
		return expr;
	}

	Expression  ComparisonExpression() {
		Expression  expr;
		Expression expr0 = AdditiveExpression();
		expr = expr0; 
		if (StartOf(1)) {
			BinaryOperator op = null; 
			switch (la.kind) {
			case 4: {
				Get();
				op = BinaryOperator.COMPARE_EQ; m_type = Symbolic.EExpressionType.EQUATION; 
				break;
			}
			case 5: {
				Get();
				op = BinaryOperator.COMPARE_NE; m_type = Symbolic.EExpressionType.INEQUALITY; 
				break;
			}
			case 6: {
				Get();
				op = BinaryOperator.COMPARE_LE; m_type = Symbolic.EExpressionType.INEQUALITY; 
				break;
			}
			case 7: {
				Get();
				op = BinaryOperator.COMPARE_GE; m_type = Symbolic.EExpressionType.INEQUALITY; 
				break;
			}
			case 8: {
				Get();
				op = BinaryOperator.COMPARE_LT; m_type = Symbolic.EExpressionType.INEQUALITY; 
				break;
			}
			case 9: {
				Get();
				op = BinaryOperator.COMPARE_GT; m_type = Symbolic.EExpressionType.INEQUALITY; 
				break;
			}
			}
			Expression expr1 = AdditiveExpression();
			if (op != null) expr = new BinaryExpression (expr0, op, expr1); 
		}
		return expr;
	}

	Expression  AdditiveExpression() {
		Expression  expr;
		Expression expr0 = MultiplicativeExpression();
		expr = expr0; 
		while (la.kind == 10 || la.kind == 11) {
			BinaryOperator op = BinaryOperator.ADD; 
			if (la.kind == 10) {
				Get();
			} else {
				Get();
				op = BinaryOperator.SUBTRACT; 
			}
			m_nFlops++; 
			Expression expr1 = MultiplicativeExpression();
			expr = new BinaryExpression (expr.clone (), op, expr1);
		}
		return expr;
	}

	Expression  MultiplicativeExpression() {
		Expression  expr;
		Expression expr0 = UnarySignExpression();
		expr = expr0; 
		while (la.kind == 12 || la.kind == 13) {
			BinaryOperator op = BinaryOperator.MULTIPLY; 
			if (la.kind == 12) {
				Get();
			} else {
				Get();
				op = BinaryOperator.DIVIDE; 
			}
			m_nFlops++; 
			Expression expr1 = UnarySignExpression();
			expr = new BinaryExpression (expr.clone (), op, expr1); 
		}
		return expr;
	}

	Expression  UnarySignExpression() {
		Expression  expr;
		boolean bIsNegative = false; 
		if (la.kind == 10 || la.kind == 11) {
			if (la.kind == 10) {
				Get();
			} else {
				Get();
				bIsNegative = true; m_nFlops++; 
			}
		}
		Expression expr1 = ExponentExpression();
		if (!bIsNegative) expr = expr1; else { 
		if (expr1 instanceof FloatLiteral) expr = new FloatLiteral (-((FloatLiteral) expr1).getValue ()); 
		if (expr1 instanceof IntegerLiteral) expr = new IntegerLiteral (-((IntegerLiteral) expr1).getValue ()); 
		else expr = new UnaryExpression (UnaryOperator.MINUS, expr1); 
		} 
		return expr;
	}

	Expression  ExponentExpression() {
		Expression  expr;
		Expression expr0 = UnaryExpression();
		expr = expr0; 
		while (la.kind == 14) {
			Get();
			Expression expr1 = UnarySignExpression();
			expr = ExpressionUtil.createExponentExpression (expr.clone (), expr1, null); 
		}
		return expr;
	}

	Expression  UnaryExpression() {
		Expression  expr;
		expr = null; 
		if (la.kind == 3) {
			double fValue = FloatLiteral();
			expr = new FloatLiteral (fValue); 
		} else if (la.kind == 2) {
			int nValue = IntegerLiteral();
			expr = new IntegerLiteral (nValue); 
		} else if (la.kind == 15) {
			Expression exprBracketed = BracketedExpression();
			expr = exprBracketed; 
		} else if (la.kind == 17 || la.kind == 18 || la.kind == 19) {
			Expression exprCast = TypeCast();
			expr = exprCast; 
		} else if (la.kind == 1) {
			Expression exprValue = FunctionCallOrIdentifier();
			expr = exprValue; 
		} else SynErr(25);
		return expr;
	}

	double  FloatLiteral() {
		double  fValue;
		fValue = 0.0; 
		Expect(3);
		fValue = Double.parseDouble (t.val); 
		return fValue;
	}

	int  IntegerLiteral() {
		int  nValue;
		nValue = 0; 
		Expect(2);
		nValue = Integer.parseInt (t.val); 
		return nValue;
	}

	Expression  BracketedExpression() {
		Expression  expr;
		Expect(15);
		expr = ComparisonExpression();
		Expect(16);
		return expr;
	}

	Expression  TypeCast() {
		Expression  expr;
		List<Specifier> listSpecs = new ArrayList<Specifier> (1); 
		if (la.kind == 17) {
			Get();
			listSpecs.add (Globals.SPECIFIER_INDEX); 
		} else if (la.kind == 18) {
			Get();
			listSpecs.add (Specifier.FLOAT); 
		} else if (la.kind == 19) {
			Get();
			listSpecs.add (Specifier.DOUBLE); 
		} else SynErr(26);
		Expect(15);
		Expression exprValue = ComparisonExpression();
		Expect(16);
		expr = new Typecast (listSpecs, exprValue); 
		return expr;
	}

	Expression  FunctionCallOrIdentifier() {
		Expression  expr;
		expr = Identifier();
		boolean bIsFunctionCall = false; List<Expression> listArgs = null; 
		if (la.kind == 15 || la.kind == 22) {
			expr = ArraySubscripts(expr);
		}
		if (la.kind == 15) {
			Get();
			bIsFunctionCall = true; listArgs = new ArrayList<Expression> (); 
			if (StartOf(2)) {
				Expression exprArg = ComparisonExpression();
				listArgs.add (exprArg); 
				while (la.kind == 20) {
					Get();
					exprArg = ComparisonExpression();
					listArgs.add (exprArg); 
				}
			}
			Expect(16);
		}
		if (bIsFunctionCall) expr = processFunctionCall (expr, listArgs); 
		return expr;
	}

	Expression  Identifier() {
		Expression  expr;
		Expect(1);
		expr = t.val.equals ("%pi") ? new FloatLiteral (Math.PI) : getIdentifier (t.val); 
		if (la.kind == 21) {
			Get();
			Expression exprSubscript = Identifier();
			expr = new AccessExpression (expr.clone (), AccessOperator.MEMBER_ACCESS, exprSubscript); 
		}
		return expr;
	}

	Expression  ArraySubscripts(Expression exprArray) {
		Expression  exprAccess;
		List<Expression> listAccesses = new ArrayList<Expression> (); 
		while (la.kind == 22) {
			Get();
			Expression exprSubscript = AdditiveExpression();
			listAccesses.add (exprSubscript); 
			Expect(23);
		}
		exprAccess = listAccesses.size () > 0 ? new ArrayAccess (exprArray, listAccesses) : exprArray; 
		return exprAccess;
	}



	public void Parse() {
		la = new Token();
		la.val = "";		
		Get();
		MaximaExpression();
		Expect(0);

	}

	private static final boolean[][] set = {
		{T,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x},
		{x,x,x,x, T,T,T,T, T,T,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x},
		{x,T,T,T, x,x,x,x, x,x,T,T, x,x,x,T, x,T,T,T, x,x,x,x, x,x}

	};
} // end Parser


class Errors {
	public int count = 0;                                    // number of errors detected
	public String errMsgFormat = "Error in expression \"{3}\":  line {0} col {1}: {2}"; // 0=line, 1=column, 2=text, 3=expression
	
	private String m_strExpression;
	
	
	public Errors (String strExpression)
	{
		m_strExpression = strExpression;
	}
	
	protected void printMsg(int line, int column, String msg) {
		StringBuffer b = new StringBuffer(errMsgFormat);
		int pos = b.indexOf("{0}");
		if (pos >= 0) { b.delete(pos, pos+3); b.insert(pos, line); }
		pos = b.indexOf("{1}");
		if (pos >= 0) { b.delete(pos, pos+3); b.insert(pos, column); }
		pos = b.indexOf("{2}");
		if (pos >= 0) b.replace(pos, pos+3, msg);
		pos = b.indexOf("{3}");
		if (pos >= 0) b.replace(pos, pos+3, m_strExpression);
		Parser.LOGGER.error(b.toString());
	}
	
	public void SynErr (int line, int col, int n) {
		String s;
		switch (n) {
			case 0: s = "EOF expected"; break;
			case 1: s = "ident expected"; break;
			case 2: s = "integer expected"; break;
			case 3: s = "float expected"; break;
			case 4: s = "\"=\" expected"; break;
			case 5: s = "\"#\" expected"; break;
			case 6: s = "\"<=\" expected"; break;
			case 7: s = "\">=\" expected"; break;
			case 8: s = "\"<\" expected"; break;
			case 9: s = "\">\" expected"; break;
			case 10: s = "\"+\" expected"; break;
			case 11: s = "\"-\" expected"; break;
			case 12: s = "\"*\" expected"; break;
			case 13: s = "\"/\" expected"; break;
			case 14: s = "\"^\" expected"; break;
			case 15: s = "\"(\" expected"; break;
			case 16: s = "\")\" expected"; break;
			case 17: s = "\"int\" expected"; break;
			case 18: s = "\"float\" expected"; break;
			case 19: s = "\"double\" expected"; break;
			case 20: s = "\",\" expected"; break;
			case 21: s = "\".\" expected"; break;
			case 22: s = "\"[\" expected"; break;
			case 23: s = "\"]\" expected"; break;
			case 24: s = "??? expected"; break;
			case 25: s = "invalid UnaryExpression"; break;
			case 26: s = "invalid TypeCast"; break;
			default: s = "error " + n; break;
		}
		printMsg(line, col, s);
		count++;
	}

	public void SemErr (int line, int col, String s) {	
		printMsg(line, col, s);
		count++;
	}
	
	public void SemErr (String s) {
		Parser.LOGGER.error(s);
		count++;
	}
	
	public void Warning (int line, int col, String s) {	
		printMsg(line, col, s);
	}
	
	public void Warning (String s) {
		Parser.LOGGER.error(s);
	}
} // Errors


class FatalError extends RuntimeException {
	public static final long serialVersionUID = 1L;
	public FatalError(String s) { super(s); }
}
