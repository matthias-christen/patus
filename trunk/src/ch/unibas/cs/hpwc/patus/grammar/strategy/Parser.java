package ch.unibas.cs.hpwc.patus.grammar.strategy;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Stack;

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FloatLiteral;
import cetus.hir.FunctionCall;
import cetus.hir.Identifier;
import cetus.hir.IDExpression;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.StringLiteral;
import cetus.hir.Symbol;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;

import ch.unibas.cs.hpwc.patus.ast.Loop;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StencilProperty;
import ch.unibas.cs.hpwc.patus.ast.StencilSpecifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;

import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.Strategy;

import ch.unibas.cs.hpwc.patus.geometry.Border;
import ch.unibas.cs.hpwc.patus.geometry.Box;
import ch.unibas.cs.hpwc.patus.geometry.Point;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.geometry.Subdomain;
import ch.unibas.cs.hpwc.patus.geometry.Vector;

import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;

import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;




public class Parser {
	public static final int _EOF = 0;
	public static final int _ident = 1;
	public static final int _integer = 2;
	public static final int _float = 3;
	public static final int maxT = 55;

	static final boolean T = true;
	static final boolean x = false;
	static final int minErrDist = 2;
	
	static final org.apache.log4j.Logger LOGGER = org.apache.log4j.Logger.getLogger (Parser.class);
	

	public Token t;    // last recognized token
	public Token la;   // lookahead token
	int errDist = minErrDist;
	
	public Scanner scanner;
	public Errors errors;

	private enum EHandSide
	{
		LEFT,
		RIGHT
	}
	
	private class SubdomainDecl
	{
        private String m_strOrigIdentifier;
		private Subdomain m_subdomain;
		private Symbol m_symbol;
		private int m_nContextID;
		
		public SubdomainDecl (String strIdentifier, Subdomain subdomain, int nContextID)
		{
            m_strOrigIdentifier = strIdentifier;
			m_subdomain = subdomain;
			m_nContextID = nContextID;

			VariableDeclarator decl = new VariableDeclarator (new NameID (StringUtil.concat (strIdentifier, m_nContextID)));
			m_cmpstmtStrategyBody.addDeclaration (new VariableDeclaration (StencilSpecifier.STENCIL_GRID, decl));
			m_symbol = decl;
		}		
		
		public String getOriginalIdentifier ()
		{
            return m_strOrigIdentifier;
		}
		
		public Subdomain getSubdomain ()
		{
			return m_subdomain;
		}
		
		public Symbol getSymbol ()
		{
			return m_symbol;
		}
		
		public int getContextID ()
		{
            return m_nContextID;
        }
	}


	///////////////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The result of the parser
	 */
	private Strategy m_strategy;
	
	/**
	 * The body of the strategy procedure.
	 * Used to declare variables.
	 */
	private CompoundStatement m_cmpstmtStrategyBody;
	
	/**
	 * Map of subdomains used in the strategy
	 */
	private Map<String, List<SubdomainDecl>> m_mapSubdomains = new HashMap<> ();
	
	/**
	 * Parallelism levels of loops
	 */
	private Stack<Integer> m_stackParallelismLevels = new Stack<> ();
	
	/**
	 * Dimension identifiers. The value in the map are the codimensions of the corresponding
	 * dimension identifiers.
	 */
	private Map<String, Integer> m_mapDimensionIdentifiers = new HashMap<> ();
	
	private Map<String, Integer> m_mapConstants = new HashMap<> ();
	
	/**
	 * The stencil calculation
	 */
	private StencilCalculation m_stencilCalculation = null;
	
	
	/**
	 * The current context identifier. Context identifiers indicate 'where' in the strategy
	 * the current token is read to provide information about/simulate variable scopes.
	 */
	private int m_nCurrentContextID;
	private int m_nLastContextID;
	private Stack<Integer> m_stackContextIDs;
		
	
	///////////////////////////////////////////////////////////////////////////
	// Custom Implementation
	
	public boolean hasErrors ()
	{
		return errors.count > 0;
	}
	
	/**
	 * Sets the stencil calculation object. This must be set before starting parsing.
	 */
	public void setStencilCalculation (StencilCalculation stencilCalculation)
	{
		m_stencilCalculation = stencilCalculation;
	}
	
	/**
	 * Returns the result after parsing.
	 */
	public Strategy getStrategy ()
	{
		return m_strategy;
	}
	
	private void initialize ()
	{
        if (m_stencilCalculation == null)
            throw new RuntimeException ("The stencil calculation object must be set before starting to parse the strategy.");

        m_strategy = new Strategy ();
        m_stackParallelismLevels.push (0);

        // initialize the context
        m_nCurrentContextID = 0;
        m_nLastContextID = 0;
        m_stackContextIDs = new Stack<> ();
        m_stackContextIDs.push (m_nCurrentContextID);
    }
    
    private void pushContext ()
    {
        m_stackContextIDs.push (m_nCurrentContextID);
        m_nLastContextID++;
        m_nCurrentContextID = m_nLastContextID;
    }
    
    private void popContext ()
    {
        m_nCurrentContextID = m_stackContextIDs.pop ();
    }
	
	/**
	 * Extracts the first identifier from the expression <code>expr</code>.
	 * @param expr The expression from which to extract the identifier
	 * @return The first identifier in <code>expr</code>
	 */
	private Expression findIdentifier (Expression expr)
	{
		if (expr instanceof IDExpression)
			return expr;
			
		for (Object o : expr.getChildren ())
		{
			if (o instanceof ArrayAccess)
				continue;
			if (o instanceof Expression)
				findIdentifier ((Expression) o);
		}
		
		return null;
	}
	
	/**
	 * Determines whether the identifier has been declared.
	 */
	private boolean isDeclared (IDExpression identifier)
	{
		// check whether the variable has already been declared within the procedure body
		if (identifier instanceof SubdomainIdentifier)
			return m_cmpstmtStrategyBody.findSymbol (new NameID (((SubdomainIdentifier) identifier).getName ())) != null;
		return m_cmpstmtStrategyBody.findSymbol (identifier) != null;
	}
	
	/**
	 * Adds a variable declaration for the identifier in <code>exprIdentifier</code> to the
	 * symbol table of the stategy.
	 * @param specifier
	 * @param exprIdentifier The identifier
	 */
	private Identifier addDeclaration (Specifier specifier, Expression exprIdentifier)
	{
		IDExpression id = (IDExpression) findIdentifier (exprIdentifier);
		if (id != null && !isDeclared (id))
		{
			VariableDeclarator decl = new VariableDeclarator (id);
			m_cmpstmtStrategyBody.addDeclaration (new VariableDeclaration (specifier, decl));
			return new Identifier (decl);
		}
		
		return null;
	}

	/**
	 * Checks whether the identifier in <code>exprIdentifier</code> has been declared.
	 * If not, a semantic error is thrown.
	 * @param exprIdentifier The identifier to check
	 */	
	private void checkDeclared (Expression exprIdentifier)
	{
		IDExpression id = (IDExpression) findIdentifier (exprIdentifier);
		if (id != null && !isDeclared (id))
			errors.SemErr (la.line, la.col, id.toString () + " has not been initialized before use");
	}
	
	/**
	 * Registers a subdomain and binds it to the identifier <code>strIdentifier</code>.
	 * @param strIdentifier The identifier to which the domain will be tied
	 * @param subdomain The subdomain that is registered
	 */
	private void registerSubdomain (String strIdentifier, Subdomain subdomain, List<Expression> listArraySizes)
	{
        List<SubdomainDecl> list = m_mapSubdomains.get (strIdentifier);
        if (list == null)
            m_mapSubdomains.put (strIdentifier, list = new LinkedList<> ());
		list.add (new SubdomainDecl (strIdentifier, subdomain, m_nCurrentContextID));
	}
	
	private SubdomainDecl getSubdomainDecl (String strIdentifier)
	{
        List<SubdomainDecl> listDecls = m_mapSubdomains.get (strIdentifier);
        if (listDecls == null || listDecls.size () == 0)
        {
            errors.SemErr (la.line, la.col, StringUtil.concat (strIdentifier, " has not been declared"));
            return null;
        }
            
        // find the decl matching the current context
        for (SubdomainDecl decl : listDecls)
        {
            // check whether the decl's context is the current one or belongs to a parent context
            if (decl.getContextID () == m_nCurrentContextID || m_stackContextIDs.contains (decl.getContextID ()))
                return decl;
        }
        
        // no decl found
        errors.SemErr (la.line, la.col, StringUtil.concat (strIdentifier, " has not been declared"));
        return null;
	}
	
	private SubdomainIdentifier getSubdomainIdentifier (String strIdentifier)
	{
        SubdomainDecl decl = getSubdomainDecl (strIdentifier);
        if (decl != null)
            return new SubdomainIdentifier (decl.getSymbol (), decl.getSubdomain ());
        return null;
	}
	
	private Subdomain getSubdomain (String strIdentifier)
	{
        SubdomainDecl decl = getSubdomainDecl (strIdentifier);
        if (decl != null)
            return decl.getSubdomain ();
        return null;
	}
	
	private void setStrategySubdomainMap ()
	{
		Map<String, Subdomain> map = new HashMap<> ();
		for (String s : m_mapSubdomains.keySet ())
			map.put (s, m_mapSubdomains.get (s).iterator ().next ().getSubdomain ());    // just get the first list entry
		m_strategy.setSubdomains (map); 
	}
	
	private boolean isSubdomainIdentifier ()
	{
		return m_mapSubdomains == null ? false : m_mapSubdomains.containsKey (la.val);
	}
	
	private boolean isDimensionParameter ()
	{
		return m_mapDimensionIdentifiers == null ? false : m_mapDimensionIdentifiers.containsKey (la.val);
	}
	
	private Expression getDimIdentifier (Expression expr, int nDim)
	{
		if (expr instanceof NameID)
			return new NameID (StringUtil.concat (((NameID) expr).getName (), "_", CodeGeneratorUtil.getDimensionName (nDim)));

		Expression exprNew = expr.clone ();
		for (DepthFirstIterator it = new DepthFirstIterator (exprNew); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof NameID)
			{
				NameID nid = (NameID) obj;
				if (m_mapDimensionIdentifiers.containsKey (nid.getName ()))
					nid.swapWith (new NameID (StringUtil.concat (nid.getName (), "_", CodeGeneratorUtil.getDimensionName (nDim))));
			}
		}
		return exprNew;
	}
	
	/**
	 * Creates an expression array of the dimension as the stencil computations to be used as
	 * spatial index from indices in <code>listHead</code> and <code>listTail</code>.
	 * <code>listHead</code> has priority over <code>listTail</code> if there isn't a
	 * <code>null</code> entry in the list, i.e., if there are more
	 * entries in total in the lists, the array will be filled up with the entries of
	 * <code>listHead</code> first before the entries in <code>listTail</code> will be used.
	 * A <code>null</code> entry in <code>listHead</code> or <code>listTail</code> means that
	 * the corresponding coordinates will be replaced by the domain's size. 
	 */
	private List<Expression> createVector (List<Expression> listHead, List<Expression> listTail, Vector vecDefault)
	{
		int nTailSize = listTail == null ? 0 : listTail.size ();
		byte nDimensionality = (nTailSize == 0 && listHead != null) ? (byte) listHead.size () : m_stencilCalculation.getDimensionality ();
		int j = 0;
		Expression[] rgIdx = new Expression[nDimensionality];
		
		// set head elements
		if (listHead != null)
		{
			for (Expression expr : listHead)
			{
	            if (expr == null)
	                break;
				if (j >= nDimensionality)
					break;
				rgIdx[j++] = expr;
			}
		}
		
		// set tail elements
		for ( ; j < nDimensionality - nTailSize; j++)
			rgIdx[j] = null;

		for (int k = nTailSize - 1; k >= 0; k--)
		{
			if (j >= nDimensionality)
				break;
			j++;
			rgIdx[nDimensionality - nTailSize + k] = listTail.get (k);
		}
		
		// fill result list
		List<Expression> listResult = new ArrayList<> (nDimensionality);
		j = 0;
		for (Expression expr : rgIdx)
		{
            listResult.add (expr == null && vecDefault != null ? vecDefault.getCoord (j) : expr);
            j++;
        }
		return listResult;
	}

	/**
	 * Creates a unary expression.
	 * @param bIsNegative Flag indicating whether to return the negative of <code>expr</code>
	 * @param expr The expression
	 */	
	private Expression createUnaryExpression (boolean bIsNegative, Expression expr)
	{
		if (expr instanceof IntegerLiteral)
			return bIsNegative ? new IntegerLiteral (-((IntegerLiteral) expr).getValue ()) : expr;
		if (expr instanceof FloatLiteral)
			return bIsNegative ? new FloatLiteral (-((FloatLiteral) expr).getValue ()) : expr;
		return bIsNegative ? new UnaryExpression (UnaryOperator.MINUS, expr) : expr;
	}
	
	private FloatLiteral floatOp (double fVal1, BinaryOperator op, double fVal2)
	{
		if (op == BinaryOperator.ADD)
			return new FloatLiteral (fVal1 + fVal2);
		if (op == BinaryOperator.SUBTRACT)
			return new FloatLiteral (fVal1 - fVal2);
		if (op == BinaryOperator.MULTIPLY)
			return new FloatLiteral (fVal1 * fVal2);
		if (op == BinaryOperator.DIVIDE)
			return new FloatLiteral (fVal1 / fVal2);
			
		errors.SemErr (la.line, la.col, "Unsupported binary integer operation.");
		return null;
	}
	
	private IntegerLiteral intOp (long nVal1, BinaryOperator op, long nVal2)
	{
		if (op == BinaryOperator.ADD)
			return new IntegerLiteral (nVal1 + nVal2);
		if (op == BinaryOperator.SUBTRACT)
			return new IntegerLiteral (nVal1 - nVal2);
		if (op == BinaryOperator.MULTIPLY)
			return new IntegerLiteral (nVal1 * nVal2);
		if (op == BinaryOperator.DIVIDE)
			return new IntegerLiteral (nVal1 / nVal2);
		if (op == BinaryOperator.MODULUS)
			return new IntegerLiteral (nVal1 % nVal2);

		errors.SemErr (la.line, la.col, "Unsupported binary integer operation.");
		return null;
	}

	private Expression createBinaryExpression (Expression expr1, BinaryOperator op, Expression expr2)
	{
		if (expr1 instanceof IntegerLiteral)
		{
			if (expr2 instanceof IntegerLiteral)
				return intOp (((IntegerLiteral) expr1).getValue (), op, ((IntegerLiteral) expr2).getValue ());
			else if (expr2 instanceof FloatLiteral)
				return floatOp (((IntegerLiteral) expr1).getValue (), op, ((FloatLiteral) expr2).getValue ());
		}
		else if (expr1 instanceof FloatLiteral)
		{
			if (expr2 instanceof IntegerLiteral)
				return floatOp (((FloatLiteral) expr1).getValue (), op, ((IntegerLiteral) expr2).getValue ());
			else if (expr2 instanceof FloatLiteral)
				return floatOp (((FloatLiteral) expr1).getValue (), op, ((FloatLiteral) expr2).getValue ());
		}

		return new BinaryExpression (expr1, op, expr2);
	}
	
	/**
	 * Returns the <code>nDimension</code>-th coordinate of the subdomain <code>sgid</code>.
	 */
	private Expression createCoordinateExpression (Expression exprSgid, int nDimension)
	{
		if (!(exprSgid instanceof SubdomainIdentifier))
		{
			errors.SemErr (la.line, la.col, "Coordinate subscripts can only be applied to subdomain identifiers.");
			return null;
		}
		
		return new NameID (StringUtil.concat (((SubdomainIdentifier) exprSgid).getName (), "_idx_", CodeGeneratorUtil.getDimensionName (nDimension + 1)));
	}
	
	/**
	 * Creates new vector by selecting the coordinates defined in <code>listCoords</code> from the
	 * vector <code>listVector</code>.
	 */
	private List<Expression> createSubscriptedVector (List<Expression> listVector, List<Expression> listCoords)
	{
        List<Expression> listResult = new ArrayList<> (listCoords.size ());
        for (Expression exprCoord : listCoords)
        {
            int nCoord = getIntValue (exprCoord);
            if (0 < nCoord && nCoord <= listVector.size ())
                listResult.add (listVector.get (nCoord - 1));
            else
            	listResult.add (null);
        }
        
        return listResult;
	}
	
	private Size createSize (List<Expression> listSize)
	{
	   Expression[] rgSize = new Expression[listSize.size ()];
	   listSize.toArray (rgSize);
	   return new Size (rgSize);
	}
	
	private Integer getIntegerValue (Expression expr)
	{
        try
        {
            return ExpressionUtil.getIntegerValue (expr);
        }
        catch (RuntimeException e)
        {
        	return null;
        }
	}

	private int getIntValue (Expression expr)
	{
		Integer nVal = getIntegerValue (expr);
		if (nVal == null)
		{
            errors.SemErr (la.line, la.col, "Compile time constant expected.");
            return 0;
		}

		return nVal;
	}
	
	private Integer getConstantValue (String strIdentifier)
	{
		return m_mapConstants.get (strIdentifier);
	}
	
//	/**
//	 * LL1 conflict resolver for assignments.
//	 */
//	private boolean isAssignment ()
//	{
//		Token t = scanner.Peek ();
//		scanner.ResetPeek ();
//		return t.val.equals ("=");
//	}
	
	/**
	 * LL1 conflict resolver for function calls.
	 */
	private boolean isSubCall ()
	{
		if (isSubdomainIdentifier () || isDimensionParameter ())
			return false;
		Token t = scanner.Peek ();
		scanner.ResetPeek ();
		return t.val.equals ("(");
	}
	
	private boolean isStencilCall ()
	{
		Token t = scanner.Peek ();
		scanner.ResetPeek ();
		return la.val.equals ("stencil") && t.val.equals ("(");		
	}
	
	/**
	 * LL1 conflict resolver for array accesses.
	 */
	private boolean isGridAccess ()
	{
		Token t = scanner.Peek ();
		scanner.ResetPeek ();
		return t.val.equals ("[");
	}
	
	/**
	 * LL1 conflict resolver checking whether there is only one identifier
	 * as spatial index.
	 */
	private boolean isSingleSpatialIndex ()
	{
		Token t = scanner.Peek ();
		scanner.ResetPeek ();
		return t.val.equals (";");
	}
	
	private boolean isCoordinateOrVector ()
	{
		Token t = scanner.Peek ();
		scanner.ResetPeek ();
		return isSubdomainIdentifier () && t.val.equals ("(");
	}
	
	private boolean checkVector (boolean bHasOpenBracket, boolean bIsBracketedVector)
	{
		int nOpenBrackets = bHasOpenBracket ? 1 : 0;
		boolean bIsVector = false;
		while (nOpenBrackets > 0 || !bIsVector)
		{
			Token t = scanner.Peek ();
			if (t.val.equals (":") || t.val.equals ("..") || t.val.equals ("...") || t.val.equals (","))
			{
				if (nOpenBrackets == 1)
				{
					bIsVector = true;
					break;
				}
			}
			else if (t.val.equals ("("))
				nOpenBrackets++;
			else if (t.val.equals (")"))
				nOpenBrackets--;
				
			if (bIsBracketedVector && nOpenBrackets == 0)
				break;
		}
		scanner.ResetPeek ();
		
		return bIsVector;
	}
	
	/**
	 * LL1 conflict resolver for vector types.
	 */
	private boolean isVector ()
	{
		if (!isSubdomainIdentifier () && !isDimensionParameter ())
			return false;
		return checkVector (false, false);
	}

	private boolean isBracketedVector ()
	{
		if (!la.val.equals ("("))
			return false;
			
		return checkVector (true, true);
	}

    /**
     * LL1 conflict resolver detecting vector-valued properties (size, min, and max).
     */
    private boolean isVectorProperty ()
    {
        Token t = scanner.Peek ();
        if (!t.val.equals ("."))
        {
            scanner.ResetPeek ();
            return false;
        }
        
        t = scanner.Peek ();
        if (!t.val.equals ("size") && !t.val.equals ("min") && !t.val.equals ("max"))
        {
            scanner.ResetPeek ();
            return false;
        }
        
        t = scanner.Peek ();
        if (t.val.equals ("("))
        {
        	// subscripted property; check whether the subscript is a vector
        	return checkVector (true, true);
        }
        
        scanner.ResetPeek ();
        return true;
    }
    
    private boolean isNoEllipsis ()
    {
    	return !la.val.equals ("...");
    }

    /**
     * LL1 conflict resolver determining whether a border specifiaction is encountered.
     * Borders either start with "stencil.box" or are a literal border specification.
     */
    private boolean isBorder ()
    {
        if (isStencilBox ())
            return true;
        if (isLiteralBorder ())
            return true;
        return false;
    }
    
	/**
	 * LL1 conflict resolver determining whether a "stencil.box" is encountered.
	 */	
	private boolean isStencilBox ()
	{
		Token t = la;
		if (!t.val.equals ("stencil"))
			return false;
			
		t = scanner.Peek ();
		if (!t.val.equals ("."))
		{
			scanner.ResetPeek ();
			return false;
		}
		
		t = scanner.Peek ();
		if (!t.val.equals ("box"))
		{
			scanner.ResetPeek ();
			return false;
		}
		
		return true;
	}
	
	/**
	 * LL1 conflict resolver for literal border expressions.
	 */
	private boolean isLiteralBorder ()
	{
		return la.val.equals ("<");
		
		/*	
		if (!la.val.equals ("("))
			return false;

		// the number of ";"-separated components; must have exactly 2			 
		int nComponents = 1;

		int nOpenBrackets = 1;
		while (nOpenBrackets > 0)
		{
			t = scanner.Peek ();
			if (t.val.equals ("("))
				nOpenBrackets++;
			else if (t.val.equals (")"))
				nOpenBrackets--;
			else if (nOpenBrackets == 1 && t.val.equals (";"))
				nComponents++;
		}
		scanner.ResetPeek ();
		
		return nComponents == 2;
		*/
	}
	
	/**
	 * LL1 conflict resolver to decide whether the next entity is a property (of a subdomain or the stencil).
	 */
	private boolean isProperty ()
	{
		Token t = scanner.Peek ();
		scanner.ResetPeek ();
		return (isSubdomainIdentifier () || la.val.equals ("stencil")) && t.val.equals (".");
	}
	
	private Expression getAutotuneListItem (List<Expression> l, int nIdx)
	{
		if (l.size () == 1)
			return l.get (0);
		return l.get (nIdx);
	}
	

///////////////////////////////////////////////////////////////////////////
// Tokens



	public Parser(Scanner scanner) {
		this.scanner = scanner;
		errors = new Errors();
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
	
	void StrategySpecification() {
		initialize (); 
		Expect(4);
		m_cmpstmtStrategyBody = new CompoundStatement (); 
		Expect(1);
		Expect(5);
		List<Declaration> listParameters  = StrategyParamList();
		m_strategy.setParameters (listParameters); for (Declaration decl : listParameters) m_cmpstmtStrategyBody.addDeclaration (decl.clone ()); 
		Expect(6);
		Statement stmtBody = StrategyCompoundStatement();
		m_cmpstmtStrategyBody.addStatement (stmtBody); m_strategy.setBody (m_cmpstmtStrategyBody); setStrategySubdomainMap (); 
	}

	List<Declaration>  StrategyParamList() {
		List<Declaration>  listParams;
		listParams = new ArrayList<> (); 
		Expect(7);
		Expect(1);
		VariableDeclarator decl = new VariableDeclarator (new NameID (t.val)); listParams.add (new VariableDeclaration (StencilSpecifier.STENCIL_GRID, decl)); 
		Point ptMin = m_stencilCalculation.getDomainSize ().getMin (); Point ptMax = m_stencilCalculation.getDomainSize ().getMax (); 
		Subdomain sg = new Subdomain (null, Subdomain.ESubdomainType.SUBDOMAIN, ptMin, new Size (ptMin, ptMax), true); 
		registerSubdomain (decl.getSymbolName (), sg, null); m_strategy.setBaseDomain (getSubdomainIdentifier (decl.getSymbolName ())); 
		while (la.kind == 8) {
			Get();
			Expect(9);
			boolean bIsDimParam = false; int nStartDimension = 0; 
			if (la.kind == 10) {
				Get();
			} else if (la.kind == 11) {
				Get();
				bIsDimParam = true; nStartDimension = 0; 
			} else if (la.kind == 12) {
				Get();
				Expect(5);
				Expect(2);
				Expect(6);
				bIsDimParam = true; nStartDimension = Math.max (0, m_stencilCalculation.getDimensionality () - Integer.parseInt (t.val)); 
			} else SynErr(56);
			Expect(1);
			String strParamName = t.val; String[] rgParamNames = null; decl = new VariableDeclarator (new NameID (strParamName)); 
			if (!bIsDimParam) listParams.add (new VariableDeclaration (StencilSpecifier.STRATEGY_AUTO, decl)); 
			else { 
			rgParamNames = new String[m_stencilCalculation.getDimensionality ()]; 
			m_mapDimensionIdentifiers.put (decl.getSymbolName (), m_stencilCalculation.getDimensionality () - nStartDimension); 
			for (int i = nStartDimension; i < m_stencilCalculation.getDimensionality (); i++) { 
			rgParamNames[i] = StringUtil.concat (decl.getSymbolName (), "_", CodeGeneratorUtil.getDimensionName (i)); 
			VariableDeclarator declNew = decl.clone (); declNew.setName (rgParamNames[i]); 
			listParams.add (new VariableDeclaration (StencilSpecifier.STRATEGY_AUTO, declNew)); 
			} 
			} 
			if (la.kind == 13) {
				Get();
				List<IAutotunerParam> listAutotuneParams = AutoTuneValues();
				if (!bIsDimParam) m_strategy.setAutotuneSpecification (strParamName, listAutotuneParams.get (0)); 
				else for (int i = nStartDimension; i < m_stencilCalculation.getDimensionality (); i++) { 
				m_strategy.setAutotuneSpecification (rgParamNames[i], listAutotuneParams.get (i - nStartDimension)); 
				} 
			}
		}
		return listParams;
	}

	Statement  StrategyCompoundStatement() {
		Statement  stmt;
		Expect(15);
		CompoundStatement cmpstmt = new CompoundStatement (); pushContext (); 
		while (StartOf(1)) {
			Statement stmt1 = StrategyStatement();
			if (stmt1 != null) { if (stmt1 instanceof DeclarationStatement) { Declaration decl = ((DeclarationStatement) stmt1).getDeclaration (); decl.setParent (null); cmpstmt.addDeclaration (decl); } else cmpstmt.addStatement (stmt1); } 
		}
		Expect(16);
		if (cmpstmt.getChildren ().size () == 1) { stmt = (Statement) cmpstmt.getChildren ().get (0); stmt.setParent (null); } else stmt = cmpstmt; popContext (); 
		return stmt;
	}

	List<IAutotunerParam>  AutoTuneValues() {
		List<IAutotunerParam>  listParams;
		listParams = null; 
		if (la.kind == 5) {
			listParams = AutoTuneVector();
		} else if (StartOf(2)) {
			listParams = AutoTuneItem();
		} else SynErr(57);
		return listParams;
	}

	Statement  StrategyStatement() {
		Statement  stmt;
		stmt = null; 
		if (la.kind == 7 || la.kind == 10) {
			Statement stmtDeclaration = StrategyDeclaration();
			while (!(la.kind == 0 || la.kind == 14)) {SynErr(58); Get();}
			Expect(14);
			stmt = stmtDeclaration; 
		} else if (la.kind == 19) {
			Statement stmtLoop = StrategyLoop();
			stmt = stmtLoop; 
		} else if (la.kind == 43) {
			Statement stmtConditional = StrategyIfStatement();
			stmt = stmtConditional; 
		} else if (la.kind == 15) {
			Statement stmtCompound = StrategyCompoundStatement();
			stmt = stmtCompound; 
		} else if (isSubCall ()) {
			Statement stmtSubCall = SubCall();
			while (!(la.kind == 0 || la.kind == 14)) {SynErr(59); Get();}
			Expect(14);
			stmt = stmtSubCall; 
		} else if (la.kind == 1) {
			Expression exprAssign = StrategyAssignment();
			while (!(la.kind == 0 || la.kind == 14)) {SynErr(60); Get();}
			Expect(14);
			stmt = new ExpressionStatement (exprAssign); 
		} else SynErr(61);
		return stmt;
	}

	Statement  StrategyDeclaration() {
		Statement  stmt;
		stmt = null; 
		if (la.kind == 7) {
			stmt = StrategyDomainDeclaration();
		} else if (la.kind == 10) {
			stmt = StrategyIntegerDeclaration();
		} else SynErr(62);
		return stmt;
	}

	Loop  StrategyLoop() {
		Loop  loop;
		loop = null; boolean bNewParallelismLevel = false; 
		while (!(la.kind == 0 || la.kind == 19)) {SynErr(63); Get();}
		Expect(19);
		if (la.kind == 24 || la.kind == 25 || la.kind == 26) {
			SubdomainIterator loopSubdomain = new SubdomainIterator (); 
			if (la.kind == 24) {
				SubdomainLoop(loopSubdomain);
			} else if (la.kind == 25) {
				PlaneLoop(loopSubdomain);
			} else {
				PointLoop(loopSubdomain);
			}
			if (la.kind == 20) {
				while (!(la.kind == 0 || la.kind == 20)) {SynErr(64); Get();}
				Get();
				loopSubdomain.setNumberOfThreads (Loop.MAX_THREADS); bNewParallelismLevel = true; 
				if (la.kind == 21) {
					while (!(la.kind == 0 || la.kind == 21)) {SynErr(65); Get();}
					Get();
					Expression exprChunkSizeTmp = StrategyExpression();
					int nDim = loopSubdomain.getDomainIdentifier ().getDimensionality (); Expression[] rgChunkSize = new Expression[nDim]; rgChunkSize[0] = exprChunkSizeTmp; for (int i = 1; i < nDim; i++) rgChunkSize[i] = Globals.ONE.clone (); loopSubdomain.setChunkSize (rgChunkSize); 
				}
			}
			loop = loopSubdomain; 
		} else if (la.kind == 1) {
			loop = RangeLoop();
			if (la.kind == 20) {
				while (!(la.kind == 0 || la.kind == 20)) {SynErr(66); Get();}
				Get();
				loop.setNumberOfThreads (Loop.MAX_THREADS); bNewParallelismLevel = true; 
				if (StartOf(3)) {
					Expression exprNumThreads = StrategyExpression();
					loop.setNumberOfThreads (exprNumThreads); bNewParallelismLevel = false; 
				}
				if (bNewParallelismLevel) m_stackParallelismLevels.push (m_stackParallelismLevels.peek () + 1); 
				if (la.kind == 21) {
					while (!(la.kind == 0 || la.kind == 21)) {SynErr(67); Get();}
					Get();
					Expression exprChunkSizeTmp = StrategyExpression();
					loop.setChunkSize (new Expression[] { exprChunkSizeTmp }); 
				}
			}
		} else SynErr(68);
		if (loop == null) loop = new RangeIterator (); 
		int nParLevel = m_stackParallelismLevels.peek (); if (bNewParallelismLevel) nParLevel++; m_stackParallelismLevels.push (nParLevel); loop.setParallelismLevel (nParLevel); 
		Statement stmtBody = StrategyStatement();
		loop.setLoopBody (stmtBody); m_stackParallelismLevels.pop (); 
		return loop;
	}

	Statement  StrategyIfStatement() {
		Statement  stmt;
		while (!(la.kind == 0 || la.kind == 43)) {SynErr(69); Get();}
		Expect(43);
		Expect(5);
		Expression exprCondition = ConditionalExpression();
		Expect(6);
		Statement stmtIf = StrategyStatement();
		stmt = new IfStatement (exprCondition, stmtIf); 
		if (la.kind == 44) {
			while (!(la.kind == 0 || la.kind == 44)) {SynErr(70); Get();}
			Get();
			Statement stmtElse = StrategyStatement();
			((IfStatement) stmt).setElseStatement (stmtElse); 
		}
		return stmt;
	}

	Statement  SubCall() {
		Statement  stmtSubCall;
		List<Expression> listArgs = new ArrayList<> (); 
		Expect(1);
		String strSubName = t.val; 
		Expect(5);
		if (StartOf(4)) {
			Expression expr = SubCallExpression();
			listArgs.add (expr); 
			while (la.kind == 8) {
				Get();
				expr = SubCallExpression();
				listArgs.add (expr); 
			}
		}
		Expect(6);
		stmtSubCall = new ExpressionStatement (new FunctionCall (new NameID (strSubName), listArgs)); 
		return stmtSubCall;
	}

	Expression  StrategyAssignment() {
		Expression  exprAssignment;
		exprAssignment = null; 
		Expression exprIdentifier = StrategyLValue();
		Expression exprRHS = StrategyAssignmentOperation();
		exprAssignment = new AssignmentExpression (exprIdentifier, AssignmentOperator.NORMAL, exprRHS); 
		return exprAssignment;
	}

	Statement  StrategyDomainDeclaration() {
		Statement  stmt;
		stmt = null; 
		while (!(la.kind == 0 || la.kind == 7)) {SynErr(71); Get();}
		Expect(7);
		Expect(1);
		String strIdentifier = t.val; 
		Expect(5);
		byte nDim = m_stencilCalculation.getDimensionality (); Size size = new Size (nDim); Border border = new Border (nDim); 
		SubdomainSize(size, border, Vector.getZeroVector (nDim));
		Expect(6);
		size.addBorder (border); List<Expression> listArraySizes = new ArrayList<Expression> (); 
		if (la.kind == 17) {
			Get();
			Expression exprArraySize = StrategyExpression();
			listArraySizes.add (exprArraySize); 
			while (la.kind == 8) {
				Get();
				exprArraySize = StrategyExpression();
				listArraySizes.add (exprArraySize); 
			}
			Expect(18);
		}
		registerSubdomain (strIdentifier, new Subdomain (null, Subdomain.ESubdomainType.SUBDOMAIN, size), listArraySizes); SubdomainIdentifier sgid = getSubdomainIdentifier (strIdentifier); addDeclaration (StencilSpecifier.STENCIL_GRID, sgid); 
		stmt = new ExpressionStatement (new AssignmentExpression (sgid, AssignmentOperator.NORMAL, new FunctionCall (Globals.FNX_MALLOC.clone (), CodeGeneratorUtil.expressions (size.getVolume ())))); 
		return stmt;
	}

	Statement  StrategyIntegerDeclaration() {
		Statement  stmt;
		stmt = null; 
		while (!(la.kind == 0 || la.kind == 10)) {SynErr(72); Get();}
		Expect(10);
		Expect(1);
		String strIdentifier = t.val; 
		if (la.kind == 13) {
			Expression expr = StrategyAssignmentOperation();
			Integer nVal = getIntegerValue (expr); if (nVal != null) m_mapConstants.put (strIdentifier, nVal); 
			else { 
			VariableDeclarator decl = new VariableDeclarator (new NameID (strIdentifier)); 
			decl.setInitializer (new ValueInitializer (expr)); 
			stmt = new DeclarationStatement (new VariableDeclaration (Globals.SPECIFIER_INDEX, decl)); } 
		}
		return stmt;
	}

	void SubdomainSize(Size size, Border border, Vector vecParentSize) {
		List<Expression> listExpressions = StrategyVector(vecParentSize );
		int i = 0; for (Expression expr : listExpressions) { size.setCoord (i, expr); i++; } 
		while (la.kind == 28 || la.kind == 29) {
			boolean bIsNegative = false; 
			if (la.kind == 28) {
				Get();
			} else {
				Get();
				bIsNegative = true; 
			}
			Border border1 = SubdomainScaledBorder();
			if (border1 != null) { 
			if (bIsNegative) { border.getMin ().subtract (border1.getMin ()); border.getMax ().subtract (border1.getMax ()); } 
			else { border.getMin ().add (border1.getMin ()); border.getMax ().add (border1.getMax ()); } 
			} 
		}
	}

	Expression  StrategyExpression() {
		Expression  expr;
		expr = StrategyAdditiveExpression();
		return expr;
	}

	Expression  StrategyAssignmentOperation() {
		Expression  expr;
		expr = null; 
		Expect(13);
		if (isStencilCall ()) {
			Expect(33);
			Expect(5);
			Expression exprArgument = StrategyGridAccess(EHandSide.RIGHT);
			expr = CodeGeneratorUtil.createStencilFunctionCall (exprArgument.clone ()); 
			Expect(6);
		} else if (StartOf(3)) {
			expr = StrategyExpression();
		} else SynErr(73);
		return expr;
	}

	void SubdomainLoop(SubdomainIterator loop) {
		while (!(la.kind == 0 || la.kind == 24)) {SynErr(74); Get();}
		Expect(24);
		Expect(1);
		String strIdentifier = t.val; byte nDimensionality = m_stencilCalculation.getDimensionality (); Size size = new Size (nDimensionality); Border border = new Border (nDimensionality); Size sizeDomn = new Size (nDimensionality); Border borderDomn = new Border (nDimensionality); 
		Expect(5);
		SubdomainSize(size,border,Vector.getOnesVector(nDimensionality));
		Expect(6);
		size.addBorder (border); 
		SubdomainIdentifier sgidDomn = IterationSpace(sizeDomn, borderDomn);
		loop.setDomainSubdomain (sgidDomn, sizeDomn, borderDomn); 
		registerSubdomain (strIdentifier, new Subdomain (sgidDomn.getSubdomain (), Subdomain.ESubdomainType.SUBDOMAIN, size), null); 
		SubdomainIdentifier sgid = getSubdomainIdentifier (strIdentifier); 
		if (sgidDomn.getTemporalIndex () != null) sgid.setTemporalIndex (sgidDomn.getTemporalIndex ()); 
		loop.setIteratorSubdomain (sgid); 
	}

	void PlaneLoop(SubdomainIterator loop) {
		while (!(la.kind == 0 || la.kind == 25)) {SynErr(75); Get();}
		Expect(25);
		Expect(1);
		String strIdentifier = t.val; byte nDimensionality = m_stencilCalculation.getDimensionality (); Size sizeDomn = new Size (nDimensionality); Border borderDomn = new Border (nDimensionality); 
		SubdomainIdentifier sgidDomn = IterationSpace(sizeDomn, borderDomn);
		loop.setDomainSubdomain (sgidDomn, sizeDomn, borderDomn); 
		Expression[] rgSize = new Expression[nDimensionality]; for (int i = 0; i < nDimensionality - 1; i++) rgSize[i] = sgidDomn.getSubdomain ().getSize ().getCoord (i).clone (); rgSize[nDimensionality - 1] = new IntegerLiteral (1); 
		registerSubdomain (strIdentifier, new Subdomain (sgidDomn.getSubdomain (), Subdomain.ESubdomainType.PLANE, new Size (rgSize)), null); 
		SubdomainIdentifier sgid = getSubdomainIdentifier (strIdentifier); 
		if (sgidDomn.getTemporalIndex () != null) sgid.setTemporalIndex (sgidDomn.getTemporalIndex ()); 
		loop.setIteratorSubdomain (sgid); 
	}

	void PointLoop(SubdomainIterator loop) {
		while (!(la.kind == 0 || la.kind == 26)) {SynErr(76); Get();}
		Expect(26);
		Expect(1);
		String strIdentifier = t.val; byte nDimensionality = m_stencilCalculation.getDimensionality (); Size sizeDomn = new Size (nDimensionality); Border borderDomn = new Border (nDimensionality); 
		SubdomainIdentifier sgidDomn = IterationSpace(sizeDomn, borderDomn);
		loop.setDomainSubdomain (sgidDomn, sizeDomn, borderDomn); 
		Expression[] rgSize = new Expression[nDimensionality]; for (int i = 0; i < nDimensionality; i++) rgSize[i] = new IntegerLiteral (1); 
		registerSubdomain (strIdentifier, new Subdomain (sgidDomn.getSubdomain (), Subdomain.ESubdomainType.POINT, new Size (rgSize)), null); 
		SubdomainIdentifier sgid = getSubdomainIdentifier (strIdentifier); 
		if (sgidDomn.getTemporalIndex () != null) sgid.setTemporalIndex (sgidDomn.getTemporalIndex ()); 
		loop.setIteratorSubdomain (sgid); 
	}

	RangeIterator  RangeLoop() {
		RangeIterator  loop;
		loop = new RangeIterator (); 
		Expect(1);
		NameID idLoopIdx = new NameID (t.val); loop.setLoopIndex (addDeclaration (Globals.SPECIFIER_INDEX, idLoopIdx)); 
		Expect(13);
		Expression exprStart = StrategyExpression();
		Expect(22);
		Expression exprEnd = StrategyExpression();
		if (exprEnd.equals (StencilProperty.getMaxTime ())) { exprEnd = m_stencilCalculation.getMaxIterations ().clone (); loop.setMainTemporalIterator (true); } 
		Expression exprStep = new IntegerLiteral (1); 
		if (la.kind == 23) {
			while (!(la.kind == 0 || la.kind == 23)) {SynErr(77); Get();}
			Get();
			Expression exprStepTmp = StrategyExpression();
			exprStep = exprStepTmp; 
		}
		loop.setRange (exprStart, exprEnd, exprStep); 
		return loop;
	}

	SubdomainIdentifier  IterationSpace(Size size, Border border) {
		SubdomainIdentifier  subdomain;
		while (!(la.kind == 0 || la.kind == 27)) {SynErr(78); Get();}
		Expect(27);
		subdomain = null; 
		Expression exprSgid = StrategySubdomainIdentifier(EHandSide.RIGHT);
		if (!(exprSgid instanceof SubdomainIdentifier)) { errors.SemErr (la.line, la.col, StringUtil.concat (exprSgid.toString (), " is not a subdomain.")); return null; } 
		Expect(5);
		subdomain = (SubdomainIdentifier) exprSgid; 
		SubdomainSize(size, border, subdomain.getSubdomain ().getSize ());
		Expect(14);
		Expression exprTimeIdx = StrategyExpression();
		if (exprTimeIdx != null) subdomain.setTemporalIndex (exprTimeIdx); 
		Expect(6);
		return subdomain;
	}

	Expression  StrategySubdomainIdentifier(EHandSide hs) {
		Expression  exprIdentifier;
		Expect(1);
		String strName = t.val; if (m_mapDimensionIdentifiers.containsKey (strName)) errors.SemErr (la.line, la.col, "Dimension identifiers cannot be used in epxressions"); 
		exprIdentifier = m_mapSubdomains.containsKey (strName) ? getSubdomainIdentifier (t.val) : new NameID (t.val); 
		if (hs == EHandSide.LEFT) 
		addDeclaration (StencilSpecifier.STENCIL_GRID, exprIdentifier); 
		else 
		checkDeclared (exprIdentifier); 
		return exprIdentifier;
	}

	List<Expression>  StrategyVector(Vector vecDefault) {
		List<Expression>  listExpressions;
		listExpressions = new LinkedList<> (); List<Expression> listTail = null; byte nDim = m_stencilCalculation.getDimensionality (); 
		List<Expression> listHead  = Subvector();
		if (la.kind == 36) {
			Get();
			if (la.kind == 8) {
				Get();
				listTail = Subvector();
			}
			Expression exprLast = listHead.get (listHead.size () - 1); int nTailSize = listTail == null ? 0 : listTail.size (); for (int i = listHead.size (); i < nDim - nTailSize; i++) listHead.add (exprLast.clone ()); 
		}
		listExpressions = createVector (listHead, listTail, vecDefault); 
		return listExpressions;
	}

	Border  SubdomainScaledBorder() {
		Border  border;
		border = null; Expression exprFactor1 = null; Expression exprFactor2 = null; 
		if (isBorder ()) {
			border = SubdomainBorder();
			if (border == null) return null; 
		} else if (StartOf(3)) {
			exprFactor1 = StrategyUnaryExpression();
			Expect(30);
			border = SubdomainBorder();
			if (border == null) return null; 
		} else SynErr(79);
		if (la.kind == 30) {
			Get();
			exprFactor2 = StrategyUnaryExpression();
		}
		Expression exprFactor = (exprFactor1 == null ? (exprFactor2 == null ? null : exprFactor2) : (exprFactor2 == null ? exprFactor1 : new BinaryExpression (exprFactor1, BinaryOperator.MULTIPLY, exprFactor2))); 
		if (exprFactor != null) border.scale (exprFactor); 
		return border;
	}

	Border  SubdomainBorder() {
		Border  border;
		border = null; 
		if (la.kind == 33) {
			border = StencilBoxBorder();
		} else if (la.kind == 31) {
			border = LiteralSubdomainBorder();
		} else SynErr(80);
		return border;
	}

	Expression  StrategyUnaryExpression() {
		Expression  expr;
		expr = null; boolean bIsNegative = false; 
		if (la.kind == 28 || la.kind == 29) {
			if (la.kind == 28) {
				Get();
			} else {
				Get();
				bIsNegative = true; 
			}
		}
		if (la.kind == 2 || la.kind == 3) {
			Number numValue = NumberLiteral();
			expr = numValue instanceof Integer ? new IntegerLiteral (bIsNegative ? -numValue.intValue () : numValue.intValue ()) : new FloatLiteral (bIsNegative ? -numValue.doubleValue () : numValue.doubleValue ()); 
		} else if (la.kind == 5) {
			Expression exprBracketed = StrategyBracketedExpression();
			expr = createUnaryExpression (bIsNegative, exprBracketed); 
		} else if (la.kind == 53) {
			Expression exprPointer = StrategyPointerExpression();
			expr = createUnaryExpression (bIsNegative, exprPointer); 
		} else if (isSubCall ()) {
			Expression exprFnxValue = StrategyFunctionCall();
			expr = createUnaryExpression (bIsNegative, exprFnxValue); 
		} else if (isGridAccess ()) {
			Expression exprGridValue = StrategyGridAccess(EHandSide.RIGHT);
			expr = createUnaryExpression (bIsNegative, exprGridValue); 
		} else if (isCoordinateOrVector ()) {
			Expression exprCoord = StrategySubdomainCoordinate();
			expr = createUnaryExpression (bIsNegative, exprCoord); 
		} else if (isProperty ()) {
			Expression exprProperty = StrategyProperty();
			expr = createUnaryExpression (bIsNegative, exprProperty); 
		} else if (la.kind == 1) {
			Get();
			Integer nVal = getConstantValue (t.val); expr = nVal != null ? new IntegerLiteral ((bIsNegative ? -1 : 1) * nVal) : createUnaryExpression (bIsNegative, new NameID (t.val)); 
		} else SynErr(81);
		return expr;
	}

	Border  StencilBoxBorder() {
		Border  border;
		border = null; Stencil stencil = m_stencilCalculation.getStencilBundle ().getFusedStencil (); 
		while (!(la.kind == 0 || la.kind == 33)) {SynErr(82); Get();}
		Expect(33);
		Expect(34);
		Expect(35);
		int[] rgMinSpaceIdx = stencil.getMinSpaceIndex (); int[] rgMaxSpaceIdx = stencil.getMaxSpaceIndex (); 
		List<Expression> listMinSpaceIdx = new ArrayList<Expression> (rgMinSpaceIdx.length); List<Expression> listMaxSpaceIdx = new ArrayList<Expression> (rgMaxSpaceIdx.length); 
		for (int i = 0; i < rgMinSpaceIdx.length; i++) { listMinSpaceIdx.add (new IntegerLiteral (-rgMinSpaceIdx[i])); listMaxSpaceIdx.add (new IntegerLiteral (rgMaxSpaceIdx[i])); } 
		if (la.kind == 5) {
			Get();
			List<Expression> listCoords = StrategyVector(null );
			listMinSpaceIdx = createSubscriptedVector (listMinSpaceIdx, listCoords); listMaxSpaceIdx = createSubscriptedVector (listMaxSpaceIdx, listCoords); 
			Expect(6);
			border = new Border (createSize (listMinSpaceIdx), createSize (listMaxSpaceIdx)); 
		}
		; 
		return border;
	}

	Border  LiteralSubdomainBorder() {
		Border  border;
		border = null; 
		Expect(31);
		if (isVector ()) {
			border = LiteralVectorBorder();
		} else if (isBorder ()) {
			border = SubdomainBorder();
			while (la.kind == 8) {
				Get();
				Border border1 = SubdomainBorder();
				int nDim0 = border.getDimensionality (); Expression rgExprMin[] = new Expression[nDim0 + border1.getDimensionality ()]; 
				Expression rgExprMax[] = new Expression[nDim0 + border1.getDimensionality ()]; 
				for (int i = 0; i < nDim0; i++) { rgExprMin[i] = border.getMin ().getCoord (i); rgExprMax[i] = border.getMax ().getCoord (i); } 
				for (int i = 0; i < border1.getDimensionality (); i++) { rgExprMin[nDim0 + i] = border1.getMin ().getCoord (i); rgExprMax[nDim0 + i] = border1.getMax ().getCoord (i); } 
				border = new Border (new Size (rgExprMin), new Size (rgExprMax)); 
			}
		} else if (StartOf(3)) {
			Expression exprMin = StrategyExpression();
			Expect(8);
			Expression exprMax = StrategyExpression();
			border = new Border (new Size (new UnaryExpression (UnaryOperator.MINUS, exprMin)), new Size (exprMax)); 
		} else SynErr(83);
		Expect(32);
		return border;
	}

	Border  LiteralVectorBorder() {
		Border  border;
		border = null; Vector v = Vector.getZeroVector (m_stencilCalculation.getStencilBundle ().getFusedStencil ().getDimensionality ()); 
		Expect(5);
		List<Expression> listMin = StrategyVector(v );
		Expect(6);
		List<Expression> listMinNeg = new ArrayList<> (listMin.size ()); 
		Expect(8);
		for (Expression expr : listMin) listMinNeg.add (new UnaryExpression (UnaryOperator.MINUS, expr)); 
		Expect(5);
		List<Expression> listMax = StrategyVector(v );
		Expect(6);
		border = new Border (createSize (listMinNeg), createSize (listMax)); 
		return border;
	}

	List<Expression>  Subvector() {
		List<Expression>  listExpressions;
		listExpressions = null; byte nDimensionality = m_stencilCalculation.getDimensionality (); 
		if (la.kind == 37) {
			Get();
			List<Expression> listExprs1 = null; 
			while (la.kind == 8) {
				Get();
				listExprs1 = ScalarList();
			}
			listExpressions = new ArrayList<> (nDimensionality); int nTailSize = listExprs1 == null ? 0 : listExprs1.size (); 
			for (int i = 0; i < nDimensionality - nTailSize; i++) listExpressions.add (null); if (listExprs1 != null) listExpressions.addAll (listExprs1); 
		} else if (isDimensionParameter ()) {
			listExpressions = DimensionIdentifier();
		} else if (isVectorProperty ()) {
			listExpressions = DomainSizeExpression();
		} else if (isBracketedVector ()) {
			Expect(5);
			listExpressions = StrategyVector(null);
			while (la.kind == 8) {
				Get();
				List<Expression> listExprs1 = StrategyVector(null );
				listExpressions.addAll (listExprs1); 
			}
			Expect(6);
		} else if (StartOf(3)) {
			listExpressions = ScalarList();
		} else SynErr(84);
		return listExpressions;
	}

	List<Expression>  ScalarList() {
		List<Expression>  listExpressions;
		listExpressions = new ArrayList<> (); 
		List<Expression> listRange  = ScalarRange();
		listExpressions.addAll (listRange); 
		while (la.kind == 8) {
			Get();
			listRange  = ScalarRange();
			listExpressions.addAll (listRange); 
		}
		return listExpressions;
	}

	List<Expression>  DimensionIdentifier() {
		List<Expression>  listExpressions;
		byte nDimensionality = m_stencilCalculation.getDimensionality (); 
		Expression exprSize = StrategyExpression();
		listExpressions = new ArrayList<> (); for (int i = 0; i < nDimensionality; i++) listExpressions.add (getDimIdentifier (exprSize, i)); 
		if (la.kind == 5) {
			Get();
			Vector v = new Vector (nDimensionality); for (int i = 0; i < nDimensionality; i++) v.setCoord (i, i + 1); 
			List<Expression> listSubscripts = StrategyVector(v );
			listExpressions = createSubscriptedVector (listExpressions, listSubscripts); 
			Expect(6);
		}
		return listExpressions;
	}

	List<Expression>  DomainSizeExpression() {
		List<Expression>  listExpressions;
		Box box = null; 
		if (la.kind == 33) {
			Get();
			box = m_stencilCalculation.getStencilBundle ().getFusedStencil ().getBoundingBox (); 
		} else if (la.kind == 1) {
			Expression exprSgid = StrategySubdomainIdentifier(EHandSide.RIGHT);
			Subdomain sg = getSubdomain (t.val); if (sg == null) return new ArrayList<> (0); box = sg.getBox (); 
		} else SynErr(85);
		Expect(34);
		listExpressions = new ArrayList<> (); if (box == null) { errors.SemErr (la.line, la.col, "No box defined"); return null; } 
		if (la.kind == 38) {
			Get();
			for (Expression expr : box.getSize ()) listExpressions.add (expr); 
		} else if (la.kind == 39) {
			Get();
			for (Expression expr : box.getMin ()) listExpressions.add (expr); 
		} else if (la.kind == 40) {
			Get();
			for (Expression expr : box.getMax ()) listExpressions.add (expr); 
		} else SynErr(86);
		if (la.kind == 5) {
			Get();
			List<Expression> listCoords = StrategyVector(null );
			Expect(6);
			listExpressions = createSubscriptedVector (listExpressions, listCoords); 
		}
		return listExpressions;
	}

	List<Expression>  ScalarRange() {
		List<Expression>  listExpressions;
		listExpressions = new ArrayList<> (); 
		Expression exprStart = StrategyExpression();
		listExpressions.add (exprStart); 
		if (la.kind == 22) {
			Get();
			int nEnd = CompileTimeConstant();
			for (int i = getIntValue (exprStart) + 1; i <= nEnd; i++) listExpressions.add (new IntegerLiteral (i)); 
		}
		return listExpressions;
	}

	int  CompileTimeConstant() {
		int  nResult;
		nResult = 0; 
		Expression expr = StrategyExpression();
		try { nResult = ExpressionUtil.getIntegerValue (expr); } catch (RuntimeException e) { errors.SemErr (la.line, la.col, "Compile time constant expected."); } 
		return nResult;
	}

	Expression  StrategyLValue() {
		Expression  expr;
		expr = null; 
		if (isGridAccess ()) {
			Expression expr0 = StrategyGridAccess(EHandSide.LEFT);
			expr = expr0; 
		} else if (la.kind == 1) {
			Get();
			expr = new NameID (t.val); 
		} else SynErr(87);
		return expr;
	}

	Expression  StrategyGridAccess(EHandSide hs) {
		Expression  exprSubdomainIdentifier;
		Expression exprSgid = StrategySubdomainIdentifier(hs);
		exprSubdomainIdentifier = exprSgid; 
		Expect(17);
		if (!(exprSubdomainIdentifier instanceof SubdomainIdentifier)) { errors.SemErr (la.line, la.col, StringUtil.concat (exprSubdomainIdentifier.toString (), " has not been declared a subdomain. Only subdomains can have array subscripts.")); return null; } 
		SubdomainIdentifier sgid = (SubdomainIdentifier) exprSubdomainIdentifier; 
		if (isSubdomainIdentifier ()) {
			Expression exprPoint = StrategySubdomainIdentifier(hs);
			exprSubdomainIdentifier = exprPoint; sgid = (SubdomainIdentifier) exprSubdomainIdentifier; 
		} else if (StartOf(5)) {
			byte nDim = m_stencilCalculation.getDimensionality (); Size size = new Size (nDim); Border border = new Border (nDim); 
			SubdomainSize(size, border, Vector.getZeroVector (nDim));
			Box box = new Box (size.getCoords (), size.getCoords ()); box.addBorder (border); sgid.setSpatialOffset (box.getMin ().getCoords ()); 
		} else SynErr(88);
		Expect(14);
		Expression exprTimeIndex = StrategyExpression();
		sgid.setTemporalIndex (exprTimeIndex); 
		while (la.kind == 14) {
			while (!(la.kind == 0 || la.kind == 14)) {SynErr(89); Get();}
			Get();
			Expression exprIdx = StrategyExpression();
			
		}
		Expect(18);
		return exprSubdomainIdentifier;
	}

	Expression  StrategySubdomainCoordinate() {
		Expression  exprCoord;
		Expression exprSgid = StrategySubdomainIdentifier(EHandSide.RIGHT);
		Expect(5);
		int nDim = CompileTimeConstant();
		Expect(6);
		exprCoord = createCoordinateExpression (exprSgid, nDim); 
		return exprCoord;
	}

	Expression  StrategyProperty() {
		Expression  expr;
		expr = null; 
		if (la.kind == 33) {
			expr = StrategyStencilProperty();
		} else if (la.kind == 1) {
			expr = StrategySubdomainProperty();
		} else SynErr(90);
		return expr;
	}

	Expression  StrategyStencilProperty() {
		Expression  expr;
		expr = null; 
		Expect(33);
		Expect(34);
		if (la.kind == 11) {
			Get();
			expr = new IntegerLiteral (m_stencilCalculation.getDimensionality ()); 
		} else if (la.kind == 41) {
			Get();
			expr = StencilProperty.getMaxTime (); 
		} else if (la.kind == 38 || la.kind == 39 || la.kind == 40) {
			Vector v = null; Box box = m_stencilCalculation.getStencilBundle ().getFusedStencil ().getBoundingBox (); 
			if (la.kind == 38) {
				Get();
				v = box.getSize (); 
			} else if (la.kind == 39) {
				Get();
				v = box.getMin (); 
			} else {
				Get();
				v = box.getMax (); 
			}
			Expect(5);
			int nIdx = CompileTimeConstant();
			Expect(6);
			expr = v.getCoord (nIdx - 1); 
		} else SynErr(91);
		return expr;
	}

	Expression  StrategySubdomainProperty() {
		Expression  expr;
		expr = null; 
		Expression exprSgid = StrategySubdomainIdentifier(EHandSide.RIGHT);
		Expect(34);
		if (!(exprSgid instanceof SubdomainIdentifier)) { errors.SemErr (la.line, la.col, "Subdomain identifier expected."); return null; } SubdomainIdentifier sgid = (SubdomainIdentifier) exprSgid; 
		if (la.kind == 11) {
			Get();
			expr = new IntegerLiteral (m_stencilCalculation.getDimensionality ()); 
		} else if (la.kind == 39) {
			Get();
			Expect(5);
			int nIdx = CompileTimeConstant();
			Expect(6);
			expr = sgid.getSubdomain ().getBox ().getMin ().getCoord (nIdx - 1); 
		} else if (la.kind == 40) {
			Get();
			Expect(5);
			int nIdx = CompileTimeConstant();
			Expect(6);
			expr = sgid.getSubdomain ().getBox ().getMax ().getCoord (nIdx - 1); 
		} else if (la.kind == 38) {
			Get();
			Expect(5);
			int nIdx = CompileTimeConstant();
			Expect(6);
			expr = sgid.getSubdomain ().getBox ().getSize ().getCoord (nIdx - 1); 
		} else if (la.kind == 42) {
			Get();
			expr = sgid.getSubdomain ().getBox ().getVolume (); /* TODO: respect padding/alignment restrictions... */ 
		} else SynErr(92);
		return expr;
	}

	Expression  ConditionalExpression() {
		Expression  expr;
		Expression expr0 = ConditionalAndExpression();
		expr = expr0; 
		while (la.kind == 45) {
			Get();
			Expression expr1 = ConditionalAndExpression();
			expr = new BinaryExpression (expr.clone (), BinaryOperator.LOGICAL_OR, expr1); 
		}
		return expr;
	}

	Expression  ConditionalAndExpression() {
		Expression  expr;
		Expression expr0 = ComparisonExpression();
		expr = expr0; 
		while (la.kind == 46) {
			Get();
			Expression expr1 = ComparisonExpression();
			expr = new BinaryExpression (expr.clone (), BinaryOperator.LOGICAL_AND, expr1); 
		}
		return expr;
	}

	Expression  ComparisonExpression() {
		Expression  expr;
		Expression expr0 = StrategyExpression();
		BinaryOperator op = null; 
		switch (la.kind) {
		case 31: {
			Get();
			op = BinaryOperator.COMPARE_LT; 
			break;
		}
		case 47: {
			Get();
			op = BinaryOperator.COMPARE_LE; 
			break;
		}
		case 48: {
			Get();
			op = BinaryOperator.COMPARE_EQ; 
			break;
		}
		case 49: {
			Get();
			op = BinaryOperator.COMPARE_GE; 
			break;
		}
		case 32: {
			Get();
			op = BinaryOperator.COMPARE_GT; 
			break;
		}
		case 50: {
			Get();
			op = BinaryOperator.COMPARE_NE; 
			break;
		}
		default: SynErr(93); break;
		}
		Expression expr1 = StrategyExpression();
		expr = new BinaryExpression (expr0, op, expr1); 
		return expr;
	}

	Expression  StrategyAdditiveExpression() {
		Expression  expr;
		Expression expr0 = StrategyMultiplicativeExpression();
		expr = expr0; 
		while (la.kind == 28 || la.kind == 29) {
			BinaryOperator op = BinaryOperator.ADD; 
			if (la.kind == 28) {
				Get();
			} else {
				Get();
				op = BinaryOperator.SUBTRACT; 
			}
			Expression expr1 = StrategyMultiplicativeExpression();
			expr = createBinaryExpression (expr.clone (), op, expr1); 
		}
		return expr;
	}

	Expression  StrategyMultiplicativeExpression() {
		Expression  expr;
		Expression expr0 = StrategyUnaryExpression();
		expr = expr0; 
		while (la.kind == 30 || la.kind == 51 || la.kind == 52) {
			BinaryOperator op = BinaryOperator.MULTIPLY; 
			if (la.kind == 30) {
				Get();
			} else if (la.kind == 51) {
				Get();
				op = BinaryOperator.DIVIDE; 
			} else {
				Get();
				op = BinaryOperator.MODULUS; 
			}
			Expression expr1 = StrategyUnaryExpression();
			expr = createBinaryExpression (expr.clone (), op, expr1); 
		}
		return expr;
	}

	Number  NumberLiteral() {
		Number  numValue;
		numValue = null; 
		if (la.kind == 2) {
			Get();
			numValue = Integer.parseInt (t.val); 
		} else if (la.kind == 3) {
			Get();
			numValue = Double.parseDouble (t.val); 
		} else SynErr(94);
		return numValue;
	}

	Expression  StrategyBracketedExpression() {
		Expression  expr;
		Expect(5);
		expr = StrategyExpression();
		Expect(6);
		return expr;
	}

	Expression  StrategyPointerExpression() {
		Expression  exprPointer;
		exprPointer = null; 
		Expect(53);
		Expression expr = StrategyExpression();
		exprPointer = new UnaryExpression (UnaryOperator.ADDRESS_OF, expr); 
		return exprPointer;
	}

	Expression  StrategyFunctionCall() {
		Expression  exprFnx;
		Expect(1);
		String strFunctionName = t.val; 
		Expect(5);
		List<Expression> listArgs = new ArrayList<> (); 
		if (StartOf(3)) {
			Expression expr = StrategyExpression();
			listArgs.add (expr); 
			while (la.kind == 8) {
				while (!(la.kind == 0 || la.kind == 8)) {SynErr(95); Get();}
				Get();
				expr = StrategyExpression();
				listArgs.add (expr); 
			}
		}
		Expect(6);
		exprFnx = new FunctionCall (new NameID (strFunctionName), listArgs); 
		return exprFnx;
	}

	Expression  SubCallExpression() {
		Expression  expr;
		expr = null; 
		if (la.kind == 54) {
			expr = StringExpression();
		} else if (StartOf(3)) {
			expr = StrategyExpression();
		} else SynErr(96);
		return expr;
	}

	Expression  StringExpression() {
		Expression  expr;
		Expect(54);
		while (StartOf(6)) {
			Get();
		}
		expr = new StringLiteral (t.val); 
		Expect(54);
		return expr;
	}

	List<IAutotunerParam>  AutoTuneVector() {
		List<IAutotunerParam>  listParams;
		listParams = new ArrayList<> (); 
		Expect(5);
		List<IAutotunerParam> l0  = AutoTuneItem();
		listParams.addAll (l0); 
		while (la.kind == 8) {
			Get();
			List<IAutotunerParam> l1  = AutoTuneItem();
			listParams.addAll (l1); 
		}
		Expect(6);
		return listParams;
	}

	List<IAutotunerParam>  AutoTuneItem() {
		List<IAutotunerParam>  listParams;
		listParams = null; 
		if (la.kind == 15) {
			listParams = AutoTuneList();
		} else if (StartOf(3)) {
			listParams = AutoTuneRange();
		} else SynErr(97);
		return listParams;
	}

	List<IAutotunerParam>  AutoTuneList() {
		List<IAutotunerParam>  listParams;
		listParams = new ArrayList<> (); 
		Expect(15);
		List<Expression> l0  = AutoTuneValue();
		listParams = new ArrayList<> (l0.size ()); 
		for (Expression expr : l0) listParams.add (new IAutotunerParam.AutotunerListParam (expr)); 
		while (la.kind == 8) {
			Get();
			List<Expression> l1  = AutoTuneValue();
			if (listParams.size () != l1.size ()) errors.SemErr (la.line, la.col, "Entries in an auto-tuner list parameter must have the same length."); 
			int i = 0; for (Expression expr : l1) { ((IAutotunerParam.AutotunerListParam) listParams.get (i)).addValue (expr); i++; }
		}
		Expect(16);
		return listParams;
	}

	List<IAutotunerParam>  AutoTuneRange() {
		List<IAutotunerParam>  listParams;
		boolean bIsMultiplicative = false; 
		List<Expression> listStart  = AutoTuneValue();
		Expect(37);
		if (la.kind == 30) {
			Get();
			bIsMultiplicative = true; 
		}
		List<Expression> listStepOrEnd  = AutoTuneValue();
		List<Expression> listEnd = null; 
		if (listStart.size () != listStepOrEnd.size () && (listStart.size () > 1 && listStepOrEnd.size () > 1)) errors.SemErr (la.line, la.col, "Vector entries in an auto-tuner range parameter must have the same length."); 
		if (la.kind == 37) {
			Get();
			listEnd = AutoTuneValue();
			if ((listStart.size () != listEnd.size () && (listStart.size () > 1 && listEnd.size () > 1)) || (listStepOrEnd.size () != listEnd.size () && (listStepOrEnd.size () > 1 && listEnd.size () > 1))) errors.SemErr (la.line, la.col, "Vector entries in an auto-tuner range parameter must have the same length."); 
		}
		listParams = new ArrayList<> (listStart.size ()); 
		int nLen = Math.max (listStart.size (), listStepOrEnd.size ()); 
		if (listEnd != null) nLen = Math.max (nLen, listEnd.size ()); 
		for (int i = 0; i < nLen; i++) 
		listParams.add (listEnd == null ? 
		new IAutotunerParam.AutotunerRangeParam (getAutotuneListItem (listStart, i), getAutotuneListItem (listStepOrEnd, i)) : 
		new IAutotunerParam.AutotunerRangeParam (getAutotuneListItem (listStart, i), getAutotuneListItem (listStepOrEnd, i), bIsMultiplicative, getAutotuneListItem (listEnd, i))); 
		return listParams;
	}

	List<Expression>  AutoTuneValue() {
		List<Expression>  listExpressions;
		listExpressions = null; 
		if (isDimensionParameter ()) {
			listExpressions = DimensionIdentifier();
		} else if (isVectorProperty ()) {
			listExpressions = DomainSizeExpression();
		} else if (StartOf(3)) {
			Expression expr = StrategyExpression();
			listExpressions = new ArrayList<Expression> (1); listExpressions.add (expr); 
		} else SynErr(98);
		return listExpressions;
	}



	public void Parse() {
		la = new Token();
		la.val = "";		
		Get();
		StrategySpecification();
		Expect(0);

	}

	private static final boolean[][] set = {
		{T,x,x,x, x,x,x,T, T,x,T,x, x,x,T,x, x,x,x,T, T,T,x,T, T,T,T,T, x,x,x,x, x,T,x,x, x,x,x,x, x,x,x,T, T,x,x,x, x,x,x,x, x,x,x,x, x},
		{x,T,x,x, x,x,x,T, x,x,T,x, x,x,x,T, x,x,x,T, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,T, x,x,x,x, x,x,x,x, x,x,x,x, x},
		{x,T,T,T, x,T,x,x, x,x,x,x, x,x,x,T, x,x,x,x, x,x,x,x, x,x,x,x, T,T,x,x, x,T,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,T,x,x, x},
		{x,T,T,T, x,T,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, T,T,x,x, x,T,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,T,x,x, x},
		{x,T,T,T, x,T,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, T,T,x,x, x,T,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,T,T,x, x},
		{x,T,T,T, x,T,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, T,T,x,x, x,T,x,x, x,T,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,T,x,x, x},
		{x,T,T,T, T,T,T,T, T,T,T,T, T,T,T,T, T,T,T,T, T,T,T,T, T,T,T,T, T,T,T,T, T,T,T,T, T,T,T,T, T,T,T,T, T,T,T,T, T,T,T,T, T,T,x,T, x}

	};
} // end Parser


class Errors {
	public int count = 0;                                    // number of errors detected
	public String errMsgFormat = "Line {0}, col {1}: {2}"; // 0=line, 1=column, 2=text
	
	protected void printMsg(int line, int column, String msg) {
		StringBuffer b = new StringBuffer(errMsgFormat);
		int pos = b.indexOf("{0}");
		if (pos >= 0) { b.delete(pos, pos+3); b.insert(pos, line); }
		pos = b.indexOf("{1}");
		if (pos >= 0) { b.delete(pos, pos+3); b.insert(pos, column); }
		pos = b.indexOf("{2}");
		if (pos >= 0) b.replace(pos, pos+3, msg);
		Parser.LOGGER.error (b.toString());
	}
	
	public void SynErr (int line, int col, int n) {
		String s;
		switch (n) {
			case 0: s = "EOF expected"; break;
			case 1: s = "ident expected"; break;
			case 2: s = "integer expected"; break;
			case 3: s = "float expected"; break;
			case 4: s = "\"strategy\" expected"; break;
			case 5: s = "\"(\" expected"; break;
			case 6: s = "\")\" expected"; break;
			case 7: s = "\"domain\" expected"; break;
			case 8: s = "\",\" expected"; break;
			case 9: s = "\"auto\" expected"; break;
			case 10: s = "\"int\" expected"; break;
			case 11: s = "\"dim\" expected"; break;
			case 12: s = "\"codim\" expected"; break;
			case 13: s = "\"=\" expected"; break;
			case 14: s = "\";\" expected"; break;
			case 15: s = "\"{\" expected"; break;
			case 16: s = "\"}\" expected"; break;
			case 17: s = "\"[\" expected"; break;
			case 18: s = "\"]\" expected"; break;
			case 19: s = "\"for\" expected"; break;
			case 20: s = "\"parallel\" expected"; break;
			case 21: s = "\"schedule\" expected"; break;
			case 22: s = "\"..\" expected"; break;
			case 23: s = "\"by\" expected"; break;
			case 24: s = "\"subdomain\" expected"; break;
			case 25: s = "\"plane\" expected"; break;
			case 26: s = "\"point\" expected"; break;
			case 27: s = "\"in\" expected"; break;
			case 28: s = "\"+\" expected"; break;
			case 29: s = "\"-\" expected"; break;
			case 30: s = "\"*\" expected"; break;
			case 31: s = "\"<\" expected"; break;
			case 32: s = "\">\" expected"; break;
			case 33: s = "\"stencil\" expected"; break;
			case 34: s = "\".\" expected"; break;
			case 35: s = "\"box\" expected"; break;
			case 36: s = "\"...\" expected"; break;
			case 37: s = "\":\" expected"; break;
			case 38: s = "\"size\" expected"; break;
			case 39: s = "\"min\" expected"; break;
			case 40: s = "\"max\" expected"; break;
			case 41: s = "\"t_max\" expected"; break;
			case 42: s = "\"volume\" expected"; break;
			case 43: s = "\"if\" expected"; break;
			case 44: s = "\"else\" expected"; break;
			case 45: s = "\"||\" expected"; break;
			case 46: s = "\"&&\" expected"; break;
			case 47: s = "\"<=\" expected"; break;
			case 48: s = "\"==\" expected"; break;
			case 49: s = "\">=\" expected"; break;
			case 50: s = "\"!=\" expected"; break;
			case 51: s = "\"/\" expected"; break;
			case 52: s = "\"%\" expected"; break;
			case 53: s = "\"&\" expected"; break;
			case 54: s = "\"\"\" expected"; break;
			case 55: s = "??? expected"; break;
			case 56: s = "invalid StrategyParamList"; break;
			case 57: s = "invalid AutoTuneValues"; break;
			case 58: s = "this symbol not expected in StrategyStatement"; break;
			case 59: s = "this symbol not expected in StrategyStatement"; break;
			case 60: s = "this symbol not expected in StrategyStatement"; break;
			case 61: s = "invalid StrategyStatement"; break;
			case 62: s = "invalid StrategyDeclaration"; break;
			case 63: s = "this symbol not expected in StrategyLoop"; break;
			case 64: s = "this symbol not expected in StrategyLoop"; break;
			case 65: s = "this symbol not expected in StrategyLoop"; break;
			case 66: s = "this symbol not expected in StrategyLoop"; break;
			case 67: s = "this symbol not expected in StrategyLoop"; break;
			case 68: s = "invalid StrategyLoop"; break;
			case 69: s = "this symbol not expected in StrategyIfStatement"; break;
			case 70: s = "this symbol not expected in StrategyIfStatement"; break;
			case 71: s = "this symbol not expected in StrategyDomainDeclaration"; break;
			case 72: s = "this symbol not expected in StrategyIntegerDeclaration"; break;
			case 73: s = "invalid StrategyAssignmentOperation"; break;
			case 74: s = "this symbol not expected in SubdomainLoop"; break;
			case 75: s = "this symbol not expected in PlaneLoop"; break;
			case 76: s = "this symbol not expected in PointLoop"; break;
			case 77: s = "this symbol not expected in RangeLoop"; break;
			case 78: s = "this symbol not expected in IterationSpace"; break;
			case 79: s = "invalid SubdomainScaledBorder"; break;
			case 80: s = "invalid SubdomainBorder"; break;
			case 81: s = "invalid StrategyUnaryExpression"; break;
			case 82: s = "this symbol not expected in StencilBoxBorder"; break;
			case 83: s = "invalid LiteralSubdomainBorder"; break;
			case 84: s = "invalid Subvector"; break;
			case 85: s = "invalid DomainSizeExpression"; break;
			case 86: s = "invalid DomainSizeExpression"; break;
			case 87: s = "invalid StrategyLValue"; break;
			case 88: s = "invalid StrategyGridAccess"; break;
			case 89: s = "this symbol not expected in StrategyGridAccess"; break;
			case 90: s = "invalid StrategyProperty"; break;
			case 91: s = "invalid StrategyStencilProperty"; break;
			case 92: s = "invalid StrategySubdomainProperty"; break;
			case 93: s = "invalid ComparisonExpression"; break;
			case 94: s = "invalid NumberLiteral"; break;
			case 95: s = "this symbol not expected in StrategyFunctionCall"; break;
			case 96: s = "invalid SubCallExpression"; break;
			case 97: s = "invalid AutoTuneItem"; break;
			case 98: s = "invalid AutoTuneValue"; break;
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
		Parser.LOGGER.error (s);
		count++;
	}
	
	public void Warning (int line, int col, String s) {	
		printMsg(line, col, s);
	}
	
	public void Warning (String s) {
		Parser.LOGGER.error (s);
	}
} // Errors


class FatalError extends RuntimeException {
	public static final long serialVersionUID = 1L;
	public FatalError(String s) { super(s); }
}
