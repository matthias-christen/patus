package ch.unibas.cs.hpwc.patus.grammar.stencil2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FloatLiteral;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;

import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.geometry.Box;
import ch.unibas.cs.hpwc.patus.geometry.Point;
import ch.unibas.cs.hpwc.patus.representation.Index;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilBundle;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.symbolic.ExpressionData;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.DomainPointEnumerator;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.IntArray;
import ch.unibas.cs.hpwc.patus.util.StringUtil;




public class Parser {
	public static final int _EOF = 0;
	public static final int _ident = 1;
	public static final int _integer = 2;
	public static final int _pi = 3;
	public static final int maxT = 56;

	static final boolean T = true;
	static final boolean x = false;
	static final int minErrDist = 2;
	
	static final org.apache.log4j.Logger LOGGER = org.apache.log4j.Logger.getLogger (Parser.class);
	

	public Token t;    // last recognized token
	public Token la;   // lookahead token
	int errDist = minErrDist;
	
	public Scanner scanner;
	public Errors errors;

	enum EStreamDirection
	{
		/**
		 * A stream from which data is read
		 */
		INPUT,
		
		/**
		 * A stream to which data is written to
		 */
		OUTPUT
	}
	
	/**
	 * Helper class for mapping multidimensional and differently named identifiers
	 * to a normalized set of identifiers (map to vector components of StencilNodes).
	 */
	protected class StreamIndex
	{
		private Specifier m_specType;
		private boolean m_bIsConstant;
		private Box m_boxStreamDimensions;
		private int m_nStartIndex;
		private Range[] m_rgDimensions;
		
		/**
		 * Constructs a new stream index.
		 */
		public StreamIndex (String strName, Specifier specType, boolean bIsConstant, Box boxDimensions, List<Range> listDimensions, EStreamDirection sd)
		{
			m_specType = specType;
			m_bIsConstant = bIsConstant;
			m_boxStreamDimensions = boxDimensions;
			
			// copy the dimensions
			m_rgDimensions = new Range[listDimensions.size ()];
			int i = 0;
			for (Range range : listDimensions)
				m_rgDimensions[i++] = range;
			
			// calculate the total number of dimensions
			int nStreamsCount = 1;
			for (Range range : listDimensions)
				nStreamsCount *= range.getSize ();

			// set the start index and the new number of total streams
			StreamIndex idx = m_mapInputStreams.get (strName);
			if (idx == null)
				idx = m_mapOutputStreams.get (strName);
			if (idx != null)
				m_nStartIndex = idx.m_nStartIndex;
			else
			{
				m_nStartIndex = m_nTotalStreamsCount;
				m_nTotalStreamsCount += nStreamsCount;
			}
		}
		
		/**
		 * Returns the index.
		 */
		public int getLinearIndex (String strIdentifier, List<Expression> listIndices)
		{
			if ((listIndices == null && m_rgDimensions.length != 0) || (listIndices != null && listIndices.size () != m_rgDimensions.length))
			{
				errors.SemErr (la.line, la.col, StringUtil.concat ("Parameter dimension of ", strIdentifier,
					" does not agree with its definition: should be ", (listIndices == null ? 0 : listIndices.size ()), ".")
				);
				return -1;
			}
			
			int nIdx = 0;
			if (listIndices != null)
			{
				int i = 0;
				for (Expression exprIdxValue : listIndices)
				{
					Integer nIdxValue = ExpressionUtil.getIntegerValueEx (exprIdxValue);
					if (nIdxValue == null)
					{
						errors.SemErr (la.line, la.col, StringUtil.concat ("Value ", exprIdxValue.toString (), " does not evaluate to an integer number"));
						continue;
					}
					
					// check whether the indices are within the defined bounds
					if (nIdxValue < m_rgDimensions[i].getStart () || nIdxValue > m_rgDimensions[i].getEnd ())
					{
						errors.SemErr (la.line, la.col, StringUtil.concat ("Index in dimension ", i, " of \"", strIdentifier, listIndices.toString (), "\" out of bounds: should be in ", m_rgDimensions[i].toString (), "."));
						return -1;
					}
					
					// caluclate index
					nIdx = nIdx * m_rgDimensions[i].getSize () + nIdxValue;
					i++;
				}
			}
						
			return nIdx + m_nStartIndex;
		}
		
		/**
		 * Returns the grid type.
		 */
		public Specifier getSpecifier ()
		{
			return m_specType;
		}
		
		/**
		 * Returns <code>true</code> iff the grid is specified to be a constant grid, i.e. does not change in time.
		 */
		public boolean isConstant ()
		{
			return m_bIsConstant;
		}
		
		/**
		 * Returns the box, i.e., the dimensions of the stream in each direction.
		 */
		public Box getStreamDimensions ()
		{
            return m_boxStreamDimensions;
		}
	}
	
	
	///////////////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The result of the parser
	 */
	private StencilCalculation m_stencil;
	
	private ArithmeticUtil m_au;
	
	/**
	 * The data streams from which data are read
	 */
	private Map<String, StreamIndex> m_mapInputStreams;
	
	/**
	 * The data streams to which data are written
	 */
	private Map<String, StreamIndex> m_mapOutputStreams;
	
	/**
	 * Scalars: Parameters passed to the stencil operation or temporary variables defined
	 * within the stencil operation
	 */
	private Map<String, StencilCalculation.ParamType> m_mapScalars;
	
	/**
	 * Constants: variables that are assigned a constant value within the stencil operation
	 */
	private Map<String, Literal> m_mapConstants;
	
	/**
	 * List of arguments to the stencil operation (in the original order)
	 */
	private List<String> m_listStencilOperationArguments;
	
	/**
	 * A list of size parameters to the stencil definition (contained in the
	 * domain size definition and the optional size parameters to the grids)
	 */
	private List<NameID> m_listSizeParameters;
	
	/**
	 * The current total number of registered streams
	 */
	private int m_nTotalStreamsCount = 0;

	/**
	 * Has <code>t_max</code> been set?
	 */
	private boolean m_bIterateWhileSet = false;
	
	private CodeGenerationOptions m_options;

		
	///////////////////////////////////////////////////////////////////////////
	// Custom Implementation
	
	public void setOptions (CodeGenerationOptions options)
	{
		m_options = options;
		m_au = new ArithmeticUtil (m_options, errors);
	}
	
	public boolean hasErrors ()
	{
		return errors.count > 0;
	}
	
	/**
	 * Returns the stencil calculation object.
	 */
	public StencilCalculation getStencilCalculation ()
	{
		return m_stencil;
	}	
	
	/**
	 * Registers a stream along with its dimensions mapping it to the normalized internal representation.
	 * @param strIdentifier The identifier by which the stream is referred to in the source code
	 * @param box The dimensions of the stream box
	 * @param listDimensions List of dimensions of the stream
	 * @param sd The stream direction specifying whether this is an input or an output stream, i.e. is read from or written to
	 */
	private void registerStream (String strIdentifier, Specifier specType, boolean bIsConstant, Box box, List<Range> listDimensions, EStreamDirection sd)
	{
		// lazily create the maps
		if (m_mapInputStreams == null)
			m_mapInputStreams = new HashMap<> ();
		if (m_mapOutputStreams == null)
			m_mapOutputStreams = new HashMap<> ();
		if (m_listStencilOperationArguments == null)
			m_listStencilOperationArguments = new ArrayList<> ();
		
		if (CodeGeneratorUtil.isDimensionIdentifier (strIdentifier))
		{
			errors.SemErr (StringUtil.concat ("The identifier \"", strIdentifier, "\" represents a dimension and cannot be used as a grid identifier."));
			return;
		}
		
		Map<String, StreamIndex> map = sd == EStreamDirection.INPUT ? m_mapInputStreams : m_mapOutputStreams;
		if (!map.containsKey (strIdentifier))
			map.put (strIdentifier, new StreamIndex (strIdentifier, specType, bIsConstant, box, listDimensions, sd));
		else
			errors.SemErr (la.line, la.col, StringUtil.concat ("Duplicate declaration of grid \"", strIdentifier, "\""));
			
		if (sd == EStreamDirection.INPUT)
			m_listStencilOperationArguments.add (strIdentifier);
	}
			
	private void registerScalar (String strIdentifier, Specifier specType, List<Range> listDimensions, boolean bIsStencilArgument)
	{
		ensureScalarsMapCreated ();
		if (m_listStencilOperationArguments == null)
			m_listStencilOperationArguments = new ArrayList<> ();
			
		if (CodeGeneratorUtil.isDimensionIdentifier (strIdentifier))
		{
			errors.SemErr (StringUtil.concat ("The identifier \"", strIdentifier, "\" represents a dimension and cannot be used as a scalar identifier."));
			return;
		}

		if (!m_mapScalars.containsKey (strIdentifier))
		{
			m_mapScalars.put (strIdentifier, new StencilCalculation.ParamType (specType, listDimensions));
			
			DomainPointEnumerator dpe = new DomainPointEnumerator ();
			for (Range range : listDimensions)
				dpe.addDimension (new DomainPointEnumerator.MinMax (range.getStart (), range.getEnd ()));
				
			if (dpe.size () == 0)
			{
				if (bIsStencilArgument)
					m_listStencilOperationArguments.add (strIdentifier);
			}
			else
			{
				// convert multi-dimensional scalars to simple scalars with the index in their name
				for (int[] rgIdx : dpe)
				{
					String strIndexedIdentifier = m_au.getIndexedIdentifier (strIdentifier, rgIdx);
					m_mapScalars.put (strIndexedIdentifier, new StencilCalculation.ParamType (specType));
					if (bIsStencilArgument)
						m_listStencilOperationArguments.add (strIndexedIdentifier);
				}
			}
		}
	}
	
	private ExpressionData registerScalarAssignment (Stencil stencil, String strIdentifier, ExpressionData edRHS, Specifier specType)
	{
		Expression exprRHS = edRHS.getExpression ();
		Expression exprSimplified = null;
		if (exprRHS instanceof FloatLiteral)
			exprSimplified = exprRHS;
		else if (!containsStencilNode (exprRHS))
			exprSimplified = Symbolic.simplify (exprRHS);
			
		if (exprSimplified instanceof FloatLiteral || exprSimplified instanceof IntegerLiteral)
		{
			registerConstant (strIdentifier, (Literal) exprSimplified);
			return null;
		}

		registerScalar (strIdentifier, specType, new ArrayList<Range> (), false);
		StencilNode node = new StencilNode (strIdentifier, specType, null);
		stencil.addOutputNode (node);
		
		return new ExpressionData (
			new AssignmentExpression (node, AssignmentOperator.NORMAL, exprRHS),
			edRHS.getFlopsCount (),
			Symbolic.EExpressionType.EXPRESSION
		);
	}
	
	private void setParamDefaultValues (String strIdentifier, Map<IntArray, Stencil> mapDefaultValues)
	{
		for (IntArray arrIdx : mapDefaultValues.keySet ())
		{
			String strIndexedIdentifier = m_au.getIndexedIdentifier (strIdentifier, arrIdx.get ());
			StencilCalculation.ParamType pt = m_mapScalars.get (strIndexedIdentifier);
			
			if (pt == null)
				errors.SemErr (StringUtil.concat ("The stencil parameter ", strIndexedIdentifier, " has not yet been registered."));
			else
				pt.setDefaultValue (mapDefaultValues.get (arrIdx).getExpression ());
		}
	}
	
	private void setScalarInitValues (StencilBundle bundle, Specifier specType, String strIdentifier, Map<IntArray, Stencil> mapInit, boolean bOffsetInSpace)
	{
		for (IntArray arrIdx : mapInit.keySet ())
		{
			Stencil stencil = mapInit.get (arrIdx);
			String strIndexedIdentifier = m_au.getIndexedIdentifier (strIdentifier, arrIdx.get ());
			
			ExpressionData edSimplified = registerScalarAssignment (stencil, strIndexedIdentifier, stencil.getExpressionData (), specType);
			if (edSimplified != null)
				stencil.setExpression (edSimplified);
			
			try
			{
				if (!stencil.isEmpty ())
					bundle.addStencil (stencil, bOffsetInSpace);
			}
			catch (NoSuchMethodException e)
			{
				e.printStackTrace ();
			}
		}	
	}
	
	private void registerConstant (String strIdentifier, Literal litValue)
	{
		if (m_mapConstants == null)
			m_mapConstants = new HashMap<> ();
	       
		if (CodeGeneratorUtil.isDimensionIdentifier (strIdentifier))
		{
			errors.SemErr (StringUtil.concat ("The identifier \"", strIdentifier, "\" represents a dimension and cannot be used as a constant identifier."));
			return;
		}
	       
		m_mapConstants.put (strIdentifier, litValue.clone ());
	}
	
	private Literal getConstantValue (String strIdentifier)
	{
		return m_au.getConstantValue (strIdentifier, m_mapConstants);
	}
	
	/**
	 * Returns the internal stream index given the identifier in the source code and indices.
	 * @param strIdentifier The identifier in the source code
	 * @param listIndices A list of indices (subscripts)
	 * @param sd The stream direction specifying whether this is an input or an output stream, i.e. is read from or written to
	 * @return The internal stream index
	 */
	private int getStreamIndex (String strIdentifier, List<Expression> listIndices, EStreamDirection sd)
	{
		Map<String, StreamIndex> map = sd == EStreamDirection.INPUT ? m_mapInputStreams : m_mapOutputStreams;
		StreamIndex idx = map.get (strIdentifier);
		
		if (idx != null)
			return idx.getLinearIndex (strIdentifier, listIndices);
			
		errors.SemErr (la.line, la.col, StringUtil.concat ("The variable \"", strIdentifier, "\" has not been defined"));
		return -1;		
	}
	
	private StreamIndex getInputStream (String strIdentifier)
	{
		StreamIndex si = m_mapInputStreams.get (strIdentifier);
		if (si == null)
			errors.SemErr (la.line, la.col, StringUtil.concat ("The grid \"", strIdentifier, "\" has not been defined"));
		return si;
	}
	
	private StreamIndex getOutputStream (String strIdentifier)
	{
		StreamIndex si = m_mapOutputStreams.get (strIdentifier);
		if (si == null)
		{
			if (m_mapInputStreams.containsKey (strIdentifier))
				errors.SemErr (la.line, la.col, StringUtil.concat ("In order to assign a value to the grid \"", strIdentifier, "\", it must not be declared as \"const\""));
			else
				errors.SemErr (la.line, la.col, StringUtil.concat ("The grid \"", strIdentifier, "\" has not been defined"));
		}
		
		return si;
	}
	
	private StreamIndex getStream (String strIdentifier, EStreamDirection dir)
	{
		return dir == EStreamDirection.INPUT ? getInputStream (strIdentifier) : getOutputStream (strIdentifier);
	}
	
	private void ensureScalarsMapCreated ()
	{
		if (m_mapScalars == null)
		{
			m_mapScalars = new HashMap<> ();
		
			// add constants
			m_mapScalars.put ("PI", new StencilCalculation.ParamType (Specifier.DOUBLE));
		} 
	}
	
	private void checkParameterIndices (String strIdentifier, Expression exprParam, LocalVars lv)
	{
		ensureScalarsMapCreated ();
		StencilCalculation.ParamType param = m_mapScalars.get (strIdentifier);
		if (param == null)
		{
			// the key hasn't been found => parameter is not defined
			// check whether it is a built-in dimension identifier
			if ("t".equals (strIdentifier))
				return;
			if (CodeGeneratorUtil.getDimensionFromName (strIdentifier) >= 0)
				return;	// error if getDimensionFromName returns -1
			
			errors.SemErr (la.line, la.col, StringUtil.concat ("The parameter \"", strIdentifier, "\" has not been defined"));
			return;
		}
		
		// check bounds
		if (exprParam instanceof ArrayAccess)
		{
			ArrayAccess arr = (ArrayAccess) exprParam;
			
			// check whether the dimensions agree
			if (param.getDimensionsCount () != arr.getNumIndices ())
			{
				errors.SemErr (la.line, la.col, StringUtil.concat ("The parameter dimension of ", strIdentifier, " does not agree with its definition: should be ", param.getDimensionsCount (), ", but is ", arr.getNumIndices (), "."));
				return;
			}
			
			// check bounds
			int i = 0;
			for (Range range : param.getRanges ())
			{
				Integer nIdx = ExpressionUtil.getIntegerValueEx (arr.getIndex (i));
				if (nIdx != null)
				{
					if (nIdx < range.getStart () || nIdx > range.getEnd ())
					{
						errors.SemErr (la.line, la.col, StringUtil.concat ("Index in dimension ", i, " of \"", exprParam.toString (), "\" out of bounds: should be in ", range.toString (), "."));
						return;
					}
				}
				else
				{
					// check whether all indices are local variables
					for (DepthFirstIterator it = new DepthFirstIterator (arr.getIndex (i)); it.hasNext (); )
					{
						Object o = it.next ();
						if (o instanceof IDExpression)
						{
							String strName = ((IDExpression) o).getName ();
							if (!lv.hasVariable (strName))
							{
								errors.SemErr (la.line, la.col, StringUtil.concat ("The index \"", strName, "\" has not been declared."));
								return;
							}
						}
					}
					
					// TODO: check symbolic indices
				}
					
				i++;
			}
		}
		else
		{
			// not an array access => the number of dimensions must be 0
			if (param.getDimensionsCount () != 0)
				errors.SemErr (la.line, la.col, StringUtil.concat ("The parameter dimension of ", strIdentifier, " does not agree with its definition: should be ", param.getDimensionsCount (), ", but is 0."));
		}
	}
	
	private void addSizeParameters (Box box)
	{
		for (Expression expr : box.getMin ())
			addSizeParameter (expr);
		for (Expression expr : box.getMax ())
			addSizeParameter (expr);
	}
	
	private void addSizeParameter (Expression expr)
	{	
		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
		{
			Object o = it.next ();
			if (o instanceof NameID)
			{
				if (m_listSizeParameters == null)
					m_listSizeParameters = new LinkedList<> ();
			
				if (!m_listSizeParameters.contains (o))
				{
					m_listSizeParameters.add ((NameID) o);
					
					ensureScalarsMapCreated ();
					m_mapScalars.put (((NameID) o).getName (), new StencilCalculation.ParamType (Globals.SPECIFIER_SIZE));
				}
			}
		}		
	}
		
	private IntArray getStart (List<Range> list)
	{
		int[] rgStart = new int[list.size ()];
		int i = 0;
		
		for (Range range : list)
		{
			rgStart[i] = range.getStart ();
			i++;
		}
		
		return new IntArray (rgStart);
	}

	/**
	 * Adds the parameters to the StencilCalculation object 
	 */	
	private void setStencilOperationArguments ()
	{
        if (m_listStencilOperationArguments != null)
        {
            for (String strIdentifier : m_listStencilOperationArguments)
            {
                StreamIndex idx =  m_mapInputStreams.get (strIdentifier);
                m_stencil.addStencilOperationArgument (
                    strIdentifier,
                    idx == null ? null : idx.getStreamDimensions (),
                    m_mapScalars == null ? null : m_mapScalars.get (strIdentifier));
            }
        }
        
        if (m_listSizeParameters != null)
        {
        	for (NameID nid : m_listSizeParameters)
        		m_stencil.addSizeParameter (nid);
        }
	}
	
	private void checkDefinitions ()
	{
		if (m_stencil.getDomainSize () == null)
			errors.SemErr ("No domainsize defined in the stencil specification.");
		//if (!m_bIterateWhileSet)
		//	errors.SemErr ("No 'iterate while ...' defined in the stencil specification.");
		if (m_stencil.getStencilBundle () == null)
			errors.SemErr ("No stencil operation defined in the stencil specification.");
	}
	
	private static boolean containsStencilNode (Expression expr)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
			if (it.next () instanceof StencilNode)
				return true;
		return false;
	}
	
	private static List<Expression> getNegSpatialCoords (int nDimensionality)
	{
		List<Expression> l = new ArrayList<> (nDimensionality);
		for (int i = 0; i < nDimensionality; i++)
			l.add (new UnaryExpression (UnaryOperator.MINUS, new NameID (CodeGeneratorUtil.getDimensionName (i))));
		return l;
	}
	
	private static List<Expression> getNegTemporalCoord ()
	{
		List<Expression> l = new ArrayList<> (1);
		l.add (new UnaryExpression (UnaryOperator.MINUS, new NameID ("t")));
		return l;
	}
			

	///////////////////////////////////////////////////////////////////////////
	// LL(1) Conflict Resolvers
	
	/**
	 * Determines whether the next token is a grid variable.
	 * @return <code>true</code> iff the next token is a grid variable
	 */
	private boolean isGridVariable ()
	{
		String strIdentifier = la.val;
		boolean bResult = m_mapInputStreams == null ? false : m_mapInputStreams.containsKey (strIdentifier);
		if (bResult)
			return true;
		bResult = m_mapOutputStreams == null ? false : m_mapOutputStreams.containsKey (strIdentifier);
		if (bResult)
			return true;
			
		return false;
	}
	
	/**
	 * Determines whether the next token is a const grid variable.
	 * @return <code>true</code> iff the next token is a const grid variable
	 */
	private boolean isConstGridVariable (String strIdentifier)
	{
		boolean bResult = m_mapInputStreams == null ? false : m_mapInputStreams.containsKey (strIdentifier);
		if (bResult)
		{
			StreamIndex idx = m_mapInputStreams.get (strIdentifier);
			return idx.isConstant ();
		}
			
		return false;
	}

	/**
	 * LL1 conflict resolver for function calls.
	 */
	private boolean isFunctionCall ()
	{
		Token token = scanner.Peek ();
		scanner.ResetPeek ();
		return token.val.equals ("(");
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
	
	void StencilSpecification() {
		while (!(la.kind == 0 || la.kind == 4)) {SynErr(57); Get();}
		Expect(4);
		Expect(1);
		m_stencil = new StencilCalculation (t.val); 
		Expect(5);
		StencilOperationParamList();
		Expect(6);
		Expect(7);
		while (StartOf(1)) {
			switch (la.kind) {
			case 9: {
				StencilOptions();
				break;
			}
			case 15: {
				Expression exprIterateWhile = StencilIterateWhile();
				if (m_bIterateWhileSet) errors.SemErr (la.line, la.col, "Found multiple 'iterate while ...' definitions. All but the first one are ignored."); else { m_stencil.setIterateWhile (exprIterateWhile); m_bIterateWhileSet = true; } 
				break;
			}
			case 17: {
				Box boxGrid = StencilDomainSize();
				if (m_stencil.getDomainSize () != null) errors.SemErr (la.line, la.col, "Found multiple grid size definitions. All but the first one are ignored."); else m_stencil.setDomainSize (boxGrid); 
				break;
			}
			case 18: {
				StencilBundle bundle = StencilOperation();
				if (m_stencil.getStencilBundle () != null) errors.SemErr (la.line, la.col, "Found multiple stencil definitions. All but the first are ignored."); else m_stencil.setStencil (bundle); 
				break;
			}
			case 19: {
				StencilBundle bundle = StencilBoundaries();
				if (m_stencil.getBoundaries () != null) errors.SemErr ("Found multiple boundaries definitions. All but the first are ignored."); else m_stencil.setBoundaries (bundle); 
				break;
			}
			case 20: {
				StencilBundle bundle = StencilInitial();
				if (m_stencil.getInitialization () != null) errors.SemErr ("Found multiple initializations. All but the first are ignored."); else m_stencil.setInitialization (bundle); 
				break;
			}
			}
		}
		Expect(8);
		setStencilOperationArguments (); checkDefinitions (); if (!m_bIterateWhileSet) m_stencil.setMaxIterations (new IntegerLiteral (1)); 
	}

	void StencilOperationParamList() {
		StencilOperationParam();
		while (la.kind == 21) {
			Get();
			StencilOperationParam();
		}
	}

	void StencilOptions() {
		while (!(la.kind == 0 || la.kind == 9)) {SynErr(58); Get();}
		Expect(9);
		Expect(7);
		while (la.kind == 11) {
			StencilOptionsCompatibility();
			Expect(10);
		}
		Expect(8);
	}

	Expression  StencilIterateWhile() {
		Expression  exprIterateWhile;
		while (!(la.kind == 0 || la.kind == 15)) {SynErr(59); Get();}
		Expect(15);
		Expect(16);
		Stencil stencilDummy = new Stencil (); 
		ExpressionData edIterateWhile = LogicalExpression(stencilDummy, null, true, false, true, true);
		exprIterateWhile = edIterateWhile.getExpression (); 
		Expect(10);
		return exprIterateWhile;
	}

	Box  StencilDomainSize() {
		Box  boxGrid;
		while (!(la.kind == 0 || la.kind == 17)) {SynErr(60); Get();}
		Expect(17);
		Expect(12);
		boxGrid = StencilBox();
		Expect(10);
		return boxGrid;
	}

	StencilBundle  StencilOperation() {
		StencilBundle  bundle;
		bundle = new StencilBundle (m_stencil); 
		while (!(la.kind == 0 || la.kind == 18)) {SynErr(61); Get();}
		Expect(18);
		Body(bundle, true, true, false);
		return bundle;
	}

	StencilBundle  StencilBoundaries() {
		StencilBundle  bundle;
		bundle = new StencilBundle (m_stencil); 
		while (!(la.kind == 0 || la.kind == 19)) {SynErr(62); Get();}
		Expect(19);
		Body(bundle, false, true, false);
		StencilSpecificationAnalyzer.normalizeStencilNodesForBoundariesAndIntial (bundle); 
		return bundle;
	}

	StencilBundle  StencilInitial() {
		StencilBundle  bundle;
		bundle = new StencilBundle (m_stencil); 
		while (!(la.kind == 0 || la.kind == 20)) {SynErr(63); Get();}
		Expect(20);
		Body(bundle, false, false, true);
		StencilSpecificationAnalyzer.normalizeStencilNodesForBoundariesAndIntial (bundle); 
		return bundle;
	}

	void StencilOptionsCompatibility() {
		while (!(la.kind == 0 || la.kind == 11)) {SynErr(64); Get();}
		Expect(11);
		Expect(12);
		if (la.kind == 13) {
			Get();
			m_options.setCompatibility (CodeGenerationOptions.ECompatibility.C); 
		} else if (la.kind == 14) {
			Get();
			m_options.setCompatibility (CodeGenerationOptions.ECompatibility.FORTRAN); 
		} else SynErr(65);
	}

	ExpressionData  LogicalExpression(Stencil stencil, LocalVars lv, boolean bIsDeclaration, boolean bIsInteger, boolean bOffsetInSpace, boolean bOffsetInTime) {
		ExpressionData  expr;
		expr = OrExpression(stencil, lv, bIsDeclaration, bIsInteger, bOffsetInSpace, bOffsetInTime);
		return expr;
	}

	Box  StencilBox() {
		Box  box;
		Expect(5);
		List<Expression> listMin = new ArrayList<> (); List<Expression> listMax = new ArrayList<> (); 
		Box box1 = StencilBoxCoordinate();
		listMin.add (box1.getMin ().getCoord (0)); listMax.add (box1.getMax ().getCoord (0)); 
		while (la.kind == 21) {
			Get();
			box1 = StencilBoxCoordinate();
			listMin.add (box1.getMin ().getCoord (0)); listMax.add (box1.getMax ().getCoord (0)); 
		}
		Expression[] rgMin = new Expression[listMin.size ()]; Expression[] rgMax = new Expression[listMax.size ()]; listMin.toArray (rgMin); listMax.toArray (rgMax); 
		Expect(6);
		box = new Box (new Point (rgMin), new Point (rgMax)); addSizeParameters (box); 
		return box;
	}

	void Body(StencilBundle bundle, boolean bOffsetInSpace, boolean bOffsetInTime, boolean bIsInitialization) {
		Expect(7);
		while (StartOf(2)) {
			AssignmentStatement(bundle, bOffsetInSpace, bOffsetInTime, bIsInitialization);
		}
		Expect(8);
	}

	void AssignmentStatement(StencilBundle bundle, boolean bOffsetInSpace, boolean bOffsetInTime, boolean bIsInitialization) {
		if (la.kind == 1) {
			StencilAssignment(bundle, bOffsetInSpace, bOffsetInTime, bIsInitialization);
		} else if (la.kind == 24 || la.kind == 25) {
			ScalarAssignment(bundle, bOffsetInSpace, bOffsetInTime, bIsInitialization);
		} else if (la.kind == 32) {
			PredicateAssignment();
		} else SynErr(66);
		while (!(la.kind == 0 || la.kind == 10)) {SynErr(67); Get();}
		Expect(10);
	}

	Box  StencilBoxCoordinate() {
		Box  box;
		ExpressionData edMin = StencilExpression(null, null, true, true, false, false);
		Expect(22);
		ExpressionData edMax = StencilExpression(null, null, true, true, false, false);
		box = new Box (new Point (edMin.getExpression ()), new Point (edMax.getExpression ())); 
		return box;
	}

	ExpressionData  StencilExpression(Stencil stencil, LocalVars lv, boolean bIsDeclaration, boolean bIsInteger, boolean bOffsetInSpace, boolean bOffsetInTime) {
		ExpressionData  expr;
		LocalVars lvNew = lv; 
		if (la.kind == 7) {
			lvNew = SetComprehension(lv);
		}
		ExpressionData exprAdd = AdditiveExpression(stencil, lvNew, bIsDeclaration, bIsInteger, bOffsetInSpace, bOffsetInTime);
		expr = exprAdd; /*expr = exprAdd == null ? null : NormalExpression.simplify (exprAdd);*/ 
		return expr;
	}

	void StencilOperationParam() {
		boolean bIsGridVariable = true; boolean bIsConstant = false; Specifier specVarType = null; Box boxGrid = null; 
		if (la.kind == 23) {
			Get();
			bIsConstant = true; 
		}
		if (la.kind == 24) {
			Get();
			specVarType = Specifier.FLOAT; 
		} else if (la.kind == 25) {
			Get();
			specVarType = Specifier.DOUBLE; 
		} else if (la.kind == 26) {
			Get();
			specVarType = Specifier.INT; 
		} else if (la.kind == 27) {
			Get();
			specVarType = Specifier.LONG; 
		} else SynErr(68);
		if (la.kind == 28) {
			Get();
		} else if (la.kind == 29) {
			Get();
			bIsGridVariable = false; 
		} else SynErr(69);
		Expect(1);
		String strIdentifier = t.val; List<Range> listDimensions = new ArrayList<> (); 
		if (la.kind == 5) {
			boxGrid = StencilBox();
			if (!bIsGridVariable) errors.SemErr (la.line, la.col, "Parameters cannot have a box size declaration"); 
		}
		if (la.kind == 30) {
			DimensionDeclarator(listDimensions);
		}
		if (bIsGridVariable) { 
		registerStream (strIdentifier, specVarType, bIsConstant, boxGrid, listDimensions, EStreamDirection.INPUT); 
		if (!bIsConstant) 
		registerStream (strIdentifier, specVarType, bIsConstant, boxGrid, listDimensions, EStreamDirection.OUTPUT); 
		} 
		else { 
		registerScalar (strIdentifier, specVarType, listDimensions, true); 
		m_stencil.preAddStencilOperationParameter (strIdentifier, specVarType); 
		} 
		if (la.kind == 12) {
			Get();
			if (bIsGridVariable) errors.SemErr (la.line, la.col, "Grid variables can't be assigned a default value. To initialize grids, use the \"initial\" method."); 
			Map<IntArray, Stencil> mapInit = new HashMap<> (); 
			ValueInitializer(mapInit, listDimensions, getStart (listDimensions), listDimensions.size () - 1, false, false);
			setParamDefaultValues (strIdentifier, mapInit); 
		}
	}

	void DimensionDeclarator(List<Range> listDimensions ) {
		Expect(30);
		Range range = RangeLiteral();
		listDimensions.add (range); 
		while (la.kind == 21) {
			while (!(la.kind == 0 || la.kind == 21)) {SynErr(70); Get();}
			Get();
			range = RangeLiteral();
			listDimensions.add (range); 
		}
		while (!(la.kind == 0 || la.kind == 31)) {SynErr(71); Get();}
		Expect(31);
	}

	void ValueInitializer(Map<IntArray, Stencil> mapInitializer, List<Range> listDimensions, IntArray arrCurrCoord, int nDim, boolean bOffsetInSpace, boolean bOffsetInTime ) {
		if (la.kind == 7) {
			Get();
			IntArray arr = new IntArray (arrCurrCoord.get (), true); arr.set (nDim, listDimensions.get (nDim).getStart ()); 
			ValueInitializer(mapInitializer, listDimensions, arr, nDim - 1, bOffsetInSpace, bOffsetInTime);
			while (la.kind == 21) {
				arr = new IntArray (arr.get (), true); arr.set (nDim, arr.get (nDim) + 1); 
				Get();
				ValueInitializer(mapInitializer, listDimensions, arr, nDim - 1, bOffsetInSpace, bOffsetInTime);
			}
			while (!(la.kind == 0 || la.kind == 8)) {SynErr(72); Get();}
			Expect(8);
		} else if (StartOf(3)) {
			Stencil stencil = new Stencil (); 
			ExpressionData ed = StencilExpression(stencil, null, false, false, bOffsetInSpace, bOffsetInTime);
			if (ed != null) stencil.setExpression (ed); 
			mapInitializer.put (arrCurrCoord, stencil); 
		} else SynErr(73);
	}

	Range  RangeLiteral() {
		Range  range;
		int nEnd = -1; boolean bHasEnd = false; 
		int nStart = PosNegIntegerLiteral();
		if (la.kind == 22) {
			Get();
			nEnd = PosNegIntegerLiteral();
			bHasEnd = true; 
		}
		range = bHasEnd ? new Range (nStart, nEnd) : new Range (0, nStart - 1); 
		return range;
	}

	void StencilAssignment(StencilBundle bundle, boolean bOffsetInSpace, boolean bOffsetInTime, boolean bIsInitialization) {
		Stencil stencil = new Stencil (); 
		StencilNode nodeLHS = StencilIdentifier(null, bIsInitialization ? EStreamDirection.INPUT : EStreamDirection.OUTPUT, bOffsetInSpace, bOffsetInTime);
		stencil.addOutputNode (nodeLHS); 
		Expect(12);
		ExpressionData edRHS = StencilExpression(stencil, null, false, false, bOffsetInSpace, bOffsetInTime);
		stencil.setExpression ((nodeLHS == null || edRHS == null) ? null : new ExpressionData (new AssignmentExpression (nodeLHS, AssignmentOperator.NORMAL, edRHS.getExpression ()), edRHS.getFlopsCount (), Symbolic.EExpressionType.EXPRESSION)); 
		try { if (!stencil.isEmpty ()) bundle.addStencil (stencil, bOffsetInSpace); } catch (NoSuchMethodException e) { e.printStackTrace (); } 
	}

	void ScalarAssignment(StencilBundle bundle, boolean bOffsetInSpace, boolean bOffsetInTime, boolean bIsInitialization) {
		Specifier specType = Specifier.FLOAT; List<Range> listDimensions = new ArrayList<> (); Map<IntArray, Stencil> mapInit = new HashMap<> (); 
		if (la.kind == 24) {
			Get();
		} else if (la.kind == 25) {
			Get();
			specType = Specifier.DOUBLE; 
		} else SynErr(74);
		Expect(1);
		String strIdentifier = t.val; 
		if (la.kind == 30) {
			DimensionDeclarator(listDimensions);
		}
		Expect(12);
		registerScalar (strIdentifier, specType, listDimensions, false); 
		ValueInitializer(mapInit, listDimensions, getStart (listDimensions), listDimensions.size () - 1, bOffsetInSpace, bOffsetInTime);
		setScalarInitValues (bundle, specType, strIdentifier, mapInit, bOffsetInSpace); 
	}

	void PredicateAssignment() {
		Expect(32);
		Expect(1);
		Expect(12);
		ExpressionData expr = LogicalExpression(null, null, true, true, false, false);
	}

	StencilNode  StencilIdentifier(LocalVars lv, EStreamDirection dir, boolean bOffsetInSpace, boolean bOffsetInTime) {
		StencilNode  node;
		Expect(1);
		String strIdentifier = t.val; node = null; Index index = new Index (); boolean bVectorIndexSet = false; 
		Expect(30);
		ExpressionData exprIdx0 = StencilExpression(null, lv, false, true, false, false);
		int nMode = 0; List<Expression> listIndices = new ArrayList<> (); listIndices.add (exprIdx0.getExpression ()); 
		while (la.kind == 10 || la.kind == 21) {
			while (!(la.kind == 0 || la.kind == 10 || la.kind == 21)) {SynErr(75); Get();}
			if (la.kind == 21) {
				Get();
			} else {
				Get();
				switch (nMode) { 
				case 0: index.setSpaceIndex (listIndices, bOffsetInSpace ? getNegSpatialCoords (listIndices.size ()) : null); if (isConstGridVariable (strIdentifier)) nMode++; break; 
				case 1: index.setTimeIndex (listIndices, bOffsetInTime ? getNegTemporalCoord () : null); break; 
				case 2: index.setVectorIndex (getStreamIndex (strIdentifier, listIndices, dir)); bVectorIndexSet = true; break; 
				default: errors.SemErr (la.line, la.col, "Grids can't have more than 3 different index types."); } 
				nMode++; listIndices = new ArrayList<> (); 
			}
			ExpressionData exprIdx1 = StencilExpression(null, lv, false, true, false, false);
			listIndices.add (exprIdx1.getExpression ()); 
		}
		switch (nMode) { 
		case 0: index.setSpaceIndex (listIndices, bOffsetInSpace ? getNegSpatialCoords (listIndices.size ()) : null); break; 
		case 1: index.setTimeIndex (listIndices, bOffsetInTime ? getNegTemporalCoord () : null); break; 
		case 2: index.setVectorIndex (getStreamIndex (strIdentifier, listIndices, dir)); bVectorIndexSet = true; break; 
		default: errors.SemErr (la.line, la.col, "Grids can't have more than 3 different index types."); } 
		if (!bVectorIndexSet) index.setVectorIndex (getStreamIndex (strIdentifier, null, dir)); 
		StreamIndex si = getStream (strIdentifier, dir); 
		node = new StencilNode (strIdentifier, si == null ? Specifier.DOUBLE : si.getSpecifier (), index); 
		if (la.kind == 33) {
			Get();
			ExpressionData exprConstr0 = LogicalExpression(null, lv, false, true, false, false);
			node.setConstraint (exprConstr0.getExpression ()); 
			while (la.kind == 21) {
				Get();
				ExpressionData exprConstr1 = LogicalExpression(null, lv, false, true, false, false);
				node.addConstraint (exprConstr1.getExpression ()); 
			}
		}
		while (!(la.kind == 0 || la.kind == 31)) {SynErr(76); Get();}
		Expect(31);
		return node;
	}

	Expression  ScalarIdentifier(LocalVars lv, boolean bIsDecl, boolean bIsInteger) {
		Expression  exprParam;
		Expect(1);
		String strIdentifier = t.val; Literal litValue = getConstantValue (strIdentifier); exprParam = litValue == null ? new NameID (strIdentifier) : litValue; int[] rgIdx = null; 
		if (la.kind == 30) {
			Get();
			if ((exprParam instanceof FloatLiteral) || (exprParam instanceof IntegerLiteral)) errors.SemErr (la.line, la.col, "Cannot subscript a scalar value"); List<Expression> listIndices = new ArrayList<> (); 
			ExpressionData exprIdx0 = StencilExpression(null, lv, false, true, false, false);
			listIndices.add (Symbolic.simplify (exprIdx0.getExpression ())); 
			while (la.kind == 21) {
				while (!(la.kind == 0 || la.kind == 21)) {SynErr(77); Get();}
				Get();
				ExpressionData exprIdx1 = StencilExpression(null, lv, false, true, false, false);
				listIndices.add (Symbolic.simplify (exprIdx1.getExpression ())); 
			}
			while (!(la.kind == 0 || la.kind == 31)) {SynErr(78); Get();}
			Expect(31);
			rgIdx = m_au.asIntArray (listIndices); 
			if (rgIdx == null) exprParam = new ArrayAccess (new NameID (strIdentifier), listIndices); 
			else { 
			String strIndexedIdentifier = m_au.getIndexedIdentifier (strIdentifier, rgIdx); 
			litValue = getConstantValue (strIndexedIdentifier); 
			if (litValue != null) exprParam = litValue; else exprParam = new NameID (strIndexedIdentifier); 
			} 
		}
		if (rgIdx == null && !(exprParam instanceof FloatLiteral) && !(exprParam instanceof IntegerLiteral) && !bIsDecl && (lv == null || !lv.hasVariable (strIdentifier))) 
		checkParameterIndices (strIdentifier, exprParam, lv); 
		return exprParam;
	}

	LocalVars  SetComprehension(LocalVars lv) {
		LocalVars  lvNew;
		Expect(7);
		lvNew = lv == null ? new LocalVars () : lv; 
		RangeDeclaration(lvNew);
		while (la.kind == 21) {
			Get();
			RangeDeclaration(lvNew);
		}
		if (la.kind == 33) {
			Get();
			ExpressionData exprConstr0 = LogicalExpression(null, lvNew, false, true, false, false);
			lvNew.addPredicate (exprConstr0.getExpression ()); 
			while (la.kind == 21) {
				Get();
				ExpressionData exprConstr1 = LogicalExpression(null, lvNew, false, true, false, false);
				lvNew.addPredicate (exprConstr1.getExpression ()); 
			}
		}
		Expect(8);
		return lvNew;
	}

	ExpressionData  AdditiveExpression(Stencil stencil, LocalVars lv, boolean bIsDecl, boolean bIsInteger, boolean bOffsetInSpace, boolean bOffsetInTime) {
		ExpressionData  expr;
		List<ExpressionData> listSummands = new LinkedList<> (); boolean bAdd = true; expr = null; 
		ExpressionData expr0 = MultiplicativeExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
		listSummands.add (expr0); 
		while (la.kind == 47 || la.kind == 48) {
			if (la.kind == 47) {
				Get();
				bAdd = true; 
			} else {
				Get();
				bAdd = false; expr = m_au.sum (listSummands, bIsInteger); listSummands.clear (); 
			}
			ExpressionData expr1 = MultiplicativeExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
			if (bAdd) listSummands.add (expr1); else listSummands.add (m_au.subtract (expr.clone (), expr1, bIsInteger)); 
		}
		expr = m_au.sum (listSummands, bIsInteger); 
		return expr;
	}

	void RangeDeclaration(LocalVars lv) {
		Expect(1);
		String strVarName = t.val; 
		Expect(12);
		Range range = RangeLiteral();
		lv.addVariable (strVarName, range); 
	}

	ExpressionData  OrExpression(Stencil stencil, LocalVars lv, boolean bIsDecl, boolean bIsInteger, boolean bOffsetInSpace, boolean bOffsetInTime) {
		ExpressionData  expr;
		ExpressionData expr0 = AndExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
		expr = expr0; 
		while (la.kind == 34 || la.kind == 35) {
			if (la.kind == 34) {
				Get();
			} else {
				Get();
			}
			ExpressionData expr1 = AndExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
			expr = new ExpressionData (new BinaryExpression (expr.getExpression (), BinaryOperator.LOGICAL_OR, expr1.getExpression ()), expr.getFlopsCount () + expr1.getFlopsCount (), Symbolic.EExpressionType.EXPRESSION); 
		}
		return expr;
	}

	ExpressionData  AndExpression(Stencil stencil, LocalVars lv, boolean bIsDecl, boolean bIsInteger, boolean bOffsetInSpace, boolean bOffsetInTime) {
		ExpressionData  expr;
		ExpressionData expr0 = NotExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
		expr = expr0; 
		while (la.kind == 36 || la.kind == 37) {
			if (la.kind == 36) {
				Get();
			} else {
				Get();
			}
			ExpressionData expr1 = NotExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
			expr = new ExpressionData (new BinaryExpression (expr.getExpression (), BinaryOperator.LOGICAL_AND, expr1.getExpression ()), expr.getFlopsCount () + expr1.getFlopsCount (), Symbolic.EExpressionType.EXPRESSION); 
		}
		return expr;
	}

	ExpressionData  NotExpression(Stencil stencil, LocalVars lv, boolean bIsDecl, boolean bIsInteger, boolean bOffsetInSpace, boolean bOffsetInTime) {
		ExpressionData  expr;
		UnaryOperator op = null; 
		if (la.kind == 38 || la.kind == 39) {
			if (la.kind == 38) {
				Get();
				op = UnaryOperator.LOGICAL_NEGATION; 
			} else {
				Get();
				op = UnaryOperator.LOGICAL_NEGATION; 
			}
		}
		expr = ComparisonExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
		if (op != null) expr = new ExpressionData (new UnaryExpression (op, expr.getExpression ()), expr.getFlopsCount (), Symbolic.EExpressionType.EXPRESSION); 
		return expr;
	}

	ExpressionData  ComparisonExpression(Stencil stencil, LocalVars lv, boolean bIsDecl, boolean bIsInteger, boolean bOffsetInSpace, boolean bOffsetInTime) {
		ExpressionData  expr;
		BinaryOperator op0 = null; BinaryOperator op1 = null; 
		ExpressionData expr0 = AdditiveExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
		expr = expr0; 
		if (StartOf(4)) {
			op0 = ComparisonOperator();
			ExpressionData expr1 = AdditiveExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
			expr = new ExpressionData (new BinaryExpression (expr.getExpression (), op0, expr1.getExpression ()), expr.getFlopsCount () + expr1.getFlopsCount (), Symbolic.EExpressionType.EXPRESSION); 
			if (StartOf(4)) {
				op1 = ComparisonOperator();
				ExpressionData expr2 = AdditiveExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
				expr = new ExpressionData (new BinaryExpression (expr.getExpression (), BinaryOperator.LOGICAL_AND, new BinaryExpression (expr1.getExpression ().clone (), op1, expr2.getExpression ())), expr.getFlopsCount () + expr2.getFlopsCount (), Symbolic.EExpressionType.EXPRESSION); 
			}
		}
		return expr;
	}

	BinaryOperator  ComparisonOperator() {
		BinaryOperator  op;
		op = null; 
		switch (la.kind) {
		case 40: {
			Get();
			op = BinaryOperator.COMPARE_LT; 
			break;
		}
		case 41: {
			Get();
			op = BinaryOperator.COMPARE_LE; 
			break;
		}
		case 42: {
			Get();
			op = BinaryOperator.COMPARE_EQ; 
			break;
		}
		case 43: {
			Get();
			op = BinaryOperator.COMPARE_GE; 
			break;
		}
		case 44: {
			Get();
			op = BinaryOperator.COMPARE_GT; 
			break;
		}
		case 45: {
			Get();
			op = BinaryOperator.COMPARE_NE; 
			break;
		}
		case 46: {
			Get();
			op = BinaryOperator.COMPARE_NE; 
			break;
		}
		default: SynErr(79); break;
		}
		return op;
	}

	ExpressionData  MultiplicativeExpression(Stencil stencil, LocalVars lv, boolean bIsDecl, boolean bIsInteger, boolean bOffsetInSpace, boolean bOffsetInTime) {
		ExpressionData  expr;
		List<ExpressionData> listFactors = new LinkedList<> (); BinaryOperator op = null; expr = null; 
		ExpressionData expr0 = UnarySignExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
		listFactors.add (expr0); 
		while (la.kind == 49 || la.kind == 50 || la.kind == 51) {
			if (la.kind == 49) {
				Get();
				op = BinaryOperator.MULTIPLY; 
			} else if (la.kind == 50) {
				Get();
				op = BinaryOperator.DIVIDE; 
			} else {
				Get();
				if (!bIsInteger) { errors.SemErr (la.line, la.col, "The % operator is only defined for integers"); } 
				else { op = BinaryOperator.MODULUS; expr = m_au.product (listFactors, bIsInteger); listFactors.clear (); } 
			}
			ExpressionData expr1 = UnarySignExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
			if (op == BinaryOperator.MULTIPLY) listFactors.add (expr1); 
			else if (op == BinaryOperator.DIVIDE) { 
			if ((expr1.getExpression () instanceof Literal) && !bIsInteger) 
			listFactors.add (new ExpressionData (new FloatLiteral (1.0 / ExpressionUtil.getFloatValue (expr1.getExpression ())), 0, Symbolic.EExpressionType.EXPRESSION)); 
			else { 
			expr = m_au.product (listFactors, bIsInteger); listFactors.clear (); 
			listFactors.add (m_au.divide (expr.clone (), expr1, bIsInteger)); 
			} 
			} 
			else if (op == BinaryOperator.MODULUS) listFactors.add (m_au.modulus (expr.clone (), expr1, bIsInteger)); 
			else errors.SemErr (la.line, la.col, "No multiplicative operator defined"); 
		}
		expr = m_au.product (listFactors, bIsInteger); 
		return expr;
	}

	ExpressionData  UnarySignExpression(Stencil stencil, LocalVars lv, boolean bIsDecl, boolean bIsInteger, boolean bOffsetInSpace, boolean bOffsetInTime) {
		ExpressionData  expr;
		boolean bIsNegative = false; 
		if (la.kind == 47 || la.kind == 48) {
			if (la.kind == 47) {
				Get();
			} else {
				Get();
				bIsNegative = true; 
			}
		}
		ExpressionData expr1 = ExponentExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
		if (!bIsNegative) expr = expr1; else { 
		if (expr1.getExpression () instanceof FloatLiteral) expr = new ExpressionData (new FloatLiteral (-((FloatLiteral) expr1.getExpression ()).getValue ()), 0, Symbolic.EExpressionType.EXPRESSION); 
		else if (expr1.getExpression () instanceof IntegerLiteral) expr = new ExpressionData (new IntegerLiteral (-((IntegerLiteral) expr1.getExpression ()).getValue ()), 0, Symbolic.EExpressionType.EXPRESSION); 
		else expr = new ExpressionData (new UnaryExpression (UnaryOperator.MINUS, expr1.getExpression ()), expr1.getFlopsCount () + 1, Symbolic.EExpressionType.EXPRESSION); 
		} 
		return expr;
	}

	ExpressionData  ExponentExpression(Stencil stencil, LocalVars lv, boolean bIsDecl, boolean bIsInteger, boolean bOffsetInSpace, boolean bOffsetInTime) {
		ExpressionData  expr;
		expr = UnaryExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
		while (la.kind == 52) {
			Get();
			ExpressionData expr1 = UnaryExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
			expr = expr == null ? null : ExpressionUtil.createExponentExpression (expr.clone (), expr1, null); 
		}
		return expr;
	}

	ExpressionData  UnaryExpression(Stencil stencil, LocalVars lv, boolean bIsDecl, boolean bIsInteger, boolean bOffsetInSpace, boolean bOffsetInTime) {
		ExpressionData  expr;
		expr = null; 
		if (StartOf(5)) {
			double fValue = NumberLiteral(bIsInteger);
			expr = new ExpressionData (m_au.createLiteral (fValue, bIsInteger), 0, Symbolic.EExpressionType.EXPRESSION); 
		} else if (isGridVariable ()) {
			StencilNode node = StencilIdentifier(lv, EStreamDirection.INPUT, bOffsetInSpace, bOffsetInTime);
			expr = new ExpressionData (node, 0, Symbolic.EExpressionType.EXPRESSION); 
			if (lv == null) stencil.addInputNode (node); else for (ExpressionData edNode : lv.expand (node)) stencil.addInputNode ((StencilNode) edNode.getExpression ()); 
		} else if (la.kind == 5) {
			ExpressionData exprBracketed = BracketedExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
			expr = exprBracketed; 
		} else if (isFunctionCall ()) {
			ExpressionData exprFnxValue = FunctionCall(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
			expr = exprFnxValue; 
		} else if (la.kind == 1) {
			Expression exprParam = ScalarIdentifier(lv, bIsDecl, bIsInteger);
			expr = new ExpressionData (exprParam, 0, Symbolic.EExpressionType.EXPRESSION); 
		} else SynErr(80);
		return expr;
	}

	double  NumberLiteral(boolean bIsInteger) {
		double  fValue;
		fValue = 0.0; 
		if (!bIsInteger) {
			fValue = FloatLiteral();
		} else if (la.kind == 2) {
			Get();
			fValue = Integer.parseInt (t.val); 
		} else SynErr(81);
		return fValue;
	}

	ExpressionData  BracketedExpression(Stencil stencil, LocalVars lv, boolean bIsDeclaration, boolean bIsInteger, boolean bOffsetInSpace, boolean bOffsetInTime) {
		ExpressionData  expr;
		Expect(5);
		expr = StencilExpression(stencil, lv, bIsDeclaration, bIsInteger, bOffsetInSpace, bOffsetInTime);
		Expect(6);
		return expr;
	}

	ExpressionData  FunctionCall(Stencil stencil, LocalVars lv, boolean bIsDecl, boolean bIsInteger, boolean bOffsetInSpace, boolean bOffsetInTime) {
		ExpressionData  fnx;
		Expect(1);
		String strFunctionName = t.val; 
		Expect(5);
		List<Expression> listArgs = new ArrayList<> (); int nFlopsCount = 0; 
		if (StartOf(6)) {
			ExpressionData expr = StencilExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
			listArgs.add (expr.getExpression ()); nFlopsCount += expr.getFlopsCount (); 
			while (la.kind == 21) {
				while (!(la.kind == 0 || la.kind == 21)) {SynErr(82); Get();}
				Get();
				expr = StencilExpression(stencil, lv, bIsDecl, bIsInteger, bOffsetInSpace, bOffsetInTime);
				listArgs.add (expr.getExpression ()); nFlopsCount += expr.getFlopsCount (); 
			}
		}
		Expect(6);
		fnx = m_au.isCompileTimeReduction (strFunctionName, listArgs, lv, la) ? 
		m_au.replaceIndexedScalars (m_au.expandCompileTimeReduction (strFunctionName, listArgs, lv, la), m_mapScalars, m_mapConstants, la) : 
		new ExpressionData (new FunctionCall (new NameID (strFunctionName), listArgs), nFlopsCount + 1, Symbolic.EExpressionType.EXPRESSION); 
		return fnx;
	}

	double  FloatLiteral() {
		double  fValue;
		fValue = 0.0; 
		if (la.kind == 3) {
			Get();
			fValue = Math.PI; 
		} else if (StartOf(7)) {
			StringBuilder sb = new StringBuilder (); 
			if (la.kind == 2) {
				Get();
				sb.append (t.val); 
			}
			if (la.kind == 53) {
				Get();
				sb.append (t.val); 
				if (la.kind == 2) {
					Get();
					sb.append (t.val); 
				}
			}
			if (la.kind == 54 || la.kind == 55) {
				if (la.kind == 54) {
					Get();
					sb.append (t.val); 
				} else {
					Get();
					sb.append (t.val); 
				}
				Expect(2);
				sb.append (t.val); 
			}
			fValue = Double.parseDouble (sb.toString ()); 
		} else SynErr(83);
		return fValue;
	}

	int  IntegerLiteral() {
		int  nValue;
		Expect(2);
		nValue = Integer.parseInt (t.val); 
		return nValue;
	}

	int  PosNegIntegerLiteral() {
		int  nValue;
		boolean bIsNeg = false; 
		if (la.kind == 47 || la.kind == 48) {
			if (la.kind == 47) {
				Get();
			} else {
				Get();
				bIsNeg = true; 
			}
		}
		nValue = IntegerLiteral();
		if (bIsNeg) nValue = -nValue; 
		return nValue;
	}



	public void Parse() {
		la = new Token();
		la.val = "";		
		Get();
		StencilSpecification();
		Expect(0);

	}

	private static final boolean[][] set = {
		{T,x,x,x, T,x,x,x, T,T,T,T, x,x,x,T, x,T,T,T, T,T,x,x, x,x,x,x, x,x,x,T, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x},
		{x,x,x,x, x,x,x,x, x,T,x,x, x,x,x,T, x,T,T,T, T,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x},
		{x,T,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, T,T,x,x, x,x,x,x, T,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x},
		{x,T,T,T, x,T,T,T, T,x,T,x, x,x,x,x, x,x,x,x, x,T,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,T, T,T,T,T, T,T,T,T, x,x},
		{x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, T,T,T,T, T,T,T,x, x,x,x,x, x,x,x,x, x,x},
		{x,x,T,T, x,x,T,x, T,x,T,x, x,x,x,x, x,x,x,x, x,T,T,x, x,x,x,x, x,x,x,T, x,T,T,T, T,T,x,x, T,T,T,T, T,T,T,T, T,T,T,T, T,T,T,T, x,x},
		{x,T,T,T, x,T,T,T, x,x,x,x, x,x,x,x, x,x,x,x, x,T,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,T, T,T,T,T, T,T,T,T, x,x},
		{x,x,T,x, x,x,T,x, T,x,T,x, x,x,x,x, x,x,x,x, x,T,T,x, x,x,x,x, x,x,x,T, x,T,T,T, T,T,x,x, T,T,T,T, T,T,T,T, T,T,T,T, T,T,T,T, x,x}

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
			case 3: s = "pi expected"; break;
			case 4: s = "\"stencil\" expected"; break;
			case 5: s = "\"(\" expected"; break;
			case 6: s = "\")\" expected"; break;
			case 7: s = "\"{\" expected"; break;
			case 8: s = "\"}\" expected"; break;
			case 9: s = "\"options\" expected"; break;
			case 10: s = "\";\" expected"; break;
			case 11: s = "\"compatibility\" expected"; break;
			case 12: s = "\"=\" expected"; break;
			case 13: s = "\"C/C++\" expected"; break;
			case 14: s = "\"Fortran\" expected"; break;
			case 15: s = "\"iterate\" expected"; break;
			case 16: s = "\"while\" expected"; break;
			case 17: s = "\"domainsize\" expected"; break;
			case 18: s = "\"operation\" expected"; break;
			case 19: s = "\"boundaries\" expected"; break;
			case 20: s = "\"initial\" expected"; break;
			case 21: s = "\",\" expected"; break;
			case 22: s = "\"..\" expected"; break;
			case 23: s = "\"const\" expected"; break;
			case 24: s = "\"float\" expected"; break;
			case 25: s = "\"double\" expected"; break;
			case 26: s = "\"int\" expected"; break;
			case 27: s = "\"long\" expected"; break;
			case 28: s = "\"grid\" expected"; break;
			case 29: s = "\"param\" expected"; break;
			case 30: s = "\"[\" expected"; break;
			case 31: s = "\"]\" expected"; break;
			case 32: s = "\"predicate\" expected"; break;
			case 33: s = "\":\" expected"; break;
			case 34: s = "\"||\" expected"; break;
			case 35: s = "\"or\" expected"; break;
			case 36: s = "\"&&\" expected"; break;
			case 37: s = "\"and\" expected"; break;
			case 38: s = "\"!\" expected"; break;
			case 39: s = "\"not\" expected"; break;
			case 40: s = "\"<\" expected"; break;
			case 41: s = "\"<=\" expected"; break;
			case 42: s = "\"==\" expected"; break;
			case 43: s = "\">=\" expected"; break;
			case 44: s = "\">\" expected"; break;
			case 45: s = "\"!=\" expected"; break;
			case 46: s = "\"/=\" expected"; break;
			case 47: s = "\"+\" expected"; break;
			case 48: s = "\"-\" expected"; break;
			case 49: s = "\"*\" expected"; break;
			case 50: s = "\"/\" expected"; break;
			case 51: s = "\"%\" expected"; break;
			case 52: s = "\"^\" expected"; break;
			case 53: s = "\".\" expected"; break;
			case 54: s = "\"e+\" expected"; break;
			case 55: s = "\"e-\" expected"; break;
			case 56: s = "??? expected"; break;
			case 57: s = "this symbol not expected in StencilSpecification"; break;
			case 58: s = "this symbol not expected in StencilOptions"; break;
			case 59: s = "this symbol not expected in StencilIterateWhile"; break;
			case 60: s = "this symbol not expected in StencilDomainSize"; break;
			case 61: s = "this symbol not expected in StencilOperation"; break;
			case 62: s = "this symbol not expected in StencilBoundaries"; break;
			case 63: s = "this symbol not expected in StencilInitial"; break;
			case 64: s = "this symbol not expected in StencilOptionsCompatibility"; break;
			case 65: s = "invalid StencilOptionsCompatibility"; break;
			case 66: s = "invalid AssignmentStatement"; break;
			case 67: s = "this symbol not expected in AssignmentStatement"; break;
			case 68: s = "invalid StencilOperationParam"; break;
			case 69: s = "invalid StencilOperationParam"; break;
			case 70: s = "this symbol not expected in DimensionDeclarator"; break;
			case 71: s = "this symbol not expected in DimensionDeclarator"; break;
			case 72: s = "this symbol not expected in ValueInitializer"; break;
			case 73: s = "invalid ValueInitializer"; break;
			case 74: s = "invalid ScalarAssignment"; break;
			case 75: s = "this symbol not expected in StencilIdentifier"; break;
			case 76: s = "this symbol not expected in StencilIdentifier"; break;
			case 77: s = "this symbol not expected in ScalarIdentifier"; break;
			case 78: s = "this symbol not expected in ScalarIdentifier"; break;
			case 79: s = "invalid ComparisonOperator"; break;
			case 80: s = "invalid UnaryExpression"; break;
			case 81: s = "invalid NumberLiteral"; break;
			case 82: s = "this symbol not expected in FunctionCall"; break;
			case 83: s = "invalid FloatLiteral"; break;
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
