package ch.unibas.cs.hpwc.patus.codegen.computation;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import cetus.hir.AnnotationStatement;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.CommentAnnotation;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.analysis.StencilAnalyzer;
import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.ast.IStatementList;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.codegen.options.StencilLoopUnrollingConfiguration;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.IntArray;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

abstract class AbstractStencilCalculationCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	protected CodeGeneratorSharedObjects m_data;

	protected CodeGeneratorRuntimeOptions m_options;

	/**
	 * The stencil expression for which to generate the C version
	 */
	protected Expression m_exprStrategy;

	/**
	 * The list of generated statements to which additional code
	 * (temporary calculations) is added
	 */
	protected StatementListBundle m_slbGenerated;

	/**
	 * The hardware description
	 */
	protected IArchitectureDescription m_hw;

	/**
	 * The least common multiple of the SIMD vector lengths of all
	 * SIMD vectors occurring in the stencil computation
	 */
	protected int m_nLcmSIMDVectorLengths;

	/**
	 * The argument to the stencil call in the strategy
	 */
	protected SubdomainIdentifier m_sdidStencilArg;

	/**
	 * Temporary shuffled stencil node variables
	 */
	protected Map<IntArray, Identifier> m_mapShuffledNodes;
	
	/**
	 * The (0,...,0) offset
	 */
	private int[] m_rgDefaultOffset;
	

	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates a new stencil calculation code generator.
	 * 
	 * @param data
	 *            The shared data object
	 * @param exprStrategy
	 *            The Strategy stencil expression (the formal stencil call)
	 * @param nLcmSIMDVectorLengths
	 *            The LCM of the SIMD vector lengths
	 * @param slbGenerated
	 *            The statement list bundle to which all the generated code will
	 *            be added
	 * @param options
	 *            Code generation options
	 */
	public AbstractStencilCalculationCodeGenerator (CodeGeneratorSharedObjects data, Expression exprStrategy, int nLcmSIMDVectorLengths, StatementListBundle slbGenerated, CodeGeneratorRuntimeOptions options)
	{
		// initialize member variables
		m_data = data;
		m_options = options;

		m_exprStrategy = exprStrategy;
		m_slbGenerated = slbGenerated;

		m_hw = m_data.getArchitectureDescription ();
		m_nLcmSIMDVectorLengths = nLcmSIMDVectorLengths;
		m_sdidStencilArg = m_exprStrategy == null ? null : (SubdomainIdentifier) StrategyAnalyzer.getStencilArgument (m_exprStrategy);

		m_mapShuffledNodes = new HashMap<> ();
		
		m_rgDefaultOffset = new int[m_data.getStencilCalculation ().getDimensionality ()];
		Arrays.fill (m_rgDefaultOffset, 0);
	}
	
	/**
	 * Generates the code for the stencil computation.
	 */
	public void generate ()
	{
		// annotate with original stencil expression
		m_slbGenerated.addStatement (new AnnotationStatement (new CommentAnnotation (m_exprStrategy.toString ())));

		// the the loop unrolling configuration (if any)
		StencilLoopUnrollingConfiguration configUnroll =
			((StencilLoopUnrollingConfiguration) m_options.getObjectValue (CodeGeneratorRuntimeOptions.OPTION_STENCILLOOPUNROLLINGFACTOR));
		byte nDimensionality = m_sdidStencilArg.getDimensionality ();
		
		boolean bSuppressVectorization = m_options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, false);
		boolean bUseNativeSIMDDatatypes = m_data.getOptions ().useNativeSIMDDatatypes () && m_data.getArchitectureDescription ().useSIMD ();
		
		// determine whether the code is generated for calculation, initialization, or validation
		// in case it is generated for validation, do not treat constant stencils specially (no loop invariant code motion)
		// in the other cases, add the constant expression to the respective initialization statement
		// (identified by the parameter paramComputationType)
		
		boolean bIsCreatingValidation = m_options.hasValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_VALIDATE);
		
		ParameterAssignment paComputationType = StencilCalculationCodeGenerator.createStencilCalculationParamAssignment (m_options);
		
		// generate the code for the stencil calculations
		for (Stencil stencil : m_data.getStencilCalculation ().getStencilBundle ())
		{
			Specifier specDatatype = getDatatype (stencil.getExpression ());
			int nSIMDVectorLength = m_data.getArchitectureDescription ().getSIMDVectorLength (specDatatype);
			int nOffsetStep = 1;
			int nEndOffset = 0;

			// if no native SIMD data types are used, but the code is vectorized, use SIMD vector length offset steps
			if (!bUseNativeSIMDDatatypes && !bSuppressVectorization)
			{
				nOffsetStep = nSIMDVectorLength;
				nEndOffset = m_nLcmSIMDVectorLengths;
			}
			else
				nEndOffset = (bSuppressVectorization ? 1 : (m_nLcmSIMDVectorLengths / nSIMDVectorLength));
			
			for (ParameterAssignment pa : m_slbGenerated)
			{
				if (!bIsCreatingValidation && StencilAnalyzer.isStencilConstant (stencil, m_data.getStencilCalculation ()))
				{
					// if the stencil is constant, add the computation to the head of the
					// function instead within the computation loop and, if the loop is unrolled,
					// avoid creating the statement multiple times
					// (loop-invariant code motion)
					
					StatementList sl = m_data.getData ().getInitializationStatements (paComputationType);
					generateSingleCalculation (stencil, specDatatype, m_rgDefaultOffset, sl, sl);
				}
				else
				{
					// regular, non-constant stencils
					// add the statement to the loop body
					
					Iterable<int[]> itOffsets = null;
					if (configUnroll == null)
						itOffsets = StencilLoopUnrollingConfiguration.getDefaultSpace (nDimensionality);
					else
						itOffsets = configUnroll.getConfigurationSpace (nDimensionality);
					
					for (int[] rgOffset : itOffsets)
					{
						int nOffsetDim0 = rgOffset[0];
						if (!bUseNativeSIMDDatatypes && !bSuppressVectorization)
							nOffsetDim0 *= nSIMDVectorLength;

						for (int nOffset = 0; nOffset < nEndOffset; nOffset += nOffsetStep)
						{
							rgOffset[0] = nOffsetDim0 + nOffset;
							
							generateSingleCalculation (
								stencil, specDatatype, rgOffset,
								getStencilComputationStatementList (pa), getAuxiliaryCalculationStatementList (pa)
							);
						}
					}
				}
			}
		}
	}
	
	protected StatementList getStencilComputationStatementList (ParameterAssignment pa)
	{
		return m_slbGenerated.getStatementList (pa);
	}
	
	protected StatementList getAuxiliaryCalculationStatementList (ParameterAssignment pa)
	{
		return m_slbGenerated.getStatementList (pa);
	}
	
	protected abstract void generateSingleCalculation (Stencil stencil, Specifier specDatatype, int[] rgOffsetIndex,
		StatementList slStencilComputation, StatementList slAuxiliaryCalculations);

	/**
	 * Determines the datatype of the expression <code>expr</code>.
	 * The datatype of the expression is determined by the stencil nodes
	 * occurring in the expression). By default, i.e., if no stencil
	 * nodes are contained in the expression, {@link Specifier#FLOAT} is
	 * returned.
	 * 
	 * @param expr
	 *            The expression to check for the datatype
	 * @return The datatype of the expression <code>expr</code>
	 */
	@SuppressWarnings("static-method")
	protected Specifier getDatatype (Expression expr)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof StencilNode)
				if (((StencilNode) obj).getSpecifier ().equals (Specifier.DOUBLE))
					return Specifier.DOUBLE;
		}

		return Specifier.FLOAT;
	}
	
	protected int[] getDefaultOffset ()
	{
		return m_rgDefaultOffset;
	}

	/**
	 * Replaces stencil node in the stencil expression <code>expr</code> by
	 * the instantiated, indexed memory objects.
	 * 
	 * @param expr
	 *            The input stencil expression (referencing stencil nodes)
	 * @param specDatatype
	 *            The datatype used to calculate the expression
	 * @param rgOffsetIndex
	 * @param slGenerated
	 *            The generated code to which the index calculations are
	 *            appended
	 * @return
	 */
	protected Expression replaceStencilNodes (Expression expr, Specifier specDatatype, int[] rgOffsetIndex, IStatementList slGenerated)
	{
		Expression exprResult = expr.clone ();
		return recursiveReplaceStencilNodes (exprResult, exprResult, specDatatype, rgOffsetIndex, slGenerated);
	}

	protected Expression recursiveReplaceStencilNodes (Expression exprMain, Traversable trv, Specifier specDatatype, int[] rgOffsetIndex, IStatementList slGenerated)
	{
		if (trv instanceof StencilNode)
		{
			// replace the stencil nodes by an indexed box expression
			return createSIMDStencilNode ((StencilNode) trv, specDatatype, rgOffsetIndex, slGenerated);
		}

		if (trv instanceof FunctionCall)
		{
			// don't change function names, only arguments
			for (Object objArg : ((FunctionCall) trv).getArguments ())
			{
				if (objArg instanceof Expression)
				{
					Expression exprReplace = recursiveReplaceStencilNodes (exprMain, (Expression) objArg, specDatatype, rgOffsetIndex, slGenerated);
					if (exprReplace != null)
						((Expression) objArg).swapWith (exprReplace);
				}
			}

			return trv == exprMain ? exprMain : null;
		}

		if ((trv instanceof Identifier) || (trv instanceof Literal) || ((trv instanceof NameID) && m_data.getStencilCalculation ().isParameter (((NameID) trv).getName ())))
		{
			return m_data.getCodeGenerators ().getSIMDScalarGeneratedIdentifiers ().createVectorizedScalar (
				(Expression) trv, specDatatype, m_slbGenerated, m_options);
		}

		if (trv instanceof NameID)
		{
			// NameID found in the expression: e.g. a variable that was assigned a value by another stencil expression:
			// float d = ...                                        <-- stencil 1
			// u[x,y,z;t+1] = ... f(d) ... (some function of d)     <-- stencil 2; we are here

			// TODO: other cases? when is there a NameID in the expression instead of an Identifier? When are there Identifiers?
			// should the above branch not have Identifier, but add to this one?

			// if the stencil node corresponding to trv is a constant-output stencil node,
			// there are no versions for unrolling (they remain the same for each unrolling index)
			int[] rgOffset = rgOffsetIndex;
			if (m_data.getStencilCalculation ().getStencilBundle ().isConstantOutputStencilNode ((NameID) trv))
				rgOffset = m_rgDefaultOffset;
				
			return m_data.getCodeGenerators ().getUnrollGeneratedIdentifiers ().createIdentifier (
				(IDExpression) trv, rgOffset, specDatatype, m_slbGenerated, m_options);
		}

		// other cases: recursively find subexpressions that match one of the above
		for (Traversable trvChild : trv.getChildren ())
		{
			if (trvChild instanceof Expression)
			{
				Expression exprReplace = recursiveReplaceStencilNodes (exprMain, trvChild, specDatatype, rgOffsetIndex, slGenerated);
				if (exprReplace != null)
					((Expression) trvChild).swapWith (exprReplace);
			}
		}

		return trv == exprMain ? exprMain : null;
	}

	/**
	 * Determines whether SIMD shuffling is needed for the stencil node
	 * <code>nodeInput</code>.
	 * 
	 * @param nodeInput
	 *            The stencil node
	 * @param specDatatype
	 *            The stencil expression data type
	 * @return <code>true</code> if shuffling is needed to address the node
	 *         <code>nodeInput</code>
	 */
	protected boolean needsShuffling (StencilNode nodeInput, Specifier specDatatype)
	{
		int nSIMDVectorLength = m_hw.getSIMDVectorLength (specDatatype);

		if (nSIMDVectorLength == 1)
			return false;

		int nCoord = ExpressionUtil.getIntegerValue (nodeInput.getIndex ().getSpaceIndex (0));
		return (nCoord % nSIMDVectorLength) != 0;
	}

	/**
	 * Returns the SIMD-aware expression for the stencil node
	 * <code>nodeInput</code>.
	 * 
	 * @param nodeInput
	 *            The input stencil node for which to retrieve the SIMD
	 *            expression
	 * @param specDatatype
	 * @return
	 */
	protected Expression createSIMDStencilNode (StencilNode nodeInput, Specifier specDatatype, int[] rgOffsetIndex, IStatementList slGenerated)
	{
		StencilNode node = nodeInput;

		if (nodeInput.isScalar ())
		{
			IDExpression idNodeTmp = m_data.getCodeGenerators ().getUnrollGeneratedIdentifiers ().createIdentifier (
				nodeInput, rgOffsetIndex, specDatatype, m_slbGenerated, m_options);

			if (idNodeTmp != node)
				node = new StencilNode (idNodeTmp.getName (), nodeInput.getSpecifier (), nodeInput.getIndex ());
		}
		else
		{
			boolean bNoVectorize = m_options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, false);
			int nSIMDVectorLength = m_hw.getSIMDVectorLength (specDatatype);

			if (nSIMDVectorLength > 1 && !bNoVectorize)
			{
				int nSpaceIdxUnitStride = ExpressionUtil.getIntegerValue (nodeInput.getIndex ().getSpaceIndex (0));

				if (m_data.getOptions ().useNativeSIMDDatatypes ())
				{
					// if native SIMD datatypes are used:
					// adjust the spatial index to account for SIMD (divide the first coordinate by the SIMD vector length)
					//
					// example: SIMD vector length = 4; in unit stride dimension:
					//
					// scalar:  -5 -4 -3 -2 -1  0  1  2  3  4
					//          --+-----------+-----------+---
					// SIMD:    -2|     -1    |     0     | 1
					//
					// node will become the "left" node (if we need to shuffle)

					node = new StencilNode (nodeInput);
					node.getIndex ().setSpaceIndex (0, new IntegerLiteral ((int) Math.floor ((double) nSpaceIdxUnitStride / (double) nSIMDVectorLength)));
				}
				else
				{
					// No native SIMD datatypes are used:
					// Stencil offsets in unit stride direction must be multiples of nSIMDVectorLength
					// since we keep the original non-vectorized grid
					// For now we assume that the number of grid points in unit stride direction
					// is divisible by nSIMDVectorLength. If not, apply padding.

					// Round the offset away from zero to the next multiple of nSIMDVectorLength
					// (e.g., 0 -> 0; 1,2,...,nSIMDVectorLength-1 -> nSIMDVectorLength;
					// -1,-2,...,-nSIMDVectorLength+1 -> -nSIMDVectorLength; ...)

					// example: SIMD vector length = 4; in unit stride dimension:
					//
					// scalar:     -5 -4 -3 -2 -1  0  1  2  3  4  5
					//             --------------------------------
					// SIMD:       -8 -4 -4 -4 -4  0  0  0  0  4  4
					// [no-native]

					//int nXCoord = node.getSpaceIndex ()[0];
					int nXCoord = ExpressionUtil.getIntegerValue (node.getIndex ().getSpaceIndex (0));
					if (nXCoord != 0)
					{
						node = new StencilNode (nodeInput);
						node.getIndex ().setSpaceIndex (0, (int) Math.floor ((double) nSpaceIdxUnitStride / (double) nSIMDVectorLength) * nSIMDVectorLength);
					}
				}

				// calculate the offset used for shuffling (the offset with respect to the left node)
				// make positive if nOffset is negative
				// determine whether we need to shuffle at all and if so, compute the shuffle expression
				int nShuffleOffset = nSpaceIdxUnitStride % nSIMDVectorLength;
				if (nShuffleOffset != 0)
				{
					if (nShuffleOffset < 0)
						nShuffleOffset += nSIMDVectorLength;
					return getShuffledNode (nodeInput, node, rgOffsetIndex, nShuffleOffset, specDatatype, nSIMDVectorLength, slGenerated);
				}
			}

			// no shuffling needed
			// create a new stencil node with the offsets incorporated
			node = new StencilNode (node);
			node.getIndex ().offsetInSpace (rgOffsetIndex);
		}

		return m_data.getData ().getMemoryObjectManager ().getMemoryObjectExpression (
			m_sdidStencilArg, node, null, true, true, false, slGenerated, m_options);
	}

	private static String toCString (int[] rgIdx)
	{
		StringBuilder sb = new StringBuilder ();
		boolean bFirst = true;
		for (int n : rgIdx)
		{
			if (!bFirst)
				sb.append ('_');
			if (n < 0)
				sb.append ('m');
			sb.append (Math.abs (n));
			bFirst = false;
		}
		return sb.toString ();
	}

	/**
	 * Creates a shuffle expression for the offset <code>nOffset</code> and
	 * stores it in an intermediate variable.
	 * 
	 * @param node
	 *            The base node from which the shuffle expression is
	 *            calculated
	 * @param rgOffset
	 *            The stencil node offset (offsets the stencil node
	 *            <code>node</code> in space; the stencil node
	 *            is treated as a SIMD vector)
	 * @param nShuffleOffset
	 *            The offset within the SIMD vector with respect to the base
	 *            node <code>node</code>.
	 *            Must be in the range 1 to (SIMD vector length - 1). (Note
	 *            that a shuffle offset of 0 means no shuffling.)
	 * @param specDatatype
	 * @param nSIMDVectorLength
	 * @return
	 */
	protected Identifier getShuffledNode (StencilNode nodeOrig, StencilNode nodeLeft, int[] rgOffset, int nShuffleOffset, Specifier specDatatype, int nSIMDVectorLength, IStatementList slbGenerated)
	{
		assert (1 <= nShuffleOffset && nShuffleOffset < nSIMDVectorLength);

		// check whether the node for the offsets rgOffset/nShuffleOffset has already been created
		// note: arrOffset serves only as key into the map m_mapShuffledNodes,
		// it consists of the array rgOffset and appended nShuffleOffset
		
		nodeLeft.getIndex ().offsetInSpace (rgOffset);
		int[] rgLeftSpaceIdx = nodeLeft.getSpaceIndex ();
		
		IntArray arrOffset = new IntArray (rgLeftSpaceIdx);
		arrOffset.append (nShuffleOffset);
		arrOffset.append (nodeLeft.getIndex ().getVectorIndex ());
		Identifier idShuffledNode = m_mapShuffledNodes.get (arrOffset);
		if (idShuffledNode != null)
			return idShuffledNode.clone ();

		// get the right stencil node that need to be joined to the right of nodeLeft for the shuffling
		StencilNode nodeRight = new StencilNode (nodeLeft);
		nodeRight.getIndex ().offsetInSpace (0, nSIMDVectorLength);

		// get the C expressions for the nodes that need to be joined
		Expression exprLeft = m_data.getData ().getMemoryObjectManager ().getMemoryObjectExpression (
			m_sdidStencilArg, nodeLeft, null, true, true, false, slbGenerated, m_options);
		Expression exprRight = m_data.getData ().getMemoryObjectManager ().getMemoryObjectExpression (
			m_sdidStencilArg, nodeRight, null, true, true, false, slbGenerated, m_options);

		// construct a name for the temporary shuffled variable
		String strName = null;
		for (DepthFirstIterator it = new DepthFirstIterator (exprLeft); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof IDExpression)
			{
				strName = ((IDExpression) obj).getName ();
				break;
			}
		}
		if (strName == null)
			strName = m_sdidStencilArg.getName ();

		int nOrigOffset = ExpressionUtil.getIntegerValue (nodeOrig.getIndex ().getSpaceIndex (0));
		NameID nid = new NameID (StringUtil.concat (
			strName, "_", nOrigOffset < 0 ? "minus_" : "plus_", Math.abs (nOrigOffset),
			"__", AbstractStencilCalculationCodeGenerator.toCString (rgLeftSpaceIdx),
			"__offs_", AbstractStencilCalculationCodeGenerator.toCString (rgOffset)));

		// create the identifier
		VariableDeclarator decl = new VariableDeclarator (nid);
		Identifier idTmpVar = new Identifier (decl);
		m_data.getData ().addDeclaration (new VariableDeclaration (m_hw.getType (specDatatype), decl));
		m_mapShuffledNodes.put (arrOffset, idTmpVar);

		Expression exprShuffle = m_data.getCodeGenerators ().getBackendCodeGenerator ().shuffle (exprLeft, exprRight, specDatatype, nShuffleOffset);
		if (exprShuffle == null)
			throw new RuntimeException ("IBackend#shuffle is required to be implemented.");
		m_slbGenerated.addStatement (new ExpressionStatement (new AssignmentExpression (idTmpVar.clone (), AssignmentOperator.NORMAL, exprShuffle)));

		return idTmpVar;
	}

	/**
	 * Returns the constraint of a stencil node, which can be used in generated
	 * code (i.e., it has all dimension identifiers replaced by the correct
	 * iterator variables).
	 * 
	 * @param node
	 *            The stencil node for which to return the constraint. If the
	 *            node doesn't have a constraint, the method will return
	 *            <code>null</code>.
	 * @return The ready-to-use node constraint or <code>null</code> if the node
	 *         doesn't have a constraint
	 */
	protected Expression getConstraintCondition (StencilNode node)
	{
		Expression exprResult = node.getConstraint ();
		if (exprResult == null)
			return null;
		exprResult = exprResult.clone ();
		
		for (DepthFirstIterator it = new DepthFirstIterator (exprResult); it.hasNext (); )
		{
			Traversable trv = (Traversable) it.next ();
			int nDim = CodeGeneratorUtil.getDimensionFromIdentifier (trv);
			if (nDim >= 0)
				((Expression) trv).swapWith (getDimensionIndexIdentifier (nDim));
		}
		
		return exprResult;
	}
	
	protected Expression getDimensionIndexIdentifier (SubdomainIdentifier sdid, int nDim)
	{
		return m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (sdid, nDim).clone ();		
	}
	
	protected Expression getDimensionIndexIdentifier (int nDim)
	{
		return getDimensionIndexIdentifier (m_sdidStencilArg, nDim);
	}
}