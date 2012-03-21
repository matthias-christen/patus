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
package ch.unibas.cs.hpwc.patus.codegen;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

import cetus.hir.AnnotationStatement;
import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CommentAnnotation;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FloatLiteral;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.StringLiteral;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.ast.IStatementList;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.codegen.options.StencilLoopUnrollingConfiguration;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.IntArray;
import ch.unibas.cs.hpwc.patus.util.MathUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class StencilCalculationCodeGenerator implements ICodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants and Static Types

	private final static Logger LOGGER = Logger.getLogger (StencilCalculationCodeGenerator.class);
	

	///////////////////////////////////////////////////////////////////
	// Inner Types

	private abstract class CodeGenerator
	{
		///////////////////////////////////////////////////////////////////
		// Member Variables

		protected CodeGeneratorRuntimeOptions m_options;

		/**
		 * The stencil expression for which to generate the C version
		 */
		protected Expression m_expr;

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

		public CodeGenerator (Expression expr, StatementListBundle slGenerated, CodeGeneratorRuntimeOptions options)
		{
			// initialize member variables
			m_options = options;

			m_expr = expr;
			m_slbGenerated = slGenerated;

			m_hw = m_data.getArchitectureDescription ();
			m_nLcmSIMDVectorLengths = getLcmSIMDVectorLengths ();
			m_sdidStencilArg = (SubdomainIdentifier) StrategyAnalyzer.getStencilArgument (m_expr);

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
			m_slbGenerated.addStatement (new AnnotationStatement (new CommentAnnotation (m_expr.toString ())));

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
			
			ParameterAssignment paComputationType = new ParameterAssignment (
				CodeGeneratorData.PARAM_COMPUTATION_TYPE,
				m_options.getIntValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL)
			);

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
					
					if (!bIsCreatingValidation && stencil.isConstant ())
					{
						// if the stencil is constant, add the computation to the head of the
						// function instead within the computation loop and, if the loop is unrolled,
						// avoid creating the statement multiple times
						// (loop-invariant code motion)
						
						generateSingleCalculation (
							stencil, specDatatype, m_rgDefaultOffset,
							m_data.getData ().getInitializationStatements (paComputationType)
						);
					}
					else
					{
						// regular, non-constant stencils
						// add the statement to the loop body
						
						for (int[] rgOffset : (configUnroll == null ?
							StencilLoopUnrollingConfiguration.getDefaultSpace (nDimensionality) :
							configUnroll.getConfigurationSpace (nDimensionality)))
						{
							int nOffsetDim0 = rgOffset[0];
							if (!bUseNativeSIMDDatatypes && !bSuppressVectorization)
								nOffsetDim0 *= nSIMDVectorLength;
	
							for (int nOffset = 0; nOffset < nEndOffset; nOffset += nOffsetStep)
							{
								rgOffset[0] = nOffsetDim0 + nOffset;
								generateSingleCalculation (stencil, specDatatype, rgOffset, m_slbGenerated.getStatementList (pa));
							}
						}
					}
				}
			}
		}
		
		protected abstract void generateSingleCalculation (Stencil stencil, Specifier specDatatype, int[] rgOffsetIndex, StatementList slGenerated);

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

			if ((trv instanceof Identifier) || (trv instanceof Literal) || ((trv instanceof NameID) && m_data.getStencilCalculation ().isArgument (((NameID) trv).getName ())))
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

			return (nodeInput.getSpaceIndex ()[0] % nSIMDVectorLength) != 0;
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
		protected Expression createSIMDStencilNode (StencilNode nodeInput, Specifier specDatatype, int[] rgOffsetIndex, IStatementList slbGenerated)
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
					int nSpaceIdxUnitStride = nodeInput.getSpaceIndex ()[0];

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
						node.getIndex ().getSpaceIndex ()[0] =
							(int) Math.floor ((double) nSpaceIdxUnitStride / (double) nSIMDVectorLength);
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

						int nXCoord = node.getSpaceIndex ()[0];
						if (nXCoord != 0)
						{
							node = new StencilNode (nodeInput);
							/*
							node.getIndex ().getSpaceIndex ()[0] =
								(int) (nXCoord < 0 ?
									Math.floor ((double) nXCoord / (double) nSIMDVectorLength) :
									Math.ceil ((double) nXCoord / (double) nSIMDVectorLength))
								* nSIMDVectorLength;
							*/
							node.getIndex ().getSpaceIndex ()[0] =
								(int) Math.floor ((double) nSpaceIdxUnitStride / (double) nSIMDVectorLength) * nSIMDVectorLength;
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
						return getShuffledNode (nodeInput, node, rgOffsetIndex, nShuffleOffset, specDatatype, nSIMDVectorLength, slbGenerated);
					}
				}

				// no shuffling needed
				// create a new stencil node with the offsets incorporated
				node = new StencilNode (node);
				node.getIndex ().offsetInSpace (rgOffsetIndex);
			}

			return m_data.getData ().getMemoryObjectManager ().getMemoryObjectExpression (
				m_sdidStencilArg, node, null, true, true, false, slbGenerated, m_options);
		}

		private String toCString (int[] rgIdx)
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
			IntArray arrOffset = new IntArray (nodeLeft.getSpaceIndex ());
			arrOffset.add (rgOffset);
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

			int nOrigOffset = nodeOrig.getSpaceIndex ()[0];
			NameID nid = new NameID (StringUtil.concat (
				strName, "_", nOrigOffset < 0 ? "minus_" : "plus_", Math.abs (nOrigOffset),
				"__", toCString (nodeLeft.getSpaceIndex ()),
				"__offs_", toCString (rgOffset)));

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
	}

	private class StencilCodeGenerator extends CodeGenerator
	{
		public StencilCodeGenerator (Expression expr, StatementListBundle slGenerated, CodeGeneratorRuntimeOptions options)
		{
			super (expr, slGenerated, options);
		}

		/**
		 * Generates the code for a single stencil calculation
		 * @param stencil
		 * @param specDatatype
		 * @param nSIMDOffsetIndex The offsets of the indices
		 * @param slGenerated
		 */
		@Override
		protected void generateSingleCalculation (Stencil stencil, Specifier specDatatype, int[] rgOffsetIndex, StatementList slGenerated)
		{
			Expression exprStencil = stencil.getExpression ();
			if (exprStencil != null)
			{
				// replace identifiers in the stencil expression
				exprStencil = exprStencil.clone ();

				// check whether the hardware / programming model supports explicit FMAs
				// TODO: support FMA when not vectorizing? (need to distinguish between non-vectorizing and vectorizing in HardwareDescription#getIntrinsicName)
//				if (m_hw.getIntrinsicName (Globals.FNX_FMA.getName (), specDatatype) != null)
//					exprStencil = m_data.getCodeGenerators ().getFMACodeGenerator ().applyFMAs (exprStencil, specDatatype);

				boolean bSuppressVectorization = m_options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, false);
				boolean bFirst = true;

				if (!bSuppressVectorization)
				{
					if (m_hw.getIntrinsic (Globals.FNX_FMA.getName (), specDatatype) != null)
						exprStencil = m_data.getCodeGenerators ().getFMACodeGenerator ().applyFMAs (exprStencil, specDatatype);
				}

				// add the stencil computation to the generated code
				for (StencilNode nodeOutput : stencil.getOutputNodes ())
				{
					// SIMDize the stencil expression
					Expression exprVectorizedStencil = m_data.getCodeGenerators ().getBackendCodeGenerator ().createExpression (
						exprStencil.clone (), specDatatype, !bSuppressVectorization);

					// replace the stencil nodes in the expression with the indexed memory object instances
					Expression exprMOStencil = replaceStencilNodes (exprVectorizedStencil, specDatatype, rgOffsetIndex, slGenerated);
					Expression exprLHS = replaceStencilNodes (nodeOutput, specDatatype, rgOffsetIndex, slGenerated);

					// create a printf statement for debugging purposes
					if (bFirst && m_data.getOptions ().isDebugPrintStencilIndices ())
						createDebugPrint (exprLHS, specDatatype, slGenerated, bSuppressVectorization);
					bFirst = false;

					// create the calculation statement
					Statement stmtStencil = new ExpressionStatement (
						new AssignmentExpression (
							exprLHS,
							/*
							m_data.getMemoryObjectManager ().getMemoryObjectExpression (
								m_sdidStencilArg, nodeOutput, true, true, slGenerated),*/
							AssignmentOperator.NORMAL,
							exprMOStencil));

					if (StencilCalculationCodeGenerator.LOGGER.isDebugEnabled ())
						StencilCalculationCodeGenerator.LOGGER.debug (StringUtil.concat ("Adding stencil ", stmtStencil.toString ()));

					slGenerated.addStatement (stmtStencil);
				}
			}
		}

		/**
		 * Insert a <code>printf</code> statement for debugging purposes
		 * printing the index into the grid array at
		 * which the result is written.
		 * 
		 * @param exprLHS
		 *            The LHS stencil expression
		 * @param slGenerated
		 *            The statement list to which the generated
		 *            <code>printf</code> statement is added
		 */
		private void createDebugPrint (Expression exprLHS, Specifier specDatatype, StatementList slGenerated, boolean bNoVectorize)
		{
			Expression exprIdx = null;
			for (DepthFirstIterator it = new DepthFirstIterator (exprLHS); it.hasNext (); )
			{
				Object obj = it.next ();
				if (obj instanceof ArrayAccess)
				{
					exprIdx = ((ArrayAccess) obj).getIndex (0);
					break;
				}
			}

			if (exprIdx == null)
				exprIdx = new IntegerLiteral (0x1D000DEF);	// IDx_unDEF

			int nSIMDVectorLengthToPrint =
				!bNoVectorize && !m_data.getOptions ().useNativeSIMDDatatypes () && m_data.getArchitectureDescription ().useSIMD () ?
				m_data.getArchitectureDescription ().getSIMDVectorLength (specDatatype) : 1;

			for (int i = 0; i < nSIMDVectorLengthToPrint; i++)
			{
				slGenerated.addStatement (new ExpressionStatement (new FunctionCall (
					new NameID ("printf"),
					CodeGeneratorUtil.expressions (
						new StringLiteral (bNoVectorize ? "%d\\n" : "%d!\\n"),
						new BinaryExpression (exprIdx.clone (), BinaryOperator.ADD, new IntegerLiteral (i))
					)
				)));
			}
		}
	}

	/**
	 * Generates initialization code.
	 */
	private class InitializeCodeGenerator extends CodeGenerator
	{
		public InitializeCodeGenerator (Expression expr, StatementListBundle slGenerated, CodeGeneratorRuntimeOptions options)
		{
			super (expr, slGenerated, options);
		}

		@Override
		protected void generateSingleCalculation (Stencil stencil, Specifier specDatatype, int[] rgOffsetIndex, StatementList slGenerated)
		{
			if (stencil.getExpression () == null)
				return;

			boolean bVectorize =
				m_data.getOptions ().useNativeSIMDDatatypes () ||
				!m_options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, false);

			// collect the stencil nodes that need to be initialized
			List<StencilNode> listNodesToInitialize = new LinkedList<> ();
			for (StencilNode nodeInput : stencil)
				if (!bVectorize || (bVectorize && !needsShuffling (nodeInput, specDatatype)))
					listNodesToInitialize.add (nodeInput);
			for (StencilNode nodeOutput : stencil.getOutputNodes ())
				listNodesToInitialize.add (nodeOutput);

			byte nDim = stencil.getDimensionality ();

			// add the initialization statements
			for (StencilNode node : listNodesToInitialize)
			{
				Expression exprLHS = bVectorize ?
					createSIMDStencilNode (node, specDatatype, rgOffsetIndex, slGenerated) :
					replaceStencilNodes (node, specDatatype, rgOffsetIndex, slGenerated);

				// initialize with some arbitrary value
				Expression exprValue = new FloatLiteral (10 * node.getIndex ().getTimeIndex () + (node.getIndex ().getVectorIndex () + 1));

				// make the values depend on the point in space
				if (node.getSpaceIndex ().length >= nDim)
				{
					// TODO: remove this restriction
					if (m_data.getOptions ().useNativeSIMDDatatypes ())
						LOGGER.error ("Initialization to non-constant fields not implemented for native SIMD datatypes");
					else
					{
						double fCoeff = 0.1;
						for (int i = 0; i < nDim; i++)
						{
							Expression exprCoord = m_data.getData ().getGeneratedIdentifiers ().getDimensionIndexIdentifier (m_sdidStencilArg, i).clone ();
							if (node.getSpaceIndex ()[i] + rgOffsetIndex[i] != 0)
								exprCoord = new BinaryExpression (exprCoord, BinaryOperator.ADD, new IntegerLiteral (node.getSpaceIndex ()[i] + rgOffsetIndex[i]));

							exprValue = new BinaryExpression (
								exprValue,
								BinaryOperator.ADD,
								new BinaryExpression (exprCoord, BinaryOperator.MULTIPLY, new FloatLiteral (fCoeff)));
							fCoeff /= 10;
						}
					}
				}

				Expression exprRHS = bVectorize ?
					m_data.getCodeGenerators ().getSIMDScalarGeneratedIdentifiers ().createVectorizedScalar (exprValue, specDatatype, m_slbGenerated, m_options) :
					exprValue;

				slGenerated.addStatement (new ExpressionStatement (new AssignmentExpression (exprLHS, AssignmentOperator.NORMAL, exprRHS)));
			}
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public StencilCalculationCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
	}

	/**
	 * Generates the code for the stencil calculation.
	 */
	@Override
	public StatementListBundle generate (Traversable trvInput, CodeGeneratorRuntimeOptions options)
	{
		if (StencilCalculationCodeGenerator.LOGGER.isDebugEnabled ())
			StencilCalculationCodeGenerator.LOGGER.debug (StringUtil.concat ("Generating code with options ", options.toString ()));

		StatementListBundle slb = new StatementListBundle (new ArrayList<Statement> ());

		if (!(trvInput instanceof Expression))
			throw new RuntimeException ("Expression as input to StencilCalculationCodeGenerator expected.");

		int nComputationType = options.getIntValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL);
		switch (nComputationType)
		{
		case CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL:
		case CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_VALIDATE:
			new StencilCodeGenerator ((Expression) trvInput, slb, options).generate ();
			break;
			
		case CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_INITIALIZE:
			new InitializeCodeGenerator ((Expression) trvInput, slb, options).generate ();
			break;
			
		default:
			throw new RuntimeException (StringUtil.concat (
				"Unknown option for ", CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION,	": ", nComputationType));
		}

		return slb;
	}
	
	/**
	 * Returns the least common multiple (LCM) of the SIMD vector lengths of
	 * all the stencil computations in the bundle.
	 * 
	 * @return The LCM of the stencil node SIMD vector lengths
	 */
	public int getLcmSIMDVectorLengths ()
	{
		int nLCM = 1;
		for (Stencil stencil : m_data.getStencilCalculation ().getStencilBundle ())
			for (StencilNode node : stencil.getOutputNodes ())
				nLCM = MathUtil.getLCM (nLCM, m_data.getArchitectureDescription ().getSIMDVectorLength (node.getSpecifier ()));
		return nLCM;
	}
}
