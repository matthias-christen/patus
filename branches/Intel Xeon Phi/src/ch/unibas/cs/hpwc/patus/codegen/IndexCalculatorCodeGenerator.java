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

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

import cetus.hir.AccessExpression;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.Statement;
import cetus.hir.Typecast;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.codegen.backend.IIndexing;
import ch.unibas.cs.hpwc.patus.codegen.backend.IIndexing.IIndexingLevel;
import ch.unibas.cs.hpwc.patus.codegen.backend.IndexingLevelUtil;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class IndexCalculatorCodeGenerator
{
	private final static boolean FIND_PREVIOUSLY_CALCULATED_STRIDES = false;


	///////////////////////////////////////////////////////////////////
	// Inner Types

	private enum ECalculationMode
	{
		CALCULATE_INDICES,
		CALCULATE_SIZES,
		CALCULATE_ALL
	}

	private class Calculator
	{
		///////////////////////////////////////////////////////////////////
		// Member Variables

		/**
		 * The compound statement to which auxiliary calculations are added or
		 * <code>null</code> if the statements are to be added to the initialization
		 * statement
		 */
		private CompoundStatement m_cmpstmt;

		private Expression[][] m_rgIndices;
		private Expression[][] m_rgBlockSizes;

		private Expression[] m_rgStrides;

		private Expression[] m_rgTargetIndices;
		private Expression[] m_rgTargetSizes;

		private Size m_sizeDomain;

		private int m_nTargetDimension;
		private int m_nIndexingLevelsCount;

		private IIndexing m_cg;

		private CodeGeneratorRuntimeOptions m_options;


		///////////////////////////////////////////////////////////////////
		// Implementation

		/**
		 * Creates the index calculator.
		 * @param size The size of the domain
		 * @param cmpstmt The compound statement to which auxiliary calculations are added, or <code>null</code>
		 * 	if the auxiliary calculations are to be added to the initialization statement of the kernel function
		 */
		public Calculator (IIndexing cg, Size size, CompoundStatement cmpstmt, ECalculationMode mode, CodeGeneratorRuntimeOptions options)
		{
			m_sizeDomain = size;
			m_cmpstmt = cmpstmt;
			m_options = options;

			// initialize variables
			m_cg = cg;

			m_nTargetDimension = m_sizeDomain.getDimensionality ();
			m_nIndexingLevelsCount = m_cg.getIndexingLevelsCount ();

			// allocate memory
			m_rgIndices = new Expression[m_nIndexingLevelsCount][m_nTargetDimension];
			m_rgBlockSizes = new Expression[m_nIndexingLevelsCount][m_nTargetDimension];
			m_rgStrides = new Expression[m_nTargetDimension];

			m_rgTargetIndices = new Expression[m_nTargetDimension];
			m_rgTargetSizes = new Expression[m_nTargetDimension];

			// calculate sizes, strides, and indices
			calculateSizes (mode != ECalculationMode.CALCULATE_INDICES);
			calculateStrides ();
			calculateIndices ();

			// calculate the target indices
			calculateTargetIndices ();
		}

		/**
		 * Returns the target indices.
		 * @return The target sizes
		 */
		public Expression[] getTargetIndices ()
		{
			return m_rgTargetIndices;
		}

		/**
		 * Returns the target sizes.
		 * @return The target sizes
		 */
		public Expression[] getTargetSizes ()
		{
			return m_rgTargetSizes;
		}

		/**
		 * Fills the sizes array with the (if necessary, calculated) block sizes.
		 * <code>bCalculateAllSizes</code> defines whether all the sizes are to be calculated
		 * or only the ones needed to compute the indices.
		 * New statements are added to the auxiliary sizes statement list only if
		 * <code>bCalculateAllSizes</code> is set to <code>true</code>.
		 *
		 * @param bCalculateAllSizes If set to <code>true</code>, all sizes are calculated,
		 * 	if set to <code>false</code> only the sizes needed for calculating the indices
		 *  (up to level L-1) are calculated.
		 */
		private void calculateSizes (boolean bCalculateAllSizes)
		{
			// block sizes:
			//
			// if l < m_nIndexingLevelsCount:
			//
			//            /  (l)                  (l)
			//   ~(l)     | n      if  1 <= j <= d,
			//   n    =  <   j
			//    j       |
			//            \  1     otherwise
			//
			// else:
			//            _                      _
			//           |         /  L-1         |
			//   ~(L)    |        /  ----- ~(l)   |                          (L)
			//   n    =  |  N    /    | |  n      |         for j = 1, ..., d    - 1
			//    j      |   j  /     l=1   j     |
			//
			//
			//              D        _           L-1         _
			//   ~(L)     -----     |        /  ----- ~(l)    |
			//   n     =   | |      |  N    /    | |  n       |
			//     (L)        (L)   |   k  /     l=1   k      |
			//    d        k=d      |     /                   |
			//

			int nMaxIndexingLevel = bCalculateAllSizes ? m_nIndexingLevelsCount : m_nIndexingLevelsCount - 1;
			for (int l = 0; l < nMaxIndexingLevel; l++)
			{
				IIndexing.IIndexingLevel level = m_cg.getIndexingLevel (l);
				int nDimensionality = level.getDimensionality ();

				// fill the sizes up to the dimensionality of the level - 1
				int j = 0;
				if (l < m_nIndexingLevelsCount - 1)
				{
					// first m_nIndexingLevelsCount-1 levels

					for ( ; j < Math.min (nDimensionality, m_nTargetDimension); j++)
						m_rgBlockSizes[l][j] = getSizeForDimension (level, j);
					for ( ; j < m_nTargetDimension; j++)
						m_rgBlockSizes[l][j] = Globals.ONE.clone ();
				}
				else
				{
					// l = L (last indexing level)
					Expression exprCeil = null;
					for (j = 0; j < m_nTargetDimension; j++)
					{
						Expression exprProduct = use (m_rgBlockSizes, 0, j);
						if (exprProduct != null)
							exprProduct = exprProduct.clone ();

						for (int k = 1; k < m_nIndexingLevelsCount - 1; k++)
							exprProduct = new BinaryExpression (exprProduct, BinaryOperator.MULTIPLY, use (m_rgBlockSizes, k, j).clone ());

						if (exprProduct == null)
							exprCeil = m_sizeDomain.getCoord (j).clone ();
						else
							exprCeil = ExpressionUtil.ceil (m_sizeDomain.getCoord (j).clone (), exprProduct);

						if (j <= nDimensionality - 1)
							m_rgBlockSizes[l][j] = exprCeil;
						else
						{
							m_rgBlockSizes[l][nDimensionality - 1] =
								new BinaryExpression (use (m_rgBlockSizes, l, nDimensionality - 1).clone (), BinaryOperator.MULTIPLY, exprCeil);
							m_rgBlockSizes[l][j] = Globals.ONE.clone ();
						}
					}
				}
			}
		}

		/**
		 *
		 * @param rgStrides
		 * @param rgSizes
		 * @param cmpstmt
		 */
		private void calculateStrides ()
		{
			// strides (number of level L blocks per grid):
			//                                                                        (L)
			//            /  1                                        for  0 <= j <= d   - 1
			//            |            _           L-1         _
			//  s    =   <            |        /  -----  ~(l)   |           (L)
			//   j        |  s     .  |  N    /    | |   n      |     for  d    <= j <= D
			//            \   j-1     |   j  /     l=1    j     |
			//

			IIndexing.IIndexingLevel level = m_cg.getIndexingLevel (m_nIndexingLevelsCount - 1);
			int j = 0;

			for ( ; j < level.getDimensionality () - 1; j++)
				m_rgStrides[j] = null;	//  null means 1 (the neutral element for multiplication)

			for ( ; j < m_nTargetDimension; j++)
			{
				Expression exprProduct = use (m_rgBlockSizes, 0, j);
				if (exprProduct != null)
					exprProduct = exprProduct.clone ();

				for (int k = 1; k < m_nIndexingLevelsCount - 1; k++)
				{
					Expression exprBlockSize = use (m_rgBlockSizes, k, j);
					if (exprBlockSize != null)
						exprProduct = new BinaryExpression (exprProduct, BinaryOperator.MULTIPLY, exprBlockSize.clone ());
				}

				Expression exprCeil = exprProduct == null ?
					m_sizeDomain.getCoord (j).clone () :
					ExpressionUtil.ceil (m_sizeDomain.getCoord (j).clone (), exprProduct);

				m_rgStrides[j] = ((j == 0) || (m_rgStrides[j - 1] == null)) ?
					exprCeil :
					new BinaryExpression (use (m_rgStrides, j - 1).clone (), BinaryOperator.MULTIPLY, exprCeil);
			}
		}

		/**
		 *
		 * @param rgIndices
		 * @param rgStrides
		 * @param cmpstmt
		 */
		private void calculateIndices ()
		{
			// indices:
			//
			// if l < m_nIndexingLevelsCount ( =: L) :
			//
			//            /  (l)                  (l)
			//   ~(l)     | i      if  1 <= j <= d,
			//   i    =  <   j
			//    j       |
			//            \  0     otherwise
			//
			// else:
			//                             D
			//           |  /            -----             \    /        |
			//   ~(L)    |  |   (L)       \     ~(L)       |   /         |                            (L)
			//   i    =  |  |  i      --  /     i    s     |  /   s      |      for j = D, D-1, ..., d
			//    j      |  \    (L)     -----   k    k-1  / /     j-1   |
			//           +--    d        k=j+1                         --+
			//
			//     ~(L)     (L)                 (L)
			// or  i    =  i     if  1 <= j <= d    - 1
			//      j       j
			//

			for (int l = 0; l < m_nIndexingLevelsCount; l++)
			{
				IIndexing.IIndexingLevel level = m_cg.getIndexingLevel (l);
				int nDimensionality = level.getDimensionality ();

				// fill the sizes up to the dimensionality of the level - 1
				if (l < m_nIndexingLevelsCount - 1)
				{
					// first m_nIndexingLevelsCount-1 levels
					int j = 0;
					for ( ; j < Math.min (nDimensionality, m_nTargetDimension); j++)
						m_rgIndices[l][j] = getIndexForDimension (level, j).clone ();
					for ( ; j < m_nTargetDimension; j++)
						m_rgIndices[l][j] = Globals.ZERO.clone ();
				}
				else
				{
					// l = last indexing level

					Expression exprLast = getIndexForDimension (level, nDimensionality - 1).clone ();
					if (m_nTargetDimension == nDimensionality)
						m_rgIndices[l][m_nTargetDimension - 1] = exprLast;
					else
					{
						// get the index that is used to emulate the missing dimensions
						Identifier idTmp = createIdentifier ("tmp", exprLast);
						m_rgIndices[l][m_nTargetDimension - 1] = divide (idTmp.clone (), use (m_rgStrides, m_nTargetDimension - 2).clone ());

						for (int j = m_nTargetDimension - 2; j >= nDimensionality - 1; j--)
						{
							// tmp -= i(j+1) * s(j)
							Expression exprStrideJ = use (m_rgStrides, j);
							Expression exprIndexJ1 = use (m_rgIndices, l, j + 1);

							addStatement (new ExpressionStatement (new AssignmentExpression (
								idTmp.clone (),
								AssignmentOperator.SUBTRACT,
								exprStrideJ == null ?
									exprIndexJ1.clone () :
									new BinaryExpression (exprIndexJ1.clone (), BinaryOperator.MULTIPLY, exprStrideJ.clone ()))));

							Expression exprStrideJ_1 = j == 0 ? null : use (m_rgStrides, j - 1);
							m_rgIndices[l][j] = createIdentifier (
								m_rgIndices,
								divide (idTmp.clone (), j == 0 ? null : (exprStrideJ_1 == null ? null : exprStrideJ_1.clone ())), l, j);
						}
					}

					for (int j = 0; j < nDimensionality - 1; j++)
						m_rgIndices[l][j] = getIndexForDimension (level, j).clone ();
				}
			}
		}

		/**
		 *
		 * @param rgTargetIndices
		 * @param rgIndices
		 * @param rgSizes
		 * @param cmpstmt
		 */
		private void calculateTargetIndices ()
		{
			// for j = 1, ...,  D (target dimension)
			//
			//
			//         ^       (1)     (1)   /   (2)     (2)   /     /   (k-1)     (k-1)     (k)  \     \\
			//         i   =  i    +  n    . |  i    +  n    . | ... |  i      +  n      .  i     | ... ||
			//          j      j       j     \   j       j     \     \   j         j         j    /     //
			//

			for (int j = 0; j < m_nTargetDimension; j++)
			{
				Expression[] rgIndices = new Expression[m_nIndexingLevelsCount];
				Expression[] rgSizes = new Expression[m_nIndexingLevelsCount];
				for (int l = 0; l < m_nIndexingLevelsCount; l++)
				{
					rgIndices[l] = use (m_rgIndices, l, j).clone ();

					if (l < m_nIndexingLevelsCount - 1)
						rgSizes[l] = use (m_rgBlockSizes, l, j).clone ();
					else
						rgSizes[l] = null;
				}

				m_rgTargetIndices[j] = IndexCalculatorCodeGenerator.calculateMultiToOne (rgIndices, rgSizes);
			}
		}

		private Identifier createIdentifier (Object objArray, Expression expr, int... rgIndices)
		{
			// find the name for the identifier
			String strName = "";
			if (objArray == m_rgIndices)
				strName = "idx";
			else if (objArray == m_rgStrides)
				strName = "stride";
			else if (objArray == m_rgBlockSizes)
				strName = "size";
			else if (objArray == m_rgTargetIndices)
				strName = "idx_target";
			else if (objArray == m_rgTargetSizes)
				strName = "size_target";
			else
				throw new RuntimeException ("Array not found in the list of predefined arrays");

			StringBuilder sbName = new StringBuilder (strName);
			for (int nIdx : rgIndices)
			{
				sbName.append ('_');
				sbName.append (nIdx);
			}

			return createIdentifier (sbName.toString (), expr.clone ());
		}

		/**
		 *
		 * @param objArray
		 * @param rgIndices
		 * @return
		 * @throws RuntimeException, ClassCastException
		 */
		private Expression use (Object objArray, int... rgIndices)
		{
			Object obj = objArray;
			Object objCo1DArray = objArray;
			for (int i = 0; i < rgIndices.length; i++)
			{
				obj = Array.get (obj, rgIndices[i]);
				if (i == rgIndices.length - 2)
					objCo1DArray = obj;
			}

			Expression expr = (Expression) obj;

			if ((expr instanceof BinaryExpression || expr instanceof Typecast || expr instanceof FunctionCall) && !(expr instanceof AccessExpression))
			{
				// replace the array entry by an identifier, which is to be created
				Identifier id = createIdentifier (objArray, expr.clone (), rgIndices);
				Array.set (objCo1DArray, rgIndices[0], id);
				return id;
			}

			return expr;
		}

		private Identifier createIdentifier (String strName, Expression exprValue)
		{
			return IndexCalculatorCodeGenerator.this.createIdentifier (strName, exprValue, m_cmpstmt);
		}

		private void addStatement (Statement stmt)
		{
			IndexCalculatorCodeGenerator.this.addStatement (stmt, m_cmpstmt);
		}

		private Expression getIndexForDimension (IIndexingLevel level, int nDimension)
		{
			Expression exprIdx = level.getIndexForDimension (nDimension);
			if (level.isVariable ())
				return exprIdx;

			return m_data.getCodeGenerators ().getConstantGeneratedIdentifiers ().getConstantIdentifier (
				exprIdx, "dimidx", Globals.SPECIFIER_INDEX, null, null, m_options);
		}

		private Expression getSizeForDimension (IIndexingLevel level, int nDimension)
		{
			Expression exprSize = level.getSizeForDimension (nDimension);
			if (level.isVariable ())
				return exprSize;

			return m_data.getCodeGenerators ().getConstantGeneratedIdentifiers ().getConstantIdentifier (
				exprSize, "dimsize", Globals.SPECIFIER_SIZE, null, null, m_options);
		}
	}


	private static class SizesAndStrides
	{
		private Expression[] m_rgSizes;
		private Expression[] m_rgStrides;

		public SizesAndStrides (Expression[] rgSizes, Expression[] rgStrides)
		{
			m_rgSizes = rgSizes;
			m_rgStrides = rgStrides;
		}

		public Expression[] getSizes ()
		{
			return m_rgSizes;
		}

		public Expression[] getStrides ()
		{
			return m_rgStrides;
		}

		/**
		 * Determines whether the sizes stored in this object match the sizes in
		 * <code>rgSizes</code>. If there are more sizes in <code>rgSizes</code> than in the
		 * internal array, always <code>false</code> is returned. If there are less sizes
		 * in <code>rgSizes</code> than in the internal array, <code>true</code> is returned
		 * if the first <code>rgSizes.length</code> sizes are equal.
		 * @param rgSizes The sizes to examine
		 * @return <code>true</code> iff the first <code>rgSizes.length</code> sizes in the
		 * 	internal array are equal to the sizes in <code>rgSizes</code>
		 */
		public boolean areSizesEqualTo (Expression[] rgSizes)
		{
			if (rgSizes.length > m_rgSizes.length)
				return false;
			for (int i = 0; i < rgSizes.length; i++)
				if (!m_rgSizes[i].equals (rgSizes[i]))
					return false;
			return true;
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;

	private List<SizesAndStrides> m_listSizesAndStrides;
	private int m_nTmpStrides;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public IndexCalculatorCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;

		m_listSizesAndStrides = new ArrayList<> ();
		m_nTmpStrides = 0;
	}

	/**
	 * Creates an expression dividing <code>exprNumerator</code> by
	 * <code>exprDenominator</code>.
	 * 
	 * @param exprNumerator
	 *            The numerator
	 * @param exprDenominator
	 *            The denominator
	 * @return Returns an expression computing <code>exprNumerator</code> /
	 *         <code>exprDenominator</code>
	 */
	protected static Expression divide (Expression exprNumerator, Expression exprDenominator)
	{
		if (exprDenominator == null || ((exprDenominator instanceof IntegerLiteral) && (((IntegerLiteral) exprDenominator).getValue () == 1)))
			return exprNumerator.clone ();
		return new BinaryExpression (exprNumerator.clone (), BinaryOperator.DIVIDE, exprDenominator.clone ());
	}

	/**
	 * Creates a new identifier named <code>strName</code> and assigns it the
	 * value <code>exprValue</code>.
	 * 
	 * @param strName
	 *            The name of the identifier to create
	 * @param exprValue
	 *            The value to assign
	 * @return The identifier object
	 */
	protected Identifier createIdentifier (String strName, Expression exprValue, CompoundStatement cmpstmt)
	{
		// create a new identifier
		VariableDeclarator decl = new VariableDeclarator (new NameID (strName));
		Identifier id = new Identifier (decl);

		// memorize the variable declaration
		m_data.getData ().addDeclaration (new VariableDeclaration (Globals.SPECIFIER_INDEX, decl));

		// add the new statement to the code
		addStatement (new ExpressionStatement (new AssignmentExpression (id, AssignmentOperator.NORMAL, exprValue)), cmpstmt);

		return id;
	}

	protected void addStatement (Statement stmt, CompoundStatement cmpstmt)
	{
		if (cmpstmt == null)
			m_data.getData ().addInitializationStatement (stmt);
		else
			cmpstmt.addStatement (stmt);
	}

	protected Expression[] findStrides (Expression[] rgSizes, CompoundStatement cmpstmt)
	{
		if (FIND_PREVIOUSLY_CALCULATED_STRIDES)
		{
			for (SizesAndStrides s : m_listSizesAndStrides)
			{
				if (s.areSizesEqualTo (rgSizes))
					return s.getStrides ();
			}
		}

		// strides matching sizes not found; create new strides
		Expression[] rgStrides = new Expression[rgSizes.length];
		rgStrides[0] = new IntegerLiteral (1);

		if (rgStrides.length > 1)
		{
			if (rgSizes[0] instanceof BinaryExpression)
			{
				rgStrides[1] = createIdentifier (
					StringUtil.concat ("tmp_stride_", m_nTmpStrides, CodeGeneratorUtil.getDimensionName (1)),
					rgSizes[0].clone (),
					cmpstmt);
			}
			else
				rgStrides[1] = rgSizes[0];
	
			for (int i = 2; i < rgSizes.length; i++)
			{
				rgStrides[i] = createIdentifier (
					StringUtil.concat ("tmp_stride_", m_nTmpStrides, CodeGeneratorUtil.getDimensionName (i)),
					new BinaryExpression (rgStrides[i - 1].clone (), BinaryOperator.MULTIPLY, rgSizes[i - 1].clone ()),
					cmpstmt);
			}
		}
		
		if (FIND_PREVIOUSLY_CALCULATED_STRIDES)
			m_listSizesAndStrides.add (new SizesAndStrides (rgSizes, rgStrides));

		return rgStrides;
	}

	private Identifier createIndexIdentifier (Expression exprIndexCalculation, Identifier[] rgIndex, int nDim, CompoundStatement cmpstmt)
	{
		Identifier idIndex = null;
		if (rgIndex == null)
			idIndex = createIdentifier (StringUtil.concat ("tmpidxa", m_nTmpStrides, "_", nDim), exprIndexCalculation, cmpstmt);
		else
		{
			idIndex = rgIndex[nDim];
			addStatement (new ExpressionStatement (new AssignmentExpression (idIndex, AssignmentOperator.NORMAL, exprIndexCalculation)), cmpstmt);
		}
		return idIndex;
	}

	/**
	 * Calculates a <i>d</i>-dimensional index from a one-dimensional one,
	 * <code>exprIndex</code>.
	 * 
	 * @param exprIndex
	 *            The one-dimensional index from which the multi-dimensional one
	 *            is calculated
	 * @param rgIndex
	 *            The multi-dimensional index; an array of identifiers to which
	 *            the calculations
	 *            will be assigned. Can be <code>null</code>; in this case,
	 *            temporary variables will be created
	 *            and returned.
	 * @param rgSizes
	 *            The array of sizes. Can be <code>null</code> if
	 *            <code>rgStrides</code> is not <code>null</code>
	 * @param rgStrides
	 *            The array of strides or <code>null</code> if not known and the
	 *            strides are
	 *            to be automatically calculated from the sizes,
	 *            <code>rgSizes</code>
	 * @param cmpstmt
	 *            The compound statement to which the temporary calculations are
	 *            added
	 * @return An array containing the identifiers holding the values of the
	 *         indices
	 */
	public Expression[] calculateOneToMulti (Expression exprIndex, Identifier[] rgIndex, Expression[] rgSizes, Expression[] rgStrides, CompoundStatement cmpstmt)
	{
		if (rgSizes == null && rgStrides == null)
			throw new RuntimeException ("Either sizes or strides must be provided.");

		int nDim = rgSizes == null ? rgStrides.length : rgSizes.length;
		if (rgIndex != null && rgIndex.length < nDim)
			throw new RuntimeException ("The length of the index identifier array rgIndex must contain at least as many elements as rgSizes / rgStrides.");

		Expression[] rgResult = new Expression[nDim];

		// if no strides are given, calculate them first
		if (rgStrides == null)
			rgStrides = findStrides (rgSizes, cmpstmt);

		// last index: i(m_nDimensionExpand-1) = i' / s(m_nDimensionExpand-1)
		// note that arguments to "divide" will automatically be cloned if necessary
		rgResult[nDim - 1] = createIndexIdentifier (divide (exprIndex, rgStrides[nDim - 1]), rgIndex, nDim - 1, cmpstmt);

		// other indices
		if (nDim > 1)
		{
			// tmp = i' - i(m_nDimensionExpand-1) * s(m_nDimensionExpand-1)
			Identifier idTmp = createIdentifier (
				StringUtil.concat ("tmpidxc", m_nTmpStrides),
				new BinaryExpression (
					exprIndex.clone (),
					BinaryOperator.SUBTRACT,
					new BinaryExpression (rgResult[nDim - 1].clone (), BinaryOperator.MULTIPLY, rgStrides[nDim - 1].clone ())
				),
				cmpstmt
			);

			// i(m_nDimensionExpand-2) = tmp / s(m_nDimensionExpand-2)
			rgResult[nDim - 2] = createIndexIdentifier (divide (idTmp, rgStrides[nDim - 2]), rgIndex, nDim - 2, cmpstmt);

			for (int l = nDim - 3; l >= 0; l--)
			{
				// tmp -= i(l+1) * s(l)
				addStatement (
					new ExpressionStatement (new AssignmentExpression (
						idTmp.clone (),
						AssignmentOperator.SUBTRACT,
						new BinaryExpression (rgResult[l + 1].clone (), BinaryOperator.MULTIPLY, rgStrides[l + 1].clone ()))),
					cmpstmt);

				// i(l) = tmp / s(l-1)
				rgResult[l] = createIndexIdentifier (divide (idTmp, rgStrides[l]), rgIndex, l, cmpstmt);
			}
		}

		m_nTmpStrides++;
		return rgResult;
	}

	/**
	 * Calculates a one-dimensional index from the multi-dimensional index
	 * <code>rgIndex</code>.
	 * 
	 * @param rgIndex
	 *            The index
	 * @param rgSizes
	 *            The sizes, must be of the same size as <code>rgIndex</code>
	 * @return The one-dimensional index obtained from the multi-index
	 *         <code>rgIndex</code>
	 */
	public static Expression calculateMultiToOne (Expression[] rgIndex, Expression[] rgSizes)
	{
		if (rgIndex.length != rgSizes.length)
			throw new RuntimeException ("Index and sizes must be of the same size.");

		Expression exprIdx = rgIndex[rgIndex.length - 1].clone ();
		for (int i = rgIndex.length - 2; i >= 0; i--)
		{
			if (!ExpressionUtil.isValue (rgSizes[i], 1))
			{
				exprIdx = new BinaryExpression (
					new BinaryExpression (rgSizes[i].clone (), BinaryOperator.MULTIPLY, exprIdx.clone ()),
					BinaryOperator.ADD,
					rgIndex[i].clone ());
			}
		}

		return exprIdx;
	}
	
	/**
	 * Collapses the multi-dimensional hardware index to a one-dimensional one.
	 * 
	 * @param nDimension
	 *            The dimension for which to collapse the hardware index
	 * @param nParallelismLevelStart
	 *            The first (lowest) parallelism level the final index should
	 *            depend on
	 * @param nParallelismLevelEnd
	 *            The last (highest) parallelism level the final index should
	 *            depend on
	 * @return A one-dimensional index computed from the possibly
	 *         multi-dimensional one
	 */
	public Expression calculateHardwareIndicesToOne (int nDimension, int nParallelismLevelStart, int nParallelismLevelEnd)
	{
		IIndexing indexing = m_data.getCodeGenerators ().getBackendCodeGenerator ();
		int nIndexingLevels = Math.min (nParallelismLevelEnd - nParallelismLevelStart + 1, indexing.getIndexingLevelsCount ());
		if (nIndexingLevels <= 0)
			throw new RuntimeException ("The end parallelism level must be at least the start parallelism level");
		
		Expression[] rgIndex = new Expression[nIndexingLevels];
		Expression[] rgSizes = new Expression[nIndexingLevels];
		
		int i = 0;
		for (int nParallelismLevel = nParallelismLevelEnd; nParallelismLevel >= nParallelismLevelStart; nParallelismLevel--)
		{
			IIndexingLevel level = indexing.getIndexingLevelFromParallelismLevel (nParallelismLevel);
			if (level == null)
				continue;
			
			if (nDimension < level.getDimensionality ())
			{
				rgIndex[i] = level.getIndexForDimension (nDimension);
				rgSizes[i] = level.getSizeForDimension (nDimension);
			}
			else
			{
				rgIndex[i] = Globals.ZERO.clone ();
				rgSizes[i] = Globals.ONE.clone ();
			}
			i++;
		}
		
		return Symbolic.simplify (IndexCalculatorCodeGenerator.calculateMultiToOne (rgIndex, rgSizes));
	}

	/**
	 * Calculates a d-dimensional index with index values restricted to &le;
	 * <code>sizeDomain</code>,
	 * where the target dimensionality d is
	 * <code>sizeDomain.getDimensionality ()</code>.<br/>
	 * Intermediate expressions that are generated to compute the index are
	 * added to <code>cmpstmt</code>.
	 * 
	 * @param sizeDomain
	 * @param cmpstmt
	 *            The compound statement to which intermediate calculations are
	 *            added. If <code>null</code>,
	 *            the statements are added to the initialization statement
	 * @return A d-dimensional index
	 */
	public Expression[] calculateIndicesFromHardwareIndices (Size sizeDomain, CompoundStatement cmpstmt, CodeGeneratorRuntimeOptions options)
	{
		return calculateIndices (m_data.getCodeGenerators ().getBackendCodeGenerator (), sizeDomain, cmpstmt, options);
	}
	
	/**
	 * Returns the total size of the addressable hardware index in dimension <code>nDimension</code>.
	 * @param nDimension
	 * @return
	 */
	public Expression calculateTotalHardwareSize (int nDimension, int nParallelismLevelStart, int nParallelismLevelEnd)
	{
		IIndexing indexing = m_data.getCodeGenerators ().getBackendCodeGenerator ();
		int nIndexingLevels = Math.min (nParallelismLevelEnd - nParallelismLevelStart + 1, indexing.getIndexingLevelsCount ());
		if (nIndexingLevels <= 0)
			throw new RuntimeException ("The end parallelism level must be at least the start parallelism level");
		
		Expression exprTotalSize = null;
		
		for (int nParallelismLevel = nParallelismLevelEnd; nParallelismLevel >= nParallelismLevelStart; nParallelismLevel--)
		{
			IIndexingLevel level = indexing.getIndexingLevelFromParallelismLevel (nParallelismLevel);
			if (level == null)
				continue;
			if (nDimension >= level.getDimensionality ())
				continue;
			
			Expression exprSize = level.getSizeForDimension (nDimension);
			if (!ExpressionUtil.isValue (exprSize, 1))
			{
				if (exprTotalSize == null)
					exprTotalSize = exprSize;
				else
					exprTotalSize = new BinaryExpression (exprTotalSize, BinaryOperator.MULTIPLY, exprSize);
			}			
		}

		return exprTotalSize == null ? Globals.ONE.clone () : exprTotalSize;
	}
	
	/**
	 * Converts the d-dimensional index <code>rgIndices</code> to a
	 * D-dimensional target index
	 * (D being the dimensionality of the domain).
	 * 
	 * @param rgIndices
	 *            The d-dimensional index to convert
	 * @param sizeDomain
	 *            The D-dimensional domain
	 * @param cmpstmt
	 *            The compound statement to which auxiliary calculations are
	 *            added as the index is converted
	 * @param options
	 *            Code generation options
	 * @return The D-dimensional target index
	 */
	public Expression[] convertIndices (final Expression[] rgIndices, final Expression[] rgSizes,
		Size sizeDomain, CompoundStatement cmpstmt, CodeGeneratorRuntimeOptions options)
	{
		return calculateIndices (
			new IIndexing ()
			{
				@Override
				public int getIndexingLevelsCount ()
				{
					return 1;
				}

				@Override
				public IIndexingLevel getIndexingLevel (int nIndexingLevel)
				{
					return new IIndexingLevel ()
					{
						@Override
						public boolean isVariable ()
						{
							for (Expression exprIdx : rgIndices)
								if (!(exprIdx instanceof IDExpression) && !(exprIdx instanceof Literal))
									return false;
							return true;
						}
						
						@Override
						public Expression getSizeForDimension (int nDimension)
						{
							return rgSizes[nDimension];
						}
						
						@Override
						public Expression getIndexForDimension (int nDimension)
						{
							return rgIndices[nDimension];
						}
						
						@Override
						public int getDimensionality ()
						{
							return rgIndices.length;
						}
						
						@Override
						public int getDefaultBlockSize (int nDimension)
						{
							return 0;
						}
					};
				}

				@Override
				public IIndexingLevel getIndexingLevelFromParallelismLevel (int nParallelismLevel)
				{
					return IndexingLevelUtil.getIndexingLevelFromParallelismLevel (this, nParallelismLevel);
				}

				@Override
				public EThreading getThreading ()
				{
					return EThreading.MULTI;
				}				
			},
			sizeDomain, cmpstmt, options);
	}
	
	public Expression[] calculateIndices (IIndexing indexing, Size sizeDomain, CompoundStatement cmpstmt, CodeGeneratorRuntimeOptions options)
	{
		return new IndexCalculatorCodeGenerator.Calculator (
			indexing, sizeDomain, cmpstmt, ECalculationMode.CALCULATE_INDICES, options
		).getTargetIndices ();
	}

	public Expression[] calculateSizes (Size sizeDomain, CompoundStatement cmpstmt, CodeGeneratorRuntimeOptions options)
	{
		return new IndexCalculatorCodeGenerator.Calculator (
			m_data.getCodeGenerators ().getBackendCodeGenerator (), sizeDomain, cmpstmt, ECalculationMode.CALCULATE_SIZES, options
		).getTargetSizes ();
	}
}
