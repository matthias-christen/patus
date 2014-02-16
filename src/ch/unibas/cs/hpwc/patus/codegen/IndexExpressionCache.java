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

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.log4j.Logger;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.IStatementList;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Point;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.geometry.Vector;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.symbolic.ExpressionData;
import ch.unibas.cs.hpwc.patus.symbolic.ExpressionOptimizer;
import ch.unibas.cs.hpwc.patus.symbolic.NotConvertableException;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Caches index expressions and creates index variables for &quot;complicated&quot; index expressions.
 * @author Matthias-M. Christen
 */
public class IndexExpressionCache
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (IndexExpressionCache.class);


	///////////////////////////////////////////////////////////////////
	// Inner Types

	private static class IndexInfo
	{
		private int[] m_rgNodeSpaceIndex;

		private Point m_ptBoxRefPoint;

		private Size m_sizeOffset;


		public IndexInfo (StencilNode node, Vector vecOffsetWithinMemObj, Point ptBoxRefPoint, Size sizeOffset)
		{
			m_rgNodeSpaceIndex = node.getIndex ().getSpaceIndex ();
			m_ptBoxRefPoint = ptBoxRefPoint;
			m_sizeOffset = new Size (sizeOffset);
			if (vecOffsetWithinMemObj != null)
				m_sizeOffset.add (vecOffsetWithinMemObj);
		}

		public int[] getNodeSpaceIndex ()
		{
			return m_rgNodeSpaceIndex;
		}

		public Point getBoxRefPoint ()
		{
			return m_ptBoxRefPoint;
		}

		public Size getLocalOriginMinusOriginMinusStencilOffset ()
		{
			return m_sizeOffset;
		}

		@Override
		public boolean equals (Object obj)
		{
			if (!(obj instanceof IndexInfo))
				return false;

			return Arrays.equals (m_rgNodeSpaceIndex, ((IndexInfo) obj).getNodeSpaceIndex ()) && m_ptBoxRefPoint.equals (((IndexInfo) obj).getBoxRefPoint ())
				&& m_sizeOffset.equals (((IndexInfo) obj).getLocalOriginMinusOriginMinusStencilOffset ());
		}

		@Override
		public int hashCode ()
		{
			return Arrays.hashCode (m_rgNodeSpaceIndex) + m_ptBoxRefPoint.hashCode () + 1009 * m_sizeOffset.hashCode ();
		}

		@Override
		public String toString ()
		{
			return StringUtil.concat ("stencil node: ", Arrays.toString (m_rgNodeSpaceIndex), ", ", "ref point: ", m_ptBoxRefPoint.toString (), ", ", "offset: ",
				m_sizeOffset.toString ());
		}
	}

	/**
	 * An index value encapsulating the calculated expression as well as the
	 * index variable to which the expression is assigned if the expression is
	 * &quot;complicated&quot;.
	 *
	 * @author Matthias-M. Christen
	 */
	private class IndexValue
	{
		/**
		 * The original, computed index expression
		 */
		private Expression m_exprIndex;

		/**
		 * The simplified index expression, i.e. the index expressed in terms of other indices
		 */
		private Expression m_exprSimplifiedIndex;

		/**
		 * The index expression as a string (key for the {@link IndexExpressionCache#m_mapIdentifiers}
		 * map that maps expressions to index variables)
		 */
		private String m_strIndexExpression;

		private ExpressionData m_dataIndexExpression;

		/**
		 * The index variable that is used in the stencil expression (rather than the original
		 * index expression)
		 */
		private Identifier m_idIndexVariable;

		/**
		 * Flag specifying whether the index variable has to be recomputed (if we enter a new loop nest)
		 */
		private boolean m_bNeedsRecomputation;

		/**
		 * Identifier on which the computation of this index value depends (note that by construction
		 * an index value can only depend on one other index identifier)
		 */
		private IndexValue m_ivDependence;


		/**
		 * Constructs the index value.
		 * @param exprIndex The expression computing the index
		 * @param stmtlist The statement list to which the computation statement is added
		 */
		public IndexValue (Expression exprIndex, IStatementList stmtlist, boolean bNoVectorize)
		{
			m_exprIndex = exprIndex;
			m_dataIndexExpression = new ExpressionData (exprIndex, ExpressionUtil.getNumberOfFlops (exprIndex), Symbolic.EExpressionType.EXPRESSION);

			m_exprSimplifiedIndex = null;
			m_idIndexVariable = null;
			m_strIndexExpression = "";
			m_bNeedsRecomputation = false;

			if (exprIndex instanceof BinaryExpression)
			{
				m_strIndexExpression = exprIndex.toString ();

				IndexValue iv = getIndexValueByExpression (m_strIndexExpression, bNoVectorize);
				if (iv == null)
				{
					m_idIndexVariable = createNewIndexIdentifier ();
					setIndexValueByExpression (m_strIndexExpression, this, bNoVectorize);
					addComputationStatement (stmtlist, bNoVectorize);
				}
				else
				{
					m_exprSimplifiedIndex = iv.m_exprSimplifiedIndex == null ? null : iv.m_exprSimplifiedIndex.clone ();
					m_idIndexVariable = iv.m_idIndexVariable == null ? createNewIndexIdentifier () : iv.m_idIndexVariable;
					m_bNeedsRecomputation = iv.m_bNeedsRecomputation;
					m_ivDependence = iv.m_ivDependence;
				}
			}
			else if (exprIndex instanceof Identifier)
				m_idIndexVariable = (Identifier) exprIndex;
			else if (exprIndex instanceof IntegerLiteral)
				;
			else
				throw new RuntimeException (StringUtil.concat ("The index expression must be a binary expression or an IDExpression (was a ", exprIndex.getClass ().getName (), ")."));
		}

		private Identifier createNewIndexIdentifier ()
		{
			VariableDeclarator decl = new VariableDeclarator (CodeGeneratorUtil.createNameID ("_idx", m_nCachedIndicesCount++));
			Identifier idIndexVariable = new Identifier (decl);
			m_data.getData ().addDeclaration (new VariableDeclaration (Globals.SPECIFIER_INDEX, decl));

			return idIndexVariable;
		}

		/**
		 * Adds the statement to the statement list <code>stmtlist</code> that
		 * recomputes the index.
		 *
		 * @param stmtlist
		 *            The list of statements to which to add the recompute
		 *            statement
		 */
		public void addComputationStatement (IStatementList stmtlist, boolean bNoVectorize)
		{
			// there's nothing to do if the index variable is null (the index expression
			// might be a literal, in which case no recomputation has to be done anyway)
			if (m_idIndexVariable == null)
				return;

			if (m_exprSimplifiedIndex == null)
			{
				try
				{
					m_exprSimplifiedIndex = computeSimplifiedIndex (m_strIndexExpression, m_dataIndexExpression, bNoVectorize, m_exprIndex);
				}
				catch (NotConvertableException e)
				{
					// something went wrong when trying to simplify... ignore and use the original expression
				}

				// computeSimplifiedIndex returns null if the index can't be simplified
				if (m_exprSimplifiedIndex == null)
					m_exprSimplifiedIndex = m_exprIndex;
			}

			// add compute statements on which this index computation depends on before this on (if the
			// index on which this index depends needs to be calculated again)
			if (m_ivDependence != null && m_ivDependence.needsRecomputation ())
				m_ivDependence.addComputationStatement (stmtlist, bNoVectorize);

			// add a statement that recomputes the index
			stmtlist.addStatement (CodeGeneratorUtil.createComment (StringUtil.concat (m_idIndexVariable.getName (), " = ", m_strIndexExpression), true));
			stmtlist.addStatement (new ExpressionStatement (new AssignmentExpression (m_idIndexVariable.clone (), AssignmentOperator.NORMAL, m_exprSimplifiedIndex.clone ())));

			// recomputation has just been done, no need to recompute now until
			// the flag is reset
			m_bNeedsRecomputation = false;

			// set all flags belonging to the same expression to false
			Map<Size, Map<IndexInfo, IndexValue>> mapOptions = m_mapCaches.get (bNoVectorize);
			if (mapOptions != null)
			{
				for (Size size : mapOptions.keySet ())
				{
					Map<IndexInfo, IndexValue> map = mapOptions.get (size);
					for (IndexInfo ii : map.keySet ())
					{
						IndexValue iv = map.get (ii);
						if (iv != null && m_strIndexExpression.equals (iv.m_strIndexExpression))
							iv.m_bNeedsRecomputation = false;
					}
				}
			}
		}

		/**
		 * @throws NotConvertableException
		 *
		 */
		private Expression computeSimplifiedIndex (String strIndexExpression, ExpressionData dataIndexExpression, boolean bNoVectorize, Expression... rgExprOrigs) throws NotConvertableException
		{
			// ignore if the expression string is empty
			if (strIndexExpression.equals (""))
				return null;
			
			// if there is only one expression in the map, we can't simplify
			Map<String, IndexValue> mapIdentifiers = m_mapIdentifiers.get (bNoVectorize);
			if (mapIdentifiers == null || mapIdentifiers.size () <= 1)
				return ExpressionOptimizer.optimize (m_exprIndex, Symbolic.ALL_VARIABLES_INTEGER);
				//return null;

			ExpressionData dataOptimized = ExpressionOptimizer.optimizeEx (strIndexExpression, Symbolic.ALL_VARIABLES_INTEGER, dataIndexExpression);

			int nMinFlops = dataOptimized.getFlopsCount ();
			Expression exprMinFlops = dataOptimized.getExpression ();
			Identifier idMinFlopsBase = null;

			for (String strExpr : mapIdentifiers.keySet ())
			{
				if (strIndexExpression.equals (strExpr))
					continue;

				// try to simplify the expression (idxN-idxM) as much as possible
				String strDerivedExpression = StringUtil.concat ("(", strIndexExpression, ") - (", strExpr, ")");
				ExpressionData dataDerived = new ExpressionData (
					new BinaryExpression (dataIndexExpression.getExpression ().clone (), BinaryOperator.SUBTRACT, mapIdentifiers.get (strExpr).getIndexVariable ().clone ()),
					dataIndexExpression.getFlopsCount () + 1,
					Symbolic.EExpressionType.EXPRESSION);

				ExpressionData ed = ExpressionOptimizer.optimizeEx (strDerivedExpression, Symbolic.ALL_VARIABLES_INTEGER, dataDerived, rgExprOrigs);
				if (ed.getFlopsCount () < nMinFlops)
				{
					nMinFlops = ed.getFlopsCount ();
					exprMinFlops = ed.getExpression ();

					m_ivDependence = mapIdentifiers.get (strExpr);
					idMinFlopsBase = m_ivDependence.getIndexVariable ();
				}

				// we can't go below that...
				if (nMinFlops <= 1)
					break;
			}

			if (IndexExpressionCache.LOGGER.isDebugEnabled ())
			{
				if (idMinFlopsBase == null)
					IndexExpressionCache.LOGGER.debug (StringUtil.concat ("Expression", strIndexExpression, " not simplified"));
				else
					IndexExpressionCache.LOGGER.debug (StringUtil.concat ("Simplifying the index expression ", strIndexExpression, " to ", idMinFlopsBase.getName (), "+", exprMinFlops.toString ()));
			}

			if (idMinFlopsBase == null)
				return exprMinFlops;
			return Symbolic.simplify (new BinaryExpression (idMinFlopsBase.clone (), BinaryOperator.ADD, exprMinFlops.clone ()));
		}

		public Expression getValue (IStatementList stmtlist, boolean bNoVectorize)
		{
			if (m_bNeedsRecomputation)
				addComputationStatement (stmtlist, bNoVectorize);
			return m_idIndexVariable == null ? m_exprIndex : m_idIndexVariable;
		}

		public Identifier getIndexVariable ()
		{
			return m_idIndexVariable;
		}

		public boolean needsRecomputation ()
		{
			return m_bNeedsRecomputation;
		}

		public void resetComputationFlag ()
		{
			m_bNeedsRecomputation = true;
		}

		@Override
		public String toString ()
		{
			return m_idIndexVariable == null ? m_exprIndex.toString () : m_idIndexVariable.toString ();
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;

	private Map<Boolean, Map<Size, Map<IndexInfo, IndexValue>>> m_mapCaches;

	/**
	 * Maps expressions to identifiers (to check whether there already is an
	 * identifier for a particular expression)
	 */
	private Map<Boolean, Map<String, IndexValue>> m_mapIdentifiers;

	/**
	 * The number of cached indices
	 */
	private int m_nCachedIndicesCount;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public IndexExpressionCache (CodeGeneratorSharedObjects data)
	{
		m_data = data;

		m_mapCaches = new HashMap<> ();
		m_mapIdentifiers = new HashMap<> ();
		m_nCachedIndicesCount = 0;
	}

	/**
	 *
	 * @param boxIterator
	 * @param node
	 * @param vecOffsetWithinMemObj An offset from the index point defined by
	 * 	<code>sdid</code> or <code>null</code> if no offset is to be added. The offset point must
	 * 	lie within the same memory object.
	 * @param sizeMemoryObject
	 * @param sizeLocalOrigin_minus_Origin_minus_StencilOffset
	 * @param mo
	 * @return
	 */
	public Expression getIndex (
		SubdomainIdentifier sdid, StencilNode node, Vector vecOffsetWithinMemObj, MemoryObject mo, Size sizeLocalOrigin_minus_Origin, IStatementList slGeneratedCode,
		CodeGeneratorRuntimeOptions options)
	{
		// calculate the reference point of the box
		// if the iterator "it" is null, assume that we're talking about the entire domain (with ref point (0,...,0))
		Point ptBoxRef = null;
		if (sdid == null)
			ptBoxRef = Point.getZero (mo.getDimensionality ());
		else
			ptBoxRef = m_data.getData ().getGeneratedIdentifiers ().getIndexPoint (sdid);

		// get keys
		IndexInfo ii = new IndexInfo (node, vecOffsetWithinMemObj, ptBoxRef, sizeLocalOrigin_minus_Origin);

		// get the child maps, create if they don't exist yet
		boolean bNoVectorize = options.getBooleanValue (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, false);
		Map<Size, Map<IndexInfo, IndexValue>> mapOptions = m_mapCaches.get (bNoVectorize);
		if (mapOptions == null)
			m_mapCaches.put (bNoVectorize, mapOptions = new HashMap<> ());
		Size sizeMemoryObject = mo.getSize ();
		Map<IndexInfo, IndexValue> mapIndexInfo = mapOptions.get (sizeMemoryObject);
		if (mapIndexInfo == null)
			mapOptions.put (sizeMemoryObject, mapIndexInfo = new HashMap<> ());

		// check whether index exists, if not, compute it
		IndexValue ivIndex = mapIndexInfo.get (ii);
		if (ivIndex == null)
		{
			// create a new index variable and return it
			mapIndexInfo.put (ii, ivIndex = new IndexValue (mo.computeIndex (sdid, node, vecOffsetWithinMemObj, options), slGeneratedCode, bNoVectorize));
			return ivIndex.getValue (slGeneratedCode, bNoVectorize);
		}

		// return the expression
		return ivIndex.getValue (slGeneratedCode, bNoVectorize);
	}

	protected IndexValue getIndexValueByExpression (String strExpression, boolean bNoVectorize)
	{
		Map<String, IndexValue> map = m_mapIdentifiers.get (bNoVectorize);
		if (map == null)
			m_mapIdentifiers.put (bNoVectorize, map = new HashMap<> ());
		return map.get (strExpression);
	}

	protected void setIndexValueByExpression (String strExpression, IndexValue iv, boolean bNoVectorize)
	{
		Map<String, IndexValue> map = m_mapIdentifiers.get (bNoVectorize);
		if (map == null)
			m_mapIdentifiers.put (bNoVectorize, map = new HashMap<> ());
		map.put (strExpression, iv);
	}

	/**
	 * Resets the flags in the cache signifying that after the reset the indices
	 * will be recomputed.
	 */
	public void resetIndices ()
	{
		for (Boolean b : m_mapCaches.keySet ())
		{
			Map<Size, Map<IndexInfo, IndexValue>> mapOptions = m_mapCaches.get (b);
			for (Size size : mapOptions.keySet ())
			{
				Map<IndexInfo, IndexValue> map = mapOptions.get (size);
				if (map != null)
				{
					for (IndexInfo ii : map.keySet ())
					{
						IndexValue iv = map.get (ii);
						if (iv != null)
							iv.resetComputationFlag ();
					}
				}
			}
		}
	}

	/**
	 * Clears the index expression cache.
	 */
	public void clear ()
	{
		m_mapCaches.clear ();
		m_mapIdentifiers.clear ();
		m_nCachedIndicesCount = 0;
	}

	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ();
		for (Boolean b : m_mapCaches.keySet ())
		{
			sb.append ("NoVectorize: ");
			sb.append (b);
			sb.append (":\n\n");

			Map<Size, Map<IndexInfo, IndexValue>> mapOptions = m_mapCaches.get (b);
			for (Size size : mapOptions.keySet ())
			{
				sb.append ("\n");
				sb.append (size.toString ());
				sb.append (":\n");

				Map<IndexInfo, IndexValue> map = mapOptions.get (size);
				for (IndexInfo ii : map.keySet ())
				{
					sb.append (ii.getBoxRefPoint ().toString ());
					sb.append (" => ");
					sb.append (map.get (ii).toString ());
					sb.append ("\n");
				}
			}
		}

		return sb.toString ();
	}
}
