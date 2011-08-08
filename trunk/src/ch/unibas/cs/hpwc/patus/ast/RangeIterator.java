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
package ch.unibas.cs.hpwc.patus.ast;

import java.util.Map;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Declaration;
import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Statement;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class RangeIterator extends Loop
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private static int m_nAnonymousLoopsCount = 0;

	protected Identifier m_idLoopIndex;

	protected Expression m_exprStart;
	protected Expression m_exprEnd;
	protected Expression m_exprStep;

	protected boolean m_bIsMainTemporalIterator;

//	/**
//	 * The original subdomain iterator from which this range iterator
//	 * has been constructed (when the loop was split for threading)
//	 */
//	protected SubdomainIterator m_itOriginalSubdomainIterator;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public RangeIterator ()
	{
		super ();
		setRange (null, null, null);
		m_bIsMainTemporalIterator = false;
//		m_itOriginalSubdomainIterator = null;
	}

	/**
	 * Constructs the loop object.
	 * @param idLoopIndex
	 * @param exprStart
	 * @param exprEnd
	 * @param exprStep
	 * @param stmtBody
	 * @param nParallelismLevel
	 */
	public RangeIterator (Identifier idLoopIndex, Expression exprStart, Expression exprEnd, Expression exprStep, Statement stmtBody, int nParallelismLevel)
	{
		this (idLoopIndex, exprStart, exprEnd, exprStep, 1, null, stmtBody, nParallelismLevel);
	}

	/**
	 * Constructs the loop object.
	 * @param idLoopIndex
	 * @param exprStart
	 * @param exprEnd
	 * @param exprStep
	 * @param exprNumThreads
	 * @param nNumThreads
	 * @param exprChunkSize
	 * @param stmtBody
	 * @param nParallelismLevel
	 */
	public RangeIterator (Identifier idLoopIndex, Expression exprStart, Expression exprEnd, Expression exprStep, int nNumThreads, Expression exprChunkSize, Statement stmtBody, int nParallelismLevel)
	{
		super (nNumThreads, exprChunkSize == null ? null : new Expression[] { exprChunkSize }, stmtBody, nParallelismLevel);
		setLoopIndex (idLoopIndex);
		setRange (exprStart, exprEnd, exprStep);
		m_bIsMainTemporalIterator = false;
//		m_itOriginalSubdomainIterator = null;
	}

	/**
	 * Determines whether this the main temporal iterator in the strategy, i.e., the
	 * strategy's outer-most temporal loop.
	 * @return
	 */
	public boolean isMainTemporalIterator ()
	{
		return m_bIsMainTemporalIterator;
	}

	/**
	 * Specifies that this range iterator is the main temporal iterator in the strategy
	 * (the outer-most temporal loop).
	 * @param bIsMainTemporalIterator
	 */
	public void setMainTemporalIterator (boolean bIsMainTemporalIterator)
	{
		m_bIsMainTemporalIterator = bIsMainTemporalIterator;
	}

	/**
	 *
	 * @param idLoopIndex
	 */
	public void setLoopIndex (Identifier idLoopIndex)
	{
		if (idLoopIndex == null)
		{
//			VariableDeclarator decl = new VariableDeclarator (new NameID ("i"));
////			addDeclaration (new VariableDeclaration (Globals.SPECIFIER_INDEX, decl))
///*
//			Traversable t = this;
//			while (!(t instanceof SymbolTable))
//				t = t.getParent ();
//			((SymbolTable) t).addDeclaration (new VariableDeclaration (Globals.SPECIFIER_INDEX, decl));
//*/
//			super.addDeclaration (new VariableDeclaration (Globals.SPECIFIER_INDEX, decl));
//
//			m_idLoopIndex = SymbolTools.getTemp (new Identifier (decl));

//			m_idLoopIndex = CodeGeneratorUtil.createIdentifier (Globals.SPECIFIER_INDEX, "i__", RangeIterator.m_nAnonymousLoopsCount++);

			VariableDeclarator decl = new VariableDeclarator (CodeGeneratorUtil.createNameID ("i__", RangeIterator.m_nAnonymousLoopsCount++));
			m_idLoopIndex = new Identifier (decl);
			// TODO: add VariableDeclaration(...)
		}
		else
		{
			/*
			m_idLoopIndex = (Identifier) idLoopIndex.clone ();
			if (m_idLoopIndex.getSymbol () == null)
				addDeclaration (new VariableDeclaration (Globals.SPECIFIER_INDEX, new VariableDeclarator (m_idLoopIndex)));
			*/
			m_idLoopIndex = idLoopIndex;
		}

		m_idLoopIndex.setParent (this);
	}

	/**
	 * Returns the loop index identifier.
	 * @return The loop index identifier
	 */
	@Override
	public Identifier getLoopIndex ()
	{
		return getLoopIndex (true);
	}

	/**
	 *
	 * @param bGetClone
	 * @return
	 */
	public Identifier getLoopIndex (boolean bGetClone)
	{
		return bGetClone ? m_idLoopIndex.clone () : m_idLoopIndex;
	}

	/**
	 *
	 * @param exprStart
	 * @param exprEnd
	 * @param exprStep
	 */
	public void setRange (Expression exprStart, Expression exprEnd, Expression exprStep)
	{
		m_exprStart = exprStart == null ? new IntegerLiteral (0) : (Expression) exprStart.clone ();
		m_exprEnd = exprEnd == null ? new IntegerLiteral (0) : (Expression) exprEnd.clone ();
		m_exprStep = exprStep == null ? new IntegerLiteral (1) : (Expression) exprStep.clone ();

		m_exprStart.setParent (this);
		m_exprEnd.setParent (this);
		m_exprStep.setParent (this);
	}

	/**
	 * Returns the start index of the loop.
	 * @return The start index
	 */
	public Expression getStart ()
	{
		return m_exprStart.clone ();
	}

	/**
	 * Returns the end index of the loop.
	 * @return The end index
	 */
	public Expression getEnd ()
	{
		return m_exprEnd.clone ();
	}

	/**
	 * Returns the loop stride.
	 * @return The stride/step
	 */
	public Expression getStep ()
	{
		return m_exprStep.clone ();
	}

	@Override
	protected Expression[] getDefaultChunkSize ()
	{
		// calculates the default chunk size:
		// chunk = ceil ((end - start + 1) / (step * #thds))
		return new Expression[] {
			ExpressionUtil.ceil (
				ExpressionUtil.increment (new BinaryExpression (m_exprEnd, BinaryOperator.SUBTRACT, m_exprStart)),
				ExpressionUtil.product (m_exprStep, m_exprNumThreads)
			)
		};
	}

//	/**
//	 * Returns the loop body.
//	 * @return The loop body
//	 */
//	@Override
//	public Statement getLoopBody ()
//	{
//		return m_stmtLoopBody.clone ();
//	}

//	/**
//	 * Returns the original {@link SubdomainIterator} from which this {@link RangeIterator}
//	 * has been constructed (when split for threading).
//	 * @return The original subdomain iterator or <code>null</code> if this loop was
//	 * 	a {@link RangeIterator} in the {@link Strategy}
//	 */
//	public SubdomainIterator getOriginalSubdomainIterator ()
//	{
//		return m_itOriginalSubdomainIterator;
//	}
//
//	/**
//	 *
//	 * @param it
//	 */
//	public void setOriginalSubdomainIterator (SubdomainIterator it)
//	{
//		m_itOriginalSubdomainIterator = it;
//	}

	@Override
	public String toString ()
	{
		return StringUtil.concat (
			"for ", m_idLoopIndex, " = ", m_exprStart, "..", m_exprEnd, " by ", m_exprStep,
			" parallel ", m_exprNumThreads, " <level ", m_nParallelismLevel, "> schedule ", m_rgChunkSize == null ? "1" : m_rgChunkSize[0], "\n",
			getLoopBody ());
	}

	@Override
	public String getLoopHeadAnnotation ()
	{
		return StringUtil.concat (
			"for ", m_idLoopIndex, " = ", m_exprStart, "..", m_exprEnd, " by ", m_exprStep,
			" parallel ", m_exprNumThreads, " <level ", m_nParallelismLevel, "> schedule ", m_rgChunkSize == null ? "1" : m_rgChunkSize[0], " { ... }");
	}

	@Override
	public Map<IDExpression, Declaration> getTable ()
	{
		return super.getTable ();
	}
}
