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

import java.io.PrintWriter;
import java.util.Map;

import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Statement;
import ch.unibas.cs.hpwc.patus.codegen.Globals;

/**
 *
 * @author Matthias-M. Christen
 */
public abstract class Loop extends CompoundStatement
{
	///////////////////////////////////////////////////////////////////
	// Constants

	/**
	 * Constant specifying that the loop uses the maximum number of threads
	 * (maximum number of hardware threads partitioned among the parallel loops)
	 */
	public final static int MAX_THREADS = -1;


	///////////////////////////////////////////////////////////////////
	// Member Variables

	protected Expression m_exprNumThreads;
	protected Expression[] m_rgChunkSize;

//	/**
//	 * The loop body
//	 */
//	protected Statement m_stmtLoopBody;

	/**
	 * Flag indicating whether this is a full parallel level, i.e. if the
	 * number of threads in this iterator is set to MAX_TRHEADS.
	 */
	protected boolean m_bIsParallelLevel;

	/**
	 * The level of parallelism this iterator is in
	 */
	protected int m_nParallelismLevel;


	///////////////////////////////////////////////////////////////////
	// Implementation

	protected Loop ()
	{
		setNumberOfThreads (1);
		setChunkSize (null);
		setLoopBody (null);
	}

	/**
	 * Constructs the loop object.
	 * @param idLoopIndex
	 * @param exprNumThreads
	 * @param exprChunkSize
	 */
	protected Loop (int nNumThreads, Expression[] rgChunkSize, Statement stmtBody, int nParallelismLevel)
	{
		setNumberOfThreads (nNumThreads);
		setChunkSize (rgChunkSize);
		setLoopBody (stmtBody);
		setParallelismLevel (nParallelismLevel);

		// add the declaration for the loop variable to the symbol table
		//addDeclaration (new VariableDeclaration (new VariableDeclarator (Globals.SPECIFIER_INDEX, m_idLoopIndex)));
	}

	@Override
	public Map<IDExpression, Declaration> getTable ()
	{
		return super.getTable ();
	}

	/**
	 * Returns the loop index identifier.
	 * @return The loop index identifier
	 */
	public abstract Identifier getLoopIndex ();

	/**
	 *
	 * @param nNumThreads
	 */
	public void setNumberOfThreads (int nNumThreads)
	{
		if (nNumThreads == Loop.MAX_THREADS)
		{
			m_exprNumThreads = Globals.NUMBER_OF_THREADS;
			m_bIsParallelLevel = true;
		}
		else
		{
			m_exprNumThreads = new IntegerLiteral (nNumThreads);
			m_bIsParallelLevel = false;
		}
	}

	public void setNumberOfThreads (Expression exprNumThreads)
	{
		m_exprNumThreads = exprNumThreads;
		m_bIsParallelLevel = Globals.NUMBER_OF_THREADS.equals (m_exprNumThreads);
	}

	/**
	 * Returns the number of threads by which this loop is executed as an expression.
	 * The expression corresponds to the return value of {@link Loop#getNumThreads()}, except that
	 * {@link Loop#getNumThreads()} will return the internal constant {@link Loop#MAX_THREADS} if
	 * all available threads (or the reminder if the parallel loop is nested) are to be used, whereas
	 * this method will return an identifier.
	 * @return The number of threads by which this loop is executed
	 */
	public Expression getNumberOfThreads ()
	{
		return m_exprNumThreads;
	}

	/**
	 * Returns <code>true</code> if the loop is to be executed sequentially.
	 * @return <code>true</code> if the loop is sequential
	 */
	public boolean isSequential ()
	{
		return (m_exprNumThreads instanceof IntegerLiteral) && ((IntegerLiteral) m_exprNumThreads).getValue () == 1;
	}

	/**
	 * Returns <code>true</code> if the loop is executed in parallel.
	 * @return <code>true</code> iff the loop is parallel
	 */
	public boolean isParallel ()
	{
		return !isSequential ();
	}

	/**
	 * Returns <code>true</code> if this loop is executed on a new parallel level, i.e.
	 * if the number of threads is set to {@link Globals#getNumberOfThreads()}.
	 * @return
	 */
	public boolean isParallelLevel ()
	{
		return m_bIsParallelLevel;
	}

	/**
	 *
	 * @param exprChunkSize
	 */
	public void setChunkSize (Expression[] rgChunkSize)
	{
		if (rgChunkSize == null)
			m_rgChunkSize = null;
		else
		{
			m_rgChunkSize = new Expression[rgChunkSize.length];
			for (int i = 0; i < rgChunkSize.length; i++)
			{
				m_rgChunkSize[i] = rgChunkSize[i].clone ();
				m_rgChunkSize[i].setParent (this);
			}
		}
	}

	/**
	 * Returns the default chunk size.
	 * @return The default chunk size
	 */
	protected abstract Expression[] getDefaultChunkSize ();

	/**
	 * Returns the chunk size, i.e. how many consecutive loop iterations are carried out by the same thread.
	 * The chunk size is only used if the loop is parallel.
	 * @return
	 */
	public Expression getChunkSize (int nDimension)
	{
		if (m_rgChunkSize == null)
			setChunkSize (getDefaultChunkSize ());
		return m_rgChunkSize[nDimension].clone ();
	}
	
	public Expression[] getChunkSizes ()
	{
		if (m_rgChunkSize == null)
			setChunkSize (getDefaultChunkSize ());
		
		Expression[] rgChunkSize = new Expression[m_rgChunkSize.length];
		for (int i = 0; i < m_rgChunkSize.length; i++)
			rgChunkSize[i] = m_rgChunkSize[i].clone ();
		return rgChunkSize;
	}

	/**
	 *
	 * @param stmtBody
	 */
	public void setLoopBody (Statement stmtBody)
	{
		if (stmtBody != null)
		{
			Statement stmtLoopBody = stmtBody.clone ();
			if (children.size () == 0)
				children.add (stmtLoopBody);
			else
				children.set (0, stmtLoopBody);
			stmtLoopBody.setParent (this);
		}
		else
		{
			//m_stmtLoopBody = null;
			children.clear ();
		}
	}

	/**
	 * Returns the loop body.
	 * @return The loop body
	 */
	public Statement getLoopBody ()
	{
		//return m_stmtLoopBody.clone ();
		return ((Statement) children.get (0));
	}

	/**
	 * Sets the loop's parallelism level (i.e. to which level of
	 * hardware parallelism this loop is assigned).
	 * @param nParallelismLevel
	 */
	public void setParallelismLevel (int nParallelismLevel)
	{
		m_nParallelismLevel = nParallelismLevel;
	}

	/**
	 * Gets the level of parallelism this loop is in.
	 * @return
	 */
	public int getParallelismLevel ()
	{
		return m_nParallelismLevel;
	}

	/**
	 * Returns the head of the loop as a string.
	 * @return The loop head as a string
	 */
	public abstract String getLoopHeadAnnotation ();

	@Override
	public void print (PrintWriter o)
	{
		o.print (toString ());
	}
}
