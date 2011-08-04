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
package ch.unibas.cs.hpwc.patus.codegen.unrollloop;

import java.util.Map;

import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.Expression;
import cetus.hir.ForLoop;
import cetus.hir.IDExpression;
import cetus.hir.Statement;
import cetus.hir.Traversable;

/**
 * A structure representing a perfect loop nest.
 */
public class LoopNest extends ForLoop
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * Indicates whether the loop nest has been initialized
	 */
	private boolean m_bIsEmpty;


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * Creates an empty loop nest.
	 */
	public LoopNest ()
	{
		this (null, null, null, null);
		m_bIsEmpty = true;
	}

	public LoopNest (Statement stmtInit, Expression exprCondition, Expression exprStep)
	{
		this (stmtInit, exprCondition, exprStep, null);
	}

	public LoopNest (Statement stmtInit, Expression exprCondition, Expression exprStep, Statement stmtBody)
	{
		super (stmtInit, exprCondition, exprStep, stmtBody);
		m_bIsEmpty = false;
	}

	/**
	 * Finds the inner most loop of the loop nest.
	 * @return The inner most loop
	 */
	public ForLoop getInnerMost ()
	{
		Statement stmtBody = getBody ();
		if (stmtBody instanceof LoopNest)
			return ((LoopNest) stmtBody).getInnerMost ();
		else if (stmtBody instanceof CompoundStatement)
		{
			for (Traversable trvChild : ((CompoundStatement) stmtBody).getChildren ())
				if (trvChild instanceof LoopNest)
					return ((LoopNest) trvChild).getInnerMost ();
		}

		return this;
	}

	/**
	 * Sets the body of the loop nest (i.e. the body of the innermost loop).
	 * @param stmtBody The loop nest's body
	 */
	public void setNestBody (Statement stmtBody)
	{
		// find the inner most loop in the nest and add the body to it
		getInnerMost ().setBody (stmtBody.getParent () == null ? stmtBody : stmtBody.clone () /*!!CHECK!!*/);
	}

	/**
	 * Appends a loop to the loop nest (at the inner most position).
	 * Attention: the body of the current inner most loop is discarded.
	 * @param loop The loop to add to the nest
	 */
	public void append (LoopNest loop)
	{
		if (loop == null)
			return;

		if (m_bIsEmpty)
		{
			if (!loop.m_bIsEmpty)
			{
				setInitialStatement (loop.getInitialStatement ().clone ());
				setCondition (loop.getCondition ().clone ());
				setStep (loop.getStep ().clone ());
				setBody (loop.getBody ().clone ());

				m_bIsEmpty = false;
			}
		}
		else
			setNestBody (loop);
	}

	@Override
	public ForLoop clone ()
	{
		if (m_bIsEmpty)
		{
			Statement stmtBody = getBody ();
			if (stmtBody == null || stmtBody.getChildren ().size () == 0)
				return new LoopNest ();
			return (ForLoop) stmtBody.clone (); /* !!CHECK!! ???? */
		}

		return super.clone ();
	}

	@Override
	public Map<IDExpression, Declaration> getTable ()
	{
		return super.getTable ();
	}
}
