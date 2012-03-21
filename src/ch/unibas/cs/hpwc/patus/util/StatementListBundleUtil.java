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
package ch.unibas.cs.hpwc.patus.util;

import java.util.LinkedList;
import java.util.List;

import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.IfStatement;
import cetus.hir.Loop;
import cetus.hir.Statement;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;

/**
 *
 * @author Matthias-M. Christen
 */
public class StatementListBundleUtil
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	///////////////////////////////////////////////////////////////////
	// Implementation

	public static void addToLoopBody (StatementListBundle slbLoop, Statement stmtToAdd)
	{
		// assuming that slbLoopOuter is a loop, try to find the loop body -- or if not, just add the contents to the list
		for (ParameterAssignment pa : slbLoop)
		{
			StatementList sl = slbLoop.getStatementList (pa);
			if (sl == null)
				continue;

			// find the last loop
			Loop loop = null;
			for (Statement stmt : sl)
				if (stmt instanceof Loop)
					loop = (Loop) stmt;

			if (loop == null)
				sl.addStatement (stmtToAdd);
			else
			{
				Statement stmtLoopBody = loop.getBody ();
				CompoundStatement cmpstmtLoopBody = null;
				if (stmtLoopBody instanceof CompoundStatement)
					cmpstmtLoopBody = (CompoundStatement) stmtLoopBody;
				else
					throw new RuntimeException ("CompoundStatement as loop body expected");

				CodeGeneratorUtil.addStatements (cmpstmtLoopBody, stmtToAdd);
			}
		}
	}

	/**
	 * Adds the statement <code>stmtToAdd</code> to all the loops in <code>slbLoop</code> that are
	 * tagged with the tag <code>strTag</code>.
	 * @param slbLoop
	 * @param strTag
	 * @param stmtToAdd
	 */
	public static void addToLoopBody (StatementListBundle slbLoop, String strTag, Statement stmtToAdd)
	{
		// assuming that slbLoopOuter is a loop, try to find the loop body -- or if not, just add the contents to the list
		for (ParameterAssignment pa : slbLoop)
		{
			StatementList sl = slbLoop.getStatementList (pa);
			if (sl == null)
				continue;

			// find the last loop
			List<Loop> listLoops = new LinkedList<> ();
			for (Statement stmt : sl)
				if (stmt instanceof Loop && sl.hasTag (stmt, strTag))
					listLoops.add ((Loop) stmt);

			if (listLoops.size () == 0)
				sl.addStatement (stmtToAdd);
			else
			{
				for (Loop loop : listLoops)
				{
					Statement stmtLoopBody = loop.getBody ();
					CompoundStatement cmpstmtLoopBody = null;
					if (stmtLoopBody instanceof CompoundStatement)
						cmpstmtLoopBody = (CompoundStatement) stmtLoopBody;
					else
						throw new RuntimeException ("CompoundStatement as loop body expected");

					CodeGeneratorUtil.addStatements (cmpstmtLoopBody, stmtToAdd);
				}
			}
		}
	}

	/**
	 *
	 * @param slbLoopOuter
	 * @param slbLoopContents
	 */
	public static void addToLoopBody (StatementListBundle slbLoopOuter, StatementListBundle slbLoopContents)
	{
		// make the outer loop and loop contents branches compatible so the contents can be added
		slbLoopOuter.compatibilize (slbLoopContents);

		// assuming that slbLoopOuter is a loop, try to find the loop body -- or if not, just add the contents to the list
		for (ParameterAssignment pa : slbLoopOuter)
		{
			StatementList sl = slbLoopOuter.getStatementList (pa);
			if (sl == null)
				continue;

			// find the last loop
			Loop loop = null;
			for (Statement stmt : sl)
				if (stmt instanceof Loop)
					loop = (Loop) stmt;

			if (loop == null)
				sl.addStatements (slbLoopContents.getStatementList (pa));
			else
			{
				Statement stmtLoopBody = loop.getBody ();
				CompoundStatement cmpstmtLoopBody = null;
				if (stmtLoopBody instanceof CompoundStatement)
					cmpstmtLoopBody = (CompoundStatement) stmtLoopBody;
				else
					throw new RuntimeException ("CompoundStatement as loop body expected");

				CodeGeneratorUtil.addStatements (cmpstmtLoopBody, slbLoopContents.getStatementList (pa));
			}
		}
	}

	public static void addToLoopBody (StatementListBundle slbLoopOuter, String strTag, StatementListBundle slbLoopContents)
	{
		// make the outer loop and loop contents branches compatible so the contents can be added
		slbLoopOuter.compatibilize (slbLoopContents);

		// assuming that slbLoopOuter is a loop, try to find the loop body -- or if not, just add the contents to the list
		for (ParameterAssignment pa : slbLoopOuter)
		{
			StatementList sl = slbLoopOuter.getStatementList (pa);
			if (sl == null)
				continue;

			// find the last loop
			List<Loop> listLoops = new LinkedList<> ();
			for (Statement stmt : sl)
				if (stmt instanceof Loop && sl.hasTag (stmt, strTag))
					listLoops.add ((Loop) stmt);

			if (listLoops.size () == 0)
				sl.addStatements (slbLoopContents.getStatementList (pa));
			else
			{
				for (Loop loop : listLoops)
				{
					Statement stmtLoopBody = loop.getBody ();
					CompoundStatement cmpstmtLoopBody = null;
					if (stmtLoopBody instanceof CompoundStatement)
						cmpstmtLoopBody = (CompoundStatement) stmtLoopBody;
					else
						throw new RuntimeException ("CompoundStatement as loop body expected");

					CodeGeneratorUtil.addStatements (cmpstmtLoopBody, slbLoopContents.getStatementList (pa));
				}
			}
		}
	}
	
	public static StatementListBundle createIfStatement (Expression exprControl, StatementListBundle slbThen, StatementListBundle slbElse)
	{
		StatementListBundle slbIf = new StatementListBundle ();
		
		if (slbElse != null)
			slbThen.compatibilize (slbElse);
		
		for (ParameterAssignment pa : slbThen)
		{
			IfStatement stmtIf = null;
			if (slbElse == null)
				stmtIf = new IfStatement (exprControl.clone (),	slbThen.getStatementList (pa).getCompoundStatement ());
			else
				stmtIf = new IfStatement (exprControl.clone (), slbThen.getStatementList (pa).getCompoundStatement (), slbElse.getStatementList (pa).getCompoundStatement ());

			slbIf.replaceStatementList (pa, new StatementList (stmtIf));
		}
		
		return slbIf;
	}
}
