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
package ch.unibas.cs.hpwc.patus.analysis;

import java.util.List;

import cetus.hir.DeclarationStatement;
import cetus.hir.Declarator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.IDExpression;
import cetus.hir.PointerSpecifier;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import cetus.hir.UserSpecifier;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.StatementList;

/**
 *
 * @author Matthias-M. Christen
 */
public class HIRAnalyzer
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Determines whether the identifier <code>id</code> is referenced within <code>trvContext</code>.
	 * @param id The identifier to check
	 * @param trvContext The top node of the AST to check whether the identifier <code>id</code> is
	 * 	referenced within that branch
	 * @return <code>true</code> iff the identifier <code>id</code> is referenced within <code>trvContext</code>
	 */
	public static boolean isReferenced (IDExpression id, Traversable trvContext)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (trvContext); it.hasNext (); )
			if (it.next ().equals (id))
				return true;
		return false;
	}

	public static boolean isReferenced (IDExpression id, StatementList sl)
	{
		for (Statement stmt : sl.getStatementsAsList ())
		{
			if (stmt instanceof DeclarationStatement && ((DeclarationStatement) stmt).getDeclaration () instanceof VariableDeclaration)
			{
				// handle declaration statements (only examine initializers)

				VariableDeclaration decl = (VariableDeclaration) ((DeclarationStatement) stmt).getDeclaration ();
				for (int i = 0; i < decl.getNumDeclarators (); i++)
				{
					Declarator d = decl.getDeclarator (i);
					if (d instanceof VariableDeclarator)
					{
						VariableDeclarator vd = (VariableDeclarator) d;
						if (vd.getInitializer () != null && HIRAnalyzer.isReferenced (id, vd.getInitializer ()))
							return true;
					}
				}
			}
			else
			{
				// handle "normal" statements
				if (HIRAnalyzer.isReferenced (id, stmt))
					return true;
			}
		}
		return false;
	}

	public static boolean isPointer (Specifier spec)
	{
		return
			spec.equals (PointerSpecifier.UNQUALIFIED) ||
			spec.equals (PointerSpecifier.CONST) ||
			spec.equals (PointerSpecifier.VOLATILE) ||
			spec.equals (PointerSpecifier.CONST_VOLATILE);
	}

	/**
	 * Determines whether the variable declared by <code>decl</code> is at least a double indirection.
	 * @param decl The variable declaration to examin
	 * @return <code>true</code> iff the variable declared by <code>decl</code> is at least a double indirection
	 */
	public static boolean isNoPointer (VariableDeclaration decl)
	{
		if (decl.getNumDeclarators () > 1)
			throw new RuntimeException ("NotImpl: multiple variable declarators");

		for (Specifier spec : decl.getSpecifiers ())
			if (HIRAnalyzer.isPointer (spec))
				return false;
		return true;
	}

	/**
	 * Determines whether the variable declared by <code>decl</code> is at least a double indirection.
	 * @param decl The variable declaration to examin
	 * @return <code>true</code> iff the variable declared by <code>decl</code> is at least a double indirection
	 */
	public static boolean isDoublePointer (VariableDeclaration decl)
	{
		if (decl.getNumDeclarators () > 1)
			throw new RuntimeException ("NotImpl: multiple variable declarators");

		List<Specifier> listSpecs = decl.getSpecifiers ();
		if (listSpecs.size () >= 2)
		{
			if (HIRAnalyzer.isPointer (listSpecs.get (listSpecs.size () - 2)) &&
				HIRAnalyzer.isPointer (listSpecs.get (listSpecs.size () - 1)))
			{
				return true;
			}
		}

		return false;
	}

	public static boolean isIntegerSpecifier (Specifier spec)
	{
		if (spec instanceof UserSpecifier)
			return true;
		if (Specifier.INT.equals (spec) || Specifier.UNSIGNED.equals (spec) || Specifier.SHORT.equals (spec) || Specifier.LONG.equals (spec) || Specifier.CHAR.equals (spec))
			return true;
		return false;
	}
}
