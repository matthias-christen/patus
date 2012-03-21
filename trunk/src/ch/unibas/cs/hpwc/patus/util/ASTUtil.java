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

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import cetus.hir.ArrayAccess;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FloatLiteral;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.PointerSpecifier;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Symbol;
import cetus.hir.Symbolic;
import cetus.hir.Traversable;
import cetus.hir.Typecast;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;

public class ASTUtil
{
	/**
	 * Returns the index of <code>stmt</code> within its tree structure.
	 * In the index array, the 0th entry denotes the branch that has to be taken at level 1
	 * (1 below the root node), the 1st entry denotes the branch on level 2 of the subtree selected
	 * by the 0th entry, and so forth.
	 * @param stmt The statement for which to determine its index
	 * @return
	 */
	public static int[] getIndex (Statement stmt)
	{
		// determine the indices
		List<Integer> listIndices = new LinkedList<> ();
		for (Traversable trvThis = stmt; ; )
		{
			Traversable trvParent = trvThis.getParent ();
			if (trvParent == null)
				break;

			listIndices.add (ASTUtil.getChildIndex (trvThis));
			trvThis = trvParent;
		}

		// copy the indices to the output array
		int[] rgIndices = new int[listIndices.size ()];
		int i = 0;
		for (int n : listIndices)
		{
			rgIndices[i] = n;
			i++;
		}
		return rgIndices;
	}

	/**
	 * Returns the index of a child within the child list of its parent.
	 * @param trvChild The child node for which to retrieve the index
	 * @return The index of the child in the child list of its parent, or -1 if the
	 * 	<code>trvChild</code> does not have a parent or can't be found in the list
	 */
	private static int getChildIndex (Traversable trvChild)
	{
		if (trvChild.getParent () == null)
			return -1;

		int i = 0;
		for (Traversable trv : trvChild.getParent ().getChildren ())
		{
			if (trv == trvChild)
				return i;
			i++;
		}

		return -1;
	}

	/**
	 * Returns a substatement within the parent statement <code>stmtParent</code>.
	 * @param stmtParent The parent statement
	 * @param rgIndex The index describing the location of the substatement as
	 * 	the indices into the subtree with root <code>stmtParent</code>
	 * @return
	 */
	public static Statement getStatement (Statement stmtParent, int[] rgIndex)
	{
		Traversable trv = stmtParent;
		//for (int nIdx : rgIndex)
		for (int i = rgIndex.length - 1; i >= 0; i--)
		{
			trv = trv.getChildren ().get (/*nIdx*/ rgIndex[i]);
			if (trv == null)
				break;
		}

		return (Statement) trv;
	}

	/**
	 * Returns the root of the statement <code>trv</code>.
	 * @param stmt
	 * @return
	 */
	public static Traversable getRoot (Traversable trv)
	{
		Traversable trvChild = trv;
		Traversable trvParent = null;

		while ((trvParent = trvChild.getParent ()) != null)
			trvChild = trvParent;

		return trvChild;
	}

	/**
	 * Takes a list of specifiers and removes the last pointer specifier if there is one.
	 * @param listSpecifiers The list of input specifiers
	 * @return A list of specifiers with the last pointer specifier removed
	 */
	public static List<Specifier> dereference (List<Specifier> listSpecifiers)
	{
		if (listSpecifiers == null)
			return null;
		
		Specifier specLast = listSpecifiers.get (listSpecifiers.size () - 1);
		if (ASTUtil.isPointer (specLast))
			return listSpecifiers.subList (0, listSpecifiers.size () - 1);

		return listSpecifiers;
	}
	
	public static boolean isPointer (Specifier spec)
	{
		return spec.equals (PointerSpecifier.UNQUALIFIED) || spec.equals (PointerSpecifier.CONST) || spec.equals (PointerSpecifier.VOLATILE) || spec.equals (PointerSpecifier.CONST_VOLATILE);
	}
	
	public static Expression getPointerTo (Expression expr)
	{
		if (expr instanceof UnaryExpression && ((UnaryExpression) expr).getOperator ().equals (UnaryOperator.DEREFERENCE))
			return ((UnaryExpression) expr).getExpression ().clone ();
		return new UnaryExpression (UnaryOperator.ADDRESS_OF, expr.clone ());
	}
	
	public static Expression castTo (Expression expr, List<Specifier> listType)
	{
		Expression exprToCast = expr;
		
		// remove any existing leading type casts
		while (exprToCast instanceof Typecast)
			exprToCast = (Expression) ((Typecast) exprToCast).getChildren ().get (0);		
		
		// try to determine the type of the expression
		List<Specifier> listExprToCastType = ASTUtil.getExpressionType (exprToCast);
		
		if (listExprToCastType.equals (listType))
			return exprToCast.clone ();
		return new Typecast (listType, exprToCast.clone ());
	}
	
	/**
	 * Tries to determine the type of the expression <code>expr</code>.
	 * @param expr
	 * @return
	 */
	@SuppressWarnings("unchecked")
	private static List<Specifier> getExpressionType (Expression expr)
	{
		if (expr instanceof IntegerLiteral)
			return CodeGeneratorUtil.specifiers (Specifier.INT);	// TODO: might be "long"
		if (expr instanceof FloatLiteral)
			return CodeGeneratorUtil.specifiers (Specifier.DOUBLE);	// TODO: might be "float"
		if (expr instanceof ArrayAccess)
		{
			Expression exprArray = ((ArrayAccess) expr).getArrayName ();
			if (exprArray instanceof Identifier)
			{
				Symbol sym = ((Identifier) exprArray).getSymbol ();
				// sym.getArraySpecifiers ()... TODO: handle array specifiers
				try
				{
					return ASTUtil.dereference (sym.getTypeSpecifiers ());
				}
				catch (Exception e)
				{
					return new ArrayList<> (0);
				}				
			}
			return ASTUtil.dereference (getExpressionType (((ArrayAccess) expr).getArrayName ()));
		}
		if (expr instanceof UnaryExpression)
		{
			UnaryOperator op = ((UnaryExpression) expr).getOperator ();
			if (op.equals (UnaryOperator.ADDRESS_OF))
			{
				List<Specifier> l = CodeGeneratorUtil.specifiers (PointerSpecifier.UNQUALIFIED);
				l.addAll (getExpressionType (((UnaryExpression) expr).getExpression ()));
				return l;
			}
			else if (op.equals (UnaryOperator.DEREFERENCE))
				return ASTUtil.dereference (getExpressionType (((UnaryExpression) expr).getExpression ()));
		}
		if (expr instanceof Typecast)
			return ((Typecast) expr).getSpecifiers ();
		
		// TODO: handle binary expressions
		
		return new ArrayList<> (0);
	}
	
	/**
	 * Tests whether the expression <code>expr</code> depends directly on <code>id</code>.
	 * @param expr
	 * @param id
	 * @return
	 */
	public static boolean dependsOn (Expression expr, IDExpression id)
	{
		if (expr == null || id == null)
			return false;

		for (DepthFirstIterator it = new DepthFirstIterator (Symbolic.simplify (expr)); it.hasNext (); )
			if (it.next ().equals (id))
				return true;
		return false;
	}

	/**
	 * Replaces all the <code>IDExpressions</code> contained in <code>exprOriginal</code>
	 * that also are contained in the set <code>setIDsToReplace</code> by new <code>NameIDs</code>
	 * constructed from the original name and <code>strSuffixToAdd</code> appended.
	 * @param exprOriginal The original expression in which to replace all the <code>IDExpressions</code>
	 * 	contained in the set <code>setIDsToReplace</code>
	 * @param strSuffixToAdd The suffix to add to each of the <code>IDExpressions</code>s to be replaced
	 * 	in <code>exprOriginal</code>
	 * @param setIDsToReplace The set of <code>IDExpressions</code>s to replace in <code>exprOriginal</code>
	 * @return A new expression in which all the <code>IDExpressions</code>s contained in
	 * 	<code>setIDsToReplace</code> are appended the suffix <code>strSuffixToAdd</code>
	 */
	public static Traversable addSuffixToIdentifiers (Traversable trvOriginal, String strSuffixToAdd, Set<IDExpression> setIDsToReplace)
	{
		if (trvOriginal instanceof IDExpression)
		{
			if (setIDsToReplace.contains (trvOriginal))
				return new NameID (StringUtil.concat (((IDExpression) trvOriginal).getName (), strSuffixToAdd));
			return trvOriginal;
		}

		for (DepthFirstIterator it = new DepthFirstIterator (trvOriginal); it.hasNext (); )
		{
			Object o = it.next ();
			if (o instanceof IDExpression && setIDsToReplace.contains (o))
				((IDExpression) o).swapWith (new NameID (StringUtil.concat (((IDExpression) o).getName (), strSuffixToAdd)));
		}
		return trvOriginal;
	}
}
