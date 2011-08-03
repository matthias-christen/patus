package ch.unibas.cs.hpwc.patus.util;

import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.NameID;
import cetus.hir.PointerSpecifier;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Symbolic;
import cetus.hir.Traversable;

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
		List<Integer> listIndices = new LinkedList<Integer> ();
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
		if (PointerSpecifier.UNQUALIFIED.equals (listSpecifiers.get (listSpecifiers.size () - 1)))
			return listSpecifiers.subList (0, listSpecifiers.size () - 1);
		return listSpecifiers;
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
