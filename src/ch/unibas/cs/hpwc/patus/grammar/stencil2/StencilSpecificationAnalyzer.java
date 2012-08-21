package ch.unibas.cs.hpwc.patus.grammar.stencil2;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilBundle;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class StencilSpecificationAnalyzer
{
	/**
	 * 
	 * @param bundle
	 */
	public static void normalizeStencilNodesForBoundariesAndIntial (StencilBundle bundle)
	{
		for (Stencil stencil : bundle)
		{
			for (StencilNode node : stencil.getAllNodes ())
			{
				Expression[] rgIdx = node.getIndex ().getSpaceIndexEx ();
				Expression[] rgIdxNew = new Expression[rgIdx.length];
				
				for (int i = 0; i < rgIdx.length; i++)
				{
					String strDimId = StencilSpecificationAnalyzer.getContainedDimensionIdentifier (rgIdx[i], i);
					
					if (strDimId != null)
					{
						// if an entry I contains the corresponding dimension identifier id, compute I-id
						// we expect this to be an integer value
						
						rgIdxNew[i] = Symbolic.simplify (new BinaryExpression (rgIdx[i].clone (), BinaryOperator.SUBTRACT, new NameID (strDimId)));
						if (!(rgIdxNew[i] instanceof IntegerLiteral))
						{
							throw new RuntimeException (StringUtil.concat ("Illegal coordinate ", rgIdx[i].toString (),
								" in grid reference ", node.toString ()," in definition ", stencil.toString ()
							));
						}
					}
					else
					{
						// if the entry doesn't contain a dimension identifier, set the corresponding spatial index coordinate
						// to 0 (=> do something when the point becomes the center point), and add a constraint setting the
						// corresponding subdomain index (dimension identifier) to the expression of the index entry
						
						rgIdxNew[i] = new IntegerLiteral (0);
						node.addConstraint (new BinaryExpression (new NameID (CodeGeneratorUtil.getDimensionName (i)), BinaryOperator.COMPARE_EQ, rgIdx[i]));
					}
				}
				
				node.getIndex ().setSpaceIndex (rgIdxNew);
			}
		}
	}
	
	/**
	 * Check that all the stencil nodes in the stencils of the bundle are legal,
	 * i.e., that the spatial index has the form [x+dx, y+dy, ...]. Note that
	 * the dimension identifiers (x, y, ...) have been subtracted already, so
	 * the spatial index should be an array of integer numbers.
	 * 
	 * @param bundle
	 *            The bundle to check
	 */
	public static void checkStencilNodesLegality (StencilBundle bundle)
	{
		for (Stencil stencil : bundle)
			for (StencilNode node : stencil)
				for (Expression exprIdx : node.getIndex ().getSpaceIndexEx ())
					if (!(exprIdx instanceof IntegerLiteral))
						throw new RuntimeException (StringUtil.concat ("Illegal grid reference", exprIdx.toString ()));	// TODO: handle in parser
	}
	
	/**
	 * Determines whether the expression <code>expr</code> contains a dimension
	 * identifier corresponding to dimension <code>nDim</code>.
	 * 
	 * @param expr
	 *            The expression to examine
	 * @param nDim
	 *            The dimension whose identifier to detect
	 * @return <code>true</code> iff <code>expr</code> contains a dimension
	 *         identifier corresponding to the dimension <code>nDim</code>
	 */
	public static String getContainedDimensionIdentifier (Expression expr, int nDim)
	{
		String strId = CodeGeneratorUtil.getDimensionName (nDim);
		String strIdAlt = CodeGeneratorUtil.getAltDimensionName (nDim);
		
		for (DepthFirstIterator it = new DepthFirstIterator (Symbolic.simplify (expr)); it.hasNext (); )
		{
			Object o = it.next ();
			if (o instanceof IDExpression)
			{
				if (((IDExpression) o).getName ().equals (strId))
					return strId;
				if (((IDExpression) o).getName ().equals (strIdAlt))
					return strIdAlt;
			}
		}
		
		return null;
	}
}
