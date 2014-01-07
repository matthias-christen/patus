package ch.unibas.cs.hpwc.patus.analysis;

import java.util.HashMap;
import java.util.Map;

import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.IDExpression;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;

public class StencilAnalyzer
{
	private static Map<StencilCalculation, Map<Stencil, Boolean>> m_mapIsConstant = new HashMap<> ();
	

	/**
	 * Determines whether the stencil is constant, i.e., does not depend on any
	 * grid points, but is an expression depending only on operation parameters
	 * and number
	 * literals.
	 * 
	 * @param stencil
	 *            The stencil to examine
	 * @param calc
	 *            The stencil calculation object in which the stencil is
	 *            contained
	 * @return <code>true</code> iff the stencil <code>stencil</code> is
	 *         constant
	 */
	public static boolean isStencilConstant (Stencil stencil, StencilCalculation calc)
	{
		Map<Stencil, Boolean> map = m_mapIsConstant.get (calc);
		if (map == null)
			m_mapIsConstant.put (calc, map = new HashMap<> ());
		
		Boolean bIsConstant = map.get (stencil);
		if (bIsConstant != null)
			return bIsConstant;
		
		// if the stencil has output nodes (i.e., if the RHS is not assigned to a temporary variable), treat as non-constant
		// => e.g., don't do loop-invariant code motion
		// TODO: check if this is OK
		if (stencil.getOutputNodes () != null && ((StencilNodeSet) stencil.getOutputNodes ()).size () > 0)
		{
			map.put (stencil, false);
			return false;
		}
		
		Expression expr = stencil.getExpression ();
		if (expr == null)
		{
			map.put (stencil, true);
			return true;
		}

		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof IDExpression)
			{
				IDExpression id = (IDExpression) obj;
				if (!calc.isParameter (id.getName ()))
				{
					map.put (stencil, false);
					return false;
				}
			}
		}
		
		map.put (stencil, true);
		return true;		
	}
}
