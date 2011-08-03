package ch.unibas.cs.hpwc.patus.codegen;

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.geometry.Vector;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;

public class ProjectionMask extends AbstractMask
{
	///////////////////////////////////////////////////////////////////
	// Implementation

	public ProjectionMask (SubdomainIterator it)
	{
		this (it.getIteratorSubdomain ().getBox ().getSize ().getCoords ());
	}

	public ProjectionMask (Vector v)
	{
		this (v.getCoords ());
	}

	public ProjectionMask (Expression[] rgExpressions)
	{
		super (rgExpressions);
	}

	@Override
	protected int[] createMask (Expression[] rgExpressions)
	{
		int[] rgMask = new int[rgExpressions.length];
		for (int i = 0; i < rgExpressions.length; i++)
			rgMask[i] = ExpressionUtil.isValue (rgExpressions[i], 1) ? 1 : 0;
		return rgMask;
	}
}
