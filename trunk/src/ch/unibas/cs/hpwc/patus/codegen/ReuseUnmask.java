package ch.unibas.cs.hpwc.patus.codegen;

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.geometry.Vector;

public class ReuseUnmask extends InverseMask
{
	///////////////////////////////////////////////////////////////////
	// Implementation

	public ReuseUnmask (SubdomainIterator it)
	{
		this (it.getIteratorSubdomain ().getBox ().getSize ().getCoords ());
	}

	public ReuseUnmask (Vector v)
	{
		this (v.getCoords ());
	}

	public ReuseUnmask (Expression[] rgExpressions)
	{
		super (rgExpressions, ReuseMask.class);
	}
}
