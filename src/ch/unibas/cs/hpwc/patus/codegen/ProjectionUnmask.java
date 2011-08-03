package ch.unibas.cs.hpwc.patus.codegen;

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.geometry.Vector;

/**
 *
 * @author Matthias-M. Christen
 */
public class ProjectionUnmask extends InverseMask
{
	///////////////////////////////////////////////////////////////////
	// Implementation

	public ProjectionUnmask (SubdomainIterator it)
	{
		this (it.getIteratorSubdomain ().getBox ().getSize ().getCoords ());
	}

	public ProjectionUnmask (Vector v)
	{
		this (v.getCoords ());
	}

	public ProjectionUnmask (Expression[] rgExpressions)
	{
		super (rgExpressions, ProjectionMask.class);
	}
}
