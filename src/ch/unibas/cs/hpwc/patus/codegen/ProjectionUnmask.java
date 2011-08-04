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
