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
