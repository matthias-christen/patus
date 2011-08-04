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
package ch.unibas.cs.hpwc.patus.symbolic;

import cetus.hir.Expression;

/**
 *
 * @author Matthias-M. Christen
 */
public class NotConvertableException extends Exception
{
	private static final long serialVersionUID = 1L;


	///////////////////////////////////////////////////////////////////
	// Member Variables


	///////////////////////////////////////////////////////////////////
	// Implementation

	public NotConvertableException ()
	{
		super ();
	}

	public NotConvertableException (String s)
	{
		super (s);
	}

	public NotConvertableException (Expression expr)
	{
		super ("The expression " + expr.toString () + " can't be converted to a Maxima expression");
	}
}
