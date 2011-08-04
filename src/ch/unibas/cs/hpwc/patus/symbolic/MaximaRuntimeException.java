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
/* $Id$
 *
 * Copyright (c) 2010, The University of Edinburgh.
 * All Rights Reserved
 */
package ch.unibas.cs.hpwc.patus.symbolic;

/**
 * Generic runtime Exception thrown to indicate an unexpected problem
 * encountered when communicating with Maxima.
 * <p>
 * This Exception is unchecked as there's nothing that can reasonably be done to
 * recover from this so ought to bubble right up to a handler near the "top" of
 * your application.
 *
 * @see MaximaConfigurationException
 *
 * @author David McKain
 * @version $Revision$
 */
public class MaximaRuntimeException extends RuntimeException
{
	private static final long serialVersionUID = 1L;

	public MaximaRuntimeException (String message)
	{
		super (message);
	}

	public MaximaRuntimeException (Throwable cause)
	{
		super (cause);
	}

	public MaximaRuntimeException (String message, Throwable cause)
	{
		super (message, cause);
	}
}
