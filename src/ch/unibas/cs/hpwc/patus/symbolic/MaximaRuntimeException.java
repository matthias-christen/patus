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
