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
 * Exception thrown when Maxima takes too long to respond to a command. Possible
 * reasons for this might be:
 * <ul>
 * <li>The command genuinely took too long to run</li>
 * <li>The command was ill-formed and left Maxima expecting more input</li>
 * <li>The command resulted in Maxima expecting interactive communication</li>
 * </ul>
 *
 * @author David McKain
 * @version $Revision$
 */
public final class MaximaTimeoutException extends Exception
{
	private static final long serialVersionUID = 1L;

	private final int timeoutSeconds;

	public MaximaTimeoutException (final int timeoutSeconds)
	{
		super ("Timeout of " + timeoutSeconds + "s exceeded waiting for response from Maxima");
		this.timeoutSeconds = timeoutSeconds;
	}

	public int getTimeoutSeconds ()
	{
		return timeoutSeconds;
	}
}
