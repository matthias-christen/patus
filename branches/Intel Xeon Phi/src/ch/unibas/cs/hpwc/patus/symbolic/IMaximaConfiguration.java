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
 * Interface for a class which provides configuration details for connecting to
 * Maxima.
 * 
 * @see HardCodedMaximaConfiguration
 * @see PropertiesMaximaConfiguration
 * 
 * @author David McKain
 * @version $Revision$
 */
public interface IMaximaConfiguration
{
	/**
	 * Returns the path where Maxima is installed.
	 * @return The path to Maxima
	 */
	String getMaximaExecutablePath ();

	/**
	 * An array of environment variables, in the format varname=value.
	 * @return
	 */
	String[] getMaximaRuntimeEnvironment ();

	/**
	 * The default call timeout in seconds.
	 * @return
	 */
	int getDefaultCallTimeout ();
}
