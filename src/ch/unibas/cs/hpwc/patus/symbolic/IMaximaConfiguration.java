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
