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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import ch.unibas.cs.hpwc.patus.config.Configuration;
import ch.unibas.cs.hpwc.patus.config.ConfigurationProperty;
import ch.unibas.cs.hpwc.patus.config.IConfigurable;

public class MaximaConfiguration implements IConfigurable, IMaximaConfiguration
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static ConfigurationProperty PROP_MAXIMA_EXECPATH = new ConfigurationProperty ("Maxima", "Executable Path", ConfigurationProperty.EPropertyType.FILE, "");
	private final static ConfigurationProperty PROP_MAXIMA_TIMEOUT = new ConfigurationProperty ("Maxima", "Timeout (ms)", ConfigurationProperty.EPropertyType.INTEGER, "1000", 100, 10000);


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private boolean m_bCalledConfigUI;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public MaximaConfiguration ()
	{
		m_bCalledConfigUI = false;

		// register in the configuration object
		Configuration.register (this);
	}

	@Override
	public int getDefaultCallTimeout ()
	{
		String strResult = PROP_MAXIMA_TIMEOUT.getValue ();

		if ((strResult == null || "".equals (strResult)) && !m_bCalledConfigUI)
		{
			if (!MaximaConfiguration.tryToSetDefaults ())
			{
				Configuration.getInstance ().showDialog ();
				strResult = PROP_MAXIMA_TIMEOUT.getValue ();
				m_bCalledConfigUI = true;
			}
			else
				strResult = PROP_MAXIMA_TIMEOUT.getValue ();
		}

		if (strResult == null)
			return 1000;

		try
		{
			return Integer.parseInt (strResult);
		}
		catch (NumberFormatException e)
		{
			return 1000;
		}
	}

	@Override
	public String getMaximaExecutablePath ()
	{
		String strResult = PROP_MAXIMA_EXECPATH.getValue ();

		if ((strResult == null || "".equals (strResult)) && !m_bCalledConfigUI)
		{
			if (!MaximaConfiguration.tryToSetDefaults ())
			{
				try
				{
					Configuration.getInstance ().showDialog ();
					strResult = PROP_MAXIMA_EXECPATH.getValue ();
					m_bCalledConfigUI = true;
				}
				catch (NoClassDefFoundError e)
				{
					// something went wrong when trying to show the UI
					return "";
				}
			}
			else
				strResult = PROP_MAXIMA_EXECPATH.getValue ();
		}

		return strResult == null ? "" : strResult;
	}

	/**
	 * Try to set the defaults by calling "which maxima" on *nix environments.
	 * @return
	 */
	private static boolean tryToSetDefaults ()
	{
		if (System.getProperty ("os.name").indexOf ("Windows") > -1)
			return false;

		try
		{
			Process process = Runtime.getRuntime ().exec ("which maxima");
			BufferedReader out = new BufferedReader (new InputStreamReader (process.getInputStream (), "ASCII"));
			String strResult = out.readLine ();
			out.close ();

			try
			{
				process.waitFor ();
			}
			catch (InterruptedException e)
			{
			}
			process.destroy ();

			if (strResult == null || "".equals (strResult))
				return false;

			// search for path
			int nPos = strResult.indexOf ('/');
			if (nPos == -1)
				return false;

			PROP_MAXIMA_EXECPATH.setValue (strResult.substring (nPos));
			PROP_MAXIMA_TIMEOUT.setValue ("1000");
		}
		catch (IOException e)
		{
			return false;
		}

		return true;
	}

	@Override
	public String[] getMaximaRuntimeEnvironment ()
	{
		return new String[] { };
	}

	@Override
	public Iterable<ConfigurationProperty> getConfigurationProperties ()
	{
		List<ConfigurationProperty> listProperties = new ArrayList<> ();
		listProperties.add (PROP_MAXIMA_EXECPATH);
		listProperties.add (PROP_MAXIMA_TIMEOUT);

		return listProperties;
	}
}
