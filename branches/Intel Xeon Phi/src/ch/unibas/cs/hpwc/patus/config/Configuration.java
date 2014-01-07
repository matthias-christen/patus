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
package ch.unibas.cs.hpwc.patus.config;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import javax.swing.JOptionPane;

/**
 *
 * @author Matthias-M. Christen
 */
public class Configuration
{
	///////////////////////////////////////////////////////////////////
	// Singleton Pattern

	private final static Configuration THIS = new Configuration ();

	/**
	 *
	 * @return
	 */
	public static Configuration getInstance ()
	{
		return THIS;
	}

	/**
	 *
	 * @param clsConfigurable
	 */
	public static void register (IConfigurable configurable)
	{
		THIS.registerConfigurable (configurable);
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * List of all the configurables
	 */
	private List<IConfigurable> m_listConfigurables;

	/**
	 * Configuration property map
	 */
	private Map<String, ConfigurationProperty> m_mapProperties;

	/**
	 * The properties object holding all the configuration data
	 */
	private Properties m_properties;

	/**
	 * The path to the config file
	 */
	private String m_strConfigFile;


	///////////////////////////////////////////////////////////////////
	// Implementation

	private Configuration ()
	{
		m_listConfigurables = new LinkedList<> ();
		m_mapProperties = new HashMap<> ();

		m_properties = new Properties ();
		String strConfigPath = Configuration.getConfigFilePath ();
		File filePath = new File (strConfigPath);
		if (!filePath.exists ())
			filePath.mkdirs ();
		m_strConfigFile = strConfigPath + File.separatorChar + "config.ini";

		try
		{
			m_properties.load (new FileInputStream (m_strConfigFile));
		}
		catch (IOException e)
		{
		}
	}

	/**
	 * Returns the path to the configuration file
	 * @return
	 */
	private static String getConfigFilePath ()
	{
		if (System.getProperty ("os.name").indexOf ("Windows") != -1)
		{
			// the program is run on Windows

			// check whether the %APPDATA% environment variable is set
			String strAppData = System.getenv ("APPDATA");
			if (strAppData != null)
				return strAppData + File.separatorChar + "Patus";

			// (if APPDATA can't be found, fall back to the "*nix" mode)
		}

		// on OSes other than Windows, use the sub-directory .patus of the user's home
		// directory as application data directory
		return System.getProperty ("user.home") + File.separatorChar + ".patus";
	}

	/**
	 * Registers a configurable.
	 * @param configurable The configurable to register
	 */
	private void registerConfigurable (IConfigurable configurable)
	{
		m_listConfigurables.add (configurable);
		for (ConfigurationProperty p : configurable.getConfigurationProperties ())
		{
			String strKey = p.getKey ();
			m_mapProperties.put (strKey, p);

			// get the value for the property from the property list
			// if the property exists in the property list, get the value and set it to the configurable's property
			// otherwise add the configurable's property to the property list, the value set to the property's default value
			String strValue = (String) m_properties.get (strKey);
			if (strValue != null)
				p.setValue (strValue);
			else
				m_properties.setProperty (strKey, p.getDefaultValue ());
		}
	}

	/**
	 * Returns the value for the configuration key <code>strConfigKey</code>
	 * @param strConfigKey The key of the configuration property
	 * @return The value associated with <code>strConfigKey</code>
	 */
	public String getValue (String strConfigKey)
	{
		return m_properties.getProperty (strConfigKey);
	}

	/**
	 * Saves the properties to file.
	 */
	public void save ()
	{
		// transfer the property values from the map to the property set
		for (String strKey : m_mapProperties.keySet ())
			m_properties.setProperty (strKey, m_mapProperties.get (strKey).getValue ());

		// save the properties as file
		try
		{
			m_properties.store (new FileOutputStream (m_strConfigFile), null);
		}
		catch (IOException e)
		{
			e.printStackTrace ();
		}
	}

	/**
	 * Shows the configuration dialog.
	 */
	public void showDialog ()
	{
		try
		{
			ConfigUI conf = new ConfigUI (m_mapProperties);
			if (conf.showDialog () == JOptionPane.OK_OPTION)
				save ();
		}
		catch (UnsatisfiedLinkError e)
		{
			System.out.println ("Can't show dialog. Please create the configuration file manually.");
		}
	}
}
