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

import java.util.LinkedList;
import java.util.List;

public abstract class AbstractConfigurable implements IConfigurable
{
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	protected List<ConfigurationProperty> m_listProperties;
	
	boolean m_bIsRegistered;
	

	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public AbstractConfigurable ()
	{
		m_listProperties = new LinkedList<> ();
		m_bIsRegistered = false;;
	}
	
	protected void ensureIsRegistered ()
	{
		if (!m_bIsRegistered)
			Configuration.register (this);
		m_bIsRegistered = true;
	}
	
	protected void addConfigurationProperty (ConfigurationProperty property)
	{
		m_listProperties.add (property);
	}

	@Override
	public Iterable<ConfigurationProperty> getConfigurationProperties ()
	{
		return m_listProperties;
	}
}
