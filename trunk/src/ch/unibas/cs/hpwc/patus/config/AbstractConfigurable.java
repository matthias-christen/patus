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
		m_listProperties = new LinkedList<ConfigurationProperty> ();
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
