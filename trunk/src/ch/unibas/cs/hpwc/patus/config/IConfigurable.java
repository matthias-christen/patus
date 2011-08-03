package ch.unibas.cs.hpwc.patus.config;

/**
 * 
 * @author Matthias-M. Christen
 */
public interface IConfigurable
{
	/**
	 * Returns an iterable over all the configuration properties this configurable
	 * provides and which can be configured by the user. 
	 */
	public abstract Iterable<ConfigurationProperty> getConfigurationProperties ();
}
