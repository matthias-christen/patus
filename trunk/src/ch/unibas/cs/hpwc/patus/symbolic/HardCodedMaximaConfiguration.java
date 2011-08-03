/* $Id$
 *
 * Copyright (c) 2010, The University of Edinburgh.
 * All Rights Reserved
 */
package ch.unibas.cs.hpwc.patus.symbolic;

/**
 * Trivial implementation of {@link IMaximaConfiguration} that allows you to hard
 * code the required configuration within your Java code.
 * <p>
 * This allows you to bash out code on your own machine without having to worry
 * too much about creating custom configuration files.
 *
 * <h2>Important Note</h2>
 *
 * Using this class automatically makes your code non-portable! Use an
 * alternative if this matters to you.
 *
 * @author David McKain
 * @version $Revision$
 */
public final class HardCodedMaximaConfiguration implements IMaximaConfiguration
{
	private String m_strMaximaExecutablePath;

	private String[] m_rgMaximaRuntimeEnvironment;

	private int m_nDefaultCallTimeout;


	@Override
	public String getMaximaExecutablePath ()
	{
		return m_strMaximaExecutablePath;
	}

	public void setMaximaExecutablePath (String maximaExecutablePath)
	{
		this.m_strMaximaExecutablePath = maximaExecutablePath;
	}

	@Override
	public String[] getMaximaRuntimeEnvironment ()
	{
		return m_rgMaximaRuntimeEnvironment;
	}

	public void setMaximaRuntimeEnvironment (String[] maximaRuntimeEnvironment)
	{
		this.m_rgMaximaRuntimeEnvironment = maximaRuntimeEnvironment;
	}

	@Override
	public int getDefaultCallTimeout ()
	{
		return m_nDefaultCallTimeout;
	}

	public void setDefaultCallTimeout (int defaultCallTimeout)
	{
		this.m_nDefaultCallTimeout = defaultCallTimeout;
	}
}
