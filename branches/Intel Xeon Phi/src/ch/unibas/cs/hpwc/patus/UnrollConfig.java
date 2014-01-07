package ch.unibas.cs.hpwc.patus;

public class UnrollConfig
{
	private boolean m_bIsMultiConfig;
	private int[] m_rgUnrollings;
	
	
	private UnrollConfig ()
	{
	}
	
	public UnrollConfig (String strConfig)
	{
		String[] rgUnrollingInDimensions = strConfig.split (":");
		
		m_rgUnrollings = new int[rgUnrollingInDimensions.length];
		for (int i = 0; i <  rgUnrollingInDimensions.length; i++)
			m_rgUnrollings[i] = Integer.parseInt (rgUnrollingInDimensions[i]);
		
		if (rgUnrollingInDimensions.length == 1)
			m_bIsMultiConfig = true;
	}
	
	public UnrollConfig (int nUnrollFactorForAllDimensions)
	{
		m_rgUnrollings = new int[] { nUnrollFactorForAllDimensions };
		m_bIsMultiConfig = true;
	}
	
	public int getUnrollingInDimension (int nDim)
	{
		if (m_bIsMultiConfig)
			return m_rgUnrollings[0];
			
		if (nDim < 0 || nDim >= m_rgUnrollings.length)
			return 1;
		return m_rgUnrollings[nDim];
	}
	
	public int[] getUnrollings ()
	{
		return m_rgUnrollings;
	}
	
	public boolean isMultiConfig ()
	{
		return m_bIsMultiConfig;
	}
	
	public UnrollConfig clone ()
	{
		UnrollConfig config = new UnrollConfig ();
		
		config.m_bIsMultiConfig = m_bIsMultiConfig;
		config.m_rgUnrollings = new int[m_rgUnrollings.length];
		System.arraycopy (m_rgUnrollings, 0, config.m_rgUnrollings, 0, m_rgUnrollings.length);
		
		return config;
	}
}
