package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.x86_64;

import java.util.ArrayList;
import java.util.List;

public class PrefetchConfig
{
	private boolean m_bDoPrefetching;
	private boolean m_bPrefetchOnlyAddressable;
	private boolean m_bRestrictToLargestNodeSets;
	private boolean m_bOmitHigherDimensions;
	private int m_nUnitStrideAlignment;
	
	
	public PrefetchConfig (
		boolean bDoPrefetching,
		boolean bPrefetchOnlyAddressable, boolean bRestrictToLargestNodeSets, boolean bOmitHigherDimensions,
		int nUnitStrideAlignment)
	{
		m_bDoPrefetching = bDoPrefetching;
		m_bPrefetchOnlyAddressable = bPrefetchOnlyAddressable;
		m_bRestrictToLargestNodeSets = bRestrictToLargestNodeSets;
		m_bOmitHigherDimensions = bOmitHigherDimensions;
		
		m_nUnitStrideAlignment = nUnitStrideAlignment;
		if (m_nUnitStrideAlignment <= 0)
			m_nUnitStrideAlignment = 1;
	}
	
	public boolean doPrefetching ()
	{
		return m_bDoPrefetching;
	}
	
	public boolean isPrefetchOnlyAddressable ()
	{
		return m_bPrefetchOnlyAddressable;
	}

	public boolean isRestrictToLargestNodeSets ()
	{
		return m_bRestrictToLargestNodeSets;
	}

	public boolean isOmitHigherDimensions ()
	{
		return m_bOmitHigherDimensions;
	}

	public int getUnitStrideAlignment ()
	{
		return m_nUnitStrideAlignment;
	}

	public int toInteger ()
	{
		if (!m_bDoPrefetching)
			return 0;
		
		return
			(m_nUnitStrideAlignment & 1023) |
			(m_bPrefetchOnlyAddressable ? 1024 : 0) |
			(m_bRestrictToLargestNodeSets ? 2048 : 0 ) |
			(m_bOmitHigherDimensions ? 4096 : 0);
	}
	
	@Override
	public boolean equals (Object obj)
	{
		if (!(obj instanceof PrefetchConfig))
			return false;
		return ((PrefetchConfig) obj).toInteger () == toInteger ();
	}
	
	@Override
	public int hashCode ()
	{
		return toInteger ();
	}

	public static Iterable<PrefetchConfig> getAllConfigs ()
	{
		List<PrefetchConfig> listConfigs = new ArrayList<> ();
		
		// new PrefetchConfig (DO_PREFETCHING, ONLY_ADDRESSABLE, ONLY_LARGEST_SETS, OMIT_HIGHER_DIMS, ALIGNMENT)
		
		// Note: "omit higher dimensions" requires that "only addressable" is false
		// (hence, some configs can be omitted)
		
		listConfigs.add (new PrefetchConfig (false, false, false, false, 0));
		
		listConfigs.add (new PrefetchConfig (true, false, false, false, 1));
		listConfigs.add (new PrefetchConfig (true, true, false, false, 1));
		listConfigs.add (new PrefetchConfig (true, false, true, false, 1));
		listConfigs.add (new PrefetchConfig (true, true, true, false, 1));
		listConfigs.add (new PrefetchConfig (true, false, false, true, 1));
		listConfigs.add (new PrefetchConfig (true, false, true, true, 1));
		
		listConfigs.add (new PrefetchConfig (true, false, false, false, 64));
		listConfigs.add (new PrefetchConfig (true, true, false, false, 64));
		listConfigs.add (new PrefetchConfig (true, false, true, false, 64));
		listConfigs.add (new PrefetchConfig (true, true, true, false, 64));
		listConfigs.add (new PrefetchConfig (true, false, false, true, 64));
		listConfigs.add (new PrefetchConfig (true, false, true, true, 64));
		
		return listConfigs;
	}
}
