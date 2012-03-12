package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.HashMap;
import java.util.Map;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;

public class Label implements IInstruction
{
	///////////////////////////////////////////////////////////////////
	// Static Types

	private static int m_nCurrentLabelIdx = 1;
	private static Map<String, Label> m_mapLabels = new HashMap<String, Label> ();
	

	///////////////////////////////////////////////////////////////////
	// Member Variables

	private int m_nLabelIdx;
	

	///////////////////////////////////////////////////////////////////
	// Implementation

	public Label (String strLabelIdentifier)
	{
		m_nLabelIdx = m_nCurrentLabelIdx;
		m_nCurrentLabelIdx++;
		m_mapLabels.put (strLabelIdentifier, this);
	}
	
	@Override
	public void issue (StringBuilder sbResult)
	{
		sbResult.append (m_nLabelIdx);
		sbResult.append (":\n\t");
	}
}
