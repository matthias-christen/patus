package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.HashMap;
import java.util.Map;

public class Label implements IInstruction
{
	///////////////////////////////////////////////////////////////////
	// Static Types

	private static int m_nCurrentLabelIdx = 1;
	private static Map<String, Label> m_mapLabels = new HashMap<String, Label> ();
	

	///////////////////////////////////////////////////////////////////
	// Member Variables

	private int m_nLabelIdx;
	private boolean m_bAddedToInstructions;
	

	///////////////////////////////////////////////////////////////////
	// Implementation

	public static Label getLabel (String strLabelIdentifier)
	{
		Label label = m_mapLabels.get (strLabelIdentifier);
		if (label == null)
			m_mapLabels.put (strLabelIdentifier, label = new Label (strLabelIdentifier));
		
		label.m_bAddedToInstructions = true;
		return label;
	}
	
	public static IOperand.LabelOperand getLabelOperand (String strLabelIdentifier)
	{
		Label label = m_mapLabels.get (strLabelIdentifier);
		return new IOperand.LabelOperand (label.m_nLabelIdx, label.m_bAddedToInstructions ? IOperand.EJumpDirection.BACKWARD : IOperand.EJumpDirection.FORWARD);
	}

	private Label (String strLabelIdentifier)
	{
		m_nLabelIdx = m_nCurrentLabelIdx;
		m_nCurrentLabelIdx++;
		
		m_bAddedToInstructions = false;
	}
		
	@Override
	public void issue (StringBuilder sbResult)
	{
		sbResult.append (m_nLabelIdx);
		sbResult.append (":\n\t");
	}

	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ();
		issue (sb);
		return sb.toString ();
	}
}
