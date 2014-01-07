package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.HashMap;
import java.util.Map;

import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class Label extends AbstractInstruction
{
	///////////////////////////////////////////////////////////////////
	// Static Types

	private static int m_nCurrentLabelIdx;
	private static Map<String, Label> m_mapLabels;
	
	static
	{
		Label.reset ();
	}
	

	///////////////////////////////////////////////////////////////////
	// Member Variables

	private int m_nLabelIdx;
	private boolean m_bAddedToInstructions;
	

	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public static void reset ()
	{
		m_nCurrentLabelIdx = 1;
		m_mapLabels = new HashMap<> ();
	}

	/**
	 * Returns a label {@link IInstruction} with identifier <code>strLabelIdentifier</code>.
	 * 
	 * @param strLabelIdentifier
	 *            The identifier of the label to create in the instruction list
	 * @return The label {@link IInstruction}
	 */
	public static Label getLabel (String strLabelIdentifier)
	{
		Label label = getOrCreateLabel (strLabelIdentifier);
		label.m_bAddedToInstructions = true;
		return label;
	}
	
	/**
	 * Creates a label operand for jump instructions to the label with identifier <code>strLabelIdentifier</code>.
	 * 
	 * @param strLabelIdentifier
	 *            The label identifier to which to jump to
	 * @return The label operand
	 */
	public static IOperand.LabelOperand getLabelOperand (String strLabelIdentifier)
	{
		Label label = getOrCreateLabel (strLabelIdentifier);
		return new IOperand.LabelOperand (label.m_nLabelIdx, label.m_bAddedToInstructions ? IOperand.EJumpDirection.BACKWARD : IOperand.EJumpDirection.FORWARD);
	}

	private static Label getOrCreateLabel (String strLabelIdentifier)
	{
		Label label = m_mapLabels.get (strLabelIdentifier);
		if (label == null)
			m_mapLabels.put (strLabelIdentifier, label = new Label (strLabelIdentifier));

		return label;
	}

	private Label (String strLabelIdentifier)
	{
		m_nLabelIdx = m_nCurrentLabelIdx;
		m_nCurrentLabelIdx++;
		
		m_bAddedToInstructions = false;
	}
	
	@Override
	public TypeBaseIntrinsicEnum getIntrinsic ()
	{
		return null;
	}
		
	@Override
	public void issue (StringBuilder sbResult)
	{
		sbResult.append (m_nLabelIdx);
		sbResult.append (":\\n\\t");
	}

	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ();
		issue (sb);
		return sb.toString ();
	}
	
	@Override
	public String toJavaCode (Map<IOperand, String> mapOperands)
	{
		return StringUtil.concat ("Label.getLabel (\"lbl", m_nLabelIdx, "\");");
	}
}
