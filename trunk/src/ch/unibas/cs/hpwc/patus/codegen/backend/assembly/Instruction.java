package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;

/**
 * This class encapsulates a single inline assembly instruction (mnemonic + operands).
 * 
 * @author Matthias-M. Christen
 */
public class Instruction implements IInstruction
{
	private String m_strIntrinsicBaseName;
	private IOperand[] m_rgOperands;
	
	public Instruction (String strIntrinsicBaseName, IOperand... rgOperands)
	{
		m_strIntrinsicBaseName = strIntrinsicBaseName;
		m_rgOperands = rgOperands;
	}
	
	public Instruction (TypeBaseIntrinsicEnum t, IOperand... rgOperands)
	{
		this (t.value (), rgOperands);
	}
	
	/**
	 * 
	 * @return
	 */
	public String getIntrinsicBaseName ()
	{
		return m_strIntrinsicBaseName;
	}

	/**
	 * 
	 * @return
	 */
	public IOperand[] getOperands ()
	{
		return m_rgOperands;
	}
	
	public void issue (StringBuilder sbResult)
	{
		sbResult.append (m_strIntrinsicBaseName);
		sbResult.append (" ");
		
		boolean bFirst = true;
		for (IOperand op : m_rgOperands)
		{
			if (!bFirst)
				sbResult.append (", ");
			sbResult.append (op.toString ());
			bFirst = false;
		}
		
		sbResult.append ("\\n\\t");
	}
	
	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder (m_strIntrinsicBaseName);
		issue (sb);
		return sb.toString ();
	}
}
