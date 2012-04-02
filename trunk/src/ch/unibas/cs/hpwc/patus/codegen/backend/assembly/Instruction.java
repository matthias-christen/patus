package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Arrays;

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
		StringBuilder sb = new StringBuilder ();
		issue (sb);
		return sb.toString ();
	}

	@Override
	public int hashCode ()
	{
		final int nPrime = 31;
		int nResult = nPrime + Arrays.hashCode (m_rgOperands);
		nResult = nPrime * nResult + ((m_strIntrinsicBaseName == null) ? 0 : m_strIntrinsicBaseName.hashCode ());

		return nResult;
	}

	@Override
	public boolean equals (Object obj)
	{
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (!(obj instanceof Instruction))
			return false;
		
		Instruction instrOther = (Instruction) obj;
		if (!Arrays.equals (m_rgOperands, instrOther.m_rgOperands))
			return false;
		if (m_strIntrinsicBaseName == null)
		{
			if (instrOther.m_strIntrinsicBaseName != null)
				return false;
		}
		else if (!m_strIntrinsicBaseName.equals (instrOther.m_strIntrinsicBaseName))
			return false;
		
		return true;
	}
}
