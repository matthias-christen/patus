package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.Arrays;
import java.util.Map;

import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

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
		return sb.substring (0, sb.length () - 4);
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
	
	@Override
	public String toJavaCode (Map<IOperand, String> mapOperands)
	{
		StringBuilder sb = new StringBuilder ();
		StringBuilder sbInstr = new StringBuilder ("il.addInstruction (new Instruction (\"");
		sbInstr.append (m_strIntrinsicBaseName);
		sbInstr.append ('"');
		
		for (IOperand op : m_rgOperands)
		{
			String strOp = mapOperands.get (op);
			if (strOp == null)
			{
				strOp = op.toJavaCode ();
				String[] strParts = strOp.split (" ");
				
				if (strParts[0].equals ("new"))
				{
					String strOperand = StringUtil.concat ("op", mapOperands.size ());
					
					sb.append (strParts[1]);
					sb.append (' ');
					sb.append (strOperand);
					sb.append (" = ");
					sb.append (strOp);
					sb.append (";\n");
					
					strOp = strOperand;
				}
				
				mapOperands.put (op, strOp);
			}
			
			sbInstr.append (", ");
			sbInstr.append (strOp);
		}
		
		sbInstr.append ("));");
		sb.append (sbInstr);
		
		return sb.toString ();
	}
}
