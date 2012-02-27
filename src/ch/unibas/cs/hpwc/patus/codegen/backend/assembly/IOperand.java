package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import ch.unibas.cs.hpwc.patus.arch.TypeRegister;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public interface IOperand
{
	///////////////////////////////////////////////////////////////////
	// Sub-Interfaces

	public interface IRegisterOperand extends IOperand
	{
	}
	

	///////////////////////////////////////////////////////////////////
	// Implementing Classes

	public static class Register implements IRegisterOperand
	{
		private TypeRegister m_register;
		
		public Register (TypeRegister register)
		{
			m_register = register;
		}
		
		public String getBaseName ()
		{
			return m_register.getName ();
		}
				
		@Override
		public String getAsString ()
		{
			return StringUtil.concat ("%%", m_register.getName ());
		}
		
		@Override
		public boolean equals (Object obj)
		{
			if (!(obj instanceof Register))
				return false;
			return ((Register) obj).m_register.getName ().equals (m_register.getName ());
		}
		
		@Override
		public int hashCode ()
		{
			return m_register.getName ().hashCode ();
		}
	}
	
	public static class InputRef implements IRegisterOperand
	{
		private int m_nIndex;
		
		public InputRef (int nIndex)
		{
			m_nIndex = nIndex;
		}
		
		@Override
		public String getAsString ()
		{
			return StringUtil.concat ("%", m_nIndex);
		}
	}
	
	public static class Immediate implements IOperand
	{
		private long m_nValue;
		
		public Immediate (long nValue)
		{
			m_nValue = nValue;
		}
		
		@Override
		public String getAsString ()
		{
			return StringUtil.concat ("$", m_nValue);
		}
	}
	
	public static class Address implements IOperand
	{
		private long m_nDisplacement;
		private IRegisterOperand m_regBase;
		private IRegisterOperand m_regIndex;
		private int m_nScale;
		
		public Address (IRegisterOperand regBase)
		{
			this (regBase, null, 1, 0);
		}
		
		public Address (IRegisterOperand regBase, long nDisplacement)
		{
			this (regBase, null, 1, nDisplacement);
		}
		
		public Address (IRegisterOperand regBase, IRegisterOperand regIndex)
		{
			this (regBase, regIndex, 1, 0);
		}
		
		public Address (IRegisterOperand regBase, IRegisterOperand regIndex, int nScale)
		{
			this (regBase, regIndex, nScale, 0);
		}
		
		public Address (IRegisterOperand regBase, IRegisterOperand regIndex, int nScale, long nDisplacement)
		{
			m_regBase = regBase;
			m_regIndex = regIndex;
			m_nScale = nScale;
			m_nDisplacement = nDisplacement;
		}
		
		/**
		 * Format: [ displ ] "(" base [ "," index [ "," scale ]] ")"
		 */
		public String getAsString ()
		{
			StringBuilder sb = new StringBuilder ();
			
			if (m_nDisplacement != 0)
				sb.append (m_nDisplacement);
			sb.append ('(');
			sb.append (m_regBase.toString ());
			
			if (m_regIndex != null)
			{
				sb.append (',');
				sb.append (m_regIndex.toString ());

				if (m_nScale != 1)
				{
					sb.append (',');
					sb.append (m_nScale);
				}
			}
			else if (m_nScale != 1)
				throw new RuntimeException ("If no index is provided, the scale has to be 1.");
			
			sb.append (')');
			
			return sb.toString ();
		}
	}
	
	public static class LabelOperand implements IOperand
	{
		public LabelOperand (String strLabelIdentifier)
		{
			
		}
		
		@Override
		public String getAsString ()
		{
			// TODO Auto-generated method stub
			return null;
		}
	}
	

	///////////////////////////////////////////////////////////////////
	// Method Definitions

	/**
	 * Returns a string representation of the operand for the generation of the assembly code.
	 * @return The assembly string representation
	 */
	public abstract String getAsString ();
}
