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
	
	public abstract static class AbstractOperand implements IOperand
	{
		@Override
		public String toString ()
		{
			return getAsString ();
		}

		@Override
		public boolean equals (Object obj)
		{
			if (obj == null)
				return false;
			if (!obj.getClass ().isInstance (this))
				return false;
			
			String strThis = getAsString ();
			String strOther = ((IOperand) obj).getAsString ();
			
			if (strThis == null)
				return strOther == null;
			
			return strThis.equals (strOther);
		}
		
		@Override
		public int hashCode ()
		{
			String s = getAsString ();
			return s == null ? 0 : s.hashCode ();
		}
	}

	public static class Register extends AbstractOperand implements IRegisterOperand
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
	}
	
	public static class InputRef extends AbstractOperand implements IRegisterOperand
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
	
	public static class PseudoRegister extends AbstractOperand implements IRegisterOperand
	{
		private static int m_nPseudoRegisterNumber = 0;
		
		public static void reset ()
		{
			m_nPseudoRegisterNumber = 0;
		}
		
		
		private int m_nNumber;
		
		public PseudoRegister ()
		{
			m_nNumber = m_nPseudoRegisterNumber++;
		}
		
		public int getNumber ()
		{
			return m_nNumber;
		}
		
		@Override
		public String getAsString ()
		{
			return StringUtil.concat ("{pseudoreg-", m_nNumber, "}");
		}
		
		@Override
		public boolean equals (Object obj)
		{
			if (!(obj instanceof PseudoRegister))
				return false;
			return ((PseudoRegister) obj).m_nNumber == m_nNumber;
		}
		
		@Override
		public int hashCode ()
		{
			return m_nNumber;
		}
	}
	
	public static class Immediate extends AbstractOperand
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
	
	public static class Address extends AbstractOperand
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
			
			sb.append (')');
			
			return sb.toString ();
		}
	}
	
	public static class LabelOperand extends AbstractOperand
	{
		private String m_strLabelIdentifier;
		
		public LabelOperand (String strLabelIdentifier)
		{
			m_strLabelIdentifier = strLabelIdentifier;
		}
		
		@Override
		public String getAsString ()
		{
			return m_strLabelIdentifier;
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
