package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import ch.unibas.cs.hpwc.patus.arch.TypeRegister;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
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
		
		public TypeRegister getRegister ()
		{
			return m_register;
		}
				
		@Override
		public String getAsString ()
		{
			return StringUtil.concat ("%%", m_register.getName ());
		}
		
		@Override
		public String toJavaCode ()
		{
			return StringUtil.concat ("new Register (new TypeRegister ())");
		}
	}
	
	public static class InputRef extends AbstractOperand implements IRegisterOperand
	{
		private String m_strRef;
		private int m_nIndex;
		
		public InputRef (String strRef)
		{
			m_strRef = strRef;
			m_nIndex = -1;
		}
		
		public void setIndex (int nIndex)
		{
			m_nIndex = nIndex;
		}
		
		public int getIndex ()
		{
			return m_nIndex;
		}
		
		@Override
		public String getAsString ()
		{
			if (m_nIndex == -1)
				return StringUtil.concat ("{in:", m_strRef, "}");
			return StringUtil.concat ("%", m_nIndex);
		}

		@Override
		public int hashCode ()
		{
			return m_strRef.hashCode ();
		}

		@Override
		public boolean equals (Object obj)
		{
			if (this == obj)
				return true;
			if (!(obj instanceof InputRef))
				return false;
			
			InputRef other = (InputRef) obj;
			if (m_nIndex == -1)
			{
				if (other.m_nIndex != -1)
					return false;
				return m_strRef.equals (other.m_strRef);
			}
			
			if (m_nIndex != other.m_nIndex)
				return false;
			
			return true;
		}
		
		@Override
		public String toJavaCode ()
		{
			return StringUtil.concat ("new InputRef (\"", m_strRef, "\")");
		}
	}
	
	public static class PseudoRegister extends AbstractOperand implements IRegisterOperand
	{
		private static int m_nPseudoRegisterNumber = 0;
		
		public static void reset ()
		{
			m_nPseudoRegisterNumber = 0;
		}
		
		public static boolean isPseudoRegisterOfType (IOperand op, TypeRegisterType regtype)
		{
			if (!(op instanceof PseudoRegister))
				return false;
			return ((PseudoRegister) op).getRegisterType ().equals (regtype);
		}
		
		
		private int m_nNumber;
		private TypeRegisterType m_regtype;
		
		public PseudoRegister (TypeRegisterType regtype)
		{
			m_nNumber = m_nPseudoRegisterNumber++;
			m_regtype = regtype;
		}
		
		public int getNumber ()
		{
			return m_nNumber;
		}
		
		public TypeRegisterType getRegisterType ()
		{
			return m_regtype;
		}
		
		@Override
		public String getAsString ()
		{
			return StringUtil.concat ("{pseudoreg-", m_nNumber, ":", m_regtype.toString (), "}");
		}
		
		@Override
		public boolean equals (Object obj)
		{
			if (this == obj)
				return true;
			if (!(obj instanceof PseudoRegister))
				return false;
			return ((PseudoRegister) obj).m_nNumber == m_nNumber;
		}
		
		@Override
		public int hashCode ()
		{
			return m_nNumber;
		}
		
		@Override
		public String toJavaCode ()
		{
			return StringUtil.concat ("new PseudoRegister (TypeRegisterType.", m_regtype.name (), ")");
		}
	}
	
	public static class Immediate extends AbstractOperand
	{
		private long m_nValue;
		
		public Immediate (long nValue)
		{
			m_nValue = nValue;
		}
		
		public long getValue ()
		{
			return m_nValue;
		}

		@Override
		public String getAsString ()
		{
			return StringUtil.concat ("$", m_nValue);
		}
		
		@Override
		public String toJavaCode ()
		{
			return StringUtil.concat ("new Immediate (", m_nValue, ")");
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
		
		public long getDisplacement ()
		{
			return m_nDisplacement;
		}

		public IRegisterOperand getRegBase ()
		{
			return m_regBase;
		}

		public IRegisterOperand getRegIndex ()
		{
			return m_regIndex;
		}

		public int getScale ()
		{
			return m_nScale;
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
		
		@Override
		public String toJavaCode ()
		{
			return StringUtil.concat ("new Address (",
				m_regBase == null ? "null" : m_regBase.toJavaCode (), ", ",
				m_regIndex == null ? "null" : m_regIndex.toJavaCode (), ", ",
				m_nScale, ", ", m_nDisplacement, ")"
			);
		}
	}
	
	public enum EJumpDirection
	{
		FORWARD ('f'),
		BACKWARD ('b');
		
		char m_chDir;
		
		private EJumpDirection (char chDir)
		{
			m_chDir = chDir;
		}
		
		@Override
		public String toString ()
		{
			return String.valueOf (m_chDir);
		}
		
		public static EJumpDirection fromString (String s)
		{
			if (Character.toString (FORWARD.m_chDir).equals (s))
				return FORWARD;
			if (Character.toString (BACKWARD.m_chDir).equals (s))
				return BACKWARD;
			return null;
		}
	}
	
	public static class LabelOperand extends AbstractOperand
	{
		private String m_strLabelIdentifier;
		
		public LabelOperand (int m_nLabelIdx, EJumpDirection dir)
		{
			m_strLabelIdentifier = StringUtil.concat (m_nLabelIdx, dir.toString ());
		}
		
		@Override
		public String getAsString ()
		{
			return m_strLabelIdentifier;
		}
		
		@Override
		public String toJavaCode ()
		{
			return StringUtil.concat ("new LabelOperand (", Integer.parseInt (m_strLabelIdentifier.substring (0, m_strLabelIdentifier.length () - 1)), "EJumpDirection.", ")");
		}
	}
	

	///////////////////////////////////////////////////////////////////
	// Method Definitions

	/**
	 * Returns a string representation of the operand for the generation of the
	 * assembly code.
	 * 
	 * @return The assembly string representation
	 */
	public abstract String getAsString ();
	
	public abstract String toJavaCode ();
	
	@Override
	public boolean equals (Object obj);
	
	@Override
	public int hashCode ();
}
