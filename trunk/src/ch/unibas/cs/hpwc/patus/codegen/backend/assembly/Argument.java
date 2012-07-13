package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * <p>This class represents an assembly instruction argument. The list of arguments
 * are defined in the architecture description.</p>
 * <p>The argument list syntax is:</p>
 * <pre>
 * 	arg1,arg2,...,argN
 * </pre>
 * <p>Each of the arguments <code>argI</code> has the following syntax:</p>
 * <pre>
 * 	[ "=" ] ( "reg" | "mem" | "reg/mem" ) [ ":" argname ]
 * </pre>
 * <p>If the <code>=</code> is present, the argument is an output argument,
 * i.e., a register to which the result of the operation is written.
 * <code>reg</code> specifies that the operand has to be a register,
 * <code>mem</code> specifies that the argument can be a memory address.
 * <code>reg/mem</code> specifies that the argument can be either a register or a memory address.
 * Optionally, an argument name can be provided after the colon.
 * For binary arithmetic operations, the arguments are "lhs" and "rhs" as defined in {@link Globals}.
 * Other operations take the argument names as returned by {@link Globals#getIntrinsicArguments(String)}.
 * 
 * @author Matthias-M. Christen
 */
public class Argument
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static Pattern PATTERN_ARGUMENT = Pattern.compile ("(=)?((reg)/?)?(mem)?(:(.+))?");

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	/**
	 * <code>true</code> iff the argument can be loaded from memory
	 */
	private boolean m_bIsMemory;
	
	/**
	 * <code>true</code> iff the argument can be passed in a register
	 */
	private boolean m_bIsRegister;
	
	/**
	 * <code>true</code> iff the argument is an output argument
	 * (i.e., if the argument string was prepended by a '=')
	 */
	private boolean m_bIsOutput;
	
	/**
	 * The name of the argument if it is given any (the string after the optional colon),
	 * or <code>null</code> if no name was given
	 */
	private String m_strName;
	
	/**
	 * The number of the argument
	 */
	private int m_nNumber;


	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public Argument (String strArgDescriptor, int nNumber)
	{
		// initialize
		m_bIsMemory = false;
		m_bIsRegister = true;
		m_bIsOutput = false;
		m_strName = null;
		m_nNumber = nNumber;
		
		// parse the argument descriptor
		Matcher m = PATTERN_ARGUMENT.matcher (strArgDescriptor);
		if (m.matches ())
		{
			m_bIsOutput = "=".equals (m.group (1));
			m_bIsRegister = "reg".equals (m.group (3));
			m_bIsMemory = "mem".equals (m.group (4));
			m_strName = m.group (6);
		}
		else
			m_strName = strArgDescriptor;
	}
	
	public Argument (boolean bIsRegister, boolean bIsMemory, boolean bIsOutput, String strName, int nNumber)
	{
		m_bIsRegister = bIsRegister;
		m_bIsMemory = bIsMemory;
		m_bIsOutput = bIsOutput;
		m_strName = strName;
		m_nNumber = nNumber;
	}
	
	/**
	 * Determines whether the operand can be loaded from a memory location.
	 * 
	 * @return <code>true</code> iff the operand can be loaded from memory
	 */
	public boolean isMemory ()
	{
		return m_bIsMemory;
	}
	
	/**
	 * Determines whether the operand can be a register.
	 * 
	 * @return <code>true</code> iff the operand can be a register
	 */
	public boolean isRegister ()
	{
		return m_bIsRegister;
	}
	
	/**
	 * Determines whether the argument is an output (a register or memory
	 * location
	 * to which the result is written).
	 * 
	 * @return <code>true</code> iff the argument is an output
	 */
	public boolean isOutput ()
	{
		return m_bIsOutput;
	}
	
	/**
	 * Returns the name of the argument or <code>null</code> if no name was
	 * given.
	 * 
	 * @return The argument's name (if any)
	 */
	public String getName ()
	{
		return m_strName;
	}
	
	/**
	 * Returns the argument's ordinal.
	 * 
	 * @return The argument's ordinal number
	 */
	public int getNumber ()
	{
		return m_nNumber;
	}
	
	public String encode ()
	{
		StringBuilder sb = new StringBuilder ();
		if (m_bIsOutput)
			sb.append ('=');
		if (m_bIsRegister)
		{
			sb.append ("reg");
			if (m_bIsMemory)
				sb.append ('/');
		}
		if (m_bIsMemory)
			sb.append ("mem");
		if (m_strName != null)
		{
			sb.append (':');
			sb.append (m_strName);
		}

		return sb.toString ();
	}

	@Override
	public String toString ()
	{
		return StringUtil.concat ((m_bIsOutput ? "=" : ""), (m_bIsRegister ? "{reg} " : ""), (m_bIsMemory ? "{mem} " : ""), (m_strName == null ? "" : ": " + m_strName), " [", m_nNumber, "]");
	}
	
	@Override
	public int hashCode ()
	{
		final int nPrime = 31;
		int nResult = 1;
		nResult = nPrime * nResult + (m_bIsMemory ? 1231 : 1237);
		nResult = nPrime * nResult + (m_bIsOutput ? 1231 : 1237);
		nResult = nPrime * nResult + (m_bIsRegister ? 1231 : 1237);
		nResult = nPrime * nResult + m_nNumber;
		nResult = nPrime * nResult + ((m_strName == null) ? 0 : m_strName.hashCode ());
		return nResult;
	}

	@Override
	public boolean equals (Object obj)
	{
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass () != obj.getClass ())
			return false;
		Argument other = (Argument) obj;
		if (m_bIsMemory != other.m_bIsMemory)
			return false;
		if (m_bIsOutput != other.m_bIsOutput)
			return false;
		if (m_bIsRegister != other.m_bIsRegister)
			return false;
		if (m_nNumber != other.m_nNumber)
			return false;
		if (m_strName == null)
		{
			if (other.m_strName != null)
				return false;
		}
		else if (!m_strName.equals (other.m_strName))
			return false;
		return true;
	}

	public static void main (String[] args)
	{
		Argument a = new Argument ("reg/mem:resultXX", 0);
		System.out.println (a.toString ());
	}
}
