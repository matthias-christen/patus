package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
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
	}
	
	/**
	 * Determines whether the operand can be loaded from a memory location.
	 * @return <code>true</code> iff the operand can be loaded from memory
	 */
	public boolean isMemory ()
	{
		return m_bIsMemory;
	}
	
	/**
	 * Determines whether the operand can be a register.
	 * @return <code>true</code> iff the operand can be a register
	 */
	public boolean isRegister ()
	{
		return m_bIsRegister;
	}
	
	/**
	 * Determines whether the argument is an output (a register or memory location
	 * to which the result is written).
	 * @return <code>true</code> iff the argument is an output
	 */
	public boolean isOutput ()
	{
		return m_bIsOutput;
	}
	
	/**
	 * Returns the name of the argument or <code>null</code> if no name was given.
	 * @return The argument's name (if any)
	 */
	public String getName ()
	{
		return m_strName;
	}
	
	/**
	 * Returns the argument's ordinal.
	 * @return The argument's ordinal number
	 */
	public int getNumber ()
	{
		return m_nNumber;
	}
	
	@Override
	public String toString ()
	{
		return StringUtil.concat ((m_bIsOutput ? "=" : ""), (m_bIsRegister ? "{reg} " : ""), (m_bIsMemory ? "{mem} " : ""), (m_strName == null ? "" : ": " + m_strName), " [", m_nNumber, "]");
	}
	
	
	public static void main (String[] args)
	{
		Argument a = new Argument ("reg/mem:resultXX", 0);
		System.out.println (a.toString ());
	}
}
