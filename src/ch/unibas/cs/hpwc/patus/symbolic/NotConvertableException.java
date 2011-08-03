package ch.unibas.cs.hpwc.patus.symbolic;

import cetus.hir.Expression;

/**
 *
 * @author Matthias-M. Christen
 */
public class NotConvertableException extends Exception
{
	private static final long serialVersionUID = 1L;


	///////////////////////////////////////////////////////////////////
	// Member Variables


	///////////////////////////////////////////////////////////////////
	// Implementation

	public NotConvertableException ()
	{
		super ();
	}

	public NotConvertableException (String s)
	{
		super (s);
	}

	public NotConvertableException (Expression expr)
	{
		super ("The expression " + expr.toString () + " can't be converted to a Maxima expression");
	}
}
