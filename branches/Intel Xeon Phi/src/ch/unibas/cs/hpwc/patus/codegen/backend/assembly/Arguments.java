package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import ch.unibas.cs.hpwc.patus.codegen.Globals;

public class Arguments
{
	///////////////////////////////////////////////////////////////////
	// Static Methods
	
	/**
	 * Parses the argument descriptor <code>strArgsDescriptor</code>.
	 * 
	 * @param strArgsDescriptor
	 *            The argument descriptor
	 * @return The array of arguments parsed from the architecture descriptor
	 *         string <code>strArgsDescription</code>
	 */
	public static Argument[] parseArguments (String strArgsDescriptor)
	{
		String[] rgArgDescriptors = strArgsDescriptor.split (",");
		Argument[] rgArgs = new Argument[rgArgDescriptors.length];
		
		for (int i = 0; i < rgArgDescriptors.length; i++)
			rgArgs[i] = new Argument (rgArgDescriptors[i], i);
		
		return rgArgs;
	}
	
	public static String encode (Argument[] rgArgs)
	{
		StringBuilder sb = new StringBuilder ();
		for (int i = 0; i < rgArgs.length; i++)
		{
			if (i > 0)
				sb.append (',');
			sb.append (rgArgs[i].encode ());
		}
		
		return sb.toString ();
	}

	/**
	 * Determines whether one of the arguments in <code>rgArgs</code> is an
	 * output argument
	 * 
	 * @param rgArgs
	 *            The array of arguments
	 * @return <code>true</code> iff one of <code>rgArgs</code> is an output
	 *         argument
	 */
	public static boolean hasOutput (Argument[] rgArgs)
	{
		for (Argument arg : rgArgs)
			if (arg.isOutput ())
				return true;
		return false;
	}
	
	/**
	 * Tries to find and return the left hand side (LHS) argument in the
	 * argument array <code>rgArgs</code>
	 * 
	 * @param rgArgs
	 *            The array of arguments to search
	 * @return The LHS argument in the array <code>rgArgs</code>
	 */
	public static Argument getLHS (Argument[] rgArgs)
	{
		Argument arg = Arguments.getNamedArgument (rgArgs, Globals.ARGNAME_LHS);
		if (arg == null)
		{
			// find the first argument which is not an output (and assume it's the LHS)
			for (Argument a : rgArgs)
			{
				if (!a.isOutput ())
				{
					arg = a;
					break;
				}
			}
		}
		
		return arg;
	}
	
	/**
	 * Tries to find and return the right hand side (RHS) argument in the
	 * argument array <code>rgArgs</code>
	 * 
	 * @param rgArgs
	 *            The array of arguments to search
	 * @return The RHS argument in the array <code>rgArgs</code>
	 */
	public static Argument getRHS (Argument[] rgArgs)
	{
		Argument arg = Arguments.getNamedArgument (rgArgs, Globals.ARGNAME_RHS);
		if (arg == null)
		{
			// find the second argument which is not an output (and assume it's the RHS)
			// or the second argument (even if it is an output) if there are only two arguments
			if (rgArgs.length <= 2)
				return rgArgs[rgArgs.length - 1];
			
			int nCount = 0;
			for (Argument a : rgArgs)
			{
				if (!a.isOutput ())
					nCount++;
				
				if (nCount == 2)
				{
					arg = a;
					break;
				}
			}
		}
		
		return arg;		
	}
	
	/**
	 * Finds and returns the first input argument in the argument array
	 * <code>rgArgs</code>.
	 * If no such argument can be found, <code>null</code> is returned.
	 * 
	 * @param rgArgs
	 *            The array of arguments to search
	 * @return The first input argument or <code>null</code> if no such argument
	 *         can be found
	 */
	public static Argument getFirstInput (Argument[] rgArgs)
	{
		if (rgArgs.length == 0)
			return null;
		if (rgArgs.length == 1)
			return rgArgs[0];
		
		for (Argument arg : rgArgs)
			if (!arg.isOutput ())
				return arg;
		
		return null;
	}

	/**
	 * Finds and returns the output argument in the argument array
	 * <code>rgArgs</code>.
	 * If no such argument can be found, <code>null</code> is returned.
	 * 
	 * @param rgArgs
	 *            The array of arguments to search
	 * @return The output argument or <code>null</code> if no output argument is
	 *         contained in <code>rgArgs</code>
	 */
	public static Argument getOutput (Argument[] rgArgs)
	{
		for (Argument arg : rgArgs)
			if (arg.isOutput ())
				return arg;
		return null;
	}
	
	/**
	 * Tries to find and return an argument with name <code>strName</code> in
	 * the argument array <code>rgArgs</code>.
	 * If no such argument can be found, <code>null</code> is returned.
	 * 
	 * @param rgArgs
	 *            The array of argument to search
	 * @param strName
	 *            The name of the argument to look for
	 * @return The argument named <code>strName</code> or <code>null</code> if
	 *         no such argument can be found
	 */
	public static Argument getNamedArgument (Argument[] rgArgs, String strName)
	{
		for (Argument arg : rgArgs)
			if (strName.equals (arg.getName ()))
				return arg;
		return null;
	}
}
