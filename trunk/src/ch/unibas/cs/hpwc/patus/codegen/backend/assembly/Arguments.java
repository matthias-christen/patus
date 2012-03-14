package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import ch.unibas.cs.hpwc.patus.codegen.Globals;

public class Arguments
{
	///////////////////////////////////////////////////////////////////
	// Static Methods
	
	/**
	 * Parses the argument descriptor.
	 * @param strArgsDescriptor
	 * @return
	 */
	public static Argument[] parseArguments (String strArgsDescriptor)
	{
		String[] rgArgDescriptors = strArgsDescriptor.split (",");
		Argument[] rgArgs = new Argument[rgArgDescriptors.length];
		
		for (int i = 0; i < rgArgDescriptors.length; i++)
			rgArgs[i] = new Argument (rgArgDescriptors[i], i);
		
		return rgArgs;
	}
	
	/**
	 * Determines whether one of the arguments in <code>rgArgs</code> is an output argument
	 * @param rgArgs The array of arguments
	 * @return <code>true</code> iff one of <code>rgArgs</code> is an output argument
	 */
	public static boolean hasOutput (Argument[] rgArgs)
	{
		for (Argument arg : rgArgs)
			if (arg.isOutput ())
				return true;
		return false;
	}
	
	/**
	 * 
	 * @param rgArgs
	 * @return
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
	 * 
	 * @param rgArgs
	 * @return
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
	 * 
	 * @param rgArgs
	 * @return
	 */
	public static Argument getOutput (Argument[] rgArgs)
	{
		for (Argument arg : rgArgs)
			if (arg.isOutput ())
				return arg;
		return null;
	}
	
	/**
	 * 
	 * @param rgArgs
	 * @param strName
	 * @return
	 */
	public static Argument getNamedArgument (Argument[] rgArgs, String strName)
	{
		for (Argument arg : rgArgs)
			if (strName.equals (arg.getName ()))
				return arg;
		return null;
	}
}
