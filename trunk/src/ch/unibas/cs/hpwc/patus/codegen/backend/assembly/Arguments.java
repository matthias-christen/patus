package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

public class Arguments
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	public final static String NAME_LHS = "lhs";
	public final static String NAME_RHS = "rhs";

	
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
		Argument arg = Arguments.getNamedArgument (rgArgs, NAME_LHS);
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
		Argument arg = Arguments.getNamedArgument (rgArgs, NAME_RHS);
		if (arg == null)
		{
			// find the second argument which is not an output (and assume it's the RHS)
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
