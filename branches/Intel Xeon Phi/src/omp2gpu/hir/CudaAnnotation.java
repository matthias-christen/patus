package omp2gpu.hir;

import java.util.*;
import cetus.hir.*;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 * CudaAnnotation is used for internally representing Cuda pragmas, which contain
 * Cuda-related user directives. Cuda pragmas are raw text right after parsing 
 * but converted to an internal annotation of this type.
 * 
 * The following user directives are supported:
 * #pragma cuda ainfo procname(proc-name) kernelid(kernel-id)
 * 
 * #pragma cuda cpurun [clause[[,] clause]...]
 * where clause is one of the following
 * 		c2gmemtr(list) 
 * 		noc2gmemtr(list) 
 * 		g2cmemtr(list) 
 *      nog2cmemtr(list)
 *      
 * #pragma cuda gpurun [clause[[,] clause]...]
 * where clause is one of the following
 * 		c2gmemtr(list) 
 * 		noc2gmemtr(list) 
 * 		g2cmemtr(list) 
 *      nog2cmemtr(list)
 * 		registerRO(list) 
 * 		registerRW(list) 
 *      noregister(list)
 * 		sharedRO(list) 
 * 		sharedRW(list) 
 *      noshared(list)
 * 		texture(list) 
 *      notexture(list)
 * 		constant(list)
 * 		noconstant(list)
 *      maxnumofblocks(nblocks)
 *      noreductionunroll(list)
 *      nocudamalloc(list)
 *      nocudafree(list)
 *      cudafree(list)
 *      procname(name)
 *      kernelid(id)
 *      noploopswap
 *      noloopcollapse
 *      threadblocksize(bsize)
 *      
 * #pragma cuda nogpurun
 * 
 */
public class CudaAnnotation extends PragmaAnnotation
{
	// Pragmas used without values
	private static final Set<String> no_values =
		new HashSet<String>(Arrays.asList("ainfo", "cpurun", "gpurun", 
				"noploopswap", "noloopcollapse", "nogpurun"));

	// Pragmas used with collection of values
	private static final Set<String> collection_values =
		new HashSet<String>(Arrays.asList("c2gmemtr", "noc2gmemtr", "g2cmemtr",
		"nog2cmemtr", "registerRO", "registerRW", "sharedRO", "sharedRW", 
		"texture", "constant", "noconstant", "maxnumofblocks", "noreductionunroll",
		"nocudamalloc", "nocudafree", "cudafree", "procname", "kernelid",
		"noregister", "noshared", "notexture", "threadblocksize"));

	/**
	 * Constructs an empty omp annotation.
	 */
	public CudaAnnotation()
	{
		super();
	}

	/**
	 * Constructs an omp annotation with the given key-value pair.
	 */
	public CudaAnnotation(String key, Object value)
	{
		super();
		put(key, value);
	}

	/**
	 * Returns the string representation of this cuda annotation.
	 * @return the string representation.
	 */
	public String toString()
	{
		if ( skip_print )
			return "";

		StringBuilder str = new StringBuilder(80);

		str.append(super.toString()+"cuda ");

		// Process "ainfo", "gpurun", or "cpurun" before any other keys are processed.
		if ( containsKey("ainfo") )
			str.append("ainfo ");
		if ( containsKey("gpurun") )
			str.append("gpurun ");
		if ( containsKey("cpurun") )
			str.append("cpurun ");
		if ( containsKey("nogpurun") )
			str.append("nogpurun ");

		for ( String key : keySet() )
		{
			if ( key.equals("gpurun") || key.equals("cpurun")
					|| key.equals("ainfo") || key.equals("nogpurun") )
				;
			else if ( no_values.contains(key) )
				str.append(key+" ");
			else if ( collection_values.contains(key) )
			{
				Object value = get(key);
				if ( value instanceof Collection )
					str.append(key+"("+
						PrintTools.collectionToString((Collection)value, ", ")+") ");
				else // e.g., schedule
					str.append(key+"("+value+") ");
			}
			else
			{
				str.append(key+" ");
				if ( get(key) != null )
					str.append(get(key)+" ");
			}
		}

		return str.toString();
	}
}
