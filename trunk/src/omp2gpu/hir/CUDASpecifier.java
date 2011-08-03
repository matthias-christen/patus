package omp2gpu.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.HashMap;
import cetus.hir.*;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 * Represents type specifiers and modifiers for CUDA.
 */
public class CUDASpecifier extends Specifier
{
  private static HashMap<String, CUDASpecifier> spec_map = new HashMap(24);

  private static String[] names =
    {   "__global__", "__shared__", "__host__",
    	"__device__", "__constant__", "__noinline__",
    	"dim3", "size_t"};
  
  /* The following type-specifiers are added to support CUDA */
  public static final CUDASpecifier CUDA_GLOBAL	= new CUDASpecifier(0);
  public static final CUDASpecifier CUDA_SHARED	= new CUDASpecifier(1);
  public static final CUDASpecifier CUDA_HOST	= new CUDASpecifier(2);
  public static final CUDASpecifier CUDA_DEVICE	= new CUDASpecifier(3);
  public static final CUDASpecifier CUDA_CONSTANT	= new CUDASpecifier(4);
  public static final CUDASpecifier CUDA_NOINLINE	= new CUDASpecifier(5);
  public static final CUDASpecifier CUDA_DIM3	= new CUDASpecifier(6);
  /* size_t is a macro, but treat it as a specifier for convenience */
  public static final CUDASpecifier SIZE_T = new CUDASpecifier(7);

  protected int cvalue;

  protected CUDASpecifier()
  {
    cvalue = -1;
  }

  private CUDASpecifier(int cvalue)
  {
    this.cvalue = cvalue;
    spec_map.put(names[cvalue], this);
  }

  //Below is old one.
 /* public String toString()
  {
		return (( cvalue < 0 )? "": names[cvalue]);
  }*/
  
  /** Prints the specifier to the print writer. */
  public void print(PrintWriter o)
  {
    if (cvalue >= 0)
      o.print(names[cvalue]);
  }

  /** Returns a string representation of the specifier. */
  @Override
  public String toString()
  {
    StringWriter sw = new StringWriter(16);
    print(new PrintWriter(sw));
    return sw.toString();
  }
}
