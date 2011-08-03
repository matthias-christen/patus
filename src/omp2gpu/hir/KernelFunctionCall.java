package omp2gpu.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;
import cetus.hir.*;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 * Represents a function or method call.
 */
public class KernelFunctionCall extends FunctionCall	
{
  private static Method class_print_method;
  private LinkedList<Traversable> configuration;

  static
  {
    Class[] params = new Class[2];

    try {
      params[0] = KernelFunctionCall.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  /**
   * Creates a function call.
   *
   * @param function An expression that evaluates to a function.
   */
  public KernelFunctionCall(Expression function)
  {
    super(function);
    configuration = new LinkedList<Traversable>();
    object_print_method = class_print_method;
  }

  /**
   * Creates a function call.
   *
   * @param function An expression that evaluates to a function.
   * @param args A list of arguments to the function.
   */
  public KernelFunctionCall(Expression function, List args)
  {
    super(function, args);
    configuration = new LinkedList<Traversable>();
    object_print_method = class_print_method;
  }

/**
   * Creates a function call.
   *
   * @param function An expression that evaluates to a function.
   * @param args A list of arguments to the function.
   * @param confargs A list of configuration arguments to the function.
   */
  public KernelFunctionCall(Expression function, List args, List confargs)
  {
    super(function, args);
    configuration = new LinkedList<Traversable>();
    object_print_method = class_print_method;
    setConfArguments(confargs);
  }

  /**
   * Prints a function call to a stream.
   *
   * @param call The call to print.
   * @param stream The stream on which to print the call.
   */
  public static void defaultPrint(KernelFunctionCall call, PrintWriter p)
  {
    if (call.needs_parens)
      p.print("(");

    call.getName().print(p);
    p.print("<<<");
    List tmp = call.getConfArguments();
    PrintTools.printListWithComma(tmp, p);
    p.print(">>>");
    p.print("(");
    tmp = (new ChainedList()).addAllLinks(call.children);
    tmp.remove(0);
    PrintTools.printListWithComma(tmp, p);
    p.print(")");

    if (call.needs_parens)
      p.print(")");
  }

	public String toString()
	{
		StringBuilder str = new StringBuilder(80);

		if ( needs_parens )
			str.append("(");

		str.append(getName());
		str.append("<<<");
		List tmp = configuration;
		str.append(PrintTools.listToString(tmp, ", "));
		str.append(">>>");
		str.append("(");
		tmp = (new ChainedList()).addAllLinks(children);
		tmp.remove(0);
		str.append(PrintTools.listToString(tmp, ", "));
		str.append(")");

		if ( needs_parens )
			str.append(")");

		return str.toString();
	}

  public Expression getConfArgument(int n)
  {
    return (Expression)configuration.get(n);
  }

  public List getConfArguments()
  {
    return configuration;
  }

  public void setConfArgument(int n, Expression expr)
  {
    configuration.set(n, expr);
  }

  public void setConfArguments(List args)
  {
    configuration.clear();
    //configuration.addAll(args);
	for(Object o : args)
	{
      Expression expr = null;
      try {
        expr = (Expression)o;
      } catch (ClassCastException e) {
        throw new IllegalArgumentException();
      }
      configuration.add(expr);
	}
  }

/**
   * Overrides the class print method, so that all subsequently
   * created objects will use the supplied method.
   *
   * @param m The new print method.
   */
  static public void setClassPrintMethod(Method m)
  {
    class_print_method = m;
  }
}
