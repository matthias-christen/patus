package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/** This class is no longer supported. */
public class NewExpression extends Expression
{
  private static Method class_print_method;

  /**
  * If blobal == true, it prints "::" before printing the new expression.
  * It doesn't print "::" by default.
  */
  protected static boolean global;

  static
  {
    Class<?>[] params = new Class<?>[2];

    try {
      params[0] = NewExpression.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  public NewExpression(List specs)
  {
    super(specs.size());

  	object_print_method = class_print_method;
    children.addAll(specs);
	global = false;
  }

  public NewExpression(List specs, Initializer init)
  {
    super(specs.size() + 1);

  	object_print_method = class_print_method;
    children.addAll(specs);
    children.add(init);

	global = false;
  }

	/**
	* force printing "::"
	*/
	public void setGlobal()
	{
		global = true;
	}

	/**
	* disable printing "::"
	*/
	public void clearGlobal()
	{
		global = false;
	}

	/**
	* returns true if it is set to be global.
	*/
	public boolean isGlobal()
	{
		return global;
	}

  @Override
	public NewExpression clone()
  {
    NewExpression o = (NewExpression)super.clone();
	o.global = this.global;

    return o;
  }

  /**
   * Prints a new expression to a stream.
   *
   * @param e The expression to print.
   * @param o The writer on which to print the expression.
   */
  public static void defaultPrint(NewExpression e, PrintWriter o)
  {
    if (e.global)
      o.print("::");
    o.print("new ");
    PrintTools.printList(e.children, o);
  }
}
