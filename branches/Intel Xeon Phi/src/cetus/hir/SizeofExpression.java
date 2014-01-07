package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/** Represents a sizeof operation in C programs. */
public class SizeofExpression extends Expression implements Intrinsic
{
  private static Method class_print_method;

  static
  {
    Class[] params = new Class[2];

    try {
      params[0] = SizeofExpression.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  private LinkedList specs;

  /** Constructs a sizeof expression with the specified expression operand */
  public SizeofExpression(Expression expr)
  {
    object_print_method = class_print_method;

    specs = null;

    addChild(expr);
  }

  /** Constructs a sizeof expression with the given specifiers */
  public SizeofExpression(List pspecs)
  {
    object_print_method = class_print_method;
// This part breaks in 176.gcc
/*
    if (!Tools.verifyHomogeneousList(specs, Specifier.class))
      throw new IllegalArgumentException();
*/

    specs = new LinkedList();
    specs.addAll(pspecs);
  }
/*
  public Object clone()
  {
    SizeofExpression o = (SizeofExpression)super.clone();
    return o;
  }
*/
  /**
   * Prints a sizeof expression to a stream.
   *
   * @param e The expression to print.
   * @param o The writer on which to print the expression.
   */
  public static void defaultPrint(SizeofExpression e, PrintWriter o)
  {
    o.print("sizeof ");
    if (e.specs != null) {
      o.print("(");
      PrintTools.printListWithSeparator(e.specs, o, " ");
      o.print(")");
    } else {
      e.getExpression().print(o);
    }
  }

  /**
   * Returns the expression.
   *
   * @return the expression or null if this sizeof operator is
   *   being applied to a type.
   */
  public Expression getExpression()
  {
    if (specs == null)
      return (Expression)children.get(0);
    else
      return null;
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

  /** Compares the sizeof expression with the specified object for equality */
  @Override
  public boolean equals(Object o)
  {
    return (
        super.equals(o) &&
        (specs == null && ((SizeofExpression)o).specs == null ||
        specs != null && specs.equals(((SizeofExpression)o).specs))
    );
  }
}
