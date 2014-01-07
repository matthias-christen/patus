package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/** Represents __builtin_va_arg() operation in C programs. */
public class VaArgExpression extends Expression implements Intrinsic
{
  private static Method class_print_method;

  static
  {
    Class<?>[] params = new Class<?>[2];

    try {
      params[0] = VaArgExpression.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  private LinkedList specs;

  /**
  * Constructs a va_arg expression with the specified expression and specs.
  *
  * @param expr the operand expression.
  * @param pspecs the list of specifiers.
  * @throws NotAnOrphanException if <b>expr</b> has a parent.
  */
  public VaArgExpression(Expression expr, List pspecs)
  {
    object_print_method = class_print_method;

    addChild(expr);

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
   * Prints a __builtin_va_arg expression to a stream.
   *
   * @param e The expression to print.
   * @param o The writer on which to print the expression.
   */
  public static void defaultPrint(VaArgExpression e, PrintWriter o)
  {
    o.print("__builtin_va_arg(");
    e.getExpression().print(o);
    o.print(",");
    PrintTools.printListWithSpace(e.specs, o);
    o.print(")");
  }

  /**
   * Returns the expression.
   *
   * @return the expression or null if this sizeof operator is
   *   being applied to a type.
   */
  public Expression getExpression()
  {
      return (Expression)children.get(0);
  }

  /**
   * Returns the type argument which is also the type of return value.
   *
   * @return the list of specifiers.
   */
  public List getSpecifiers()
  {
    return specs;
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

  /** Compares the va_arg expression with the specified object for equality. */
  @Override
  public boolean equals(Object o)
  {
    return (
        super.equals(o) &&
        specs.equals(((VaArgExpression)o).specs)
    );
  }
}
