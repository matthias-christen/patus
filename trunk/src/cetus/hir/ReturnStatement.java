package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/**
* Represents a return statement within a procedure.
*/
public class ReturnStatement extends Statement
{
  private static Method class_print_method;

  static
  {
    Class<?>[] params = new Class<?>[2];

    try {
      params[0] = ReturnStatement.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  /** Creates a "return nothing" statement.
   */
  public ReturnStatement()
  {
    object_print_method = class_print_method;
  }

  /** Creates a statement that returns an expression.
   *
   * @param expr The expression to return.
   * @throws IllegalArgumentException if <b>expr</b> is null.
   * @throws NotAnOrphanException if <b>expr</b> has a parent.
   */  
  public ReturnStatement(Expression expr)
  {
    object_print_method = class_print_method;
    addChild(expr);
  }

  /**
   * Prints a break statement to a stream.
   *
   * @param s The statement to print.
   * @param o The writer on which to print the statement.
   */
  public static void defaultPrint(ReturnStatement s, PrintWriter o)
  {
    o.print("return ");
    if (!s.children.isEmpty())
      s.children.get(0).print(o);
    o.print(";");
  }

  /**
   * Returns the expression that is being returned by this statement,
   * or null if nothing is being returned.
   *
   * @return the expression that is being returned by this statement,
   *   or null if nothing is being returned.
   */
  public Expression getExpression()
  {
    if (children.size() > 0)
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

  /** Returns a clone of the return statement. */
  @Override
  public ReturnStatement clone()
  {
      ReturnStatement rs = (ReturnStatement) super.clone();
      return rs;
  }
}
