package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/** Represents expressions separated by the comma
 * operator in C or C++.  The entire expression evaluates
 * to the last expression in the list.
 */
public class CommaExpression extends Expression
{
  private static Method class_print_method;

  static
  {
    Class<?>[] params = new Class<?>[2];

    try {
      params[0] = CommaExpression.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  /**
  * Constructs a comma expression from the specified list of expressions.
  *
  * @param expr_list the list of new expressions.
  * @throws IllegalArgumentException if <b>expr_list</b> is a singleton or null.
  * @throws NotAnOrphanException if an element of <b>expr_list</b> has a parent
  * object.
  */
  public CommaExpression(List expr_list)
  {
    object_print_method = class_print_method;

    if (expr_list == null || expr_list.size() < 2)
      throw new IllegalArgumentException();

    for(Expression expr : (List<Expression>)expr_list)
      addExpression(expr);
  }

  /**
  * Inserts a new expression at the end of the expression list.
  *
  * @param expr the new expression to be inserted.
  * @throws IllegalArgumentException if <b>expr</b> is null.
  * @throws NotAnOrphanException if <b>expr</b> has a parent object.
  */
  public void addExpression(Expression expr)
  {
    addChild(expr);
  }

  @Override
  public CommaExpression clone()
  {
    CommaExpression o = (CommaExpression)super.clone();
    return o; 
  }

  /**
   * Prints a CommaExpression to a stream.
   *
   * @param e The expression to print.
   * @param o The writer on which to print the expression.
   */
  public static void defaultPrint(CommaExpression e, PrintWriter o)
  {
    o.print("(");
    PrintTools.printListWithComma(e.children, o);
    o.print(")");
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
