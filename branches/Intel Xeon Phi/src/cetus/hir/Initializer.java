package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

public class Initializer implements Cloneable, Traversable 
{
  /* two types of initializers - value initializers (single
     values and array of values) and constructors

     = Expression
     = { List of Expressions }

     What about...?
     my_class y(1, 2);
  */

  private static Method class_print_method;
  private Method object_print_method;

  protected Traversable parent;
  protected LinkedList<Traversable> children;
  private boolean is_list;

  static
  {
    Class<?>[] params = new Class<?>[2];

    try {
      params[0] = Initializer.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  protected Initializer()
  {
    children = new LinkedList<Traversable>();
  }

  public Initializer(Expression value)
  {
    object_print_method = class_print_method;

    children = new LinkedList<Traversable>();
    children.add(value);
    value.setParent(this);
    is_list = false;
  }

  public Initializer(List values)
  {
    object_print_method = class_print_method;

    children = new LinkedList<Traversable>();

		for(Traversable t : (List<Traversable>)values)
		{
      // children must be expressions, or initializers of the list form 

      if (t instanceof Expression
          || t instanceof Initializer)
      {
        children.add(t);
        t.setParent(this);
      }
      else
      {
        System.out.println("Class = " + t.getClass().getName());
        if (((Initializer)t).children == null)
        {
          System.out.println("Null children !");
        }
        throw new IllegalArgumentException();
      }
		}
    is_list = true;
  }

  @Override
  public Initializer clone()
  {
    Initializer o = null;

    try {
      o = (Initializer)super.clone();
    } catch (CloneNotSupportedException e) {
      throw new InternalError();
    }

    o.object_print_method = object_print_method;
    o.parent = null;

    o.children = new LinkedList<Traversable>();
    Iterator iter = children.iterator();
    while (iter.hasNext())
    {
      Object c = iter.next();

      if (c instanceof Expression)
      {
        Expression expr = ((Expression)c).clone();
        o.children.add(expr);
        expr.setParent(o);
      }
      else
      {
        Initializer init = ((Initializer)c).clone();
        o.children.add(init);
        init.setParent(o);
      }
    }

    return o;
  }

  /**
   * Prints an initializer to a stream.
   *
   * @param i The initializer to print.
   * @param o The writer on which to print the initializer.
   */
  public static void defaultPrint(Initializer i, PrintWriter o)
  {
    if (i.parent==null || !(i.parent instanceof Initializer))
      o.print(" = ");
    if ( i.is_list ) {
      o.print(" { ");
      PrintTools.printListWithComma(i.children, o);
      o.print(" } ");
    } else
      PrintTools.printList(i.children, o);
  }

  /**
  * Converts this initializer to a string by calling the default print method.
  * All sub classes will be using this method unless specialized.
  */
  @Override
  public String toString()
  {
    StringWriter sw = new StringWriter(80);
    print(new PrintWriter(sw));
    return sw.toString();
  }

  public List<Traversable> getChildren()
  {
    return children;
  }

  public Traversable getParent()
  {
    return parent;
  }

  /** Prints an initializer object by calling its default print method. */
  public void print(PrintWriter o)
  {
    if (object_print_method == null)
      return;
    try {
      object_print_method.invoke(null, new Object[] {this, o});
    } catch (IllegalAccessException e) {
      throw new InternalError();
    } catch (InvocationTargetException e) {
      throw new InternalError();
    }
  }

  public void removeChild(Traversable child)
  {
    throw new UnsupportedOperationException("Initializers do not support removal of arbitrary children.");
  }

  public void setChild(int index, Traversable t)
  {
    children.get(index).setParent(null);
    t.setParent(this);
    children.set(index, t);
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

  public void setParent(Traversable t)
  {
    parent = t;
  }

  /**
   * Overrides the print method for this object only.
   *
   * @param m The new print method.
   */
  public void setPrintMethod(Method m)
  {
    object_print_method = m;
  }
}
