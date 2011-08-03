package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/** Represents a typecast expression in C programs. */
public class Typecast extends Expression
{
  private static Method class_print_method;

  static
  {
    Class<?>[] params = new Class<?>[2];

    try {
      params[0] = Typecast.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  /** Represents a cast type */
  public static class Cast
  {
    private static String[] name =
      { "", "dynamic_cast", "static_cast",
        "reinterpret_cast", "const_cast" };

    private int value;

    /** Constructs a cast type with the specified value */
    public Cast(int value)
    {
      this.value = value;
    }

    /** Prints the cast type */
    public void print(PrintWriter o)
    {
      o.print(name[value]);
    }
  }

  public static final Cast NORMAL  = new Cast(0);
  public static final Cast DYNAMIC = new Cast(1);
  public static final Cast STATIC  = new Cast(2);
  public static final Cast REINTERPRET = new Cast(3);
  public static final Cast CONST = new Cast(4);

  private Cast kind;
  private LinkedList specs;

  /**
   * Create a normal typecast.
   *
   * @param specs A list of type specifiers.
   * @param expr The expression to cast.
   * @throws NotAnOrphanException if <b>expr</b> has a parent.
   */
  public Typecast(List specs, Expression expr)
  {
    object_print_method = class_print_method;

    kind = NORMAL;
    this.specs = (new ChainedList()).addAllLinks(specs);

    addChild(expr);
  }

  /**
   * Create a special typecast.
   *
   * @param kind One of <var>NORMAL, DYNAMIC, STATIC, REINTERPRET,</var> or <var>CONST</var>.
   * @param specs A list of type specifiers.
   * @param expr The expression to cast.
   */
  public Typecast(Cast kind, List specs, Expression expr)
  {
    object_print_method = class_print_method;

    this.kind = kind;
    this.specs = (new ChainedList()).addAllLinks(specs);

    addChild(expr);
  }

  /**
  * Constructs a typecast with the specified kind, specifier, and list of
  * expressions.
  */
  public Typecast(Cast kind, Specifier spec, List expr_list)
  {
    object_print_method = class_print_method;

    this.kind = kind;
    this.specs = (new ChainedList()).addLink(spec);

    for (Object o : expr_list)
      addChild((Traversable)o);
  }

  /**
   * Prints a typecast expression to a stream.
   *
   * @param c The cast to print.
   * @param o The writer on which to print the cast.
   */
  public static void defaultPrint(Typecast c, PrintWriter o)
  {
    if (c.needs_parens)
      o.print("(");
    if (c.kind == NORMAL) {
      if (c.children.size() == 1) {
        o.print("(");
        PrintTools.printListWithSpace(c.specs, o);
        o.print(")");
        c.children.get(0).print(o);
      } else {
        PrintTools.printListWithSpace(c.specs, o);
        o.print("(");
        PrintTools.printListWithSeparator(c.children, o, ",");
        o.print(")");
      }
    } else {
      c.kind.print(o);
      o.print("<");
      PrintTools.printList(c.specs, o);
      o.print(">(");
      c.children.get(0).print(o);
      o.print(")");
    }
    if (c.needs_parens)
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

  /** Returns the list of specifiers of the typecast. */
  public List getSpecifiers()
  {
    return specs;
  }

  /** Compares the typecast with the specified object for equality. */
  @Override
  public boolean equals(Object o)
  {
    return (
        super.equals(o) &&
        kind == ((Typecast)o).kind &&
        specs.equals(((Typecast)o).specs)
    );
  }
}
