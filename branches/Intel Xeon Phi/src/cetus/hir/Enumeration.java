package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/**
* <b>Enumeration</b> represents a C or C++ enumeration.
*/
public class Enumeration extends Declaration 
{
  private static Method class_print_method;

  static
  {
    Class<?>[] params = new Class<?>[2];

    try {
      params[0] = Enumeration.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  /** The name of the enumeration expressed in terms of an ID */
  private IDExpression name;

  /** The specifier created from this enumeration */
  private UserSpecifier specifier;

  /**
   * Creates an enumeration.
   *
   * @param name The name of the enumeration.
   * @param declarators A list of declarators to use as the enumerators.
   *   For enumerations that are not consecutive, initializers should be
   *   placed on the declarators.
   */
  public Enumeration(IDExpression name, List declarators)
  {
    object_print_method = class_print_method;

    if (name == null
        || !Tools.verifyHomogeneousList(declarators, Declarator.class))
      throw new IllegalArgumentException();

    this.name = name;
    this.specifier = new UserSpecifier(new NameID("enum "+name.toString()));

    for(Declarator d : (List<Declarator>)declarators)
    {
      children.add(d);
      d.setParent(this);
    }
  }

  /**
   * Prints an enumeration to a stream.
   *
   * @param e The enumeration to print.
   * @param o The writer on which to print the enumeration.
   */
  public static void defaultPrint(Enumeration e, PrintWriter o)
  {
    e.specifier.print(o);
    o.print(" { ");
    PrintTools.printListWithComma(e.children, o);
    o.print(" }");
  }

  /* Declaration.getDeclaredIDs() */
  public List getDeclaredIDs()
  {
    List ret = new LinkedList();
    // TODO: check if enum name is not necessary in the look-up table.
    //ret.add(name);
    for (Traversable child : children)
      if (child instanceof Declarator)
        ret.add(((Declarator)child).getID());
    return ret;
  }

  /**
  * Returns the name ID of this enumeration.
  *
  * @return the identifier holding the enum name.
  */
  public IDExpression getName()
  {
    return name;
  }

  /**
  * Returns the specifier created from this enumeration.
  *
  * @return the user specifier {@code enum name}.
  */
  public Specifier getSpecifier()
  {
    return specifier;
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
