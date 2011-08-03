package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/**
 * Represents a statement that introduces a new variable
 * or function prototype.  A variable declaration is
 * composed of specifiers and declarators.  For example,
 * in the declaration <var>unsigned int x, y</var> the specifiers
 * would be <var>unsigned</var> and <var>int</var> while the
 * declarators would be <var>x</var> and <var>y</var>.
 */
public class VariableDeclaration extends Declaration
{
  private static Method class_print_method;

  static
  {
    Class<?>[] params = new Class<?>[2];

    try {
      params[0] = VariableDeclaration.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  private List specs;

  /**
   * Creates a variable declaration that does not use specifiers, such
   * as for a constructor prototype.
   *
   * @param declarator The only declarator in this declaration.
   */
  public VariableDeclaration(Declarator declarator)
  {
    object_print_method = class_print_method;

    this.specs = new LinkedList();
    addDeclarator(declarator);
  }

  /**
   * Creates a parameter declaration for a procedure, where the 
   * parameter is unnamed but its type is specified.
   *
   * @param specs A list of specifiers.
   */
  public VariableDeclaration(List specs)
  {
    object_print_method = class_print_method;

    if (!Tools.verifyHomogeneousList(specs, Specifier.class))
      throw new IllegalArgumentException();

    this.specs = (new ChainedList()).addAllLinks(specs);
  }

  /**
   * Creates a variable declaration that has only one specifier, such
   * as for <var>int x</var>.
   *
   * @param spec A single specifier.
   * @param declarator The only declarator in this declaration.
   */
  public VariableDeclaration(Specifier spec, Declarator declarator)
  {
    object_print_method = class_print_method;

    this.specs = (new ChainedList()).addLink(spec);
    addDeclarator(declarator);
  }

  /**
   * Creates a variable declaration that has multiple specifiers, such
   * as for <var>unsigned int x</var>.
   *
   * @param specs A list of specifiers.
   * @param declarator The only declarator in this declaration.
   */
  public VariableDeclaration(List specs, Declarator declarator)
  {
    object_print_method = class_print_method;

    if (!Tools.verifyHomogeneousList(specs, Specifier.class))
      throw new IllegalArgumentException();
 
    this.specs = (new ChainedList()).addAllLinks(specs);
    addDeclarator(declarator);
  }

  /**
   * Creates a declaration of multiple variables that has only one specifier, such
   * as for <var>int x, y, z</var>.
   *
   * @param spec A single specifier. 
   * @param declarator_list A list of declarators.
   */
  public VariableDeclaration(Specifier spec, List declarator_list)
  {
    object_print_method = class_print_method;

    if (!Tools.verifyHomogeneousList(declarator_list, Declarator.class))
      throw new IllegalArgumentException();

    this.specs = (new ChainedList()).addLink(spec);
		for(Declarator decl : (List<Declarator>)declarator_list)
      addDeclarator(decl);

  }

  /**
   * Creates a declaration of multiple variables that has multiple specifiers, such
   * as for <var>unsigned int x, y, z</var>.
   *
   * @param specs A list of specifiers.
   * @param declarator_list A list of declarators.
   */
  public VariableDeclaration(List specs, List declarator_list)
  {
    object_print_method = class_print_method;

    if (!Tools.verifyHomogeneousList(specs, Specifier.class))
      throw new IllegalArgumentException();

    if (!Tools.verifyHomogeneousList(declarator_list, Declarator.class))
      throw new IllegalArgumentException();

    this.specs = (new ChainedList()).addAllLinks(specs);
		for(Declarator decl : (List<Declarator>)declarator_list)
      addDeclarator(decl);
  }

  /**
   * Adds another declarator to this declaration.  For example,
   * if the declaration is <var>int x;</var> and a declarator for
   * <var>y</var> is added, the declaration becomes <var>int x, y;</var>.
   *
   * @param declarator The declarator to add.
   */
  public void addDeclarator(Declarator declarator)
  {
    children.add(declarator);
    declarator.setParent(this);
  }

  @Override
  public VariableDeclaration clone()
  {
    VariableDeclaration d = (VariableDeclaration)super.clone();
    d.specs = (new ChainedList()).addAllLinks(specs);
    return d;
  }

  /**
   * Prints a declaration to a stream.
   *
   * @param d The declaration to print.
   * @param o The stream on which to print the declaration.
   */
  public static void defaultPrint(VariableDeclaration d, PrintWriter o)
  {
    PrintTools.printListWithSpace(d.specs, o);
    o.print(" ");
    PrintTools.printListWithComma(d.children, o);
  }

  /**
   * Returns the <var>nth</var> declarator in this declaration.
   *
   * @throws IndexOutOfBoundsException if the <var>nth</var> declarator does not exist.
   * @return the <var>nth</var> declarator in this declaration.
   */
  public Declarator getDeclarator(int n)
  {
    return (Declarator)children.get(n);
  }

  public List getDeclaredIDs()
  {
    LinkedList list = new LinkedList();
		for(Traversable d : children)
      list.add(((Declarator)d).getID());
    return list;
  }

  /**
   * Returns the number of declarators in this declaration.
   */
  public int getNumDeclarators()
  {
    return children.size();
  }

  public List<Specifier> getSpecifiers()
  {
    return specs;
  }

  public boolean isTypedef()
  {
    return (Tools.indexByReference(specs, Specifier.TYPEDEF) != -1); 
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
