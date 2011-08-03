package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/** Represents a declarator for a variable in a VariableDeclaration. */
public class VariableDeclarator extends Declarator implements Symbol
{
  private static Method class_print_method;

  static
  {
    Class<?>[] params = new Class<?>[2];

    try {
      params[0] = VariableDeclarator.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  /** Common initialization process for constructors. */
  private void initialize(IDExpression direct_decl)
  {
    object_print_method = class_print_method;
    trailing_specs = new LinkedList();

    if (direct_decl.getParent() != null)
      throw new NotAnOrphanException();

    // Forces use of NameID instead of Identifier.
    if (!(direct_decl instanceof NameID))
      direct_decl = new NameID(direct_decl.toString());
    children.add(direct_decl);
    direct_decl.setParent(this);
  }

  /**
  * Constructs a new VariableDeclarator with the given ID.
  * It is highly recommended to use a {@link NameID} object for
  * <b>direct_decl</b> since the constructor internally assigns a new
  * <b>NameID</b> if <b>direct_decl</b> is not an instanceof <b>NameID</b>.
  *
  * @param direct_decl the given name ID for the new variable declarator.
  */
  public VariableDeclarator(IDExpression direct_decl)
  {
    initialize(direct_decl);
  }

  /**
  * Constructs a new variable declarator with the given ID and the trailing
  * specifiers.
  * It is highly recommended to use a {@link NameID} object for
  * <b>direct_decl</b> since the constructor internally assigns a new
  * <b>NameID</b> if <b>direct_decl</b> is not an instanceof <b>NameID</b>.
  *
  * @param direct_decl the given name ID.
  * @param trailing_specs the list of trailing specifiers.
  */
  public VariableDeclarator(IDExpression direct_decl, List trailing_specs)
  {
    initialize(direct_decl);
    this.trailing_specs.addAll(trailing_specs);
  }

  /**
  * Constructs a new variable declarator with the given name ID and the
  * trailing specifier.
  * It is highly recommended to use a {@link NameID} object for
  * <b>direct_decl</b> since the constructor internally assigns a new
  * <b>NameID</b> if <b>direct_decl</b> is not an instanceof <b>NameID</b>.
  *
  * @param direct_decl the given name ID.
  * @param spec the given trailing specifier.
  */
  public VariableDeclarator(IDExpression direct_decl, Specifier spec)
  {
    initialize(direct_decl);
    this.trailing_specs.add(spec);
  }

  /**
  * Constructs a new variable declarator with the given leading specifiers and
  * the name ID.
  * It is highly recommended to use a {@link NameID} object for
  * <b>direct_decl</b> since the constructor internally assigns a new
  * <b>NameID</b> if <b>direct_decl</b> is not an instanceof <b>NameID</b>.
  *
  * @param leading_specs the list of leading specifiers.
  * @param direct_decl the given name ID.
  */
  public VariableDeclarator(List leading_specs, IDExpression direct_decl)
  {
    initialize(direct_decl);
    this.leading_specs.addAll(leading_specs);
  }

  /**
  * Constructs a new variable declarator with the given leading specifiers,
  * the name ID, and the trailing specifiers.
  * It is highly recommended to use a {@link NameID} object for
  * <b>direct_decl</b> since the constructor internally assigns a new
  * <b>NameID</b> if <b>direct_decl</b> is not an instanceof <b>NameID</b>.
  *
  * @param leading_specs the list of leading specifiers.
  * @param direct_decl the given name ID.
  * @param trailing_specs the list of trailing specifiers.
  */
  public VariableDeclarator(List leading_specs, IDExpression direct_decl,
    List trailing_specs)
  {
    initialize(direct_decl);
    this.leading_specs.addAll(leading_specs);
    this.trailing_specs.addAll(trailing_specs);
  }

  /**
  * Constructs a new variable declarator with the given leading specifier and
  * the name ID.
  * It is highly recommended to use a {@link NameID} object for
  * <b>direct_decl</b> since the constructor internally assigns a new
  * <b>NameID</b> if <b>direct_decl</b> is not an instanceof <b>NameID</b>.
  *
  * @param spec the given leading specifier.
  * @param direct_decl the given name ID.
  */
  public VariableDeclarator(Specifier spec, IDExpression direct_decl)
  {
    initialize(direct_decl);
    this.leading_specs.add(spec);
  }

  /**
  * Returns a clone of this variable declarator.
  */
  @Override
  public VariableDeclarator clone()
  {
    VariableDeclarator d = (VariableDeclarator)super.clone();
    if (children.size() > 0)
    {
      IDExpression id  = getDirectDeclarator().clone();
      d.children.add(id);  
      id.setParent(d);
      if(getInitializer() != null){
      
        Initializer init = getInitializer().clone();
        d.setInitializer(init);
      }
    }
    d.leading_specs = (new ChainedList()).addAllLinks(leading_specs);
    d.trailing_specs = (new ChainedList()).addAllLinks(trailing_specs);

    return d;
  }

  /**
  * Prints a variable declarator to a stream.
  *
  * @param d The declarator to print.
  * @param o The writer on which to print the declarator.
  */
  public static void defaultPrint(VariableDeclarator d, PrintWriter o)
  {
    if (!d.leading_specs.isEmpty()) {
      PrintTools.printListWithSpace(d.leading_specs, o);
      //o.print(" ");
    }
    d.getDirectDeclarator().print(o);
    if (!d.trailing_specs.isEmpty()) {
      PrintTools.printListWithSpace(d.trailing_specs, o);
    }
    if (d.getInitializer() != null)
      d.getInitializer().print(o);
  }

  /**
  * Returns the name ID of this variable declarator.
  */
  protected IDExpression getDirectDeclarator()
  {
    return (IDExpression)children.get(0);
  }

  /**
  * Returns the name ID of this variable declarator.
  */
  public IDExpression getID()
  {
    return getDirectDeclarator();
  }

  /**
  * Returns the list of leading specifiers of this variable declarator.
  */
  public List getSpecifiers()
  {
    return leading_specs;
  }

  /**
  * Returns the list of trailing specifiers of this variable declarator.
  */
  public List getTrailingSpecifiers()
  {
    return trailing_specs;
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
  
  /**
  * Returns the initializer of this variable declarator.
  *
  * @return the initializer if it exists, null otherwise.
  */
  public Initializer getInitializer()
  {
    if (children.size() > 1)
      return (Initializer)children.get(1);
    else
      return null;
  }
  
  /**
  * Assigns a new initializer <b>init</b> for the variable declarator.
  * The existing initializer is discarded if exists.
  */
  public void setInitializer(Initializer init)
  {
    if (getInitializer() != null)
    {
      getInitializer().setParent(null);
      
      if (init != null)
      {
        children.set(1, init);
        init.setParent(this);
      }
      else {
        children.remove(1);
      }
    }
    else
    {
      if (init != null)
      {
        children.add(init);
        init.setParent(this);
      }
    }
  }

  /* Symbol interface */
  public String getSymbolName()
  {
    return getID().toString();
  }

  /* Symbol interface */
  public List getTypeSpecifiers()
  {
    Traversable t = this;
    while ( !(t instanceof Declaration) )
      t = t.getParent();
    List ret = new LinkedList();
    if ( t instanceof VariableDeclaration )
      ret.addAll(((VariableDeclaration)t).getSpecifiers());
    else if ( t instanceof Enumeration )
      ret.add(((Enumeration)t).getSpecifier());
    else
      return null;
    ret.addAll(leading_specs);

    return ret;
  }

  /* Symbol interface */
  public List getArraySpecifiers()
  {
    return trailing_specs;
  }

  /** Sets the direct declarator of the variable declarator */
  @Override
  protected void setDirectDeclarator(IDExpression id)
  {
    if (id.getParent() != null)
      throw new NotAnOrphanException();
    children.get(0).setParent(null);
    children.set(0, id);
    id.setParent(this);
  }

  /* Symbol interface */
  public void setName(String name)
  {
    this.setDirectDeclarator(new NameID(name));  
  }

  /* Symbol interface */
  public Declaration getDeclaration()
  {
    return IRTools.getAncestorOfType(this, Declaration.class);
  }
}
