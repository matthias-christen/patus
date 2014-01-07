package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/** Represents a declarator for a Procedure in a VariableDeclaration. */
public class ProcedureDeclarator extends Declarator implements Symbol
{
  /** Default method for printing procedure declarator */
  private static Method class_print_method;

  /** Default print method assignment */
  static
  {
    Class<?>[] params = new Class<?>[2];

    try {
      params[0] = ProcedureDeclarator.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  /** Not used in C */
  private ExceptionSpecification espec;

  /** Common initialization process with the given input */
  private void initialize(IDExpression direct_decl, List params)
  {
    object_print_method = class_print_method;
    trailing_specs = new LinkedList();

    if (direct_decl.getParent() != null)
      throw new NotAnOrphanException();
    // Forces use of NameID.
    if (!(direct_decl instanceof NameID))
      direct_decl = new NameID(direct_decl.toString());

    children.add(direct_decl);
    direct_decl.setParent(this);

    for(Declaration decl : (List<Declaration>)params)
    {
      if (decl.getParent() != null)
        throw new NotAnOrphanException();
      children.add(decl);
      decl.setParent(this);
    }
  }

  /** 
  * Constructs a new procedure declarator with the given ID and the list of
  * parameters.
  * 
  * @param direct_decl the IDExpression used for this procedure name; it is
  * highly recommended to use {@link NameID} since this constructor internally
  * replaces the parameter with an equivalent <b>NameID</b> object.
  * @param params the list of function parameters.
  */
  public ProcedureDeclarator(IDExpression direct_decl, List params)
  {
    super(1 + params.size());
    initialize(direct_decl, params);
  }

  /**
  * Consturcts a new procedure declarator with the given ID, list of parameters,
  * and the trailing specifiers.
  *
  * @param direct_decl the IDExpression used for this procedure name; it is
  * highly recommended to use {@link NameID} since this constructor internally
  * replaces the parameter with an equivalent <b>NameID</b> object.
  * @param params the list of function parameters.
  * @param trailing_specs the list of trailing specifiers.
  */
  public ProcedureDeclarator(IDExpression direct_decl, List params, List trailing_specs)
  {
    super(1 + params.size());
    initialize(direct_decl, params);
    this.trailing_specs.addAll(trailing_specs); 
  }

  /**
  * Constructs a new procedure declarator with the given leading specifiers,
  * the ID, and the list of parameters. This is the most commonly used
  * constructor for C input language.
  *
  * @param leading_specs the list of leading specifiers.
  * @param direct_decl the IDExpression used for this procedure name; it is
  * highly recommended to use {@link NameID} since this constructor internally
  * replaces the parameter with an equivalent <b>NameID</b> object.
  * @param params the list of function parameters.
  */
  public ProcedureDeclarator(List leading_specs, IDExpression direct_decl, List params)
  {
    super(1 + params.size());
    initialize(direct_decl, params);
    this.leading_specs.addAll(leading_specs);
  }

  /**
  * Constructs a new procedure declarator with the given leading specifiers,
  * the ID, the trailing specifiers, and the exception specification. This
  * constructor is not used for C programs.
  *
  * @param leading_specs the list of leading specifiers.
  * @param direct_decl the IDExpression used for this procedure name; it is
  * highly recommended to use {@link NameID} since this constructor internally
  * replaces the parameter with an equivalent <b>NameID</b> object.
  * @param params the list of function parameters.
  * @param trailing_specs the list of trailing specifiers.
  * @param espec the exception specification.
  */
  public ProcedureDeclarator(List leading_specs, IDExpression direct_decl,
    List params, List trailing_specs, ExceptionSpecification espec)
  {
    super(1 + params.size());
    initialize(direct_decl, params);
    this.leading_specs.addAll(leading_specs);
    this.trailing_specs.addAll(trailing_specs);
    this.espec = espec;
  }

  /**
  * Inserts a new parameter declaration at the end of the parameter list.
  * 
  * @param decl the new parameter declaration to be added.
  */
  public void addParameter(Declaration decl)
  {
    if (decl.getParent() != null)
      throw new NotAnOrphanException();

    children.add(decl);
    decl.setParent(this);
  }

  /**
  * Inserts a new parameter declaration before the specified reference
  * parameter declaration.
  *
  * @param ref the reference parameter declaration.
  * @param decl the new parameter declaration to be added.
  */
  public void addParameterBefore(Declaration ref, Declaration decl)
  {
    int index = Tools.indexByReference(children, ref);

    if (index == -1)
      throw new IllegalArgumentException();

    if (decl.getParent() != null)
      throw new NotAnOrphanException();

    children.add(index, decl);
    decl.setParent(this);
  }

  /**
  * Inserts a new parameter declaration after the specified reference parameter
  * declaration.
  *
  * @param ref the reference parameter declaration.
  * @param decl the new parameter declaration to be added.
  */
  public void addParameterAfter(Declaration ref, Declaration decl)
  {
    int index = Tools.indexByReference(children, ref);

    if (index == -1)
      throw new IllegalArgumentException();

    if (decl.getParent() != null)
      throw new NotAnOrphanException();

    children.add(index + 1, decl);
    decl.setParent(this);
  }

  /** Returns a clone of the procedure declarator. */
  @Override
  public ProcedureDeclarator clone()
  {
    ProcedureDeclarator d = (ProcedureDeclarator)super.clone();
    IDExpression id  = getDirectDeclarator().clone();
    d.children.add(id);  
    id.setParent(d);
    if (children.size() > 1)
    {
      List tmp = (new ChainedList()).addAllLinks(children);
      tmp.remove(0);
      Iterator iter = tmp.iterator();
      while (iter.hasNext())
      {
         Declaration decl = (Declaration)iter.next();

         decl = decl.clone();
         d.children.add(decl);
         decl.setParent(d);
     }
      
    }
    d.leading_specs = (new ChainedList()).addAllLinks(leading_specs);
    d.trailing_specs = (new ChainedList()).addAllLinks(trailing_specs);
    d.espec = espec;

    return d;
  }

  /**
   * Prints a procedure declarator to a stream.
   *
   * @param d The declarator to print.
   * @param o The writer on which to print the declarator.
   */
  public static void defaultPrint(ProcedureDeclarator d, PrintWriter o)
  {
    PrintTools.printList(d.leading_specs, o);
    d.getDirectDeclarator().print(o);
    o.print("(");
    if (d.children.size() > 1)
      PrintTools.printListWithComma(d.children.subList(1, d.children.size()), o);
    o.print(")");
    PrintTools.printListWithSeparator(d.trailing_specs, o, " ");
  }

  /** Returns the name ID of the procedure declarator. */
  protected IDExpression getDirectDeclarator()
  {
    return (IDExpression)children.get(0);
  }

  /** Returns the name ID of the procedure declarator. */
  public IDExpression getID()
  {
    return getDirectDeclarator();
  }

  /**
  * Returns the list of parameter declaration of the procedure declarator.
  */
  public List<Declaration> getParameters()
  {
    List tmp = (new ChainedList()).addAllLinks(children);
    tmp.remove(0);
    return tmp;
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

  /* Symbol interface */
  public String getSymbolName()
  {
    return getDirectDeclarator().toString();
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
    else
      return null;
    ret.addAll(leading_specs);

    return ret;
  }

  /* Symbol interface */
  public List getArraySpecifiers()
  {
    return null;
  }

  /** Sets the direct declarator with the specified new ID */
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

  /**
  * Returns the parent declaration of the procedure declarator if one exists.
  * The procedure declarators that are not included in the IR tree
  * (e.g., field of a procedure object) does not have any parent declaration
  * since search is not posssible (hence null is returned), whereas a procedure
  * declarator that appears as a child of a variable declaration has a specific
  * parent declaration.
  *
  * @return the parent declaration if one exists, null otherwise. 
  */
  public Declaration getDeclaration()
  {
    return IRTools.getAncestorOfType(this, Declaration.class);
  }
}
