package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/**
 * Base class for all statements. 
 * Statement is the base class of numerous specific statement classes.
 */
public abstract class Statement implements Cloneable, Traversable, Annotatable
{
  /** The print method for the statement */
  protected Method object_print_method;

  /** The parent traversable object */
  protected Traversable parent;

  /** The list of children of the statement */
  protected LinkedList<Traversable> children;

  /** The position of the statement */
  protected int line_number = -1;

  /** The list of annotations attached to the statement */
  protected List<Annotation> annotations;

  /** Constructor for derived classes. */
  protected Statement()
  {
    parent = null;
    children = new LinkedList<Traversable>();
    annotations = null;
  }

  /**
   * Constructor for derived classes that preallocates
   * space for multiple children.
   *
   * @param size The expected number of children for this statement.
   */
  protected Statement(int size)
  {
    parent = null;
    children = new LinkedList<Traversable>();
    annotations = null;
  }

  /** Returns a clone of the statement */
  @Override
  public Statement clone()
  {
    Statement o = null;

    try {
      o = (Statement)super.clone();
    } catch (CloneNotSupportedException e) {
      throw new InternalError();
    }

    o.object_print_method = object_print_method;
    o.parent = null;

    if (children != null)
    {
      o.children = new LinkedList<Traversable>();
      Iterator iter = children.iterator();
      while (iter.hasNext())
      {
        Traversable new_child = null;
        Object tmp = iter.next();

        if (tmp instanceof Statement)
          new_child = ((Statement)tmp).clone();
        else if (tmp instanceof Expression)
          new_child = ((Expression)tmp).clone();
        else if (tmp instanceof Declaration)
          new_child = ((Declaration)tmp).clone();
        else
        {
          System.err.println(tmp.getClass().toString());
        }

        new_child.setParent(o);
        o.children.add(new_child);
      }
    }
    else
      o.children = null;

    // Clone annotations after removing shallow copies.
    o.annotations = null;
    List<Annotation> notes = getAnnotations();
    if ( notes != null )
      for ( Annotation note : notes )
        o.annotate( note.clone() );

    return o;
  }

  /**
  * Compares the statement with the specified object for equality.
  *
  * @param o the object to be compared.
  * @return true if {@code (o == this)}, false otherwise.
  */
  @Override
  public boolean equals(Object o)
  {
    return (o == this);
  }

  /**
  * Returns the hash code of the statement. It returns the identity hash code
  * of the current statement object.
  *
  * @return the identity hash code of the statement.
  */
  @Override
  public int hashCode()
  {
    return System.identityHashCode(this);
  }

  /**
   * Detaches this statement from it's parent, if it has one.
   */
  public void detach()
  {
    if (parent != null)
    {
      parent.removeChild(this);
      setParent(null);
    }    
  }

  public List<Traversable> getChildren()
  {
    return children;
  }

  public Traversable getParent()
  {
    return parent;
  }

  /**
   * Returns the procedure in which this statement is located.
   *
   * @return the procedure in which this statement is located,
   *   or null if it is not in a procedure.
   */
  public Procedure getProcedure()
  {
    Traversable p = getParent();

    while (p != null)
    {
      if (p instanceof Procedure)
        return (Procedure)p;
      else
        p = p.getParent();
    } 

    return null;
  }

  /**
  * Prints the statement on the specified print writer.
  *
  * @param o the target print writer.
  */
  public void print(PrintWriter o)
  {
    if (object_print_method == null)
      return;
    try {
      List<Annotation> notes = null;
      if (!(notes=getAnnotations(Annotation.BEFORE)).isEmpty())
        o.print(PrintTools.listToStringWithSkip(notes, "\n"));
      if (this instanceof AnnotationStatement)
        return; // nothing to print.
      if (!notes.isEmpty())
        o.println("");
      object_print_method.invoke(null, new Object[] {this, o});
      if ( !(notes=getAnnotations(Annotation.WITH)).isEmpty() )
        o.print(" "+PrintTools.listToStringWithSkip(notes, "\n"));
      if ( !(notes=getAnnotations(Annotation.AFTER)).isEmpty() ) {
        o.println("");
        o.println(PrintTools.listToStringWithSkip(notes, "\n"));
      }
    } catch (IllegalAccessException e) {
      throw new InternalError(e.getMessage());
    } catch (InvocationTargetException e) {
      throw new InternalError(e.getMessage());
    }
  }

  /**
   * Removes a specific child of this statement;
   * some statements do not support this method.
   *
   * @param child The child to remove.
   */
  public void removeChild(Traversable child)
  {
    throw new UnsupportedOperationException("This statement does not support removal of arbitrary children.");
  }

  public void setChild(int index, Traversable t)
  {
    if (t == null || index < 0 || index >= children.size())
      throw new IllegalArgumentException();
    if (t.getParent() != null)
      throw new NotAnOrphanException();
    /* Detach the old child */
    if (children.get(index) != null)
      children.get(index).setParent(null);
    children.set(index, t);
    t.setParent(this);
  }

  public void setParent(Traversable t)
  {
    parent = t;
  }

  /**
  * Inserts the specified traversable object at the end of the child list.
  *
  * @param t the traversable object to be inserted.
  * @throws IllegalArgumentException if <b>t</b> is null.
  * @throws NotAnOrphanException if <b>t</b> has a parent.
  */
  protected void addChild(Traversable t)
  {
    if (t == null)
      throw new IllegalArgumentException("invalid child inserted.");
    if (t.getParent() != null)
      throw new NotAnOrphanException(this.getClass().getName());
    children.add(t);
    t.setParent(this);
  }

  /**
  * Inserts the specified traversable object at the specified position.
  *
  * @param t the traversable object to be inserted.
  * @throws IllegalArgumentException if <b>t</b> is null or index is
  * out-of-bound.
  * @throws NotAnOrphanException if <b>t</b> has a parent.
  */
  protected void addChild(int index, Traversable t)
  {
    if (t == null || index < 0 || index > children.size())
      throw new IllegalArgumentException("invalid child inserted.");
    if (t.getParent() != null)
      throw new NotAnOrphanException(this.getClass().getName());
    children.add(index, t);
    t.setParent(this);
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

  /**
   * Swaps two statements on the IR tree.  If neither
   * this statement nor <var>stmt</var> has a parent,
   * then this function has no effect.  Otherwise,
   * each statement ends up with the other's parent.
   *
   * @param stmt The statement with which to swap this statement.
   * @throws IllegalArgumentException if <var>stmt</var> is null.
   * @throws IllegalStateException if the types of the statements
   *   are such that they would create inconsistent IR when swapped.
   */
  public void swapWith(Statement stmt)
  {
    if (stmt == null)
      throw new IllegalArgumentException();

    if (this == stmt)
      /* swap with self does nothing */
      return;

    /* The rest of this must be done in a very particular order.
       Be EXTREMELY careful changing it. */

    Traversable this_parent = this.parent;
    Traversable stmt_parent = stmt.parent;

    int this_index = -1, stmt_index = -1;

    if (this_parent != null)
    {
      this_index = Tools.indexByReference(this_parent.getChildren(), this);
      if (this_index == -1)
        throw new IllegalStateException();
    }

    if (stmt_parent != null)
    {
      stmt_index = Tools.indexByReference(stmt_parent.getChildren(), stmt);
      if (stmt_index == -1)
        throw new IllegalStateException();
    }

    /* detach both so setChild won't complain */
    stmt.parent = null; this.parent = null;

    if (this_parent != null) {
      this_parent.getChildren().set(this_index, stmt);
      stmt.setParent(this_parent);
    }

    if (stmt_parent != null) {
      stmt_parent.getChildren().set(stmt_index, this);
      this.setParent(stmt_parent);
    }
  }

  /**
   * Sets the line number of this statement
   * This function is to be used only for parser development
   *
   * @param line The line number 
   */
  public void setLineNumber(int line)
  {
    line_number = line;
  }

  /** Returns a string representation of the statement */
  @Override
  public String toString()
  {
    StringWriter sw = new StringWriter(80);
    print(new PrintWriter(sw));
    return sw.toString();
  }

  /**
   * Returns the line number of this statement if the
   * statement was present in the original source file.
   * Line numbers are not available for statements that
   * are added by compiler passes.
   *
   * @return the line number of this statement.
   */
  public int where()
  {
    /* TODO - add code for returning the line number and
       (probably) throwing an exception if the line number
       is unknown */
    // line_number # is -1 if not defined
    return line_number;
  }

  /**
   * Verifies three properties of this object:
   * (1) All children are not null, (2) the parent object has this
   * object as a child, (3) all children have this object as the parent.
   *
   * @throws IllegalStateException if any of the properties are not true.
   */
  public void verify() throws IllegalStateException
  {
    if (parent != null && !parent.getChildren().contains(this))
      throw new IllegalStateException("parent does not think this is a child");

    if (children != null)
    {
      if (children.contains(null))
        throw new IllegalStateException("a child is null");

      for(Traversable child : children)
      {
        if (child.getParent() != this)
          throw new IllegalStateException("a child does not think this is the parent");

        if (child instanceof Statement)
          ((Statement)child).verify();
      }
    }
  }

  /**
  * Inserts the given annotation to this statement. The default position of the
  * annotation is before the statement.
  * @param annotation the annotation to be inserted.
  */
  public void annotate(Annotation annotation)
  {
    annotation.attach(this);
    if ( annotations == null )
      annotations = new LinkedList<Annotation>();
    annotations.add(annotation);
  }

  /**
  * Inserts the given annotation after this statement.
  * @param annotation the annotation to be inserted.
  */
  public void annotateAfter(Annotation annotation)
  {
    annotation.setPosition(Annotation.AFTER);
    annotate(annotation);
  }

  /**
  * Inserts the given annotation before this statement.
  * @param annotation the annotation to be inserted.
  */
  public void annotateBefore(Annotation annotation)
  {
    annotation.setPosition(Annotation.BEFORE);
    annotate(annotation);
  }

  /**
  * Returns the list of annotations attached to this statement. It returns the
  * direct handle of the annotation list, so any modifications to the returned
  * list should be carefully done.
  * @return the list of attached annotations (null if none exists).
  */
  public List<Annotation> getAnnotations()
  {
    return annotations;
  }

  /**
  * Returns the list of annotations with the specified type, attached to this
  * statement.
  * @param type the annotation type of intereset.
  * @return the list of annotations with the specified type.
  */
  @SuppressWarnings("unchecked")
  public <T extends Annotation> List<T> getAnnotations(Class<T> type)
  {
    List<T> ret = new LinkedList<T>();
    if ( annotations == null )
      return ret;
    for ( Annotation annotation : annotations )
      if ( type.isInstance(annotation) )
        ret.add((T)annotation);
    return ret;
  }

  /**
  * Checks if this statement contains any annotation with the specified type
  * and the string key.
  * @param type the annotation type of interest.
  * @param key the key to be searched for.
  * @return the search result.
  */
  public boolean
      containsAnnotation(Class<? extends Annotation> type, String key)
  {
    return (getAnnotation(type, key) != null);
  }

  /**
  * Returns the first occurrence of the annotation with the specified type
  * and the string key.
  * @param type the annotation type of interest.
  * @param key the key to be searched for.
  * @return the annotation if one exist, null otherwise.
  */
  public <T extends Annotation> T getAnnotation(Class<T> type, String key)
  {
    for ( T annotation : getAnnotations(type) )
      if ( annotation.containsKey(key) )
        return annotation;
    return null;
  }

  /**
  * Returns the list of annotations attached at the specified position of this
  * statement.
  * @param position the annotation position to be searched.
  * @return the list of annotations with the specified position.
  */
  public List<Annotation> getAnnotations(int position)
  {
    List<Annotation> ret = new LinkedList<Annotation>();
    if ( annotations == null )
      return ret;
    for ( Annotation annotation : annotations )
      if ( annotation.position == position )
        ret.add(annotation);
    return ret;
  }

  /** Removes all annotations attached to this statement. */
  public void removeAnnotations()
  {
    annotations = null;
  }

  /**
  * Removes all annotations with the specified type.
  * @param type the annotation type to be removed.
  */
  public void removeAnnotations(Class<?> type)
  {
    if ( annotations == null )
      return;
    Iterator<Annotation> iter = annotations.iterator();
    while ( iter.hasNext() )
      if ( type.isInstance(iter.next()) )
        iter.remove();
  }

}
