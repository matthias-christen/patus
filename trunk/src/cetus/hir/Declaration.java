package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/**
* Base class for all declarations. Every declaration object should have a
* <b>SymbolTable</b> object that defines the scope, as an ancestor node in the
* IR tree; {@link TranslationUnit} and {@link CompoundStatement} are the two
* typical symbol table objects. It is highly recommended to use the high-level
* interface methods of {@link SymbolTable}, not the low-level methods of
* {@link Traversable}, for consistent management of symbol table features. 
*/
public abstract class Declaration implements Cloneable, Traversable, Annotatable
{
  /** The print method of the declaration */
  protected Method object_print_method;

  /** The parent object of the declaration */
  protected Traversable parent;

  /** The list of the child objects */
  protected LinkedList<Traversable> children;

  /** The list of the annotations attached to the declaration */
  protected List<Annotation> annotations;

  /** This declaration needs semi colon when printed if this field is true */
  protected boolean needs_semi_colon;

  /**
   * Base constructor for derived classes.
   */
  protected Declaration()
  {
    parent = null;
    children = new LinkedList<Traversable>();
    annotations = null;
    needs_semi_colon = false;
  }

  /**
   * Base constructor for derived classes; sets an initial size for the
   * list of children of this declaration.
   *
   * @param size The initial size for the child list.
   */
  protected Declaration(int size)
  {
    this();
  }

  /**
   * Creates and returns a deep copy of this declaration.
   *
   * @return a deep copy of this declaration.
   */
  @Override
  public Declaration clone()
  {
    Declaration o = null;

    try {
      o = (Declaration)super.clone();
    } catch (CloneNotSupportedException e) {
      throw new InternalError();
    }

    o.object_print_method = object_print_method;
    o.parent = null;

    if (children != null)
    {
      o.children = new LinkedList<Traversable>();
      for(Traversable t : children)
      {
        if(t instanceof Declarator){
          Declarator new_child = ((Declarator)t).clone();
          o.children.add(new_child);
          new_child.setParent(o);
        }
        // following is required when Procedure's clone() is called and it in turn calls Declaration.clone() 
        else if(t instanceof CompoundStatement){
          CompoundStatement stmt = ((CompoundStatement)t).clone();
          o.children.add(stmt);
          stmt.setParent(o);
        }
        // following is required when ClassDeclaration's clone() is called and it in turn calls Declaration.clone()
        else if(t instanceof DeclarationStatement){
          DeclarationStatement stmt = ((DeclarationStatement)t).clone();
          o.children.add(stmt);
          stmt.setParent(o);
        }
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
  * Checks if the declaration is equal to the given object.
  *
  * @param o the object to be compared.
  * @return true if {@code o == this}.
  */
  @Override
  public boolean equals(Object o)
  {
    return (o == this);
  }

  /**
  * Returns the hash code of the declaration. It returns the identity hash code
  * of the declaration.
  *
  * @return the identity hash code of the declaration.
  */
  @Override
  public int hashCode()
  {
    return System.identityHashCode(this);
  }

  /**
   * Detaches this declaration from it's parent, if it has one.
   */
  public void detach()
  {
    if (parent != null)
    {
      parent.removeChild(this);
      setParent(null);
    }
  }

  /* Traversable interface */
  public List<Traversable> getChildren()
  {
    return children;
  }

  /**
  * Returns a list of name ID introduced by this declaration.
  *
  * @return a list of IDExpressions. The list will not
  *   be null but may be empty.
  */
  public abstract List getDeclaredIDs();

  /* Traversable interface */
  public Traversable getParent()
  {
    return parent;
  }

  /**
  * Prints this declaration to the specified print writer.
  * @param o the target print writer.
  */
  public void print(PrintWriter o)
  {
    if (object_print_method==null)
      return;
    try {
      List<Annotation> notes = null;
      if (!(notes=getAnnotations(Annotation.BEFORE)).isEmpty())
        o.print(PrintTools.listToString(notes, "\n"));
      if (this instanceof AnnotationDeclaration)
        return; // nothing to print.
      if (!notes.isEmpty())
        o.println("");
      object_print_method.invoke(null, new Object[] {this, o});
      if (needs_semi_colon) 
        o.print(";");
      if ( !(notes=getAnnotations(Annotation.WITH)).isEmpty() )
        o.print(" "+PrintTools.listToString(notes, "\n"));
      if ( !(notes=getAnnotations(Annotation.AFTER)).isEmpty() ) {
        o.println("");
        o.println(PrintTools.listToString(notes, "\n"));
      }
    } catch (IllegalAccessException e) {
      throw new InternalError(e.getMessage());
    } catch (InvocationTargetException e) {
      throw new InternalError(e.getMessage());
    }
  }

  /**
  * Converts this declaration to a string by calling the default print method.
  * All sub classes will be using this method unless special handling is
  * necessary.
  */
  @Override
  public String toString()
  {
    StringWriter sw = new StringWriter(80);
    print(new PrintWriter(sw));
    return sw.toString();
  }

  /**
  * This operation is not allowed.
  *
  * @throws UnsupportedOperationException always.
  */
  public void removeChild(Traversable child)
  {
    throw new UnsupportedOperationException("Declarations do not support removal of arbitrary children.");
  }

  /* Traversable interface */
  public void setChild(int index, Traversable t)
  {
    /* t must not be a child of something else */
    if (t.getParent() != null)
      throw new NotAnOrphanException();

    /* only certain types of objects can be children
       of declarations, so check for them */
    if (t instanceof Declarator
        || t instanceof Declaration
        || t instanceof Statement)
    {
      /* detach the old child at position index */
      children.get(index).setParent(null);

      /* set the new child */
      children.set(index, t);
      t.setParent(this);
    }
    else
      throw new IllegalArgumentException();
  }

  /* Traversable interface */
  public void setParent(Traversable t)
  {
    if (t == null)
      parent = null;
    else
    {
      /* parent must already have accepted this object as a child */
      // FIXME - Temporarily disabled until new Declarators are in use
      //    if (Tools.indexByReference(t.getChildren(), this) < 0)
      //      throw new NotAChildException();

      /* only certain types of objects can be the parent of
         a declaration, so check for them */
      if (t instanceof Declaration
          || t instanceof Declarator // parameter declarations in declarators
          || t instanceof Statement
          || t instanceof TemplateID
          || t instanceof TranslationUnit)
        parent = t;
      else
        throw new IllegalArgumentException();
    }
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

      for(Traversable t : children)
        if(t.getParent() != this)
          throw new IllegalStateException("a child does not think this is the parent");

    }
  }

  /**
  * Inserts the given annotation to this declaration. The default position of
  * the annotation is before the declaration.
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
  * Inserts the given annotation after this declaration.
  * @param annotation the annotation to be inserted.
  */
  public void annotateAfter(Annotation annotation)
  {
    annotation.setPosition(Annotation.AFTER);
    annotate(annotation);
  }

  /**
  * Inserts the given annotation before this declaration.
  * @param annotation the annotation to be inserted.
  */
  public void annotateBefore(Annotation annotation)
  {
    annotation.setPosition(Annotation.BEFORE);
    annotate(annotation);
  }

  /**
  * Returns the list of annotations attached to this declaration. It returns the
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
  * declaration.
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
  * Checks if this declaration contains any annotation with the specified type
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
    List<T> annotations = getAnnotations(type);
    for ( T annotation : annotations )
      if ( annotation.containsKey(key) )
        return annotation;
    return null;
  }

  /**
  * Returns the list of annotations attached at the specified position of this
  * declaration.
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

  /** Removes all annotations attached to this declaration. */
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

  /** Marks this declaration needs semi colon when printed. */
  public void setSemiColon(boolean needs_semi_colon)
  {
    this.needs_semi_colon = needs_semi_colon;
  }

}
