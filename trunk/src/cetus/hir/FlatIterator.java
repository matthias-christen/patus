package cetus.hir;

import java.util.*;

/**
 * Iterates over the immediate children of a Traversable object.
 */
public class FlatIterator extends IRIterator
{
  private ListIterator<Traversable> iter;

  /**
   * Creates a new iterator with the specified traversable object.
   *
   * @param parent The parent of the children to visit.
   */
  public FlatIterator(Traversable parent)
  {
    super(parent);
    iter = parent.getChildren().listIterator();
  }

  /**
   * Adds an object after the last object that was returned
   * by next or previous.
   *
   * @param t The object to add.
   */
  public void add(Traversable t)
  {
    iter.add(t);
    t.setParent(root);
    root.setChild(iter.nextIndex() - 1, t);
  }

  public boolean hasNext()
  {
    return iter.hasNext();
  }

  /** Checks if the iterator has previous element */
  public boolean hasPrevious()
  {
    return iter.hasPrevious();
  }
 
  public Object next()
  {
    return iter.next();
  }

  /** Returns the previous element of the current position */
  public Object previous()
  {
    return iter.previous();
  }

  /**
   * Removes the last object that was returned by next or previous.
   */
  public void remove()
  {
    root.getChildren().get(iter.nextIndex() - 1).setParent(null);
    iter.remove();
  }

  public void reset()
  {
    iter = root.getChildren().listIterator();
  }

  /**
    * Returns a linked list of objects of Class c in the IR
    */
  public List getList(Class c)
  {
    LinkedList list = new LinkedList();

    while (hasNext())
    {
      Object obj = next();
      if (c.isInstance(obj))
      {
        list.add(obj);
      }
    }
    return list;
  }

  /**
    * Returns a set of objects of Class c in the IR
    */
  public Set getSet(Class c)
  {
    HashSet set = new HashSet();

    while (hasNext())
    {
      Object obj = next();

      if (c.isInstance(obj))
      {
        set.add(obj);
      }
    }
    return set;
  }
 
}
