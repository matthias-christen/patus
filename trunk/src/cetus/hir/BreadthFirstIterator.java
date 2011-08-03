package cetus.hir;

import java.util.*;

/**
 * Iterates over Traversable objects in breadth-first order.
 */
public class BreadthFirstIterator extends IRIterator
{
  private Vector<Traversable> queue;
  private HashSet<Class> prune_set;

  /**
   * Creates a new iterator with the specified initial object and the
   * optional list of pruned types.
   *
   * @param init The first object to visit.
   */
  public BreadthFirstIterator(Traversable init, Class... pruned_types)
  {
    super(init);
    queue = new Vector<Traversable>();
    queue.add(init);
    prune_set = new HashSet<Class>();
    for (Class pruned_type : pruned_types)
      prune_set.add(pruned_type);
  }

  public boolean hasNext()
  {
    return !queue.isEmpty();
  }

  public Object next()
  {
    Traversable t = null;

    try {
      t = queue.remove(0);
    } catch (ArrayIndexOutOfBoundsException e) {
      throw new NoSuchElementException();
    }

    if (t.getChildren() != null
        && !containsCompatibleClass(prune_set, t.getClass()))
    {
      for(Traversable o : t.getChildren())
        if(o != null)
          queue.add(o);
    }

    return t;
  }

  /**
  * Disables traversal from an object having the specified type. For example,
  * if traversal reaches an object with type <b>c</b>, it does not visit the
  * children of the object.
  *
  * @param c the object type to be pruned on.
  */
  public void pruneOn(Class c)
  {
    prune_set.add(c);
  }

  /**
    * Returns a linked list of objects of Class c in the IR
    *
    * @param c the object type to be collected.
    * @return the collected list.
    */
  public LinkedList getList(Class c)
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
    *
    * @param c the object type to be collected.
    * @return the collected set.
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

  /**
  * Resets the iterator by setting the current position to the root object.
  * The pruned types are not cleared.
  */
  public void reset()
  {
    queue.clear();
    queue.add(root);
  }

  /**
  * Unlike the <b>reset</b> method, <b>clear</b> method also clears the
  * pruned types.
  */
  public void clear()
  {
    queue.clear();
    prune_set.clear();
    queue.add(root);
  }
}
