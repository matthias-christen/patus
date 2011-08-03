package cetus.hir;

import java.util.*;

/**
 * Iterates over Traversable objects in depth-first order. The iteration starts
 * from the root object that was specified in the constructor.
 */
public class DepthFirstIterator extends IRIterator
{
  private Vector<Traversable> stack;
  private HashSet<Class> prune_set;

  /**
   * Creates a new iterator with the specified initial traversable object and
   * the optional pruned types.
   *
   * @param init The first object to visit.
   * @param pruned_types The traversable types that are skipped.
   */
  public DepthFirstIterator(Traversable init, Class... pruned_types)
  {
    super(init);
    stack = new Vector<Traversable>();
    stack.add(init);
    prune_set = new HashSet<Class>();
    for (Class pruned_type : pruned_types)
      prune_set.add(pruned_type);
  }

  public boolean hasNext()
  {
    return !stack.isEmpty();
  }

  public Object next()
  {
    Traversable t = null;

    try {
      t = stack.remove(0);
    } catch(ArrayIndexOutOfBoundsException e){ // catching ArrayIndexOutofBoundsException, as remove method throws this exception
      throw new NoSuchElementException();
    }

    if ( !containsCompatibleClass(prune_set, t.getClass()) &&
      t.getChildren() != null )
    {
      int i = 0;
      for(Traversable o : t.getChildren())
        if (o != null)
          stack.add(i++, o);
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
    stack.clear();
    stack.add(root);
  }

  /**
  * Unlike the <b>reset</b> method, <b>clear</b> method also clears the
  * pruned types.
  */
  public void clear()
  {
    stack.clear();
    prune_set.clear();
    stack.add(root);
  }
}
