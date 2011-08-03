package cetus.hir;

import java.util.*;

/**
 * Peforms a post-order traversal over a Traversable object. This type
 * of traversal usually requires larger internal storage space than
 * <b>DepthFirstIterator</b> or <b>BreadthFirstIterator</b> does since it
 * needs to store all the visited objects while reaching the first element
 * to be visited which is usually a leaf node.
 */
public class PostOrderIterator extends IRIterator
{
  private LinkedList<Traversable> queue;
  private HashSet<Class> prune_set;

  /**
   * Creates a new iterator with the specified root object and the optional
   * list of pruned types.
   *
   * @param root The root object for the traversal.
   * @param pruned_types The list of object types to be pruned on.
   */
  public PostOrderIterator(Traversable root, Class... pruned_types)
  {
    super(root);
    queue = new LinkedList<Traversable>();
    prune_set = new HashSet<Class>();
    for (Class pruned_type : pruned_types)
      prune_set.add(pruned_type);
    populate(root);
  }

  public boolean hasNext()
  {
    return !queue.isEmpty();
  }
 
  public Object next()
  {
    Traversable t = null;
    return queue.remove(); // will throw NoSuchElementException on failure.
  }

  private void populate(Traversable t)
  {
    if (t.getChildren() != null
        && !containsCompatibleClass(prune_set, t.getClass()))
    {
      for (Traversable obj : t.getChildren())
      {
        if (obj != null)
          populate(obj);
      }
    }

    queue.add(t);
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
  * Resets the iterator by setting the current position to the root object.
  * The pruned types are not cleared.
  */
  public void reset()
  {
    queue.clear();
    populate(root);
  }

  /**
  * Unlike the <b>reset</b> method, <b>clear</b> method also clears the
  * pruned types.
  */
  public void clear()
  {
    queue.clear();
    prune_set.clear();
    populate(root);
  }
}
