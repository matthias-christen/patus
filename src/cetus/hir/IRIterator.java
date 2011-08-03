package cetus.hir;

import java.util.*;

/**
 * An iterator implementing this interface supports special versions of next.
 */
public abstract class IRIterator implements Iterator
{
  /** The root traversable object */
  protected Traversable root;

  /** Constructs a base IRIterator */
  protected IRIterator(Traversable root)
  {
    this.root = root;
  }

  /**
   * Checks if s contains a superset of c.
   *
   * @return true if the Set s contains a Class object that is the same
   * as or a superset of Class c, and false otherwise.
   */
  protected boolean containsCompatibleClass(Set<Class> s, Class c)
  {
    for (Class x : s)
    {
      if (x.isAssignableFrom(c))
        return true;
    }

    return false;
  }

  /**
   * Returns true if the iteration has more elements.
   * (In other words, returns true if a call to next would return
   * an element rather than throwing an exception.)
   *
   * @return true if the iterator has more elements.
   */
  public abstract boolean hasNext();

  /**
   * Returns the next element in the iteration.
   *
   * @return the next element in the iteration.
   * @throws NoSuchElementException if there is no element left.
   */
  public abstract Object next();

  /**
   * Returns the next element in the iteration that is an instance of a certain class.
   *
   * @param c The class of which the next element must be a instance.
   * @throws NoSuchElementException if there are no elements of class c.
   * @return the next element that is of class c.
   */
  public Object next(Class c) throws NoSuchElementException
  {
    Object obj = null;

    while (hasNext())
    {
      obj = next();
      if (c.isInstance(obj))
        return obj;
    }

    throw new NoSuchElementException();
  }

  /**
   * Returns the next element in the iteration that is an instance of a class in the set.
   *
   * @param s A set of classes of which the next element must be an instance.
   * @throws NoSuchElementException if there are no elements of classes in s.
   * @return the next element that is an instance of a class in s.
   */
  public Object next(Set s) throws NoSuchElementException
  {
    while (hasNext())
    {
      Object obj = next();

      if (containsCompatibleClass(s, obj.getClass()))
        return obj;
    }

    throw new NoSuchElementException();
  }

  /**
   * Returns the next element in the iteration that is not an instance of a class in the set.
   *
   * @param s A set of classes of which the next element must not be an instance.
   * @throws NoSuchElementException if there are no elements that are not of classes in s.
   * @return the next element that is not an instance of a class in s.
   */
  public Object nextExcept(Set s) throws NoSuchElementException
  {
    while (hasNext())
    {
      Object obj = next();

      if (!containsCompatibleClass(s, obj.getClass()))
        return obj;
    }

    throw new NoSuchElementException();
  }

  /**
   * This operation is not supported.
   */
  public void remove()
  {
    throw new UnsupportedOperationException();
  }

  /**
   * Moves the iterator back to the beginning of the sequence.
   */
  public abstract void reset();
}
