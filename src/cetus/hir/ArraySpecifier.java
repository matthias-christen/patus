package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/**
 * Represents an array specifier, for example the bracketed
 * parts of <var>int array[20][30];</var>
 */
public class ArraySpecifier extends Specifier
{
  /** The unbounded specifier [] */
  public static final ArraySpecifier UNBOUNDED = new ArraySpecifier();

  private List dimensions;

  public ArraySpecifier()
  {
    dimensions = new LinkedList();
    dimensions.add(null);
  }

  public ArraySpecifier(Expression expr)
  {
    dimensions = new LinkedList();
    dimensions.add(expr);
  }
  
  public ArraySpecifier(List dimensions)
  {
    setDimensions(dimensions);
  }

  /**
   * Gets the nth dimension of this array specifier.
   *
   * @param n The position of the dimension.
   * @throws IndexOutOfBoundsException if there is no expression at that position.
   * @return the nth dimension, which may be null.  A null dimension occurs
   *   for example with int array[][8].
   */
  public Expression getDimension(int n)
  {
    return (Expression)dimensions.get(n);
  }

  /**
   * Returns the number of index expressions used in this array specifier.
   *
   * @return the number of index expressions.
   */
  public int getNumDimensions()
  {
    return dimensions.size();
  }

  public void print(PrintWriter o)
  {
    for (Expression dim : (List<Expression>)dimensions)
    {
      o.print("[");
      if (dim != null)
        dim.print(o);
      o.print("]");
    }
  }

  /**
   * Sets the nth dimension of this array specifier.
   *
   * @param n The position of the dimension.
   * @param expr The expression defining the size of the dimension.
   * @throws IndexOutOfBoundsException if there is no dimension at that position.
   */
  public void setDimension(int n, Expression expr)
  {
    dimensions.set(n, expr);
  }

  /**
   * Set the list of dimension expressions.
   *
   * @param dimensions A list of expressions.
   */
  public void setDimensions(List dimensions)
  {
    if (dimensions == null)
      throw new IllegalArgumentException();

    this.dimensions = new LinkedList();
    Iterator iter = dimensions.iterator();
    while (iter.hasNext())
    {
      Object o = iter.next();

      if (o == null || o instanceof Expression)
        this.dimensions.add(o);
      else
        throw new IllegalArgumentException("all list items must be Expressions or null; found a " + o.getClass().getName() + " instead");
    }
  }
}
