package omp2gpu.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;
import cetus.hir.*;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 * Represents an dim3 type specifier in CUDA, for example the round-bracketed
 * parts of <var>dim3 dimBlock(256,1,1);</var>
 */
public class Dim3Specifier extends Specifier
{
  /** The unbounded specifier [] */
  public static final Dim3Specifier UNBOUNDED = new Dim3Specifier();

  private List dimensions;

  public Dim3Specifier()
  {
    dimensions = new LinkedList<Expression>();
    dimensions.add(new IntegerLiteral(1));
    dimensions.add(new IntegerLiteral(1));
    dimensions.add(new IntegerLiteral(1));
  }

  public Dim3Specifier(Expression expr)
  {
    dimensions = new LinkedList<Expression>();
    dimensions.add(expr);
    dimensions.add(new IntegerLiteral(1));
    dimensions.add(new IntegerLiteral(1));
  }
  
  public Dim3Specifier(Expression expr1, Expression expr2)
  {
    dimensions = new LinkedList<Expression>();
    dimensions.add(expr1);
    dimensions.add(expr2);
    dimensions.add(new IntegerLiteral(1));  
  }
  
  public Dim3Specifier(Expression expr1, Expression expr2, Expression expr3)
  {
    dimensions = new LinkedList<Expression>();
    dimensions.add(expr1);
    dimensions.add(expr2);
    dimensions.add(expr3);  
  }
  
  public Dim3Specifier(List dimensions)
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

  public void print(PrintWriter p)
  {
    p.print("(");
    PrintTools.printListWithComma(dimensions, p);
    p.print(")");
  }

  //Below is old one
/*	public String toString()
	{
		StringBuilder str = new StringBuilder(80);
		str.append("(" + PrintTools.listToString(dimensions, ", ") + ")");

		return str.toString();
	}*/
	
  /** Returns a string representation of the specifier. */
  @Override
  public String toString()
  {
	  StringWriter sw = new StringWriter(80);
	  print(new PrintWriter(sw));
	  return sw.toString();
  }

  /**
   * Sets the nth dimension of this dim3 specifier.
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
