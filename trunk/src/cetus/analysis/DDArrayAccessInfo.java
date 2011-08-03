package cetus.analysis;

import java.util.*;

import cetus.hir.*;

public class DDArrayAccessInfo
{
	static final int write_type = 0;
	static final int read_type = 1;
	
	private int access_type; // write = 0, read = 1
	private ArrayAccess expr;
	private Loop enclosing_loop;
	private Statement parent_stmt;

	/**
	 * Constructs a new array access information for the specified array access,
   * access type, enclosing loop, and enclosing statement.
	 *
	 * @param expr the array access expression.
   * @param type the type of array access (0=write, 1=read).
   * @param loop the enclosing loop.
   * @param stmt the enclosing statement.
	 */
	public DDArrayAccessInfo(ArrayAccess expr, int type, Loop loop, Statement stmt)
	{
		this.expr = expr;
		this.access_type = type;

		this.enclosing_loop = loop;
		this.parent_stmt = stmt;
	}

	/**
	 * Returns the array access expression
	 */
	public ArrayAccess getArrayAccess()
	{
		return this.expr;
	}

	/**
	 * Set the array access expression
	 */
	public void setArrayAccess(ArrayAccess e)
	{
		this.expr = e;
	}
	
	/**
	 * Returns a boolean value indicating whether this array
	 * access is a write(0) or a read(1) within the parent
	 * statement
	 */
	public int getAccessType()
	{
		return this.access_type;
	}

	/**
	 * Returns the enclosing loop
	 */
	public Loop getAccessLoop()
	{
		return this.enclosing_loop;
	}

	/**
	 * Returns entire nest of enclosing loops
	 */
	public LinkedList<Loop> getAccessEnclosingLoops()
	{
		return (LoopTools.calculateLoopNest(this.enclosing_loop));
	}
	
	/** 
	 * Returns the statement that contains the array access
	 */
	public Statement getParentStatement()
	{
		return this.parent_stmt;
	}

	public Expression getArrayAccessName()
	{
		return (this.expr.getArrayName());
	}
	
	public String toString()
	{
		return new String("ArrayAccess: " + expr.toString() +
				" AccessType: " + ((Integer)access_type).toString());
	}

}
