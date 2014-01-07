package cetus.analysis;

import java.util.*;
import java.io.*;
import cetus.hir.*;

/**
 * Points to relationships represent the link from the 
 * head of the relationship to the tail. The head is the 
 * pointer variable/symbol and the tail is the variable/location/
 * symbol being pointed to. The relationship can be 
 * definite or possible.
 */
public class PointsToRel implements Cloneable {

	// Points-to relationship pair
	// Head
	private Symbol pointer;
	// Tail
	private Symbol points_to;
	// Type of relationship
	private boolean definite;

	/**
	 * Constructor
	 * @param head Symbol for head location, always a pointer
	 * @param tail Symbol for tail location
	 * @param type The definiteness of the relationship
	 */
	public PointsToRel(Symbol head, Symbol tail, boolean type)
	{
		this.pointer = head;
		this.points_to = tail;
		this.definite = type;
	}

	/**
	 * Constructor - default false relationship
	 * @param head Symbol for head location, always a pointer
	 * @param tail Symbol for tail location
	 */
	public PointsToRel(Symbol head, Symbol tail)
	{
		this(head, tail, false);
	}

	public PointsToRel clone()
	{
		PointsToRel clone = new PointsToRel(getPointerSymbol(),
									getPointedToSymbol(),
									isDefinite());
		return clone;
	}
	
	public int hashCode()
	{
		// TODO: check if this is enough to separate different PointsToRel.
		return (pointer.hashCode() + points_to.hashCode() + ((definite)? 1: 0));
	}

	public boolean equals(Object o)
	{
		return (
				o != null &&
				o instanceof PointsToRel &&	
				getPointerSymbol().equals(((PointsToRel)o).getPointerSymbol()) &&
				getPointedToSymbol().equals(((PointsToRel)o).getPointedToSymbol()) &&
				isDefinite()==((PointsToRel)o).isDefinite());
	}
	
	/**
	 * Return the head of the relationship
	 */
	public Symbol getPointerSymbol()
	{
		return this.pointer;
	}

	/**
	 * Return the tail of the relationship
	 */
	public Symbol getPointedToSymbol()
	{
		return this.points_to;
	}

	/**
	 * Is the relationship definitely valid
	 */
	public boolean isDefinite()
	{
		return this.definite;
	}

	/**
	 * Set to definite
	 */
	public void setDefinite()
	{
		this.definite = true;
	}

	/**
	 * Set to possible
	 */
	public void setPossible()
	{
		this.definite = false;
	}

	/**
	 * Merge this rel with the input rel. Handles 
	 * merging the relationship types.
	 * @param rel input relationship
	 * @return new merged relationship
	 */
	public PointsToRel mergeRel(PointsToRel rel)
	{
		PointsToRel return_rel = new PointsToRel(
						this.getPointerSymbol(),
						this.getPointedToSymbol());
		if ((this.isDefinite()) && (rel.isDefinite()))
			return_rel.setDefinite();

		return return_rel;
	}

	public String toString()
	{
		String type = "";
		if (isDefinite())
			type = "D";
		else
			type = "P";
		return ("(" + 
			pointer.getSymbolName() + "," +
			points_to.getSymbolName() + "," +
			type +
			")");
	}
}
