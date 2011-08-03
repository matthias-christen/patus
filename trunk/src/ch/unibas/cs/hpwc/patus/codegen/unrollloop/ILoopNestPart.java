package ch.unibas.cs.hpwc.patus.codegen.unrollloop;

import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ForLoop;
import cetus.hir.IDExpression;

/**
 * Interface for a single loop within a perfect loop nest and facilities
 * to manipulate the underlying <code>for</code> loop.
 *
 * @author Matthias-M. Christen
 */
interface ILoopNestPart
{
	/**
	 * Initializes the loop nest part.
	 * @param data The shared data object
	 * @param loop The <code>for</code> loop associated with this nest part
	 * @param nLoopNumber The number of the loop (counting from the outermost to the innermost loop)
	 */
	public abstract void init (UnrollLoopSharedObjects data, ForLoop loop, int nLoopNumber);

	/**
	 * Returns the loop associated to this data object.
	 * @return The loop object
	 */
	public abstract ForLoop getLoop ();

	/**
	 * Returns the parent of this loop nest part, i.e. the loop containing this one.
	 * @return The parent loop
	 */
	public abstract ILoopNestPart getParent ();

	/**
	 * Sets the parent loop.
	 * @param lnpParent The parent loop
	 */
	public abstract void setParent (ILoopNestPart lnpParent);

	/**
	 * Determines whether the associated loop object has child loops,
	 * i.e. is further nested.
	 * @return <code>true</code> iff there are child loops in the nest
	 */
	public abstract boolean hasChildLoops ();

	/**
	 * Returns the child object if any (<code>null</code> if the loop data
	 * object doesn't have any children, i.e. if there are no further nested
	 * loops in the nest.
	 * @return The child loop
	 */
	public abstract ILoopNestPart getChild ();

	/**
	 * Sets the child loop.
	 * @param lnpChild The child loop
	 */
	public abstract void setChild (ILoopNestPart lnpChild);

	/**
	 * Returns the identifier of the loop index.
	 * @return The loop index
	 */
	public abstract IDExpression getLoopIndex ();

	/**
	 * Returns the identifier to which the value of the loop index is assigned after
	 * the first run of the loop.
	 * This value is used as start value for the cleanup loops.<br/>
	 * <b>Note:</b> This assumes that the trip count of this loop does not
	 * depend on the loop indices of the outer loops!
	 * @param nUnrollFactor The unrolling factor for which to retrieve the end value identifier
	 * @return the identifier of the end value of the loop
	 */
	public abstract IDExpression getEndValueIdentifier ();

	/**
	 * Tries to find a {@link ForLoop} nested within this one and returns it.
	 * If no nested {@link ForLoop} can be found, <code>null</code> is returned.
	 * @return The child {@link ForLoop} within this loop or <code>null</code> if there is no such loop
	 */
	public abstract ForLoop getNestedLoop ();

	/**
	 * Returns a {@link CompoundStatement} with all the statements in <code>cmpstmtBody</code>
	 * unrolled according to the unrolling factor of this object.
	 * @param cmpstmtBody
	 * @return
	 */
	public abstract CompoundStatement unrollBody (CompoundStatement cmpstmtBody, int nUnrollIndex);

	/**
	 * Determines whether the expression <code>expr</code> depends on a loop index variable in the loop nest.
	 * @param expr The expression to check
	 * @return <code>true</code> iff the expression <code>expr</code> depends on one of the loop index variables
	 */
	public abstract boolean dependsOnLoopIndex (Expression expr);

	/**
	 * Determines whether <code>expr</code> depends solely on the loop index, i.e. not
	 * on other variables.
	 * @param expr The expression to check
	 * @return <code>true</code> if the expression <code>expr</code> only depends on the loop index
	 */
	public abstract boolean dependsOnlyOnLoopIndices (Expression expr);

	/**
	 * Returns the loop head of the unrolled loop.
	 * @return The loop head of the unrolled loop
	 */
	public abstract LoopNest getUnrolledLoopHead (int nUnrollFactor);

	/**
	 * Returns the loop head of the cleanup loop.
	 * @return The loop head of the cleanup loop
	 */
	public LoopNest getCleanupLoopHead (int nUnrollFactor);
}