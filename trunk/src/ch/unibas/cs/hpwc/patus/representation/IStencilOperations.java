/**
 *
 */
package ch.unibas.cs.hpwc.patus.representation;

import java.util.NoSuchElementException;
import java.util.Set;

import cetus.hir.Expression;

/**
 * @author Matthias-M. Christen
 *
 */
public interface IStencilOperations
{
	///////////////////////////////////////////////////////////////////
	// Stencil Structure Operations

	/**
	 * Offsets the index by <code>rgSpaceOffset</code> in space.
	 * @param rgSpaceOffset The spatial offset
	 */
	public abstract void offsetInSpace (int[] rgSpaceOffset);
	
	public abstract void offsetInSpace (Expression[] rgSpaceOffset);

	/**
	 * Advances the stencil by one step in space.
	 */
	public abstract void advanceInSpace (int nDirection);

	/**
	 * Offsets the index by <code>nTimeOffset</code> in time.
	 * @param nTimeOffset The temporal offset
	 */
	public abstract void offsetInTime (int nTimeOffset);

	/**
	 * Advances the stencil by one step in time.
	 */
	public abstract void advanceInTime ();


	///////////////////////////////////////////////////////////////////
	// Stencil Structure Information

	/**
	 * Returns the dimensionality of the stencil, i.e. the number of dimensions in which the
	 * stencil is defined.
	 * @return The stencil's dimensionality
	 */
	public abstract byte getDimensionality ();

	/**
	 * Returns the minimum spatial index that occurs in the stencil description.
	 * @return The minimum space index
	 * @see IndexSetUtil#getMinSpaceIndex(Set)
	 */
	public abstract int[] getMinSpaceIndex ();

	/**
	 * Returns the minimum spatial index that occurs in the stencil description
	 * for the time index <code>nTimeIndex</code>
	 * @return The minimum space index
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 * @see IndexSetUtil#getMinSpaceIndexByTimeIndex(Set, int)
	 */
	public abstract int[] getMinSpaceIndexByTimeIndex (int nTimeIndex);

	/**
	 * Returns the minimum spatial index that occurs in the stencil description
	 * for the time index <code>nTimeIndex</code>
	 * @return The minimum space index
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 * @see IndexSetUtil#getMinSpaceIndexByTimeIndex(Set, int)
	 */
	public abstract int[] getMinSpaceIndexByVectorIndex (int nVectorIndex);

	/**
	 * Returns the minimum spatial index that occurs in the stencil description
	 * for the time index <code>nTimeIndex</code>
	 * @return The minimum space index
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 * @see IndexSetUtil#getMinSpaceIndexByTimeIndex(Set, int)
	 */
	public abstract int[] getMinSpaceIndex (int nTimeIndex, int nVectorIndex);

	/**
	 * Returns the maximum spatial index that occurs in the stencil description.
	 * @return The maximum space index
	 * @see IndexSetUtil#getMaxSpaceIndex(Set)
	 */
	public abstract int[] getMaxSpaceIndex ();

	/**
	 * Returns the maximum spatial index that occurs in the stencil description
	 * for the time index <code>nTimeIndex</code>
	 * @return The maximum space index
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 * @see IndexSetUtil#getMaxSpaceIndexByTimeIndex(Set, int)
	 */
	public abstract int[] getMaxSpaceIndexByTimeIndex (int nTimeIndex);

	/**
	 * Returns the maximum spatial index that occurs in the stencil description
	 * for the time index <code>nTimeIndex</code>
	 * @return The maximum space index
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 * @see IndexSetUtil#getMaxSpaceIndexByTimeIndex(Set, int)
	 */
	public abstract int[] getMaxSpaceIndexByVectorIndex (int nVectorIndex);

	/**
	 * Returns the maximum spatial index that occurs in the stencil description
	 * for the time index <code>nTimeIndex</code>
	 * @return The maximum space index
	 * @throws NoSuchElementException if there is no index with time index
	 * 	equal to <code>nTimeIndex</code> in the set <code>setIndices</code>
	 * @see IndexSetUtil#getMaxSpaceIndexByTimeIndex(Set, int)
	 */
	public abstract int[] getMaxSpaceIndex (int nTimeIndex, int nVectorIndex);

	/**
	 * Returns the minimum time index that occurs in the stencil description.
	 * @return The minimum time index
	 */
	public abstract int getMinTimeIndex ();

	/**
	 * Returns the maximum time index that occurs in the stencil description.
	 * @return The maximum time index
	 */
	public abstract int getMaxTimeIndex ();

	/**
	 * Determines whether time blocking is applicable for this stencil.
	 * It is applicable if the set of time-advanceable input indices is
	 * contained in the set of output indices.
	 * @return <code>true</code> if and only if time blocking is applicable.
	 */
	public abstract boolean isTimeblockingApplicable ();

//	/**
//	 * Returns the size of the ghost zone of the input plane distinguished by the time index
//	 * <code>nTimeIndex</code> and the vector component index <code>nInputVectorComponentIndex</code>
//	 * that is required to calculate the stencil.
//	 * @param nTimeIndex
//	 * @param nInputVectorComponentIndex
//	 * @return a {@link GhostZoneSize} object that identifies the widths of the ghost zones
//	 */
//	public abstract GhostZoneSize getPlaneGhostZoneSize (int nTimeIndex, int nInputVectorComponentIndex);
}
