/**
 *
 */
package ch.unibas.cs.hpwc.patus.representation;

import java.util.Iterator;
import java.util.Set;

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;

/**
 * @author Matthias-M. Christen
 */
public interface IStencilStructure extends Iterable<StencilNode>
{
	///////////////////////////////////////////////////////////////////
	// Node Operations

	/**
	 * Adds an input node to the stencil description.
	 * <p>Note that indices with same spatial indices should have distinct vectorial components, even if the
	 * &quot;time advanceability&quot; differs, i.e. if two or more nodes are added to the same spatial
	 * point, each of these nodes should have a different vectorial index. Unexpected behavior might occur
	 * if this is not the case.</p>
	 * @param node The stencil node to add to the input of the stencil structure
	 * @see Stencil#addNode(StencilNode, int)
	 */
	public abstract void addInputNode (StencilNode node);

	/**
	 * Adds an output node to the stencil description.
	 * @param node The stencil node to add to the output of the stencil structure
	 */
	public abstract void addOutputNode (StencilNode node);

	/**
	 * Copies the description of <code>stencil</code> into <code>this</code>
	 * stencil object.
	 * @param stencil The stencil to copy
	 */
	public abstract void set (Stencil stencil);

	/**
	 * Clears the stencil description.
	 */
	public abstract void clear ();


	///////////////////////////////////////////////////////////////////
	// Node Querying Methods for Stencil Structures

	/**
	 * Returns a copy of the stencil's index set.
	 * @return The index set
	 */
	public abstract Set<Index> getIndexSet ();

	/**
	 * Returns an iterator over the input nodes of the stencil structure.
	 * @return an iterator over the input nodes
	 */
	@Override
	public abstract Iterator<StencilNode> iterator ();

	/**
	 * Returns an iterable over the output nodes of the stencil structure.
	 * @return an iterable over the output nodes
	 */
	public abstract Iterable<StencilNode> getOutputNodes ();

	/**
	 * Determines the spatial location of the output index. If the output
	 * indices are not on the same spatial location, an {@link IllegalArgumentException}
	 * is thrown.
	 * @return The spatial location of the output index.
	 */
	public abstract Expression[] getSpatialOutputIndex ();

	/**
	 * Returns the output indices of the stencil structure.
	 * @return The output indices
	 */
	public abstract Iterable<Index> getOutputIndices ();

	/**
	 * Returns a set of all indices referenced in the stencil, i.e. both
	 * the input and the output indices.
	 * @return A set containing both input and output indices
	 */
	public abstract StencilNodeSet getAllNodes ();

	/**
	 * Returns a set of all nodes contained in the stencil, i.e. both
	 * the input and output nodes.
	 * @return A set containing both input and output nodes
	 */
	public abstract Set<Index> getAllIndices ();

	/**
	 * Returns an iterator over the input indices that are associated with a given
	 * output node of the stencil structure.
	 * @param nodeOutput The output node for which to retrieve the iterator
	 * @return An iterator iterating over the indices associated to a specific
	 * 	output node of the stencil structure
	 */
	public abstract Iterable<StencilNode> getNodeIteratorForOutputNode (StencilNode nodeOutput);

	/**
	 * Returns an iterator over the input indices that are associated with a given
	 * output index of the stencil structure.
	 * @param idxOutput The output index for which to retrieve the iterator
	 * @return An iterator iterating over the indices associated to a specific
	 * 	output node of the stencil structure
	 */
	public abstract Iterable<StencilNode> getNodeIteratorForOutputIndex (Index idxOutput);

	/**
	 * Returns an iterator for a given vector component index of the stencil output node.
	 * @param nVectorComponentIndex The vector component index for which to retrieve the iterator
	 * @return An iterator iterating over the indices associated to a specific vector
	 * 	component of the output node of the stencil structure
	 */
	public abstract Iterable<StencilNode> getNodeIteratorForVectorComponent (int nVectorComponentIndex);

	/**
	 * Returns the total number of indices in the stencil.
	 * @return The total number of indices
	 */
	public abstract int getNumberOfNodes ();

	/**
	 * Returns the number of nodes belonging to the output index <code>nodeOutput</code>.
	 * @param nodeOutput The output index for which to retrieve the number of
	 * 	nodes belonging to the corresponding output index of the stencil
	 * @return The number of nodes belonging to <code>nodeOutput</code>
	 */
	public abstract int getNumberOfNodes (StencilNode nodeOutput);

	/**
	 * Returns the number of nodes belonging to the vector component index <code>nVectorComponentIndex</code>.
	 * @param nVectorComponentIndex The vector component index for which to retrieve the number of
	 * 	nodes belonging to the corresponding vector component of the stencil
	 * @return The number of nodes belonging to <code>nVectorComponentIndex</code>
	 */
	public abstract int getNumberOfNodes (int nVectorComponentIndex);

	/**
	 * Returns the number of vector components that the stencil structure consists of.
	 * @return The number of vector components
	 */
	public abstract int getNumberOfVectorComponents ();

	/**
	 * Determines whether the index <code>idx</code> is contained in the stencil.
	 * @param idx The index to check
	 * @return <code>true</code> if and only if <code>idx</code> is contained in
	 * 	the stencil description
	 */
	public abstract boolean contains (Index idx);

	/**
	 * Returns the dimensionality of the stencil.
	 * @return The stencil's dimensionality
	 */
	public abstract byte getDimensionality ();
}
