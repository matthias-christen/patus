/**
 *
 */
package ch.unibas.cs.hpwc.patus.representation;

import cetus.hir.Expression;
import cetus.hir.Identifier;
import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.util.StringUtil;


/**
 * @author Matthias-M. Christen
 */
public class StencilNode extends Identifier implements ISpaceIndexable
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The data type of the stencil node
	 */
	private Specifier m_specType;

	/**
	 * The index indicating where (in space, in time, in which vector
	 * component) the node is located relative to the center node
	 * (0, 0, 0) of the stencil
	 */
	private Index m_index;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates a new stencil node locate at index <code>index</code> (which
	 * comprises
	 * the spatial, temporal, and vectorial components.
	 * 
	 * @param strIdentifier
	 *            The identifier by which this node is referred to in code
	 * @param index
	 *            The node index
	 */
	public StencilNode (String strIdentifier, Specifier specType, Index index)
	{
		super (strIdentifier);
		m_specType = specType;
		m_index = index == null ? new Index () : new Index (index);
	}

	public StencilNode (StencilNode node)
	{
		this (node.getName (), node.getSpecifier (), new Index (node.getIndex ()));
	}

	public Specifier getSpecifier ()
	{
		return m_specType;
	}

	/**
	 * Returns the stencil index consisting of the spatial, temporal, and
	 * vectorial index (relative to the center node (0, 0, 0) of the stencil.
	 * 
	 * @return The stencil node index
	 */
	public Index getIndex ()
	{
		return m_index;
	}

	@Override
	public int[] getSpaceIndex ()
	{
		return m_index.getSpaceIndex ();
	}

	/**
	 * Returns <code>true</code> iff this stencil node represents a scalar variable.
	 * @return
	 */
	public boolean isScalar ()
	{
		return m_index.getSpaceIndex ().length == 0 && m_index.getTimeIndex () == 0 && m_index.getVectorIndex () == 0;
	}

//	/**
//	 * Returns the name of the grid corresponding to this stencil node.
//	 * @return The grid identifier to which this stencil node corresponds
//	 */
//	public String getGridIdentifier ()
//	{
//		StringBuilder sb = new StringBuilder ();
//		sb.append (getName ());
//		if (m_index.getTimeIndex () < 0)
//			sb.append ('_');
//		sb.append (Math.abs (m_index.getTimeIndex ()));
//
//		///
//		// TODO check whether this still works with vector index in the name
//		sb.append ('_');
//		sb.append (m_index.getVectorIndex ());
//		///
//
//		return sb.toString ();
//	}

	@Override
	public boolean equals (Object obj)
	{
		if (obj instanceof StencilNode)
			return m_index.equals (((StencilNode) obj).getIndex ());
		if (obj instanceof Index)
			return m_index.equals (obj);

		return false;
	}

	@Override
	public int hashCode ()
	{
		return m_index.hashCode ();
	}

	@Override
	public String toString ()
	{
		return StringUtil.concat (getName (), m_index.toString ());
	}


	///////////////////////////////////////////////////////////////////
	// Comparable Implementation

	@Override
	public int compareTo (Expression expr)
	{
		if (expr instanceof StencilNode)
			return m_index.compareTo (((StencilNode) expr).getIndex ());
		return super.compareTo (expr);
	}
}
