/**
 *
 */
package ch.unibas.cs.hpwc.patus.representation;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import cetus.hir.Expression;
import cetus.hir.IDExpression;
import ch.unibas.cs.hpwc.patus.analysis.StencilAnalyzer;
import ch.unibas.cs.hpwc.patus.util.StringUtil;
import ch.unibas.cs.hpwc.patus.util.VectorUtil;

/**
 * This class provides a means to working with multiple stencils simultaneously.
 * <p>It includes a &quot;fused&quot; stencil that contains all the nodes of the
 * stencils that are part of the bundle (without the expression information),
 * which is used to assess the planes that are needed to calculate the stencils
 * and the width of the ghost zone layers.</p>
 *
 * @author Matthias-M. Christen
 */
public class StencilBundle implements IStencilOperations, Iterable<Stencil>
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private StencilCalculation m_calc;
	
	/**
	 * The &quot;fused&quot; stencil that contains the nodes of all the
	 * stencils that have been added to the bundle
	 */
	private Stencil m_stencilFused;

	/**
	 * The list of stencils in the bundle
	 */
	private List<Stencil> m_listStencils;
	
	/**
	 * The set of stencil output nodes to which a constant expression (i.e., depending
	 * only on <code>operation</code> parameters and number literals) is assigned.
	 * The set just stores the names of the stencil nodes.
	 */
	private Set<String> m_setConstantOutputNodes;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Constructs a new stencil bundle.
	 */
	public StencilBundle (StencilCalculation calc)
	{
		m_calc = calc;
		m_stencilFused = null;
		m_listStencils = new LinkedList<> ();
		m_setConstantOutputNodes = new HashSet<> ();
	}

	/**
	 * Returns the fused stencil containing all the nodes of the stencils
	 * that have been added to the bundle.
	 * 
	 * @return The fused stencil
	 */
	public Stencil getFusedStencil ()
	{
		return m_stencilFused;
	}

	/**
	 * Returns an iterable over all the stencils contained in this bundle.
	 * 
	 * @return An iterable over the stencils in this bundle
	 */
	public Iterable<Stencil> getStencils ()
	{
		return m_listStencils;
	}

	/**
	 * Count the number of Flops performed by one stencil bundle calculation.
	 * The method doesn't count Flops in constant stencils.
	 * 
	 * @return The number of Flops performed by one stencil evaluation
	 */
	public int getFlopsCount ()
	{
		int nFlopsCount = 0;
		
		for (Stencil stencil : m_listStencils)
			if (!StencilAnalyzer.isStencilConstant (stencil, m_calc))
				nFlopsCount += stencil.getFlopsCount ();
		
		return nFlopsCount;
	}

	/**
	 * Adds a stencil to the bundle.
	 * 
	 * @param stencil
	 *            The stencil to add to the bundle
	 * @throws NoSuchMethodException
	 * @throws SecurityException
	 */
	public void addStencil (Stencil stencil, boolean bAllowOffsetInSpace) throws NoSuchMethodException, SecurityException
	{
		// add the stencil to the list
		addStencilToList (stencil);

		// add the stencil nodes to the fused stencil
		addStencilToFused (stencil, bAllowOffsetInSpace);
	}
	
	private void addStencilToList (Stencil stencil)
	{
		// add the stencil to the list of stencils
		m_listStencils.add (stencil);
		
		// if the stencil is constant (i.e., depends only on parameters and number literals),
		// add its output nodes to the set of constant ouput nodes
		if (StencilAnalyzer.isStencilConstant (stencil, m_calc))
			for (StencilNode nodeOut : stencil.getOutputNodes ())
				m_setConstantOutputNodes.add (nodeOut.getName ());
	}

	/**
	 * Adds the stencil <code>stencil</code> to the fused stencil.
	 * 
	 * @param stencil
	 *            The stencil to add
	 */
	private void addStencilToFused (Stencil stencil, boolean bAllowOffsetInSpace) throws NoSuchMethodException, SecurityException
	{
/*		
		// Determine whether we need to shift the stencils in space: Shifting is necessary
		// if the output indices of the fused stencil and the stencil to add are not
		// aligned. Note that the stencils in the list will be shifted too.

		ensureFusedStencilCreated (stencil);
		Expression[] rgSpaceIndexFusedStencil = m_stencilFused.getSpatialOutputIndex ();
		Expression[] rgSpaceIdxNewStencil = stencil.getSpatialOutputIndex ();

		// offset the new stencil so that the output index is at the origin
		m_stencilFused.offsetInSpace (VectorUtil.negate (rgSpaceIndexFusedStencil));
		stencil.offsetInSpace (VectorUtil.negate (rgSpaceIdxNewStencil));

		// add the new stencil to the fused stencil
		for (int i = 0; i < stencil.getNumberOfVectorComponents (); i++)
			for (StencilNode node : stencil.getNodeIteratorForVectorComponent (i))
				m_stencilFused.addInputNode (new StencilNode (node));

		// TODO: check output node
		for (StencilNode node : stencil.getOutputNodes ())
			if (node.getIndex ().getSpaceIndexEx ().length > 0)
				m_stencilFused.addOutputNode (new StencilNode (node));

		// offset the stencils that are in the bundle
		// TODO: do we really need this??
		Expression[] rgMinSpaceIndex = VectorUtil.getMinimum (rgSpaceIndexFusedStencil, rgSpaceIdxNewStencil);
		m_stencilFused.offsetInSpace (rgMinSpaceIndex);
		for (Stencil s : m_listStencils)
			s.offsetInSpace (rgMinSpaceIndex);
*/
		
		// shift stencil indices so that the spatial coordinate of the center point is always at (0,...,0)
		ensureFusedStencilCreated (stencil);
		if (bAllowOffsetInSpace)
			stencil.offsetInSpace (VectorUtil.negate (stencil.getSpatialOutputIndex ()));
		
		// add the new stencil to the fused stencil
		for (int i = 0; i < stencil.getNumberOfVectorComponents (); i++)
			for (StencilNode node : stencil.getNodeIteratorForVectorComponent (i))
				m_stencilFused.addInputNode (new StencilNode (node));

		// TODO: check output node
		for (StencilNode node : stencil.getOutputNodes ())
			if (node.getIndex ().getSpaceIndexEx ().length > 0)
				m_stencilFused.addOutputNode (new StencilNode (node));		
	}

	/**
	 * Rebuilds the structure of the fused stencil.
	 * <p>Use this method after properties of the stencils within this bundle are
	 * changes manually (i.e. other than using the methods provided by the
	 * {@link StencilBundle} class).
	 * </p>
	 */
	public void rebuild (boolean bAllowOffsetInSpace) throws NoSuchMethodException, SecurityException
	{
		m_stencilFused.clear ();
		for (Stencil stencil : m_listStencils)
			addStencilToFused (stencil, bAllowOffsetInSpace);
	}

	/**
	 * Returns the number of stencils in the bundle.
	 * @return The number of stencils
	 */
	public int getStencilsCount ()
	{
		return m_listStencils.size ();
	}

	@Override
	public Iterator<Stencil> iterator ()
	{
		return m_listStencils.iterator ();
	}

	/**
	 * Ensures that the fused stencil object has been created
	 * 
	 * @param stencilTemplate
	 * @throws NoSuchMethodException
	 * @throws SecurityException
	 */
	private void ensureFusedStencilCreated (Stencil stencilTemplate) throws NoSuchMethodException, SecurityException
	{
		if (m_stencilFused == null)
			m_stencilFused = StencilBundle.createStencilFromTemplate (stencilTemplate, false);

	}

	/**
	 * Creates a new stencil of the same class as <code>stencilTemplate</code>
	 * and copies the data (the stencil nodes) from <code>stencilTemplate</code>
	 * into the newly created instance if <code>bCopyNodes</code> is set to
	 * <code>true</code>.
	 * 
	 * @param stencilTemplate
	 *            The template used to create a new stencil instance
	 * @param bCopyNodes
	 *            Flag determining whether to copy the stencil nodes of
	 *            <code>stencilTemplate</code> into the newly created instance
	 * @return A new instance of a stencil based on <code>stencilTemplate</code>
	 * @throws NoSuchMethodException
	 * @throws SecurityException
	 */
	@SuppressWarnings("unchecked")
	private static Stencil createStencilFromTemplate (Stencil stencilTemplate, boolean bCopyNodes) throws NoSuchMethodException, SecurityException
	{
		Class<Stencil> clsStencil = (Class<Stencil>) stencilTemplate.getClass ();
		Constructor<Stencil> mthConstructor = bCopyNodes ? clsStencil.getConstructor (clsStencil) : clsStencil.getConstructor ();
		Stencil stencil = null;

		try
		{
			if (bCopyNodes)
				stencil = mthConstructor.newInstance (stencilTemplate);
			else
				stencil = mthConstructor.newInstance ();
		}
		catch (IllegalArgumentException e)
		{
		}
		catch (InstantiationException e)
		{
		}
		catch (IllegalAccessException e)
		{
		}
		catch (InvocationTargetException e)
		{
		}

		return stencil;
	}


	///////////////////////////////////////////////////////////////////
	// Stencil Structure Operations

	@Override
	public void offsetInSpace (int[] rgSpaceOffset)
	{
		if (m_stencilFused == null)
			return;

		m_stencilFused.offsetInSpace (rgSpaceOffset);
		for (Stencil stencil : m_listStencils)
			stencil.offsetInSpace (rgSpaceOffset);
	}
	
	@Override
	public void offsetInSpace (Expression[] rgSpaceOffset)
	{
		if (m_stencilFused == null)
			return;

		m_stencilFused.offsetInSpace (rgSpaceOffset);
		for (Stencil stencil : m_listStencils)
			stencil.offsetInSpace (rgSpaceOffset);
	}

	@Override
	public void advanceInSpace (int nDirection)
	{
		if (m_stencilFused == null)
			return;

		m_stencilFused.advanceInSpace (nDirection);
		for (Stencil stencil : m_listStencils)
			stencil.advanceInSpace (nDirection);
	}

	@Override
	public void offsetInTime (int nTimeOffset)
	{
		if (m_stencilFused == null)
			return;

		m_stencilFused.offsetInTime (nTimeOffset);
		for (Stencil stencil : m_listStencils)
			stencil.offsetInTime (nTimeOffset);
	}

	@Override
	public void advanceInTime ()
	{
		if (m_stencilFused == null)
			return;

		m_stencilFused.advanceInTime ();
		for (Stencil stencil : m_listStencils)
			stencil.advanceInTime ();
	}


	///////////////////////////////////////////////////////////////////
	// Stencil Structure Information

	@Override
	public byte getDimensionality ()
	{
		return m_stencilFused == null ? 0 : m_stencilFused.getDimensionality ();
	}

	@Override
	public int[] getMinSpaceIndex ()
	{
		return m_stencilFused == null ? new int[0] : m_stencilFused.getMinSpaceIndex ();
	}

	@Override
	public int[] getMinSpaceIndexByTimeIndex (int nTimeIndex)
	{
		return m_stencilFused == null ? new int[0] : m_stencilFused.getMinSpaceIndexByTimeIndex (nTimeIndex);
	}

	@Override
	public int[] getMinSpaceIndexByVectorIndex (int nVectorIndex)
	{
		return m_stencilFused == null ? new int[0] : m_stencilFused.getMinSpaceIndexByTimeIndex (nVectorIndex);
	}

	@Override
	public int[] getMinSpaceIndex (int nTimeIndex, int nVectorIndex)
	{
		return m_stencilFused == null ? new int[0] : m_stencilFused.getMinSpaceIndex (nTimeIndex, nVectorIndex);
	}

	@Override
	public int[] getMaxSpaceIndex ()
	{
		return m_stencilFused == null ? new int[0] : m_stencilFused.getMaxSpaceIndex ();
	}

	@Override
	public int[] getMaxSpaceIndexByTimeIndex (int nTimeIndex)
	{
		return m_stencilFused == null ? new int[0] : m_stencilFused.getMaxSpaceIndexByTimeIndex (nTimeIndex);
	}

	@Override
	public int[] getMaxSpaceIndexByVectorIndex (int nVectorIndex)
	{
		return m_stencilFused == null ? new int[0] : m_stencilFused.getMaxSpaceIndexByVectorIndex (nVectorIndex);
	}

	@Override
	public int[] getMaxSpaceIndex (int nTimeIndex, int nVectorIndex)
	{
		return m_stencilFused == null ? new int[0] : m_stencilFused.getMaxSpaceIndex (nTimeIndex, nVectorIndex);
	}

	@Override
	public int getMinTimeIndex ()
	{
		return m_stencilFused == null ? 0 : m_stencilFused.getMinTimeIndex ();
	}

	@Override
	public int getMaxTimeIndex ()
	{
		return m_stencilFused == null ? 0 : m_stencilFused.getMaxTimeIndex ();
	}

	@Override
	public boolean isTimeblockingApplicable ()
	{
		return m_stencilFused == null ? false : m_stencilFused.isTimeblockingApplicable ();
	}

	public boolean isConstantOutputStencilNode (IDExpression node)
	{
		return isConstantOutputStencilNode (node.getName ());
	}
	
	public boolean isConstantOutputStencilNode (String strNodeName)
	{
		return m_setConstantOutputNodes.contains (strNodeName);
	}


	///////////////////////////////////////////////////////////////////
	// Object Overrides

	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ("Fused stencil:\n==============\n\n");
		sb.append (m_stencilFused == null ? "(null)" : m_stencilFused.toString ());
		sb.append ("\n\n\nStencils:\n=========\n\n");
		StringUtil.joinAsBuilder (m_listStencils, "\n", sb);

		return sb.toString ();
	}
}
