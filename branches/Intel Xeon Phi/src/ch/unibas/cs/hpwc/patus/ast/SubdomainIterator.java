/*******************************************************************************
 * Copyright (c) 2011 Matthias-M. Christen, University of Basel, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Matthias-M. Christen, University of Basel, Switzerland - initial API and implementation
 ******************************************************************************/
package ch.unibas.cs.hpwc.patus.ast;

import java.util.Map;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Declaration;
import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Statement;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.geometry.Border;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.geometry.Subdomain;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class SubdomainIterator extends Loop
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The domain subdomain over which is iterated
	 */
	protected SubdomainIdentifier m_sdidDomain;

	/**
	 * The border around the domain subdomain
	 */
	protected Border m_borderDomain;

	/**
	 * The total domain subdomain (m_sdidDomain.getDomain () + m_borderDomain)
	 */
	protected Subdomain m_sdTotalDomain;

	/**
	 * The subdomain that is iterated over the domain subdomain
	 */
	protected SubdomainIdentifier m_sdidIterator;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates an empty subdomain iterator.
	 */
	public SubdomainIterator ()
	{
		super ();

		setIteratorSubdomain (null);
		setDomainSubdomain (null, null, null);
	}

	/**
	 *
	 * @param sdIterator
	 * @param sdDomain
	 * @param nNumThreads
	 * @param exprChunkSize
	 * @param stmtBody
	 * @param nParallelismLevel
	 */
	public SubdomainIterator (SubdomainIdentifier sdIterator, SubdomainIdentifier sdDomain, Border borderDomain, int nNumThreads, Expression[] rgChunkSize, Statement stmtBody, int nParallelismLevel)
	{
		super (nNumThreads, rgChunkSize, stmtBody, nParallelismLevel);

		setIteratorSubdomain (sdIterator);
		setDomainSubdomain (sdDomain, sdDomain.getSubdomain ().getSize (), borderDomain);
	}

	/**
	 * Returns the iterator.
	 * @return The iterator
	 */
	public SubdomainIdentifier getIterator ()
	{
		return m_sdidIterator;
	}

	/**
	 * Synonym for {@link SubdomainIterator#getIterator()}.
	 */
	@Override
	public Identifier getLoopIndex ()
	{
		return m_sdidIterator;
	}

	/**
	 * Returns the subdomain of the iterator.
	 * @return The iterator subdomain
	 */
	public Subdomain getIteratorSubdomain ()
	{
		return m_sdidIterator.getSubdomain ();
	}

	/**
	 * <p>Sets the subdomain that is iterated within the domain subdomain.</p>
	 * <p>E.g. the domain could be a 3D subdomain and the iterator can also be a 3D subdomain
	 * to iterate blockwise over the domain subdomain, or the iterator can be a plane to
	 * iterate planewise, etc.
	 * @param sgIterator The iterator subdomain.
	 */
	public void setIteratorSubdomain (SubdomainIdentifier sdidIterator)
	{
		m_sdidIterator = sdidIterator;
	}

	/**
	 * Returns the domain of the iterator, i.e. over which subdomain the iterator iterates.
	 * @return
	 */
	public SubdomainIdentifier getDomainIdentifier ()
	{
		return m_sdidDomain;
	}

	/**
	 * Returns the subdomain of the domain of the iterator, i.e. over which subdomain the
	 * iterator subdomain iterates.
	 * @return
	 */
	public Subdomain getDomainSubdomain ()
	{
		return m_sdidDomain.getSubdomain ();
	}

	public Border getDomainBorder ()
	{
		return m_borderDomain;
	}

	public Subdomain getTotalDomainSubdomain ()
	{
		return m_sdTotalDomain;
	}

	/**
	 * Sets the domain over which this iterator is iterating.
	 * @param sdidDomain The domain subdomain identifier in which the iterator subdomain is iterated
	 * @param size
	 * @param border The border added to the domain subdomain
	 */
	public void setDomainSubdomain (SubdomainIdentifier sdidDomain, Size size, Border border)
	{
		m_sdidDomain = sdidDomain;
		m_borderDomain = border;

		if (m_sdidDomain != null)
		{
			// create the domain size object
			Size sizeDomain = new Size (size);
			for (int i = 0; i < sizeDomain.getDimensionality (); i++)
				if (sizeDomain.getCoord (i) == null)
					sizeDomain.setCoord (i, m_sdidDomain.getSubdomain ().getSize ().getCoord (i));

			// create a new subdomain for the total domain
			Subdomain subdomain = m_sdidDomain.getSubdomain ();
			m_sdTotalDomain = new Subdomain (subdomain.getParentSubdomain (), subdomain.getType (), subdomain.getLocalCoordinates (), sizeDomain, subdomain.isBaseGrid ());
			if (m_borderDomain != null)
				m_sdTotalDomain.addBorder (border);
		}
	}

	@Override
	protected Expression[] getDefaultChunkSize ()
	{
		// default is one block per thread
		Expression[] rgDefaultChunk = new Expression[m_sdidDomain.getDimensionality ()];
		for (int i = 0; i < m_sdidDomain.getDimensionality (); i++)
			rgDefaultChunk[i] = Globals.ONE.clone ();
		return rgDefaultChunk;
	}

	/**
	 * Returns the number of blocks stride in dimension <code>nDim</code>.
	 * For the unit stride direction (<code>nDim</code> = 0), the number of blocks is 1,
	 * for <code>nDim</code> = 1, the number is the number of blocks in unit stride direction,
	 * for <code>nDim</code> = 2, the number is the number of blocks in unit stride direction
	 * times the number of blocks in the next direction, etc.
	 */
	public Expression getNumberOfBlocksStride (int nDim)
	{
		if (nDim == 0)
			return new IntegerLiteral (1);

		Expression exprBlocksCount = null;
		for (int i = 0; i < nDim; i++)
		{
			// calculate size for one dimension
			Expression exprBlocksCountDim = getNumberOfBlocksInDimension (i);

			// multiply
			if (exprBlocksCount == null)
				exprBlocksCount = exprBlocksCountDim;
			else
				exprBlocksCount = new BinaryExpression (exprBlocksCount.clone (), BinaryOperator.MULTIPLY, exprBlocksCountDim);
		}

		return exprBlocksCount;
	}

	/**
	 * Returns the number of blocks the iterator traverses when iterating through the
	 * domain subdomain.
	 * @return
	 */
	public Expression getNumberOfBlocks ()
	{
		return getNumberOfBlocksStride (m_sdidDomain.getDimensionality ());
	}

	/**
	 * Returns the number of blocks in a certain dimension, <code>nDim</code>.
	 * @param nDim
	 * @return
	 */
	public Expression getNumberOfBlocksInDimension (int nDim)
	{
		return ExpressionUtil.ceil (
			m_sdTotalDomain.getSize ().getCoord (nDim).clone (),
			m_sdidIterator.getSubdomain ().getSize ().getCoord (nDim).clone ());
	}
	
	public Size getNumberOfBlocksPerDimension ()
	{
		Size sizeNumBlocks = new Size (m_sdidDomain.getDimensionality ());
		for (int i = 0; i < m_sdidDomain.getDimensionality (); i++)
			sizeNumBlocks.setCoord (i, getNumberOfBlocksInDimension (i));
		return sizeNumBlocks;
	}


	/**
	 * Returns the <code>nDim</code>-th coordinate of the N-dimensional index from
	 * the linear index <code>exprLinearIndex</code>.
	 * @param exprLinearIndex
	 * @param nDim
	 * @return
	 */
	public Expression getBoxIndexFromLinearIndex (Expression exprLinearIndex, int nDim)
	{
		Expression exprIdx = exprLinearIndex.clone ();
		if (nDim > 0)
			exprIdx = new BinaryExpression (exprIdx, BinaryOperator.DIVIDE, getNumberOfBlocksStride (nDim));

		return new BinaryExpression (
			Symbolic.simplify (exprIdx, Symbolic.ALL_VARIABLES_INTEGER),
			BinaryOperator.MODULUS,
			Symbolic.simplify (getNumberOfBlocksInDimension (nDim), Symbolic.ALL_VARIABLES_INTEGER));
	}

	@Override
	public String toString ()
	{
		return StringUtil.concat (
			"for ", m_sdidIterator.getSubdomain ().getType (), " ", m_sdidIterator, m_sdidIterator.getSubdomain ().getSize (), " in ", m_sdidDomain,
			(m_borderDomain != null ? " + " + m_borderDomain.toString () : ""),
			" parallel ", m_exprNumThreads, " <level ", m_nParallelismLevel , "> schedule ", m_rgChunkSize == null ? "default" : StringUtil.toString (m_rgChunkSize), "\n",
			getLoopBody ());
	}

	@Override
	public String getLoopHeadAnnotation ()
	{
		return StringUtil.concat (
			"for ", m_sdidIterator.getSubdomain ().getType (), " ", m_sdidIterator, " of size ", m_sdidIterator.getSubdomain ().getSize (), " in ", m_sdidDomain,
			(m_borderDomain != null ? " + " + m_borderDomain.toString () : ""),
			" parallel ", m_exprNumThreads, " <level ", m_nParallelismLevel , "> schedule ", m_rgChunkSize == null ? "default" : StringUtil.toString (m_rgChunkSize), " { ... }");
	}

	@Override
	public Map<IDExpression, Declaration> getTable ()
	{
		return super.getTable ();
	}

	@Override
	public SubdomainIterator clone ()
	{
		Expression[] rgChunkSize = null;
		if (m_rgChunkSize != null)
		{
			rgChunkSize = new Expression[m_rgChunkSize.length];
			for (int i = 0; i < m_rgChunkSize.length; i++)
				rgChunkSize[i] = m_rgChunkSize[i].clone ();
		}
		
		SubdomainIterator sgit = new SubdomainIterator (
			m_sdidIterator.clone (),
			m_sdidDomain.clone (),
			new Border (m_borderDomain.getMin ().clone (), m_borderDomain.getMax ().clone ()),
			0,
			rgChunkSize,
			getLoopBody (),	// will be cloned in Loop
			m_nParallelismLevel);
		sgit.setNumberOfThreads (m_exprNumThreads.clone ());

		return sgit;
	}
}
