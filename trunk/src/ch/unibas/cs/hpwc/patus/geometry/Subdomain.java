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
package ch.unibas.cs.hpwc.patus.geometry;

import cetus.hir.IntegerLiteral;
import ch.unibas.cs.hpwc.patus.util.StringUtil;


/**
 *
 * @author Matthias-M. Christen
 */
public class Subdomain
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public enum ESubdomainType
	{
		SUBDOMAIN,
		PLANE,
		POINT
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The type of this subdomain
	 */
	private ESubdomainType m_type;

	/**
	 * The parent grid
	 */
	private Subdomain m_sgParent;

	/**
	 * The size of the subdomain; the amount of data that is actually computed
	 * in the output grid
	 */
	private Size m_size;

	/**
	 * The box including the ghost nodes that are required for the computation.
	 * In local coordinates, i.e. the min coordinate of the box is amount of
	 * necessary padding in direction of the negative axes, the max coordinate
	 * of the box is the amount of padding required by the stencil in direction
	 * of the positive axes.
	 */
	private Box m_boxInput;

	/**
	 * The amount of padding appended in the direction of the positive axes
	 */
	private Size m_sizePadding;

	/**
	 * The total size of the subdomain, including the array padding, i.e.
	 * <code>m_boxTotal = m_boxInput + m_sizePadding</code>
	 */
	private Box m_boxTotal;

	/**
	 * The coordinates of the reference point in the parent frame.
	 * The reference point of a subdomain is the point with local coordinates (0, 0, ..., 0)
	 * (i.e. the upper left corner).
	 */
	private Point m_ptRefParent;

	/**
	 * The reference point in absolute coordinates (in the global coordinate frame).
	 * The reference point of a subdomain is the point with local coordinates (0, 0, ..., 0)
	 * (i.e. the upper left corner).
	 */
	private Point m_ptRefGlobal;

	/**
	 * Flag indicating whether this subdomain is a base grid, i.e. a grid on which the address calculation
	 * of the child grids are based.
	 * If a base grid (i.e. a subdomain with the {@link Subdomain#m_bIsBaseGrid} flag set to <code>true</code>)
	 * in the grid hierarchy is encountered, this signifies that data between two memory hierarchies has
	 * to be transferred, e.g. from main memory to a local memory.
	 */
	private boolean m_bIsBaseGrid;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 *
	 * @param sgParent
	 * @param type
	 * @param size
	 */
	public Subdomain (Subdomain sgParent, ESubdomainType type, Size size)
	{
		this (sgParent, type, new Point (size.getDimensionality ()), size);
	}

	/**
	 *
	 * @param sgParent
	 * @param type
	 * @param ptRefParent
	 * @param size
	 */
	public Subdomain (Subdomain sgParent, ESubdomainType type, Point ptRefParent, Size size)
	{
		this (sgParent, type, ptRefParent, size, false);
	}

	/**
	 *
	 * @param sgParent
	 * @param type
	 * @param ptRefParent
	 * @param size
	 * @param bIsBaseGrid
	 */
	public Subdomain (Subdomain sgParent, ESubdomainType type, Point ptRefParent, Size size, boolean bIsBaseGrid)
	{
		m_sgParent = sgParent;
		m_type = type;
		m_size = size;
		m_bIsBaseGrid = bIsBaseGrid;

//		Point ptZero = new Point (Vector.getZeroVector (size.getDimensionality ()).getCoords ());
//		m_boxInput = new Box (ptZero, size);
//		m_boxTotal = new Box (ptZero, size);
		m_boxInput = new Box (ptRefParent, size);
		m_boxTotal = new Box (ptRefParent, size);

		m_ptRefParent = new Point (ptRefParent);
		m_ptRefGlobal = new Point (m_sgParent == null ? m_ptRefParent : m_sgParent.getGlobalCoordinates ());
	}

	/**
	 * Copy constructor.
	 * @param grid
	 */
	public Subdomain (Subdomain grid)
	{
		m_sgParent = grid.m_sgParent;
		m_type = grid.m_type;
		m_size = grid.m_size.clone ();
		m_bIsBaseGrid = grid.m_bIsBaseGrid;

		m_boxInput = grid.m_boxInput.clone ();
		m_boxTotal = grid.m_boxTotal.clone ();

		m_ptRefParent = grid.m_ptRefParent.clone ();
		m_ptRefGlobal = grid.m_ptRefGlobal.clone ();
	}

	public void setPadding ()
	{

	}

	/**
	 * Returns the reference dimensions of the subdomain, i.e. the dimensions
	 * that will be the computed output, without extras like padding.
	 * @return
	 */
	public Box getBox ()
	{
		return m_boxInput;
	}

	/**
	 * Returns the total size of the box, including extras like padding.
	 * @return
	 */
	public Box getTotalBox ()
	{
		return m_boxTotal;
	}

	/**
	 * Moves the subdomain by the vector <code>sizeOffset</code>
	 * @param sizeOffset The vector by which to offset the subdomain
	 */
	public void move (Size sizeOffset)
	{
		m_ptRefParent.add (sizeOffset);
		m_ptRefGlobal.add (sizeOffset);
	}

	/**
	 * Adds a border to the subdomain.
	 * @param border
	 */
	public void addBorder (Border border)
	{
		m_boxInput.addBorder (border);
		m_boxTotal.addBorder (border);

		Size sizeOffset = new Size (border.getMin ());
		sizeOffset.scale (new IntegerLiteral (-1));
		move (sizeOffset);
	}

	/**
	 * Returns the type of the subdomain specifying whether the subdomain is a point, a plane, or a subdomain (3D slice).
	 * @return The type of the subdomain
	 */
	public ESubdomainType getType ()
	{
		return m_type;
	}

	/**
	 * Returns the reference point within the parent coordinate frame.
	 * @return The local coordinates of the reference point
	 */
	public Point getLocalCoordinates ()
	{
		return m_ptRefParent;
	}

	/**
	 * Returns the reference point in global coordinates.
	 * @return The global coordinates of the reference point
	 */
	public Point getGlobalCoordinates ()
	{
		return m_ptRefGlobal;
	}

	public Point getBaseGridCoordinates ()
	{
		if (m_bIsBaseGrid || m_sgParent == null)
			return m_ptRefParent;
		return m_sgParent.getBaseGridCoordinates ();
	}

	/**
	 * Returns the size of the subdomain.
	 * @return The subdomain size
	 */
	public Size getSize ()
	{
		return m_size;
	}

	/**
	 * Returns the parent grid or <code>null</code> if there is no parent.
	 * @return The parent grid
	 */
	public Subdomain getParentSubdomain ()
	{
		return m_sgParent;
	}

	/**
	 * Defines that this grid is a base grid, i.e. a grid on which the child grids
	 * base their address calculation.
	 * @param bIsBaseGrid Flag determining whether or not this grid is a base grid
	 */
	public void setBaseGrid (boolean bIsBaseGrid)
	{
		m_bIsBaseGrid = bIsBaseGrid;
	}

	/**
	 * Returns the base grid that is &quot;next&quot; base grid up in the hierarchy.
	 * @return The subdomain that is the base grid for this grid
	 */
	public Subdomain getBaseGrid ()
	{
		Subdomain sgBase = this;
		while (!sgBase.isBaseGrid ())
			sgBase = sgBase.getParentSubdomain ();
		return sgBase;
	}

	/**
	 * Returns <code>true</code> iff this subdomain is a base grid, i.e. a grid on which the address calculation
	 * of the child grids are based.
	 * If a base grid in the grid hierarchy is encountered, this signifies that data between two memory
	 * hierarchies has to be transferred, e.g. from main memory to a local memory.
	 * @return <code>true</code> iff this grid is a base subdomain
	 */
	public boolean isBaseGrid ()
	{
		return m_bIsBaseGrid;
	}

	@Override
	public String toString ()
	{
		return StringUtil.concat (m_type, "{", m_ptRefGlobal, " + ", m_size, "}");
	}

	@Override
	public Subdomain clone ()
	{
		return new Subdomain (this);
	}
}
