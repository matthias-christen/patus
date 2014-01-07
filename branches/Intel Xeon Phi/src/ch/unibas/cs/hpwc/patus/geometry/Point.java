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

import java.util.Arrays;

import cetus.hir.Expression;

/**
 *
 * @author Matthias-M. Christen
 */
public class Point extends Vector
{
	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates a new point with coordinates set to the origin.
	 */
	public Point ()
	{
		super ();
	}

	public Point (byte nDimensionality)
	{
		super (nDimensionality);
	}

	/**
	 * Creates a point with coordinates set to the elements of <code>rgCoords</code>.
	 * @param rgCoords The coordinates
	 */
	public Point (int... rgCoords)
	{
		super (rgCoords);
	}

	/**
	 * Creates a point with coordinates set to the elements of <code>rgCoords</code>.
	 * @param rgCoords The coordinate expressions
	 */
	public Point (Expression... rgCoords)
	{
		super (rgCoords);
	}

	/**
	 * Copy constructor.
	 * @param pt The point to copy
	 */
	public Point (Point pt)
	{
		super (pt.getCoords ());
		m_bCoordsSimplified = pt.m_bCoordsSimplified;
	}

	/**
	 * Returns a new point object whose coordinates are the ones of <code>this</code> point,
	 * shifted by <code>v</code>.
	 * @param v The offset by which to move the point
	 * @return A new point instance, <code>this</code> + <code>v</code>
	 */
	public Point offset (Vector v)
	{
		Point point = new Point (this);
		point.add (v);
		return point;
	}

	/**
	 * Returns a new point object whose coordinates are the ones of <code>this</code> point,
	 * shifted by <code>rgOffset</code>.
	 * @param rgOffset The offset by which to move the point
	 * @return A new point instance, <code>this</code> + <code>rgOffset</code>
	 */
	public Point offset (Expression[] rgOffset)
	{
		return offset (new Vector (rgOffset));
	}

	/**
	 * Returns a new point object whose coordinates are the ones of <code>this</code> point,
	 * shifted by <code>rgOffset</code>.
	 * @param rgOffset The offset by which to move the point
	 * @return A new point instance, <code>this</code> + <code>rgOffset</code>
	 */
	public Point offset (int[] rgOffset)
	{
		return offset (new Vector (rgOffset));
	}

//	@Override
//	public boolean equals (Object obj)
//	{
//		if (obj instanceof Point)
//		{
//			Point point = (Point) obj;
//
//			if (point.getDimensionality () != getDimensionality ())
//				return false;
//			for (int i = 0; i < getDimensionality (); i++)
//				if (!point.getCoord (i).equals (getCoord (i)))
//					return false;
//
//			return true;
//		}
//
//		return false;
//	}

	@Override
	public Point clone ()
	{
		return new Point (this);
	};

	/**
	 * Returns a zero vector of dimension <code>nDim</code>.
	 * @param nDim The dimension of the vector
	 * @return
	 */
	public static Point getZero (int nDim)
	{
		int[] rgCoords = new int[nDim];
		Arrays.fill (rgCoords, 0);
		return new Point (rgCoords);
	}
}
