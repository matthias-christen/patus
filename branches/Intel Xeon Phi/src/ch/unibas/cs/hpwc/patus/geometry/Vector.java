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
import java.util.Iterator;
import java.util.NoSuchElementException;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.IntArray;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class Vector extends Primitive implements Iterable<Expression>
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The coordinate array
	 */
	protected Expression[] m_rgCoords;

	/**
	 * Flag indicating whether the coordinates have been simplified
	 */
	protected boolean m_bCoordsSimplified;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates a new vector with coordinates set to the origin.
	 */
	public Vector ()
	{
		//ensureDimensionality (GlobalConstants.DIMENSIONALITY);
		m_bCoordsSimplified = false;
	}

	public Vector (byte nDimensionality)
	{
		this (IntArray.getArray (nDimensionality, 0));
	}

	/**
	 * Creates a vector with coordinates set to the elements of <code>rgCoords</code>.
	 * @param rgCoords The coordinates
	 */
	public Vector (int... rgCoords)
	{
		setCoords (rgCoords);
	}

	public Vector (byte nDimensionality, int nFillValue)
	{
		int[] rgCoords = new int[nDimensionality];
		Arrays.fill (rgCoords, nFillValue);
		setCoords (rgCoords);
	}

	/**
	 * Creates a vector with coordinates set to the elements of <code>rgCoords</code>.
	 * @param rgCoords The coordinate expressions
	 */
	public Vector (Expression... rgCoords)
	{
		setCoords (rgCoords);
	}

	public Vector (byte nDimensionality, Expression exprFillValue)
	{
		Expression[] rgCoords = new Expression[nDimensionality];
		for (int i = 0; i < nDimensionality; i++)
			rgCoords[i] = exprFillValue == null ? null : exprFillValue.clone ();
		setCoords (rgCoords);
	}

	/**
	 * Copy constructor.
	 * @param v The vector to copy
	 */
	public Vector (Vector v)
	{
		// note that setCoords clones the expressions
		setCoords (v.getCoords ());
	}

	/**
	 * Creates a vector between the start point <code>ptStart</code> and the end point <code>ptEnd</code>.
	 * @param ptStart The start point
	 * @param ptEnd The end point
	 */
	public Vector (Point ptStart, Point ptEnd)
	{
		this (ptEnd);
		subtract (ptStart);
	}

	/**
	 * Returns the coordinates.
	 * @return
	 */
	public Expression[] getCoords ()
	{
		// simplify if required and return the coordinates
		simplify ();
		return m_rgCoords;
	}

	/**
	 * Returns the <code>nDim</code>-th coordinate. If <code>nDim</code>
	 * is higher than the dimensionality of the vector, a {@link NoSuchElementException}
	 * is thrown.
	 * @param nDim
	 * @return
	 */
	public Expression getCoord (int nDim)
	{
		if (nDim >= getDimensionality ())
			throw new NoSuchElementException (StringUtil.concat ("No entry in vector ", toString (), " at coordinate ", nDim));

		simplify ();
		return m_rgCoords[nDim];
	}

	/**
	 * Sets the vector's coordinates.
	 * @param rgCoords The coordinates
	 */
	public void setCoords (int... rgCoords)
	{
		ensureDimensionality ((byte) rgCoords.length);
		for (int i = 0; i < getDimensionality (); i++)
			m_rgCoords[i] = new IntegerLiteral (i < rgCoords.length ? rgCoords[i] : 0);

		// coordinates are integers, can't be simplified any further
		m_bCoordsSimplified = true;
	}

	/**
	 * Sets the vector's coordinates.
	 * @param rgCoords The coordinates expressions
	 */
	public void setCoords (Expression... rgCoords)
	{
		if (rgCoords == null)
		{
			m_rgCoords = null;
			m_bCoordsSimplified = true;
		}
		else
		{
			ensureDimensionality ((byte) rgCoords.length);
			for (int i = 0; i < getDimensionality (); i++)
				m_rgCoords[i] = rgCoords[i] == null ? null : (i < rgCoords.length ? (Expression) rgCoords[i].clone () : new IntegerLiteral (0));

			m_bCoordsSimplified = false;
		}
	}

	/**
	 * Sets the <code>nDim</code>-th coordinate to <code>nValue</code>.
	 * @param nDim The number of the coordinate
	 * @param nValue The new value
	 */
	public void setCoord (int nDim, int nValue)
	{
		//ensureDimensionality (nDim + 1);
		// TODO: fix ensure dimensionality!!!

		m_rgCoords[nDim] = new IntegerLiteral (nValue);

		// don't change the simplified flag, if already simplified, everything is ok
		// since the integer can't be simplified any further
	}

	/**
	 * Sets the <code>nDim</code>-th coordinate to <code>exprValue</code>.
	 * @param nDim The number of the coordinate
	 * @param exprValue The new value
	 */
	public void setCoord (int nDim, Expression exprValue)
	{
		//ensureDimensionality (nDim + 1);
		// TODO: fix ensure dimensionality!!!

		if (exprValue == null)
		{
			m_rgCoords[nDim] = null;
			m_bCoordsSimplified = true;
		}
		else
		{
			m_rgCoords[nDim] = exprValue.clone ();
			m_bCoordsSimplified = false;
		}
	}

	/**
	 * Moves the vector by adding a {@link Vector} object.
	 * @param v The vector to add to this vector
	 */
	public void add (Vector v)
	{
		ensureDimensionality (v.getDimensionality ());

		for (int i = 0; i < getDimensionality (); i++)
		{
			if (i < v.getDimensionality () && m_rgCoords[i] != null && v.getCoord (i) != null)
			{
				m_rgCoords[i] = new BinaryExpression (
					m_rgCoords[i].clone (),
					BinaryOperator.ADD,
					v.getCoord (i).clone ());
				m_bCoordsSimplified = false;
			}
		}
	}

	public void add (Expression... rgAdd)
	{
		ensureDimensionality ((byte) rgAdd.length);

		for (int i = 0; i < getDimensionality (); i++)
		{
			if (i < rgAdd.length && m_rgCoords[i] != null && rgAdd[i] != null)
			{
				m_rgCoords[i] = new BinaryExpression (
					m_rgCoords[i].clone (),
					BinaryOperator.ADD,
					rgAdd[i].clone ());
				m_bCoordsSimplified = false;
			}
		}
	}

	public void add (int... rgAdd)
	{
		ensureDimensionality ((byte) rgAdd.length);

		for (int i = 0; i < getDimensionality (); i++)
		{
			if (i < rgAdd.length && m_rgCoords[i] != null)
			{
				m_rgCoords[i] = new BinaryExpression (
					m_rgCoords[i].clone (),
					BinaryOperator.ADD,
					new IntegerLiteral (rgAdd[i]));
				m_bCoordsSimplified = false;
			}
		}
	}

	/**
	 * Moves the vector by subtracting a {@link Vector} object.
	 * @param v The vector to subtract from this vector
	 */
	public void subtract (Vector v)
	{
		ensureDimensionality (v.getDimensionality ());

		for (int i = 0; i < getDimensionality (); i++)
		{
			if (i < v.getDimensionality () && m_rgCoords[i] != null && v.getCoord (i) != null)
			{
				m_rgCoords[i] = new BinaryExpression (
					m_rgCoords[i].clone (),
					BinaryOperator.SUBTRACT,
					v.getCoord (i).clone ());
				m_bCoordsSimplified = false;
			}
		}
	}

	public void subtract (Expression... rgSubtract)
	{
		ensureDimensionality ((byte) rgSubtract.length);

		for (int i = 0; i < getDimensionality (); i++)
		{
			if (i < rgSubtract.length && m_rgCoords[i] != null && rgSubtract[i] != null)
			{
				m_rgCoords[i] = new BinaryExpression (
					m_rgCoords[i].clone (),
					BinaryOperator.SUBTRACT,
					rgSubtract[i].clone ());
				m_bCoordsSimplified = false;
			}
		}
	}

	public void subtract (int... rgSubtract)
	{
		ensureDimensionality ((byte) rgSubtract.length);

		for (int i = 0; i < getDimensionality (); i++)
		{
			if (i < rgSubtract.length && m_rgCoords[i] != null)
			{
				m_rgCoords[i] = new BinaryExpression (
					m_rgCoords[i].clone (),
					BinaryOperator.SUBTRACT,
					new IntegerLiteral (rgSubtract[i]));
				m_bCoordsSimplified = false;
			}
		}
	}

	/**
	 * Scales the vector by a (scalar) factor of <code>exprScale</code>.
	 * @param exprScale The scaling factor
	 */
	public void scale (Expression exprScale)
	{
		if (exprScale == null || ExpressionUtil.isValue (exprScale, 1))
			return;

		for (int i = 0; i < getDimensionality (); i++)
			if (m_rgCoords[i] != null)
				m_rgCoords[i] = new BinaryExpression (m_rgCoords[i].clone (), BinaryOperator.MULTIPLY, exprScale.clone ());
	}

	@Override
	public void ensureDimensionality (byte nDimensionality)
	{
		m_rgCoords = createArray (m_rgCoords, nDimensionality);
		super.ensureDimensionality (nDimensionality);
	}

	/**
	 * Simplifies the coordinate expressions (if necessary).
	 */
	protected void simplify ()
	{
		// we only need to simplify if we haven't done it already
		if (m_bCoordsSimplified)
			return;

		// do a simplify run
		// (note that Symbolic#simplify is smart enough not to do simplifications on literals and identifiers
		for (int i = 0; i < getDimensionality (); i++)
			if (m_rgCoords[i] != null)
				m_rgCoords[i] = Symbolic.simplify (m_rgCoords[i].clone (), Symbolic.ALL_VARIABLES_INTEGER);

		m_bCoordsSimplified = true;
	}

	@Override
	public Iterator<Expression> iterator ()
	{
		return new Iterator<Expression>()
		{
			private int m_nIdx = 0;

			@Override
			public boolean hasNext ()
			{
				return m_nIdx < getDimensionality ();
			}

			@Override
			public Expression next ()
			{
				simplify ();
				return m_rgCoords[m_nIdx++];
			}

			@Override
			public void remove ()
			{
			}
		};
	}

	@Override
	public boolean equals (Object obj)
	{
		if (obj instanceof Vector)
		{
			Vector v = (Vector) obj;

			if (v.getDimensionality () != getDimensionality ())
				return false;

			simplify ();
			v.simplify ();
			for (int i = 0; i < getDimensionality (); i++)
				if (!v.getCoord (i).equals (getCoord (i)))
					return false;

			return true;
		}

		return false;
	}

	@Override
	public int hashCode ()
	{
		simplify ();
		int nHashCode = 0;
		for (Expression e : m_rgCoords)
			nHashCode += e == null ? 0 : e.hashCode ();

		return nHashCode;
	}

	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder ("[");

		// add the minimum coordinates to the string
		for (int i = 0; i < getDimensionality (); i++)
		{
			if (i > 0)
				sb.append (", ");
			sb.append (m_rgCoords[i]);
		}
		sb.append ("]");

		return sb.toString ();
	}

	@Override
	protected Vector clone ()
	{
		return new Vector (this);
	}

	/**
	 * Returns a zero vector of dimension <code>nDim</code>.
	 * @param nDim The dimension of the vector
	 * @return
	 */
	public static Vector getZeroVector (byte nDim)
	{
		return new Vector (nDim, 0);
	}

	/**
	 * Returns the vector (1, 1, ..., 1) of dimension <code>nDim</code>.
	 * @param nDim The dimension of the vector
	 * @return
	 */
	public static Vector getOnesVector (byte nDim)
	{
		return new Vector (nDim, 1);
	}
}
