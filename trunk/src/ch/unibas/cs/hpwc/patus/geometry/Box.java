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

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import ch.unibas.cs.hpwc.patus.codegen.IMask;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * A bounding box.
 * @author Matthias-M. Christen
 */
public class Box extends Primitive
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The minimum coordinates (top left corner)
	 */
	private Point m_ptMin;

	/**
	 * The maximum coordinates (bottom right corner)
	 */
	private Point m_ptMax;

	private Point m_ptMinOld;
	private Point m_ptMaxOld;

	/**
	 * The size of the box. This is a calculated value.
	 * It is calculated lazily when {@link Box#getSize()} is called
	 * and cached for subsequent calls. If min and max coordinates are
	 * changed (changing the size of the box), the variable is invalidated.
	 */
	private Size m_size;

	/**
	 * The volume of the box
	 */
	private Expression m_exprVolume;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates an &quot;empty&quot; box (with min and max coordinates
	 * set to 0).
	 */
	public Box ()
	{
		m_ptMin = new Point ();
		m_ptMax = new Point ();
		//ensureDimensionality (GlobalConstants.DIMENSIONALITY);

		m_ptMinOld = null;
		m_ptMaxOld = null;

		m_size = null;
		m_exprVolume = null;
	}

	/**
	 * Creates a box.
	 * @param rgMin The minimum coordinates (top left corner)
	 * @param rgMax The maximum coordinates (bottom right corner)
	 */
	public Box (int[] rgMin, int[] rgMax)
	{
		super ();
		setMin (new Point (rgMin));
		setMax (new Point (rgMax));

		m_ptMinOld = null;
		m_ptMaxOld = null;

		m_size = null;
		m_exprVolume = null;
		ensureDimensionality ((byte) Math.max (m_ptMin.getDimensionality (), m_ptMax.getDimensionality ()));
	}

	/**
	 * Creates a box.
	 * @param rgMin The minimum coordinates (top left corner)
	 * @param rgMax The maximum coordinates (bottom right corner)
	 */
	public Box (Expression[] rgMin, Expression[] rgMax)
	{
		super ();
		setMin (new Point (rgMin));
		setMax (new Point (rgMax));

		m_ptMinOld = null;
		m_ptMaxOld = null;

		m_size = null;
		m_exprVolume = null;
		ensureDimensionality ((byte) Math.max (m_ptMin.getDimensionality (), m_ptMax.getDimensionality ()));
	}

	/**
	 * Constructs a box from the left upper point <code>ptMin</code>
	 * and the lower right point <code>ptMax</code>.
	 * @param ptMin
	 * @param ptMax
	 */
	public Box (Point ptMin, Point ptMax)
	{
		super ();
		setMin (ptMin);
		setMax (ptMax);

		m_ptMinOld = null;
		m_ptMaxOld = null;

		m_size = null;
		m_exprVolume = null;
		ensureDimensionality ((byte) Math.max (m_ptMin.getDimensionality (), m_ptMax.getDimensionality ()));
	}

	/**
	 * Constructs a box from the lower left point <code>ptMin</code>
	 * and the size <code>size</code>.
	 * @param ptMin
	 * @param size
	 */
	public Box (Point ptMin, Size size)
	{
		super ();
		setMin (ptMin);
		m_size = new Size (size);
		m_exprVolume = null;

		Expression[] rgMax = new Expression[ptMin.getDimensionality ()];
		for (int i = 0; i < ptMin.getDimensionality (); i++)
		{
			rgMax[i] = Symbolic.simplify (
				ExpressionUtil.decrement (new BinaryExpression (ptMin.getCoord (i).clone (), BinaryOperator.ADD, m_size.getCoord (i).clone ())),
				Symbolic.ALL_VARIABLES_INTEGER
			);
		}
		setMax (new Point (rgMax));

		ensureDimensionality ((byte) Math.max (m_ptMin.getDimensionality (), m_ptMax.getDimensionality ()));

		m_ptMinOld = null;
		m_ptMaxOld = null;
	}

	/**
	 * Copy constructor.
	 * @param box The box to copy
	 */
	public Box (Box box)
	{
		super ();

		m_ptMin = box.m_ptMin.clone ();
		m_ptMax = box.m_ptMax.clone ();
		m_ptMinOld = null;
		m_ptMaxOld = null;
		m_size = box.m_size == null ? null : box.m_size.clone ();
		m_exprVolume = box.m_exprVolume == null ? null : box.m_exprVolume.clone ();
		ensureDimensionality (box.getDimensionality ());
	}

	/**
	 * Returns the minimum coordinates (the lower left corner)
	 * @return
	 */
	public Point getMin ()
	{
		return m_ptMin;
	}

	/**
	 * Sets the min coordinates (the upper left corner).
	 * @param ptMin The min coordinate
	 */
	public void setMin (Point ptMin)
	{
		m_ptMin = ptMin;
		//m_ptMin.ensureDimensionality (GlobalConstants.DIMENSIONALITY);

		m_size = null;
		m_exprVolume = null;
	}

	/**
	 * Returns the max coordinates (the lower right corner).
	 * @return The max coordinates
	 */
	public Point getMax ()
	{
		return m_ptMax;
	}

	/**
	 * Sets the max coordinates (the lower right corner).
	 * @param ptMax The max coordinates
	 */
	public void setMax (Point ptMax)
	{
		m_ptMax = ptMax;
		//m_ptMax.ensureDimensionality (GlobalConstants.DIMENSIONALITY);

		m_size = null;
		m_exprVolume = null;
	}

	/**
	 * Returns the size (the dimensions) of the box.
	 * @return
	 */
	public Size getSize ()
	{
		if (m_size == null)
			m_size = new Size (m_ptMin, m_ptMax);

		return m_size;
	}

	/**
	 * Sets the box's size leaving at the current position (i.e. doesn't change
	 * the min coordinates, but changes the max coordinates).
	 * @param size The new size of the box
	 */
	public void setSize (Size size)
	{
		// make sure that the min/max coordinates are large enough
		//size.ensureDimensionality (GlobalConstants.DIMENSIONALITY);
		ensureDimensionality (size.getDimensionality ());

		// set max to min+size
		m_ptMax = new Point (m_ptMin);
		m_ptMax.add (size);
		m_ptMax.subtract (Vector.getOnesVector (m_ptMax.getDimensionality ()));

		m_size = size;
		m_exprVolume = null;
	}

	/**
	 * Determines whether this box is a point.
	 * @return <code>true</code> iff this box is a point
	 */
	public boolean isPoint ()
	{
		Size size = getSize ();
		for (Expression exprCoord : size.getCoords ())
			if (!ExpressionUtil.isValue (exprCoord, 1))
				return false;
		return true;
	}

	/**
	 * Moves the box so that its new min coordinates are at
	 * <code>ptMin</code> without changing its size.
	 * @param ptMin The new min coordinates
	 */
	public void moveTo (Point ptMin)
	{
		//ptMin.ensureDimensionality (GlobalConstants.DIMENSIONALITY);
		ensureDimensionality (ptMin.getDimensionality ());

		// set the max coords to the old min coords plus an offset
		m_ptMax = new Point (m_ptMin);
		m_ptMax.add (new Size (m_ptMin, ptMin));
		m_ptMax.subtract (Vector.getOnesVector (m_ptMax.getDimensionality ()));
		m_ptMin = new Point (ptMin);

		// (don't touch size!)
	}

	/**
	 * Moves the box by <code>szOffset</code>.
	 * @param sizeOffset The box movement
	 */
	public void offset (Vector vecOffset)
	{
		ensureDimensionality (vecOffset.getDimensionality ());
		m_ptMin.add (vecOffset);
		m_ptMax.add (vecOffset);

		// (don't touch size!)
	}

	/**
	 * Offsets the box, but remembers the old coordinates.
	 * The offset can be undone by invoking {@link Box#undoOffset()}.
	 * @param vecOffset
	 */
	public void offsetTentatively (Vector vecOffset)
	{
		m_ptMinOld = m_ptMin.clone ();
		m_ptMaxOld = m_ptMax.clone ();

		ensureDimensionality (vecOffset.getDimensionality ());
		m_ptMin.add (vecOffset);
		m_ptMax.add (vecOffset);
	}

	/**
	 * Undoes the offset previously carried out by {@link Box#offsetTentatively(Size)}.
	 */
	public void undoOffset ()
	{
		if (m_ptMinOld != null && m_ptMaxOld != null)
		{
			m_ptMin = m_ptMinOld;
			m_ptMax = m_ptMaxOld;
		}
	}

	/**
	 *
	 * @param border
	 */
	public void addBorder (Border border)
	{
		enlarge (border.getMin (), border.getMax ());
	}

//	/**
//	 * Enlarges the box by <code>nTimes</code> the <code>border</code>.
//	 * @param exprTimes The factor by which to enlarge the box
//	 * @param border The border by which to enlarge the box
//	 */
//	public void addBorder (Expression exprTimes, Border border)
//	{
//		Size sizeMin = border.getMin ();
//		Size sizeMax = border.getMax ();
//
//		Expression[] rgExprMinEnlarged = new Expression[border.getDimensionality ()];
//		Expression[] rgExprMaxEnlarged = new Expression[border.getDimensionality ()];
//		for (int i = 0; i < border.getDimensionality (); i++)
//		{
//			rgExprMinEnlarged[i] = new BinaryExpression (sizeMin.getCoord (i).clone (), BinaryOperator.MULTIPLY, exprTimes.clone ());
//			rgExprMaxEnlarged[i] = new BinaryExpression (sizeMax.getCoord (i).clone (), BinaryOperator.MULTIPLY, exprTimes.clone ());
//		}
//
//		enlarge (new Size (rgExprMinEnlarged), new Size (rgExprMaxEnlarged));
//	}

	/**
	 * Enlarges the box by <code>nTimes</code> the <code>border</code>.
	 * @param exprTimes The factor by which to enlarge the box
	 * @param border The border by which to enlarge the box
	 * @param nBorderWithInUnitStrideDirectionMultipleOf Alignment restriction: border width must be a multiple of this number
	 */
	public void addBorder (/*Expression exprTimes,*/ Border border, int nBorderWithInUnitStrideDirectionMultipleOf, IMask mask)
	{
		Size sizeMin = border.getMin ();
		Size sizeMax = border.getMax ();

		Expression[] rgExprMinEnlarged = new Expression[border.getDimensionality ()];
		Expression[] rgExprMaxEnlarged = new Expression[border.getDimensionality ()];

		int[] rgMask = new int[border.getDimensionality ()];
		Arrays.fill (rgMask, 1);
		rgMask = mask.apply (rgMask);

		for (int i = 0; i < border.getDimensionality (); i++)
		{
			if (rgMask[i] == 0)
			{
				rgExprMinEnlarged[i] = new IntegerLiteral (0);
				rgExprMaxEnlarged[i] = new IntegerLiteral (0);
			}
			else
			{
//				if (exprTimes instanceof IntegerLiteral)
//				{
//					int nTimes = (int) ((IntegerLiteral) exprTimes).getValue ();
//					if (nTimes == 0)
//					{
//						rgExprMinEnlarged[i] = new IntegerLiteral (0);
//						rgExprMaxEnlarged[i] = new IntegerLiteral (0);
//					}
//					else if (nTimes == 1)
//					{
						rgExprMinEnlarged[i] = sizeMin.getCoord (i).clone ();
						rgExprMaxEnlarged[i] = sizeMax.getCoord (i).clone ();
//					}
//					else
//					{
//						rgExprMinEnlarged[i] = null;
//						rgExprMaxEnlarged[i] = null;
//					}
//				}
//
//				if (rgExprMinEnlarged[i] == null)
//				{
//					rgExprMinEnlarged[i] = new BinaryExpression (sizeMin.getCoord (i).clone (), BinaryOperator.MULTIPLY, exprTimes.clone ());
//					rgExprMaxEnlarged[i] = new BinaryExpression (sizeMax.getCoord (i).clone (), BinaryOperator.MULTIPLY, exprTimes.clone ());
//				}

				if (i == 0 && nBorderWithInUnitStrideDirectionMultipleOf > 1)
				{
					rgExprMinEnlarged[0] = new BinaryExpression (
						ExpressionUtil.ceil (rgExprMinEnlarged[0], new IntegerLiteral (nBorderWithInUnitStrideDirectionMultipleOf)),
						BinaryOperator.MULTIPLY,
						new IntegerLiteral (nBorderWithInUnitStrideDirectionMultipleOf));
					rgExprMaxEnlarged[0] = new BinaryExpression (
						ExpressionUtil.ceil (rgExprMaxEnlarged[0], new IntegerLiteral (nBorderWithInUnitStrideDirectionMultipleOf)),
						BinaryOperator.MULTIPLY,
						new IntegerLiteral (nBorderWithInUnitStrideDirectionMultipleOf));
				}
			}
		}

		enlarge (new Size (rgExprMinEnlarged), new Size (rgExprMaxEnlarged));
	}

	/**
	 * Enlarges the box at the min coordinates (top left corner) by <code>rgMin</code> along the negative axes
	 * and at the max coordinates (bottom right corner) by <code>rgMax</code> along the positive axes
	 * @param rgMin
	 * @param rgMax
	 */
	public void enlarge (Size sizeMin, Size sizeMax)
	{
		ensureDimensionality ((byte) Math.max (sizeMin.getDimensionality (), sizeMax.getDimensionality ()));
		m_ptMin.subtract (sizeMin);
		m_ptMax.add (sizeMax);

		// size needs to be recalculated
		m_size = null;
		m_exprVolume = null;
	}

	/**
	 * Tests whether the box is large enough to fit the vector <code>rgVector</code> in it.
	 * @param rgVector
	 * @return <code>true</code> if the vector <code>rgVector</code> fits in the box.
	 * 	If the result is not decided (because of symbolics) <code>true</code> is returned.
	 * 	If the vector doesn't fit, <code>false</code> is returned.
	 */
	public boolean fits (int[] rgVector)
	{
		// if the vector is too large, check for zeros in the exceeding dimensions
		// if all coordinates in the exceeding dimensions are zero, go on with the check
		// otherwise the vector doesn't fit
		for (int i = m_size.getDimensionality (); i < rgVector.length; i++)
			if (rgVector[i] != 0)
				return false;

		// check all the dimensions of the box
		int nDim = Math.min (m_size.getDimensionality (), rgVector.length);
		for (int i = 0; i < nDim; i++)
		{
			// if the expression is FALSE, return false: the vector doesn't fit
			// in the other cases (TRUE or UNKNOWN is returned) continue checking
			if (Symbolic.isTrue (
				new BinaryExpression (new IntegerLiteral (rgVector[i]), BinaryOperator.COMPARE_LT, m_size.getCoord (i).clone ()),
				Symbolic.ALL_VARIABLES_POSITIVE) == Symbolic.ELogicalValue.FALSE)
			{
				return false;
			}
		}

		// we couldn't prove that the vector didn't fit: assume it fits...
		return true;
	}

	/**
	 * Checks whether the point <code>point</code> is contained in the box.
	 * @param point The point to check
	 * @return <code>true</code> iff the point <code>point</code> is contained in the box
	 */
	public boolean contains (Point point)
	{
		// if the dimension exceeds the dimension of the box, check whether the exceeding
		// dimensions are zero
		for (int i = m_ptMin.getDimensionality (); i < point.getDimensionality (); i++)
			if (!ExpressionUtil.isZero (point.getCoord (i)))
				return false;

		// check the coordinates of the point...
		int nDim = Math.min (m_ptMin.getDimensionality (), point.getDimensionality ());
		for (int i = 0; i < nDim; i++)
		{
			// check the lower bound
			if (Symbolic.isTrue (
				new BinaryExpression (m_ptMin.getCoord (i), BinaryOperator.COMPARE_LE, point.getCoord (i)),
				Symbolic.ALL_VARIABLES_POSITIVE) == Symbolic.ELogicalValue.FALSE)
			{
				return false;
			}

			// check the upper bound
			if (Symbolic.isTrue (
				new BinaryExpression (point.getCoord (i), BinaryOperator.COMPARE_LT, m_ptMax.getCoord (i)),
				Symbolic.ALL_VARIABLES_POSITIVE) == Symbolic.ELogicalValue.FALSE)
			{
				return false;
			}
		}

		// we couldn't prove that the point is not in the box...
		return true;
	}

	/**
	 * Calculates and returns the volume of the box.
	 * @return
	 */
	public Expression getVolume ()
	{
		if (m_exprVolume == null)
			m_exprVolume = ExpressionUtil.product (getSize ().getCoords ());

		return m_exprVolume.clone ();
	}


	///////////////////////////////////////////////////////////////////
	// Private Methods

	@Override
	protected void ensureDimensionality (byte nDimensionality)
	{
		m_ptMin.ensureDimensionality (nDimensionality);
		m_ptMax.ensureDimensionality (nDimensionality);
		super.ensureDimensionality (nDimensionality);
	}


	///////////////////////////////////////////////////////////////////
	// Object Implementation

	@Override
	public boolean equals (Object obj)
	{
		if (obj instanceof Box)
		{
			Box box = (Box) obj;
			if (!box.getMin ().equals (m_ptMin))
				return false;
			if (!box.getMax ().equals (m_ptMax))
				return false;

			return true;
		}

		return false;
	}

	@Override
	public int hashCode ()
	{
		return m_ptMin.hashCode () + m_ptMax.hashCode ();
	}

	@Override
	public String toString ()
	{
		return StringUtil.concat (m_ptMin, " x ", m_ptMax);
	}

	@Override
	public Box clone ()
	{
		return new Box (this);
	}
}
