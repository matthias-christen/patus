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

import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;

/**
 *
 * @author Matthias-M. Christen
 */
public abstract class Primitive
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The dimensionality of the box
	 */
	private byte m_nDimension;


	///////////////////////////////////////////////////////////////////
	// Implementation

	protected Primitive ()
	{
		m_nDimension = 0;
	}

	/**
	 * Returns the dimensionality of the object.
	 * @return The dimensionality of the object
	 */
	public byte getDimensionality ()
	{
		return m_nDimension;
	}

	///////////////////////////////////////////////////////////////////
	// Private Methods

	/**
	 * Creates the coordinate arrays/enlarges them if they are not large enough.
	 * Subclasses should overwrite this method.
	 * @param nDimensionality The minimum dimensionality that the coordinate
	 * 	arrays must have. If {@link Primive#m_nDimension} is greater, no action is taken
	 */
	protected void ensureDimensionality (byte nDimensionality)
	{
		// enlarge the arrays if they have less than nDimensionalityToAssure elements
		m_nDimension = nDimensionality;
	}

	protected Expression[] createArray (Expression[] rgArray, int nDimensionalityToEnsure)
	{
		// enlarge the array if they have less than nDimensionalityToAssure elements
		if (nDimensionalityToEnsure > m_nDimension)
		{
			Expression[] rgTmp = null;
			int nDimOld = 0;

			if (rgArray != null)
			{
				rgTmp = new Expression[rgArray.length];
				System.arraycopy (rgArray, 0, rgTmp, 0, rgArray.length);
				nDimOld = rgArray.length;
			}
			rgArray = new Expression[nDimensionalityToEnsure];
			if (nDimOld > 0)
				System.arraycopy (rgTmp, 0, rgArray, 0, Math.min (nDimOld, nDimensionalityToEnsure));
			for (int i = nDimOld; i < nDimensionalityToEnsure; i++)
				rgArray[i] = new IntegerLiteral (0);
		}
		else
		{
			// create the array if it doesn't exist yet
			if (rgArray == null)
			{
				rgArray = new Expression[m_nDimension];
				for (int i = 0; i < m_nDimension; i++)
					rgArray[i] = new IntegerLiteral (0);
			}
		}

		return rgArray;
	}
}
