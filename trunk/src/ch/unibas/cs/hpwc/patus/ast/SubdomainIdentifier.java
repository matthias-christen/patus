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

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.Symbol;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.geometry.Subdomain;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.AnalyzeTools;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class SubdomainIdentifier extends Identifier
{
	///////////////////////////////////////////////////////////////////
	// Static Members

	/**
	 * Default print method for Identifier object
	 */
	private static Method class_print_method;

	// Assigns default print method
	static
	{
		Class<?>[] params = new Class<?>[2];

		try
		{
			params[0] = SubdomainIdentifier.class;
			params[1] = PrintWriter.class;
			SubdomainIdentifier.class_print_method = params[0].getMethod ("defaultPrint", params);
		}
		catch (NoSuchMethodException e)
		{
			throw new InternalError (e.getMessage ());
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The subdomain referenced by this identifier
	 */
	private Subdomain m_subdomain;

	/**
	 * The temporal index (including the time index variable)
	 */
	private Expression m_exprTemporalIndex;

	/**
	 * The spatial offset (relative to the origin), an array of {@link Expression}s
	 */
	private Expression[] m_rgSpatialOffset;

	/**
	 * The loop index variable of the time loop
	 */
	private IDExpression m_idTimeIndexVariable;

	/**
	 * The vector indices
	 */
	private List<Expression> m_listVectorIndices;

	/**
	 * A list of all the indices
	 */
	private List<Expression> m_listIndices;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Constructs a new subdomain identifier and associates it with the grid <code>subdomain</code>.
	 * @param strName The identifier name
	 * @param subdomain The grid to associate this identifier with
	 */
	public SubdomainIdentifier (String strName, Subdomain subdomain)
	{
		super (strName);
		m_subdomain = subdomain;

		m_idTimeIndexVariable = null;

		// create the index list
		m_listVectorIndices = new ArrayList<> ();
		m_listIndices = new ArrayList<> ();

		object_print_method = SubdomainIdentifier.class_print_method;
	}

	/**
	 * Constructs a new subdomain identifier and associates it with the grid <code>subdomain</code>.
	 * @param strName The identifier name
	 * @param subdomain The grid to associate this identifier with
	 */
	public SubdomainIdentifier (Symbol symbol, Subdomain subdomain)
	{
		super (symbol);
		m_subdomain = subdomain;

		m_idTimeIndexVariable = null;

		// create the index list
		m_listVectorIndices = new ArrayList<> ();
		m_listIndices = new ArrayList<> ();

		object_print_method = SubdomainIdentifier.class_print_method;
	}

	/**
	 * Finds the identifier of the innermost enclosing temporal loop.
	 */
	private void findTimeIndexVariable ()
	{
		// nothing to do if the time index variable has already been found
		if (m_idTimeIndexVariable != null)
			return;

		// the time index of the "for t=1..t_max" loop
		IDExpression idTimeIndex = null;
		RangeIterator loopOuterMostTimeLoop = null;

		// find this time index
		Traversable trvParent = this;
		while (trvParent != null)
		{
			if (trvParent instanceof RangeIterator)
			{
				RangeIterator loop = (RangeIterator) trvParent;
				if (loop.isMainTemporalIterator ())
				{
					idTimeIndex = loop.getLoopIndex ();
					loopOuterMostTimeLoop = loop;
					break;
				}
			}

			trvParent = trvParent.getParent ();
		}

		// exit if time index has not been found
		if (idTimeIndex == null || loopOuterMostTimeLoop == null)
			return;

		// find the inner most time loop
		List<IDExpression> listTimeIndices = new LinkedList<> ();
		listTimeIndices.add (idTimeIndex);
		RangeIterator loopInnerMostTimeLoop = loopOuterMostTimeLoop;
		for ( ; ; )
		{
			// remember the current loop that has been found
			RangeIterator loopPrevInnerMost = loopInnerMostTimeLoop;

			// find a time loop contained within the previously found ones
			trvParent = this;
			while (trvParent != null && trvParent != loopInnerMostTimeLoop && loopPrevInnerMost == loopInnerMostTimeLoop)
			{
				if (trvParent instanceof RangeIterator)
				{
					RangeIterator loop = (RangeIterator) trvParent;

					for (IDExpression id : listTimeIndices)
					{
						if (AnalyzeTools.dependsExpressionOnIdentifier (loop.getStart (), id))
						{
							loopInnerMostTimeLoop = loop;
							break;
						}
						if (AnalyzeTools.dependsExpressionOnIdentifier (loop.getEnd (), id))
						{
							loopInnerMostTimeLoop = loop;
							break;
						}
					}
				}

				trvParent = trvParent.getParent ();
			}


			// if the loop hasn't changed, exit
			if (loopPrevInnerMost == loopInnerMostTimeLoop)
				break;
		}

		// set the time index variable
		m_idTimeIndexVariable = loopInnerMostTimeLoop.getLoopIndex ();
	}

	/**
	 * Returns the spatial index, an array of {@link Expression}s.
	 * @return The spatial index
	 */
	public Expression[] getSpatialOffset ()
	{
		return m_rgSpatialOffset;
	}

	/**
	 *
	 * @param rgIndex
	 */
	public void setSpatialOffset (Expression... rgSpatialOffset)
	{
		if (rgSpatialOffset == null)
			m_rgSpatialOffset = null;
		else
		{
			m_rgSpatialOffset = new Expression[rgSpatialOffset.length];
			for (int i = 0; i < m_rgSpatialOffset.length; i++)
				m_rgSpatialOffset[i] = rgSpatialOffset[i].clone ();
		}
	}

	/**
	 *
	 * @param exprTemporalIndex
	 */
	public void setTemporalIndex (Expression exprTemporalIndex)
	{
		m_exprTemporalIndex = Symbolic.simplify (exprTemporalIndex, Symbolic.ALL_VARIABLES_INTEGER);
	}

	/**
	 * Offsets the temporal index by <code>exprTemporalOffset</code>.
	 * @param exprTemporalOffset The temporal offset
	 */
	public void offsetTemporalIndex (Expression exprTemporalOffset)
	{
		m_exprTemporalIndex = Symbolic.simplify (
			new BinaryExpression (m_exprTemporalIndex.clone (), BinaryOperator.ADD, exprTemporalOffset),
			Symbolic.ALL_VARIABLES_INTEGER
		);
	}

	/**
	 * Returns the temporal index.
	 * @return The temporal index
	 */
	public Expression getTemporalIndex ()
	{
		return m_exprTemporalIndex;
	}

	/**
	 * Returns the temporal offset, e.g., if the time variable is t and the subdomain identifier's
	 * temporal index is t+1, the offset is 1.
	 * @return The temporal offset
	 */
	public Expression getTemporalOffset ()
	{
		if (m_idTimeIndexVariable == null)
			return m_exprTemporalIndex;

		return Symbolic.simplify (
			new BinaryExpression (m_exprTemporalIndex, BinaryOperator.SUBTRACT, m_idTimeIndexVariable),
			Symbolic.ALL_VARIABLES_INTEGER
		);
	}

	/**
	 * Determines whether the temporal index is zero.
	 * @return <code>true</code> iff the temporal index is zero
	 */
	public boolean isTemporalOffsetZero ()
	{
		findTimeIndexVariable ();
		return m_exprTemporalIndex.equals (m_idTimeIndexVariable);
	}

	public List<Expression> getVectorIndices ()
	{
		return m_listVectorIndices;
	}

	/**
	 * Sets the list of vector indices. Note that this function overwrites
	 * already existing vector indices.
	 * @param listVectorIndex The new list of vector indices
	 */
	public void setVectorIndex (List<Expression> listVectorIndex)
	{
		m_listVectorIndices = listVectorIndex;
	}

	/**
	 * Appends a vector index to the list of existing vector indices.
	 * @param exprIndex The vector index to add
	 */
	public void addVectorIndex (Expression exprIndex)
	{
		m_listVectorIndices.add (exprIndex);
	}

	/**
	 * Returns the grid object.
	 * @return The grid
	 */
	public Subdomain getSubdomain ()
	{
		return m_subdomain;
	}

	/**
	 * Returns the spatial dimensionality of the object this identifier represents.
	 * @return The spatial dimensionality of the object attached to this identifier
	 */
	public byte getDimensionality ()
	{
		return m_subdomain.getBox ().getDimensionality ();
	}

	@Override
	public String toString ()
	{
		/* XXX commented out after commenting out m_rgSpatialIndex
		StringBuilder sbSpatialIndex = m_rgSpatialIndex != null ? new StringBuilder ("(") : null;
		if (sbSpatialIndex != null)
		{
			boolean bFirst = true;
			for (Expression expr : m_rgSpatialIndex)
			{
				if (!bFirst)
					sbSpatialIndex.append (", ");
				sbSpatialIndex.append (expr == null ? ":" : expr.toString ());
				bFirst = false;
			}
			sbSpatialIndex.append (")");
		}
		*/

		return StringUtil.concat (
			getName (),
			"[t=", m_exprTemporalIndex == null ? "?" : m_exprTemporalIndex.toString (), "]",//", ",
			//"s=", m_rgSpatialIndex == null ? "?" : (sbSpatialIndex == null ? m_rgSpatialIndex.toString () : sbSpatialIndex.toString ()), "]",
			m_listVectorIndices == null ? "" : (m_listVectorIndices.size () == 0 ? "[0]" : m_listVectorIndices.toString ()));
	}

	@Override
	public boolean equals (Object o)
	{
		if (o == null)
			return false;
		if (!(o instanceof SubdomainIdentifier))
			return false;

		SubdomainIdentifier sdidOther = (SubdomainIdentifier) o;

		// compare symbols
		Symbol symbol = getSymbol ();
		Symbol symbolOther = sdidOther.getSymbol ();
		if (symbol == null)
		{
			if (symbolOther != null)
				return false;
		}
		else
		{
			if (!symbol.equals (sdidOther.getSymbol ()))
				return false;
		}

		// compare indices
		/*
		if (m_listIndices == null && (sdidOther.m_listIndices != null || sdidOther.m_listIndices.size () > 0))
			return false;
		if (m_listIndices.size () != sdidOther.m_listIndices.size ())
			return false;
		Iterator<Expression> itOther = sdidOther.m_listIndices.iterator ();
		for (Expression expr : m_listIndices)
		{
			Expression exprOther = itOther.next ();
			if (!expr.equals (exprOther))
				return false;
		}*/

		// compare the temporal index
		/*
		if (m_exprTemporalIndex == null)
		{
			if (sdidOther.getTemporalIndex () != null)
				return false;
		}
		else*/
//		/**/if (m_exprTemporalIndex != null && sdidOther.getTemporalIndex () != null)
//		{
//			if (m_exprTemporalIndex != null && !m_exprTemporalIndex.equals (sdidOther.getTemporalIndex ()))
//				return false;
//		}

		// don't compare the spatial index

		// compare the vectorial indices
//		/**/if (m_listVectorIndices != null && sdidOther.m_listVectorIndices != null)
//		{
//			/**/if (m_listVectorIndices.size () > 0 && sdidOther.m_listVectorIndices.size () > 0)
//			{
//
//				if (m_listVectorIndices == null && (sdidOther.m_listVectorIndices != null || sdidOther.m_listVectorIndices.size () > 0))
//					return false;
//				if (m_listVectorIndices.size () != sdidOther.m_listVectorIndices.size ())
//					return false;
//				Iterator<Expression> itOther = sdidOther.m_listVectorIndices.iterator ();
//				for (Expression expr : m_listVectorIndices)
//				{
//					Expression exprOther = itOther.next ();
//					if (!expr.equals (exprOther))
//						return false;
//				}
//			}
//		}

		return true;
	}

	@Override
	public int hashCode ()
	{
		Symbol s = getSymbol ();
		int nHash = s == null ? 0 : s.hashCode ();

		/*
		if (m_listIndices != null)
			for (Expression e : m_listIndices)
				nHash += e.hashCode ();
		*/

		/*
		if (m_exprTemporalIndex != null)
			nHash += m_exprTemporalIndex.hashCode ();
		if (m_listVectorIndices != null)
			for (Expression e : m_listVectorIndices)
				nHash += e.hashCode ();
		*/

		return nHash;
	}

	@Override
	public SubdomainIdentifier clone ()
	{
		SubdomainIdentifier sgid = (SubdomainIdentifier) super.clone ();

		sgid.m_subdomain = m_subdomain;

		sgid.m_exprTemporalIndex = m_exprTemporalIndex == null ? null : m_exprTemporalIndex.clone ();
		sgid.m_idTimeIndexVariable = m_idTimeIndexVariable == null ? null : m_idTimeIndexVariable.clone ();

		/*
		if (m_rgSpatialIndex != null)
		{
			sgid.m_rgSpatialIndex = new Expression[m_rgSpatialIndex.length];
			for (int i = 0; i < m_rgSpatialIndex.length; i++)
				sgid.m_rgSpatialIndex[i] = m_rgSpatialIndex[i] == null ? null : m_rgSpatialIndex[i].clone ();
		}
		*/

		sgid.m_listVectorIndices = new ArrayList<> (m_listVectorIndices.size ());
		for (Expression exprVectorIndex : m_listVectorIndices)
			sgid.m_listVectorIndices.add (exprVectorIndex == null ? null : exprVectorIndex.clone ());

		sgid.m_listIndices = new ArrayList<> (m_listIndices.size ());
		for (Expression exprIndex : m_listIndices)
			sgid.m_listIndices.add (exprIndex == null ? null : exprIndex.clone ());

		return sgid;
	}

	/**
	 * Prints an identifier to a stream.
	 * @param i The identifier to print.
	 * @param o The writer on which to print the identifier.
	 */
	public static void defaultPrint (SubdomainIdentifier i, PrintWriter o)
	{
		o.print (i.toString ());
	}
}
