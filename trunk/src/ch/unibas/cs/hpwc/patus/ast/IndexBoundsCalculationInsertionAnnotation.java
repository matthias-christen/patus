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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cetus.hir.Annotation;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class IndexBoundsCalculationInsertionAnnotation extends Annotation
{
	private static final long serialVersionUID = 1L;


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The map containing all the instantiations of this class, grouped by subdomain iterators
	 */
	private static Map<String, List<IndexBoundsCalculationInsertionAnnotation>> m_mapAnnotations = new HashMap<> ();

	/**
	 * The iterator of the parent loop
	 */
	private SubdomainIdentifier m_sdidIterator;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates the annotation for the subdomain iterator <code>loop</code>.
	 * @param loop
	 */
	public IndexBoundsCalculationInsertionAnnotation (SubdomainIterator loop)
	{
		m_sdidIterator = loop.getIterator ();

		// add the new instance to the map
		List<IndexBoundsCalculationInsertionAnnotation> listAnnotations = m_mapAnnotations.get (loop.getIterator ().getName ());
		if (listAnnotations == null)
			m_mapAnnotations.put (loop.getIterator ().getName (), listAnnotations = new ArrayList<> ());
		listAnnotations.add (this);
	}

	/**
	 * Determines whether the index calculations for loop <code>loop</code> are
	 * to be placed below this annotation.
	 * @param loop The loop to examine
	 * @return <code>true</code> iff the index calculations for <code>loop</code>
	 * 	are to be placed below this annotation
	 */
	public boolean isIndexBoundCalculationLocationFor (SubdomainIterator loop)
	{
		return loop.getDomainIdentifier ().equals (m_sdidIterator);
	}

	/**
	 * Returns a list of all the annotation locations where index calculations are to be placed.
	 * @param loop The loop for which the index calculation locations are determined
	 * @return A list of annotations below which the index calculations occur, or <code>null</code>
	 * 	if the calculations are to be placed in the initialization block
	 */
	public static List<IndexBoundsCalculationInsertionAnnotation> getIndexBoundCalculationLocationsFor (SubdomainIterator loop)
	{
		return m_mapAnnotations.get (loop.getDomainIdentifier ().getName ());
	}

	@Override
	public String toString ()
	{
		return StringUtil.concat ("/* Index bounds calculations for iterators in ", m_sdidIterator.toString (), " */");
	}
}
