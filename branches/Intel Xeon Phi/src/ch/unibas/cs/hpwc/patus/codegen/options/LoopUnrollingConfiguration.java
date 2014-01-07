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
package ch.unibas.cs.hpwc.patus.codegen.options;

import java.util.ArrayList;
import java.util.List;

import cetus.analysis.LoopTools;
import cetus.hir.Loop;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class LoopUnrollingConfiguration implements Cloneable
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	private static class LoopUnrolling implements Cloneable
	{
		private Loop m_loop;
		private int m_nUnrollingFactor;

		public LoopUnrolling (Loop loop, int nUnrollingFactor)
		{
			m_loop = loop;
			m_nUnrollingFactor = nUnrollingFactor;
		}

		public final Loop getLoop ()
		{
			return m_loop;
		}

		public final int getUnrollingFactor ()
		{
			return m_nUnrollingFactor;
		}

		public void setUnrollingFactor (int nUnrollingFactor)
		{
			m_nUnrollingFactor = nUnrollingFactor;
		}

		@Override
		public String toString ()
		{
			return StringUtil.concat (
				"loop ", LoopTools.getLoopIndexSymbol (m_loop).toString (),
				": unroll ", m_nUnrollingFactor);
		}

		@Override
		public LoopUnrolling clone ()
		{
			return new LoopUnrolling (m_loop, m_nUnrollingFactor);
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private List<LoopUnrolling> m_listUnrollings;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public LoopUnrollingConfiguration ()
	{
		m_listUnrollings = new ArrayList<> ();
	}

	public void addLoopToUnroll (Loop loop, int nUnrollingFactor)
	{
		m_listUnrollings.add (new LoopUnrolling (loop, nUnrollingFactor));
	}

	public void setUnrollingFactor (Loop loop, int nUnrollingFactor)
	{
		LoopUnrolling lu = findLoopUnrolling (loop);
		if (lu != null)
			lu.setUnrollingFactor (nUnrollingFactor);
	}

	public int getUnrollingFactor (Loop loop)
	{
		LoopUnrolling lu = findLoopUnrolling (loop);
		return lu == null ? 1 : lu.getUnrollingFactor ();
	}

	public boolean isLoopUnrolled (Loop loop)
	{
		return findLoopUnrolling (loop) != null;
	}

	private LoopUnrolling findLoopUnrolling (Loop loop)
	{
		for (LoopUnrolling lu : m_listUnrollings)
			if (lu.getLoop () == loop)
				return lu;
		return null;
	}

	@Override
	public String toString ()
	{
		return StringUtil.concat ("{ ", StringUtil.join (m_listUnrollings, ", "), " }");
	}

	@Override
	public LoopUnrollingConfiguration clone ()
	{
		LoopUnrollingConfiguration config = new LoopUnrollingConfiguration ();
		for (LoopUnrolling lu : m_listUnrollings)
			config.m_listUnrollings.add (lu.clone ());
		return config;
	}
}
