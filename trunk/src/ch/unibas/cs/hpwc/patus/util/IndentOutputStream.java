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
package ch.unibas.cs.hpwc.patus.util;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/**
 * @author Matthias-M. Christen
 *
 */
public class IndentOutputStream extends BufferedOutputStream
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private int m_nIndentationLevel;

	private byte[] m_rgTabs;

	private boolean m_bPendingTabsWrite;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * @param out
	 */
	public IndentOutputStream (OutputStream out)
	{
		super (out);

		m_nIndentationLevel = 0;
		m_rgTabs = new byte[0];
		m_bPendingTabsWrite = false;
	}

	private void writeTabs () throws IOException
	{
		if (m_rgTabs.length < m_nIndentationLevel)
		{
			m_rgTabs = new byte[m_rgTabs.length + 10];
			for (int i = 0; i < m_rgTabs.length; i++)
				m_rgTabs[i] = (byte) '\t';
		}

		if (m_nIndentationLevel > 0)
			super.write (m_rgTabs, 0, m_nIndentationLevel);
	}

	@Override
	public void write (byte[] b) throws IOException
	{
		write (b, 0, b.length);
	}

	@Override
	public synchronized void write (int b) throws IOException
	{
		super.write (new byte[] { (byte) b }, 0, 1);
	}

	@Override
	public synchronized void write (byte[] b, int off, int len) throws IOException
	{
		int nLastIdxWritten = off;
		for (int i = off; i < off + len; i++)
		{
			if (b[i] == '\n')
			{
				super.write (b, nLastIdxWritten, i - nLastIdxWritten + 1);
				nLastIdxWritten = i + 1;

				if (i < off + len - 1)
				{
					if (b[i + 1] == '{')
					{
						writeTabs ();
						m_nIndentationLevel++;
					}
					else if (b[i + 1] == '}')
					{
						m_nIndentationLevel--;
						writeTabs ();
					}
					else if (b[i + 1] != '#')
						writeTabs ();
				}
				else
					m_bPendingTabsWrite = true;
			}
			else if (len == 1 && (b[i] == '{' || b[i] == '}'))
			{
				if (b[i] == '}')
					m_nIndentationLevel--;

				if (m_bPendingTabsWrite)
				{
					writeTabs ();
					m_bPendingTabsWrite = false;
				}

				super.write (b, nLastIdxWritten, i - nLastIdxWritten + 1);
				nLastIdxWritten = i + 1;
				writeTabs ();

				if (b[i] == '{')
					m_nIndentationLevel++;
			}
			else if (m_bPendingTabsWrite)
			{
				writeTabs ();
				m_bPendingTabsWrite = false;
			}
		}

		super.write (b, nLastIdxWritten, len + off - nLastIdxWritten);
	}
}
