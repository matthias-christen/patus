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
package ch.unibas.cs.hpwc.patus.compiler;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;

/**
 *
 * @author Matthias-M. Christen
 */
public class Compile
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	class StreamGobbler extends Thread
	{
		InputStream m_in;

		String m_strType;

		StreamGobbler (InputStream is, String type)
		{
			this.m_in = is;
			this.m_strType = type;
		}

		@Override
		public void run ()
		{
			try
			{
				InputStreamReader isr = new InputStreamReader (m_in);
				BufferedReader br = new BufferedReader (isr);
				String line = null;
				while ((line = br.readLine ()) != null)
					System.out.println (m_strType + ">" + line);
			}
			catch (IOException ioe)
			{
				ioe.printStackTrace ();
			}
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private IArchitectureDescription m_hw;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public Compile (IArchitectureDescription hw)
	{
		m_hw = hw;
	}

	public void compile (File fileDir)
	{
		// --> http://kylecartmell.com/?p=9

		try
		{
			ProcessBuilder builder = new ProcessBuilder (m_hw.getBuild ().getCompilerCommand ());
			builder.directory (fileDir);
			Process process = builder.start ();
			process.waitFor ();

			// http://mcoder.wordpress.com/2008/02/04/launch-a-process-from-java-and-wait-for-it-to-complete/
			BufferedReader inputStreamReader = new BufferedReader (new InputStreamReader (process.getInputStream ()));
			BufferedReader errStreamReader = new BufferedReader (new InputStreamReader (process.getErrorStream ()));

			StringBuffer output = new StringBuffer ();
			StringBuffer error = new StringBuffer ();
			for (String line; (line = inputStreamReader.readLine ()) != null;)
				output.append (line);
			for (String line; (line = errStreamReader.readLine ()) != null;)
				error.append (line);

			System.out.println ("output = " + output);
			System.out.println ("error = " + error);

			process.destroy ();
		}
		catch (InterruptedException e)
		{
			e.printStackTrace ();
		}
		catch (IOException e)
		{
			e.printStackTrace ();
		}
		finally
		{

		}
	}
}
