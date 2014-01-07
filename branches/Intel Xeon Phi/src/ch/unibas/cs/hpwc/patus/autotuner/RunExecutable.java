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
package ch.unibas.cs.hpwc.patus.autotuner;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.log4j.Logger;

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class RunExecutable extends AbstractRunExecutable
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private static final Pattern PATTERN_TIMING = Pattern.compile ("^[0-9.]+$");

	private static final Logger LOGGER = Logger.getLogger (RunExecutable.class);


	///////////////////////////////////////////////////////////////////
	// Inner Types

	private static class Executable
	{
		/**
		 * The file descriptor for the executable
		 */
		private String[] m_rgExecutableFilenameAndArgs;

		private double m_fWeight;


		public Executable (List<String> listArgs, double fWeight)
		{
			m_rgExecutableFilenameAndArgs = new String[listArgs.size ()];
			listArgs.toArray (m_rgExecutableFilenameAndArgs);
			m_fWeight = fWeight;
		}

		public double getWeight ()
		{
			return m_fWeight;
		}

		public void setWeight (double fWeight)
		{
			m_fWeight = fWeight;
		}

		public double runProgram (int[] rgActualParams, StringBuilder sbResult)
		{
			// try to launch the executable
			Process process = null;
			try
			{
				// build the command line
				String[] rgCmdParams = new String[m_rgExecutableFilenameAndArgs.length + rgActualParams.length];
				rgCmdParams[0] = new File (m_rgExecutableFilenameAndArgs[0]).getAbsolutePath ();
				for (int i = 1; i < m_rgExecutableFilenameAndArgs.length; i++)
					rgCmdParams[i] = m_rgExecutableFilenameAndArgs[i];
				for (int i = 0; i < rgActualParams.length; i++)
					rgCmdParams[m_rgExecutableFilenameAndArgs.length + i] = String.valueOf (rgActualParams[i]);

				RunExecutable.LOGGER.info (StringUtil.concat ("Executing ", Arrays.toString (rgCmdParams), "..."));

				// run the executable
				process = Runtime.getRuntime ().exec (rgCmdParams);
			}
			catch (IOException e)
			{
				System.out.println ("Couldn't launch executable " + m_rgExecutableFilenameAndArgs[0]);
				return Double.MAX_VALUE;
			}

			// read the output stream
			// the timing result is expected in the last line
			double fResult = Double.MAX_VALUE;
			Matcher matcher = null;
			try
			{
				BufferedReader out = new BufferedReader (new InputStreamReader (process.getInputStream (), "ASCII"));
				String strLine = null;
				for ( ; ; )
				{
					strLine = out.readLine ();
					if (strLine == null)
						break;

					if (matcher == null)
						matcher = PATTERN_TIMING.matcher (strLine);
					else
						matcher.reset (strLine);
					if (matcher.matches ())
						fResult = Double.parseDouble (strLine);

					if (sbResult != null)
					{
						sbResult.append (strLine);
						sbResult.append ('\n');
					}

					RunExecutable.LOGGER.info (strLine);
				}
			}
			catch (UnsupportedEncodingException e)
			{
				System.out.println ("Encoding not supported.");
				return Double.MAX_VALUE;
			}
			catch (IOException e)
			{
				System.out.println ("An error occurred while reading the output of the process.");
				return Double.MAX_VALUE;
			}

			// wait for the executable to terminate
			int nReturnValue = -1;
			try
			{
				nReturnValue = process.waitFor ();
			}
			catch (InterruptedException e)
			{
				// if an interruption occurred, kill the process
			}
			finally
			{
				process.destroy ();
			}

			return nReturnValue != 0 ? Double.MAX_VALUE : fResult;
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The file descriptor for the executable
	 */
	private List<Executable> m_listExecutables;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates the executor.
	 * @param fileExecutable The file descriptor of the executable to run
	 * @param rgParamsLowerBounds The lower bounds for the parameters
	 * @param rgParamsUpperBounds The upper bounds for the parameters
	 */
	public RunExecutable (String strExecutableFilename, List<int[]> listParamSets, List<Expression> listConstraints)
	{
		super (listParamSets, listConstraints);

		m_listExecutables = new LinkedList<> ();
		parseCommandLine (strExecutableFilename);
		if (m_listExecutables.size () == 0)
			throw new RuntimeException ("No executable specified");

		setExecutableDefaultWeights ();
	}

	protected void parseCommandLine (String strCmd)
	{
		boolean bHasWeight = false;

		List<String> listArgs = new LinkedList<> ();
		StringBuilder sb = new StringBuilder ();
		boolean bInQuotes = false;

		for (int i = 0; i < strCmd.length (); i++)
		{
			char c = strCmd.charAt (i);
			switch (c)
			{
			case ' ':
			case '\t':
				if (bInQuotes)
					sb.append (c);
				else
				{
					if (sb.length () > 0)
					{
						listArgs.add (sb.toString ());
						sb.setLength (0);
					}
				}
				break;

			case '\"':
				if (i > 0 && strCmd.charAt (i - 1) == '\\')
					break;
				bInQuotes = !bInQuotes;
				break;

			case ',':
				// next executable
				if (!bHasWeight && sb.length () > 0)
					listArgs.add (sb.toString ());

				m_listExecutables.add (new Executable (listArgs, bHasWeight ? Double.parseDouble (sb.toString ()) : 0));

				sb.setLength (0);
				listArgs.clear ();
				bHasWeight = false;
				break;

			case ':':
				if (sb.length () > 0)
				{
					listArgs.add (sb.toString ());
					sb.setLength (0);
				}
				bHasWeight = true;
				break;

			default:
				sb.append (c);
			}
		}

		if (!bHasWeight && sb.length () > 0)
			listArgs.add (sb.toString ());

		m_listExecutables.add (new Executable (listArgs, bHasWeight ? Double.parseDouble (sb.toString ()) : 0));
	}

	/**
	 * Sets default weights for executables for which no weight has been specified.
	 */
	protected void setExecutableDefaultWeights ()
	{
		double fMinWeight = Double.MAX_VALUE;
		for (Executable exe : m_listExecutables)
			if (exe.getWeight () > 0)
				fMinWeight = Math.min (fMinWeight, exe.getWeight ());

		if (fMinWeight == Double.MAX_VALUE)
			fMinWeight = 1.0;

		for (Executable exe : m_listExecutables)
			if (exe.getWeight () <= 0)
				exe.setWeight (fMinWeight);
	}

	@Override
	protected double runPrograms (int[] rgActualParams, StringBuilder sbResult)
	{
		double fResultTotal = 0;

		for (Executable exe : m_listExecutables)
		{
			double fResult = exe.runProgram (rgActualParams, sbResult);
			if (fResult == Double.MAX_VALUE)
				return Double.MAX_VALUE;

			fResultTotal += exe.getWeight () * fResult;
		}

		return fResultTotal;
	}
}
