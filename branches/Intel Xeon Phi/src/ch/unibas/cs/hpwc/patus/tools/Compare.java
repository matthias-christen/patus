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
package ch.unibas.cs.hpwc.patus.tools;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * <p>Compares all the numbers in the two files provided as command line arguments.
 * The tool assumes that both files have the same structures.</p>
 *
 * <p>An optional <code>--tol={tolerance}</code> command line argument can be specified.</p>
 *
 * @author Matthias-M. Christen
 */
public class Compare
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (Compare.class);

	private final static Pattern PATTERN_NUMBER = Pattern.compile ("[+-]*\\d[\\d\\.]*([dDeE][+-]\\d+)?");
	private final static Pattern PATTERN_ARGUMENT = Pattern.compile ("^--([\\w-]+)=(.*)$");

	private final static double DEFAULT_TOL = 1e-8;


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private Matcher m_matcher;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public Compare ()
	{
		m_matcher = null;
	}

	public void compareFiles (File file1, File file2, double fTol) throws IOException
	{
		BufferedReader in1 = new BufferedReader (new FileReader (file1));
		BufferedReader in2 = new BufferedReader (new FileReader (file2));
		int nDifferencesFound = 0;

		for (int nLine = 1; ; nLine++)
		{
			// read lines
			String strLine1 = in1.readLine ();
			String strLine2 = in2.readLine ();

			// check whether end of file is reached
			if (strLine1 == null || strLine2 == null)
			{
				if (strLine1 != null || strLine2 != null)
					LOGGER.info ("Files do not have the same length");
				break;
			}

			// find numbers
			List<Double> list1 = findNumbers (strLine1);
			List<Double> list2 = findNumbers (strLine2);

			if (list1.size () != list2.size ())
				LOGGER.info (StringUtil.concat ("Line ", nLine, ": file 1 has ", list1.size (), " numbers, file 2 has ", list2.size (), " numbers."));
			for (int j = 0; j < Math.min (list1.size (), list2.size ()); j++)
			{
				double f1 = list1.get (j);
				double f2 = list2.get (j);
				if (Math.abs (f1 - f2) > fTol)
				{
					LOGGER.error (StringUtil.concat ("Line ", nLine, ": numbers ", j + 1, " do not match: file 1: ", f1, ", file 2: ", f2));
					nDifferencesFound++;
				}
			}
		}

		in1.close ();
		in2.close ();

		LOGGER.info (StringUtil.concat (nDifferencesFound, " differences found. The tolerance was ", fTol, "."));
	}

	private List<Double> findNumbers (String strLine)
	{
		if (m_matcher == null)
			m_matcher = PATTERN_NUMBER.matcher (strLine);
		else
			m_matcher.reset (strLine);

		List<Double> list = new ArrayList<> ();
		while (m_matcher.find ())
			list.add (Double.parseDouble (m_matcher.group ()));

		return list;
	}

	private static void printUsage ()
	{
		System.err.println ("Usage:  Compare [--tol=<tolerance>] file1 file2");
	}

	public static void main (String[] args)
	{
		File file1 = null;
		File file2 = null;
		Double fTol = null;

		// read command line arguments
		Matcher matcher = null;
		for (String strArg : args)
		{
			if (matcher == null)
				matcher = PATTERN_ARGUMENT.matcher (strArg);
			else
				matcher.reset (strArg);

			if (matcher.matches ())
			{
				String strOption = matcher.group (1);
				String strValue = matcher.group (2);

				if ("tol".equals (strOption))
					fTol = Double.parseDouble (strValue);
			}
			else
			{
				if (file1 == null)
					file1 = new File (strArg);
				else if (file2 == null)
					file2 = new File (strArg);
				else
					printUsage ();
			}
		}

		// check whether files have been defined
		if (file1 == null || file2 == null)
		{
			printUsage ();
			return;
		}

		// compare the files
		Compare compare = new Compare ();
		try
		{
			compare.compareFiles (file1, file2, fTol == null ? DEFAULT_TOL : fTol);
		}
		catch (IOException e)
		{
			LOGGER.error (e.getMessage ());
		}
	}
}
