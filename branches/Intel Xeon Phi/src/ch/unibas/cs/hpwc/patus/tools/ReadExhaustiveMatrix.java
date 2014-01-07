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
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 *
 * @author Matthias-M. Christen
 */
public class ReadExhaustiveMatrix
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Pattern PATTERN_PARAM = Pattern.compile ("params (\\d+) (\\d+)");
	private final static Pattern PATTERN_PERFORMANCE = Pattern.compile ("Performance:\\s+([0-9\\.]+)\\s+GFlop/s");


	///////////////////////////////////////////////////////////////////
	// Member Variables

	///////////////////////////////////////////////////////////////////
	// Implementation


	/**
	 * @param args
	 */
	public static void main (String[] args) throws Exception
	{
		if (args.length != 1)
		{
			System.err.println ("Usage: ReadExhaustiveMatrix <input file>");
			return;
		}

		File fileIn = new File (args[0]);
		BufferedReader in = new BufferedReader (new FileReader (fileIn));
		PrintWriter out = new PrintWriter (new File (fileIn.getAbsoluteFile ().getParentFile (), "autotuneresult.m"));

		out.println ("Perf = [");

		String strLine = null;
		Matcher mParam = null;
		Matcher mPerformance = null;
		int nParamLineNum = 0;
		boolean bIsFirstLine = true;

		List<Integer> listX = new ArrayList<> ();
		List<Integer> listY = new ArrayList<> ();

		while ((strLine = in.readLine ()) != null)
		{
			if (mParam == null || mPerformance == null)
			{
				mParam = PATTERN_PARAM.matcher (strLine);
				mPerformance = PATTERN_PERFORMANCE.matcher (strLine);
			}
			else
			{
				mParam.reset (strLine);
				mPerformance.reset (strLine);
			}

			if (mParam.matches ())
			{
				int nCurParamLineNum = Integer.parseInt (mParam.group (2));

				if (nCurParamLineNum != nParamLineNum)
				{
					if (nParamLineNum != 0)
					{
						out.println ();
						bIsFirstLine = false;
					}

					nParamLineNum = nCurParamLineNum;
					listY.add (nCurParamLineNum);
				}

				if (bIsFirstLine)
					listX.add (Integer.parseInt (mParam.group (1)));
			}
			if (mPerformance.matches ())
			{
				out.print (mPerformance.group (1));
				out.print (" ");
			}
		}

		out.println ("\n];");

		out.print ("X = [");
		for (int x : listX)
		{
			out.print (x);
			out.print (" ");
		}
		out.println ("];");

		out.print ("Y = [");
		for (int y : listY)
		{
			out.print (y);
			out.print (" ");
		}
		out.println ("];");

		out.println ("figure (1)");
		out.println ("c=pcolor (X, Y, Perf);");
		//out.println ("caxis([min(min(min(min(Zgpu1, Zgpu2), Zcpu))) max(max(max(max (Zgpu1, Zgpu2), Zcpu)))]);");
		//out.println ("xlabel ('n');");
		//out.println ("ylabel ('m');");
		out.println ("title('Performances in GFLOP/s For Varying Block Sizes');");
		out.println ("colorbar");
		out.println ("set(c, 'LineStyle', 'none');");

		in.close ();
		out.close ();
	}
}
