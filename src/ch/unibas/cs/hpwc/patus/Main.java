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
package ch.unibas.cs.hpwc.patus;

import java.util.Arrays;

import ch.unibas.cs.hpwc.patus.autotuner.StandaloneAutotuner;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.preprocessor.Preprocessor;
import ch.unibas.cs.hpwc.patus.tools.Compare;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class Main
{
	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * @param args
	 */
	public static void main (String[] args) throws Exception
	{
		if (args.length < 1)
		{
			System.out.println ("No mode selected.\nSyntax:\n\tPatus <mode> <mode-specific params>\nwhere <mode> is one of 'codegen', 'codegen-x', 'autotune', or 'compare'.");
			System.exit (-1);
		}

		String strMode = args[0];
		if (strMode.equals ("codegen"))
			CodeGeneratorMain.main (Arrays.copyOfRange (args, 1, args.length));
		else if (strMode.equals ("codegen-x"))
			Preprocessor.main (Arrays.copyOfRange (args, 1, args.length));
		else if (strMode.equals ("codegen-trapezoid"))
			TrapezoidCodeGeneratorMain.main (Arrays.copyOfRange (args, 1, args.length));
		else if (strMode.equals ("autotune"))
			StandaloneAutotuner.main (Arrays.copyOfRange (args, 1, args.length));
		else if (strMode.equals ("compare"))
			Compare.main (Arrays.copyOfRange (args, 1, args.length));
		else
			System.out.println (StringUtil.concat ("Mode '", strMode, "' not recognized. Available modes are: 'codegen', 'codegen-x', 'autotune', 'compare'."));
		
		Globals.EXECUTOR_SERVICE.shutdown ();
		System.exit (0);
	}
}
