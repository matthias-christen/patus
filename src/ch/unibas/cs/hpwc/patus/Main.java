package ch.unibas.cs.hpwc.patus;

import java.util.Arrays;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.autotuner.ReadQsubResults;
import ch.unibas.cs.hpwc.patus.autotuner.StandaloneAutotuner;
import ch.unibas.cs.hpwc.patus.tools.Compare;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class Main
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * @param args
	 */
	public static void main (String[] args) throws Exception
	{
		// set logger level
		Logger.getRootLogger ().setLevel (Level.DEBUG);

		if (args.length < 1)
		{
			System.out.println ("No mode selected.\nSyntax:\n\tPatus <mode> <mode-specific params>\nwhere <mode> is one of 'codegen', 'autotune', or 'compare'.");
			System.exit (-1);
		}

		String strMode = args[0];
		if (strMode.equals ("codegen"))
			CodeGeneratorMain.main (Arrays.copyOfRange (args, 1, args.length));
		else if (strMode.equals ("autotune"))
			StandaloneAutotuner.main (Arrays.copyOfRange (args, 1, args.length));
		//else if (strMode.equals ("qsubautotune"))
		//	QsubStandaloneAutotuner.main (Arrays.copyOfRange (args, 1, args.length));
		else if (strMode.equals ("readqsubresults"))
			ReadQsubResults.main (Arrays.copyOfRange (args, 1, args.length));
		else if (strMode.equals ("compare"))
			Compare.main (Arrays.copyOfRange (args, 1, args.length));
		else
			System.out.println (StringUtil.concat ("Mode '", strMode, "' not recognized. Available modes are: 'codegen', 'autotune', 'readqsubresults', 'compare'."));
	}
}
