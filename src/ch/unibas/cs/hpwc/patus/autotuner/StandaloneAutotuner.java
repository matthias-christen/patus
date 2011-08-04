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

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.apache.log4j.Logger;

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.autotuner.HybridOptimizer.HybridRunExecutable;
import ch.unibas.cs.hpwc.patus.symbolic.ExpressionParser;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class StandaloneAutotuner
{
	///////////////////////////////////////////////////////////////////
	// Constants

	/**
	 * Maximum possible values per parameter
	 */
	private static final int MAX_PARAM_VALUES = 100;

	private static final Logger LOGGER = Logger.getLogger (StandaloneAutotuner.class);


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private boolean m_bInitSuccessful;

	private String m_strOptimizerKey;

	/**
	 * The command line parameters
	 */
	private String[] m_rgParams;

	/**
	 * The filename of the executable
	 */
	private String m_strFilename;

	/**
	 * The parameter set
	 */
	private List<ParamSet> m_listParamSets;

	/**
	 * Set of constraints
	 */
	private List<Expression> m_listConstraints;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Instantiates and starts the stand-alone autotuner.
	 * @param rgParams The command line parameters
	 */
	public StandaloneAutotuner (String[] rgParams)
	{
		m_bInitSuccessful = false;
		m_rgParams = rgParams;
		m_strOptimizerKey = "";

		// check whether the call is valid
		if (!checkCommandLine ())
			return;

		try
		{
			// parse the command line
			parseCommandLine ();
			m_bInitSuccessful = true;
		}
		catch (RuntimeException e)
		{
			StandaloneAutotuner.LOGGER.error (e.getMessage ());
		}
	}

	/**
	 * Runs the optimizer.
	 */
	public void run ()
	{
		if (!m_bInitSuccessful)
			return;

		try
		{
			// run the optimizer
			optimize ();
		}
		catch (RuntimeException e)
		{
			if (StandaloneAutotuner.LOGGER.isDebugEnabled ())
				e.printStackTrace ();
			StandaloneAutotuner.LOGGER.error (e.getMessage ());
		}
	}

	/**
	 * Checks whether the command line parameters are valid.
	 * Print the usage of the program if the number of arguments is not correct.
	 * @param rgParams The command line parameters
	 * @return <code>true</code> iff the usage is correct
	 */
	private boolean checkCommandLine ()
	{
		if (m_rgParams.length <= 1)
		{
			System.out.println ("Usage: ");
			System.out.println ();
			System.out.println ("    StandaloneAutotuner ExecutableFilename Param1 Param2 ... ParamN");
			System.out.println ("           [ Constraint1 Constraint2 ... ConstraintM ] [ -mMethod ]");
			System.out.println ();
			System.out.println ("where ParamI has the following syntax:");
			System.out.println ("    startvalue:[[*]step:]endvalue[!]");
			System.out.println (" - or -");
			System.out.println ("    value1[,value2[,value3...]][!]");
			System.out.println ("The first version enumerates all the values in");
			System.out.println ("    startvalue + k * step");
			System.out.println ("or");
			System.out.println ("    startvalue * step^k");
			System.out.println ("(if the <step> is preceded by a '*')  such  that  the expression is");
			System.out.println ("<= <endvalue>. If no step is given, it defaults to 1.");
			System.out.println ("If the optional  !  is appended to the  value  range specification,");
			System.out.println ("each of the specified values is guaranteed to be used, i.e., an ex-");
			System.out.println ("haustive search is used for the corresponding parameter.");
			System.out.println ("");
			System.out.println ("Optional  constraints  can  be  specified  that  restrict the param");
			System.out.println ("values. The constraints syntax is");
			System.out.println ("    C<comparison expression>");
			System.out.println ("The comparison expression can contain  variables  $1, ..., $N  that");
			System.out.println ("correspond  to  the  values  of  the  parameters  when  a parameter");
			System.out.println ("assignment is chosen.\n");
			System.out.println ("With -mMethod the optimization method can be set. Method can be one");
			System.out.println ("of");
			for (String s : OptimizerFactory.getOptimizerKeys ())
				System.out.println (StringUtil.concat ("    ", s));

			return false;
		}
		if (m_rgParams.length == 1)
		{
			System.out.println ("");
			return false;
		}

		return true;
	}

	/**
	 * Parses the command line parameters.
	 */
	private void parseCommandLine ()
	{
		StandaloneAutotuner.LOGGER.info (StringUtil.join (m_rgParams, " "));

		m_strFilename = m_rgParams[0];

		m_listParamSets = new ArrayList<ParamSet> (m_rgParams.length - 1);
		m_listConstraints = new ArrayList<Expression> ();

		for (int i = 1; i < m_rgParams.length; i++)
		{
			// expects the command line argument to be in the form "startvalue:[[*]step:]endvalue[!]", or "value1[,value2[,value3...]][!]".
			try
			{
				if (m_rgParams[i].charAt (0) == 'C')
				{
					// this is a constraint
					Expression exprConstraint = ExpressionParser.parse (m_rgParams[i].substring (1));
					LOGGER.info (StringUtil.concat ("Constraint ", exprConstraint.toString ()));
					m_listConstraints.add (exprConstraint);
				}
				else if (m_rgParams[i].startsWith ("-m"))
					m_strOptimizerKey = m_rgParams[i].substring (2);
				else if (m_rgParams[i].indexOf (':') >= 0)
				{
					// variant "startvalue:[[*]step:]endvalue"

					// parse the input
					String[] rgValues = m_rgParams[i].split (":");

					boolean bUseExhaustive = false;
					if (rgValues[rgValues.length - 1].endsWith ("!"))
					{
						rgValues[rgValues.length - 1] = rgValues[rgValues.length - 1].substring (0, rgValues[rgValues.length - 1].length () - 1);
						bUseExhaustive = true;
					}

					int nStartValue = Integer.parseInt (rgValues[0]);
					int nEndValue = 0;
					int nStep = 1;
					boolean bIsStepMultiplicative = false;

					if (rgValues.length == 2)
						nEndValue = Integer.parseInt (rgValues[1]);
					else if (rgValues.length == 3)
					{
						bIsStepMultiplicative = rgValues[1].charAt (0) == '*';
						nStep = Integer.parseInt (bIsStepMultiplicative ? rgValues[1].substring (1) : rgValues[1]);
						nEndValue = Integer.parseInt (rgValues[2]);
					}
					else
						throw new RuntimeException ("Malformed argument " + m_rgParams[i]);

					// assign the parameter list
					List<Integer> listValues = new LinkedList<Integer> ();
					int nParamsCount = 0;
					for (int k = nStartValue; k <= nEndValue; k = bIsStepMultiplicative ? k * nStep : k + nStep)
					{
						if (nParamsCount > StandaloneAutotuner.MAX_PARAM_VALUES)
							break;
						listValues.add (k);
						nParamsCount++;
					}

					int[] rgParamSet = new int[listValues.size ()];
					int j = 0;
					for (int nValue : listValues)
					{
						rgParamSet[j] = nValue;
						j++;
					}
					
					ParamSet ps = new ParamSet (rgParamSet, bUseExhaustive);
					LOGGER.info (ps.toString ());
					m_listParamSets.add (ps);
				}
				else if (m_rgParams[i].indexOf (',') >= 0)
				{
					// variant "value1[,value2[,value3...]]"

					String[] rgValues = m_rgParams[i].split (",");

					boolean bUseExhaustive = false;
					if (rgValues[rgValues.length - 1].endsWith ("!"))
					{
						rgValues[rgValues.length - 1] = rgValues[rgValues.length - 1].substring (0, rgValues[rgValues.length - 1].length () - 2);
						bUseExhaustive = true;
					}

					int[] rgParamSet = new int[rgValues.length];
					int j = 0;
					for (String strValue : rgValues)
					{
						rgParamSet[j] = Integer.parseInt (strValue);
						j++;
					}
					
					ParamSet ps = new ParamSet (rgParamSet, bUseExhaustive);
					LOGGER.info (ps.toString ());
					m_listParamSets.add (ps);
				}
				else
				{
					// assume this is only a single number
					ParamSet ps = new ParamSet (new int[] { Integer.parseInt (m_rgParams[i]) }, false);
					LOGGER.info (ps.toString ());
					m_listParamSets.add (ps);
				}
			}
			catch (NumberFormatException e)
			{
				throw new RuntimeException ("Malformed argument " + m_rgParams[i]);
			}
		}
	}

	/**
	 * Runs the optimizer and prints the result (parameter set and timing) to <code>stdout</code>.
	 */
	private void optimize ()
	{
		// run the optimizer
		IOptimizer optimizer = OptimizerFactory.getOptimizer (m_strOptimizerKey);
		StandaloneAutotuner.LOGGER.info (StringUtil.concat ("Using optimizer: ", optimizer.getName ()));

		HybridOptimizer optMain = new HybridOptimizer (optimizer);

		IRunExecutable run = new HybridRunExecutable (m_strFilename, m_listParamSets, m_listConstraints);
		optMain.optimize (run);

		// print the result to stdout
		for (int nParam : run.getParameters (optMain.getResultParameters ()))
		{
			System.out.print (nParam);
			System.out.print (' ');
		}
		System.out.println ();
		System.out.println (optMain.getResultTiming ());

		System.out.println ("\nProgram output of the optimal run:");
		System.out.println (optMain.getProgramOutput ());

		if (run instanceof AbstractRunExecutable)
		{
			System.out.println ("\n\n\nHistogram:\n\n");
			((AbstractRunExecutable) run).createHistogram ().print ();
		}
	}

	public static void printEnvironment ()
	{
		for (String strVariable : System.getenv ().keySet ())
			System.out.println (StringUtil.concat (strVariable, " = ", System.getenv (strVariable)));
	}

	public static void main (String[] args)
	{
		//StandaloneAutotuner.printEnvironment ();
		new StandaloneAutotuner (args).run ();
	}
}
