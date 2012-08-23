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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Symbolic;
import ch.unibas.cs.hpwc.patus.autotuner.HybridOptimizer.HybridRunExecutable;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
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
	// Inner Types
	
	private enum EValuePosition
	{
		NONE,
		START,
		END
	}
	

	///////////////////////////////////////////////////////////////////
	// Member Variables

	private boolean m_bInitSuccessful;

	private IOptimizer m_optimizer;

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
	
	/**
	 * Maps named auto-tuner parameters (arguments like '@NAME=4:4:100') to the argument index
	 */
	private Map<String, Integer> m_mapNamedAutotuneParams;


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
		m_optimizer = null;
		m_mapNamedAutotuneParams = new HashMap<> ();

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
		
		// get the default optimizer if not set on the command line
		if (m_optimizer == null)
			m_optimizer = OptimizerFactory.getOptimizer ();

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
			System.out.println ("    StandaloneAutotuner ExecutableFilename(s) Param1 Param2 ... ParamN");
			System.out.println ("           [ Constraint1 Constraint2 ... ConstraintM ] [ -mMethod ]");
			System.out.println ();
			System.out.println ("You can provide multiple executables, which will be auto-tuned simultaneously,");
			System.out.println ("i.e., the auto-tuner then tries to find the parameter set that minimizes the");
			System.out.println ("execution time for all the executables at the same time.");
			System.out.println ("Executable file names have to be separated by commas.");
			System.out.println ("Optionally, each executable can be weighted with a real number. The syntax for");
			System.out.println ("adding a weight to an executable is");
			System.out.println ("    ExecutableFilename:weight");
			System.out.println ("In this case, the auto-tuner tries to minimize the sum");
			System.out.println ("    weight_1*time_1 + ... weight_k*time_k.");
			System.out.println ();
			System.out.println ("ParamI has the following syntax:");
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

		m_listParamSets = new ArrayList<> (m_rgParams.length - 1);
		m_listConstraints = new ArrayList<> ();
		
		Map<String, Integer> mapVariables = new HashMap<> ();
		int nIdxArgs = 0;

		for (int i = 1; i < m_rgParams.length; i++)
		{
			// expects the command line argument to be in the form "startvalue:[[*]step:]endvalue[!]", or "value1[,value2[,value3...]][!]".
			boolean bArgParsed = true;
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
					m_optimizer = OptimizerFactory.getOptimizer (m_rgParams[i].substring (2));
				
				else if (m_rgParams[i].indexOf (':') >= 0)
				{
					String strValues = getArgValues (m_rgParams[i], nIdxArgs);
					
					// variant "startvalue:[[*]step:]endvalue"

					// parse the input
					String[] rgValues = strValues.split (":");

					boolean bUseExhaustive = false;
					if (rgValues[rgValues.length - 1].endsWith ("!"))
					{
						rgValues[rgValues.length - 1] = rgValues[rgValues.length - 1].substring (0, rgValues[rgValues.length - 1].length () - 1);
						bUseExhaustive = true;
					}

					int nStartValue = getValue (nIdxArgs, rgValues[0], mapVariables, EValuePosition.START);
					int nEndValue = 0;
					int nStep = 1;
					boolean bIsStepMultiplicative = false;

					if (rgValues.length == 2)
						nEndValue = getValue (nIdxArgs, rgValues[1], mapVariables, EValuePosition.END);
					else if (rgValues.length == 3)
					{
						bIsStepMultiplicative = rgValues[1].charAt (0) == '*';
						nStep = getValue (nIdxArgs, bIsStepMultiplicative ? rgValues[1].substring (1) : rgValues[1], mapVariables, EValuePosition.NONE);
						nEndValue = getValue (nIdxArgs, rgValues[2], mapVariables, EValuePosition.END);
					}
					else
						throw new RuntimeException ("Malformed argument " + m_rgParams[i]);

					// assign the parameter list
					List<Integer> listValues = new LinkedList<> ();
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
					
					nIdxArgs++;
				}
				else if (m_rgParams[i].indexOf (',') >= 0)
				{
					// variant "value1[,value2[,value3...]]"

					String strValues = getArgValues (m_rgParams[i], nIdxArgs);
					String[] rgValues = strValues.split (",");

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
						rgParamSet[j] = getValue (nIdxArgs, strValue, mapVariables, EValuePosition.NONE);
						j++;
					}
					
					ParamSet ps = new ParamSet (rgParamSet, bUseExhaustive);
					LOGGER.info (ps.toString ());
					m_listParamSets.add (ps);
					
					nIdxArgs++;
				}
				else
				{
					// assume this is only a single number
					int nValue = Integer.parseInt (m_rgParams[i]);
					ParamSet ps = new ParamSet (new int[] { nValue }, false);
					LOGGER.info (ps.toString ());
					m_listParamSets.add (ps);
					
					// save this number as a variable
					mapVariables.put ("$" + nIdxArgs, nValue);
					nIdxArgs++;
				}
			}
			catch (NumberFormatException e)
			{
				bArgParsed = false;
				//throw new RuntimeException ("Malformed argument " + m_rgParams[i]);
			}
			
			if (!bArgParsed)
			{
				// the argument hasn't been parsed yet; assume it is of the form ".*({<arg>})*.*" (regex; i.e., contains
				// some text and the parameter to vary in curly braces
				
				// TODO
			}
		}
	}
	
	/**
	 * Parses the string <code>strVal</code> and substitutes any variables in
	 * <code>mapVariables</code> by their values.
	 * 
	 * @param strVal
	 *            The string to parse
	 * @param mapVariables
	 *            A map of variable names to their values
	 * @return The value of <code>strVal</code>
	 */
	private int getValue (int nArgIdx, String strVal, Map<String, Integer> mapVariables, EValuePosition pos)
	{
		boolean bDependsOnOtherAutotuneParams = false;
		String s = strVal;
		for ( ; ; )
		{
			int nPos = s.indexOf ('$');
			if (nPos == -1)
				break;
			
			int nEnd = nPos + 1;
			while (nEnd < s.length () && Character.isDigit (s.charAt (nEnd)))
				nEnd++;
			
			String strVar = s.substring (nPos, nEnd);
			int nVarIdx = Integer.parseInt (strVar.substring (1));

			int nValue = 0;
			if (mapVariables.containsKey (strVar))
			{
				// variable reference
				Integer nVal = mapVariables.get (strVar);
				if (nVal == null)
					throw new RuntimeException (StringUtil.concat ("No variable ", strVar));
				nValue = nVal;
			}
			else if (nVarIdx < m_listParamSets.size ())
			{
				// auto-tuning parameter reference
				// use the min/max value of the auto-tuning parameter and add a constraint (see below)
				ParamSet ps = m_listParamSets.get (nVarIdx);
				switch (pos)
				{
				case START:
					nValue = ps.getParams ()[0];
					break;
				case END:
					nValue = ps.getParams ()[ps.getParams ().length - 1];
					break;
				default:
					throw new RuntimeException ("If an auto-tuning parameter depends on another auto-tuning parameter, it must occur only within a start:step:end construct.");
				}

				bDependsOnOtherAutotuneParams = true;
			}
			else
				throw new RuntimeException (StringUtil.concat (strVar, " is references neither a variable nor an auto-tuning parameter (or the definition of the reference does not occur before its use)"));
			
			s = StringUtil.concat (s.substring (0, nPos), String.valueOf (nValue), s.substring (nEnd));
		}
		
		// add constraints if necessary
		if (bDependsOnOtherAutotuneParams)
		{
			Expression exprConstraint = null;
			switch (pos)
			{
			case START:
				exprConstraint = new BinaryExpression (new NameID (StringUtil.concat ("$", nArgIdx)), BinaryOperator.COMPARE_GE, ExpressionParser.parse (strVal));
				break;
			case END:
				exprConstraint = new BinaryExpression (new NameID (StringUtil.concat ("$", nArgIdx)), BinaryOperator.COMPARE_LE, ExpressionParser.parse (strVal));
				break;
			}
			
			if (exprConstraint != null)
			{
				m_listConstraints.add (exprConstraint);
				LOGGER.info (StringUtil.concat ("Constraint ", exprConstraint.toString ()));
			}
		}

		// calculate the result
		Expression exprResult = Symbolic.simplify (ExpressionParser.parse (s));
		if (exprResult instanceof IntegerLiteral)
			return (int) ((IntegerLiteral) exprResult).getValue ();
				
		return 0;
	}
	
	private String getArgValues (String strArg, int nIdxArgs)
	{
		String[] rgParts = strArg.split ("=");
		String strValues = rgParts[rgParts.length - 1];
		if (rgParts.length > 1)
			m_mapNamedAutotuneParams.put (rgParts[0].replace ("@", ""), nIdxArgs);
	
		return strValues;
	}

	/**
	 * Runs the optimizer and prints the result (parameter set and timing) to <code>stdout</code>.
	 */
	private void optimize ()
	{
		if (m_optimizer == null)
			throw new RuntimeException ("Optimizer not found.");
		
		// run the optimizer
		StandaloneAutotuner.LOGGER.info (StringUtil.concat ("Using optimizer: ", m_optimizer.getName ()));

		HybridOptimizer optMain = new HybridOptimizer (m_optimizer);

		IRunExecutable run = new HybridRunExecutable (m_strFilename, m_listParamSets, m_listConstraints);
		optMain.optimize (run);

		if (run instanceof AbstractRunExecutable)
		{
			System.out.println ("\n\n\nHistogram:\n\n");
			((AbstractRunExecutable) run).createHistogram ().print ();
		}

		// print the result to stdout
		System.out.println ("\n\nOptimal parameter configuration found:");
		for (int nParam : run.getParameters (optMain.getResultParameters ()))
		{
			System.out.print (nParam);
			System.out.print (' ');
		}

		System.out.println ("\n\nTiming information for the optimal run:");
		System.out.println (optMain.getResultTiming ());

		System.out.println ("\nProgram output of the optimal run:");
		System.out.println (optMain.getProgramOutput ());
		
		System.out.print ("\nWriting results to ");
		System.out.print (CodeGenerationOptions.DEFAULT_TUNEDPARAMS_FILENAME);
		System.out.println ("...");
		writeTunedParametersFile (run.getParameters (optMain.getResultParameters ()));
	}
	
	private void writeTunedParametersFile (int[] rgOptimizedParamValues)
	{
		if (m_mapNamedAutotuneParams.size () == 0)
			return;
		
		try
		{
			PrintWriter out = new PrintWriter (new File (new File (m_strFilename).getParentFile (), CodeGenerationOptions.DEFAULT_TUNEDPARAMS_FILENAME));
			
			for (String strVarName : m_mapNamedAutotuneParams.keySet ())
			{
				out.print ("#define ");
				out.print (strVarName);
				out.print (' ');
				
				Integer nIdx = m_mapNamedAutotuneParams.get (strVarName);
				if (nIdx != null && nIdx >= 0 && nIdx < rgOptimizedParamValues.length)
					out.println (rgOptimizedParamValues[nIdx]);
				else
					out.println ("0");
			}
			
			out.flush ();
			out.close ();
		}
		catch (FileNotFoundException e)
		{
			e.printStackTrace ();
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
