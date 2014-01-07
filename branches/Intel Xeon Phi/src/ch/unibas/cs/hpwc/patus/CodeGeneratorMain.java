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

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.analysis.StrategyFix;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions.ETarget;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.KernelSourceFile;
import ch.unibas.cs.hpwc.patus.codegen.Strategy;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.symbolic.Maxima;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * The main entry point class for Patus.
 * @author Matthias-M. Christen
 */
public class CodeGeneratorMain
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (CodeGeneratorMain.class);


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private IArchitectureDescription m_hardwareDescription;
	private File m_fileOutputDirectory;

	private CodeGenerationOptions m_options;

	/**
	 * The stencil calculation (containing the specification for the stencil
	 * structure, boundary treatment, stopping criteria, etc. Parsed from the
	 * stencil description file.
	 */
	private StencilCalculation m_stencil;

	/**
	 * The parallelization strategy
	 */
	private Strategy m_strategy;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public CodeGeneratorMain (
		StencilCalculation stencil, Strategy strategy, IArchitectureDescription hardwareDescription, File fileOutputDirectory,
		CodeGenerationOptions options)
	{
		m_stencil = stencil;
		m_strategy = strategy;
		m_hardwareDescription = hardwareDescription;
		m_fileOutputDirectory = fileOutputDirectory;
		m_options = options;
	}

	/**
	 * Runs Patus.
	 */
	public void run ()
	{
		boolean bResult = true;

		// initialize Patus
		initialize ();

		// run Patus
		try
		{
			generateCode ();
		}
		catch (Exception e)
		{
			if (CodeGeneratorMain.LOGGER.isDebugEnabled ())
				e.printStackTrace ();
			CodeGeneratorMain.LOGGER.error (e);
			CodeGeneratorMain.LOGGER.info ("Failed.");
			bResult = false;
		}
		finally
		{
			// terminate Patus
			terminate ();
		}

		if (bResult)
			CodeGeneratorMain.LOGGER.info ("Completed successfully.");
	}

	/**
	 * Creates the list of output kernel source files.
	 * @return
	 */
	private List<KernelSourceFile> createOutputsList ()
	{
		List<KernelSourceFile> listOutputs = new ArrayList<> ();

		// create a benchmarking version?
		if (m_options.getTargets ().contains (ETarget.BENCHMARK_HARNESS))
		{
			KernelSourceFile ksf = new KernelSourceFile (getOutputFile (CodeGenerationOptions.DEFAULT_KERNEL_FILENAME));
			ksf.setCompatibility (CodeGenerationOptions.ECompatibility.C);
			ksf.setCreateInitialization (true);
			ksf.setCreateBenchmarkingHarness (true);

			listOutputs.add (ksf);
		}

		if (m_options.getTargets ().contains (ETarget.KERNEL_ONLY))
		{
			KernelSourceFile ksf = new KernelSourceFile (getOutputFile (m_options.getKernelFilename ()));
			ksf.setCompatibility (m_options.getCompatibility ());
			ksf.setCreateInitialization (m_options.getCreateInitialization ());
			ksf.setCreateBenchmarkingHarness (false);

			listOutputs.add (ksf);
		}

		return listOutputs;
	}

	/**
	 * Runs the code generator.
	 */
	public CodeGeneratorSharedObjects generateCode ()
	{
		// show stencil and strategy codes
		CodeGeneratorMain.LOGGER.debug (StringUtil.concat ("Stencil Calculation:\n", m_stencil.toString ()));
		CodeGeneratorMain.LOGGER.debug (StringUtil.concat ("Strategy:\n", m_strategy.toString ()));

		// create the code generator object
		CodeGeneratorMain.LOGGER.info (StringUtil.concat ("Creating code generator for ", m_hardwareDescription.getBackend ()));
		CodeGeneratorSharedObjects data = new CodeGeneratorSharedObjects (m_stencil, m_strategy, m_hardwareDescription, m_options);
		CodeGenerator cg = new CodeGenerator (data);

		data.getCodeGenerators ().getBackendCodeGenerator ().initializeNonKernelFunctionCG ();

		// generate the code
		CodeGeneratorMain.LOGGER.info ("Generating code...");
		List<KernelSourceFile> listOutputs = createOutputsList ();
		cg.generate (listOutputs, m_fileOutputDirectory, true);

		return data;
	}

	private String getOutputFile (String strKernelBaseName)
	{
		StringBuilder sb = new StringBuilder (strKernelBaseName);

		// remove the suffix from the file name if there is any
		int nSearchFrom = sb.lastIndexOf (File.separator);
		int nDotPos = sb.lastIndexOf (".");
		if (nDotPos >= Math.max (0, nSearchFrom))
			sb.delete (nDotPos + 1, sb.length ());
		else
			sb.append ('.');
		sb.append (m_hardwareDescription.getGeneratedFileSuffix ());

		return sb.toString ();
	}

	/**
	 * Initializes the Patus driver.
	 */
	@SuppressWarnings("static-method")
	private void initialize ()
	{
		// start Maxima
		Maxima.getInstance ();
	}

	/**
	 * Terminates the Patus driver.
	 */
	@SuppressWarnings("static-method")
	private void terminate ()
	{
		// end Maxima
		Maxima.getInstance ().close ();
	}

	/**
	 * The main entry point of Patus.
	 * 
	 * @param args
	 *            Command line arguments
	 */
	public static void main (String[] args)
	{
		try
		{
			CommandLineOptions options = new CommandLineOptions (args, true);
	
			if (options.getStencilFile () == null || options.getStrategyFile () == null || options.getArchitectureDescriptionFile () == null || options.getArchitectureName () == null)
			{
				CommandLineOptions.printHelp ();
				return;
			}
	
			// parse the input files
	
			// try to parse the stencil file
			CodeGeneratorMain.LOGGER.info (StringUtil.concat ("Reading stencil specification ", options.getStencilFile ().getName (), "..."));
			StencilCalculation stencil = StencilCalculation.load (options.getStencilFile ().getAbsolutePath (), options.getStencilDSLVersion (), options.getOptions ());
			options.getOptions ().checkSettings (stencil);
	
			// try to parse the strategy file
			CodeGeneratorMain.LOGGER.info (StringUtil.concat ("Reading strategy ", options.getStrategyFile ().getName (), "..."));
			Strategy strategy = Strategy.load (options.getStrategyFile ().getAbsolutePath (), stencil);
			StrategyFix.fix (strategy, options.getHardwareDescription (), options.getOptions ());
	
			// create the Main object
			new CodeGeneratorMain (stencil, strategy, options.getHardwareDescription (), options.getOutputDir (), options.getOptions ()).run ();
		}
		catch (Exception e)
		{
			if (CodeGeneratorMain.LOGGER.isDebugEnabled ())
				e.printStackTrace ();
			CodeGeneratorMain.LOGGER.error (e.getMessage ());
		}
	}
}
