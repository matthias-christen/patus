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
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.log4j.Logger;

import cetus.hir.TranslationUnit;
import ch.unibas.cs.hpwc.patus.analysis.StrategyFix;
import ch.unibas.cs.hpwc.patus.arch.ArchitectureDescriptionManager;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.Strategy;
import ch.unibas.cs.hpwc.patus.codegen.benchmark.BenchmarkHarness;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.symbolic.Maxima;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.IndentOutputStream;
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

	private final static Pattern PATTERN_ARGUMENT = Pattern.compile ("^--([\\w-]+)=(.*)$");


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private File m_fileStencil;
	private File m_fileStrategy;
	private IArchitectureDescription m_hardwareDescription;
	private File m_fileOutputDirectory;

	private CodeGeneratorSharedObjects m_data;
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
		File fileStencil, File fileStrategy, IArchitectureDescription hardwareDescription, File fileOutputDirectory,
		CodeGenerationOptions options)
	{
		m_fileStencil = fileStencil;
		m_fileStrategy = fileStrategy;
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
			compile ();
			autotune ();
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
	 * Runs the code generator.
	 */
	private void generateCode ()
	{
		// parse the input files

		// try to parse the stencil file
		CodeGeneratorMain.LOGGER.info (StringUtil.concat ("Reading stencil specification ", m_fileStencil.getName (), "..."));
		m_stencil = StencilCalculation.load (m_fileStencil.getAbsolutePath (), m_options);
		checkSettings ();

		// try to parse the strategy file
		CodeGeneratorMain.LOGGER.info (StringUtil.concat ("Reading strategy ", m_fileStrategy.getName (), "..."));
		m_strategy = Strategy.load (m_fileStrategy.getAbsolutePath (), m_stencil);
		StrategyFix.fix (m_strategy, m_hardwareDescription, m_options);

		// show stencil and strategy codes
		CodeGeneratorMain.LOGGER.debug (StringUtil.concat ("Stencil Calculation:\n", m_stencil.toString ()));
		CodeGeneratorMain.LOGGER.debug (StringUtil.concat ("Strategy:\n", m_strategy.toString ()));

		// create the code generator object
		CodeGeneratorMain.LOGGER.info (StringUtil.concat ("Creating code generator for ", m_hardwareDescription.getBackend ()));
		m_data = new CodeGeneratorSharedObjects (m_stencil, m_strategy, m_hardwareDescription, m_options);
		CodeGenerator cg = new CodeGenerator (m_data);

		m_data.getCodeGenerators ().getBackendCodeGenerator ().initializeNonKernelFunctionCG ();

		// generate the code
		CodeGeneratorMain.LOGGER.info ("Generating code...");
		TranslationUnit unit = new TranslationUnit (getOutputFile ());
		cg.generate (unit, true);

		// generate the benchmark harness
		CodeGeneratorMain.LOGGER.info ("Creating benchmarking harness...");
		BenchmarkHarness bh = new BenchmarkHarness (m_data);
		bh.generate (m_fileOutputDirectory);

		// write the code
		try
		{
			PrintWriter outFile = new PrintWriter (new IndentOutputStream (new FileOutputStream (new File (m_fileOutputDirectory, unit.getOutputFilename ()))));
			writeCode (outFile, cg, unit);
		}
		catch (IOException e)
		{
			e.printStackTrace ();
		}
	}

	/**
	 * Check whether the code generation options are compatible.
	 */
	private void checkSettings ()
	{
		if (m_options.getCompatibility () == CodeGenerationOptions.ECompatibility.FORTRAN)
		{
			if (!ExpressionUtil.isValue (m_stencil.getMaxIterations (), 1))
				CodeGeneratorMain.LOGGER.error ("In Fortran compatiblity mode, the only permissible t_max is 1.");
		}
	}

	/**
	 * Compiles the generated code.
	 */
	private void compile ()
	{
		CodeGeneratorMain.LOGGER.info ("Compiling generated code...");
		//new Compile (m_hardwareDescription).compile (m_fileOutputDirectory);
	}

	/**
	 * Runs the autotuner on the executable.
	 */
	private void autotune ()
	{
		CodeGeneratorMain.LOGGER.info ("Starting autotuner...");
	}

	/**
	 * Writes the code file for the translation unit <code>unit</code>.
	 * @param out The output writer
	 * @param cg The code generator
	 * @param unit The translation unit to write to file
	 */
	private void writeCode (PrintWriter out, CodeGenerator cg, TranslationUnit unit)
	{
		out.println (cg.getIncludesAndDefines (true));
		
		String strAdditionalKernelSpecificCode = m_data.getCodeGenerators ().getBackendCodeGenerator ().getAdditionalKernelSpecificCode ();
		if (strAdditionalKernelSpecificCode != null)
		{
			out.println (strAdditionalKernelSpecificCode);
			out.println ();
		}

		unit.print (out);
		out.flush ();
		out.close ();
	}

	private String getOutputFile ()
	{
		StringBuilder sb = new StringBuilder (m_options.getKernelFilename ());

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
	private void initialize ()
	{
		// start Maxima
		Maxima.getInstance ();
	}

	/**
	 * Terminates the Patus driver.
	 */
	private void terminate ()
	{
		// end Maxima
		Maxima.getInstance ().close ();
	}

	private static void printHelp ()
	{
		System.out.println ("Usage: Patus  --stencil=<Stencil File>  --strategy=<Strategy File>");
		System.out.println ("              --architecture=<Architecture Description File>,<Hardware Name>");
		System.out.println ("              [--outdir=<Output Directory>] [--generate=<Target>]");
		System.out.println ("              [--kernel-file=<Kernel Output File Name>] [--compatibility={C|Fortran}]");
		System.out.println ("              [--unroll=<UnrollFactor1,UnrollFactor2,...>]");
		System.out.println ("              [--use-native-simd-datatypes={yes|no}]");
		System.out.println ("              [--create-validation={yes|no}] [--validation-tolerance=<Tolerance>]");
		System.out.println ("              [--debug=<Debug Option 1>,[<Debug Option 2>,[...,[<Debug Option N>]...]]");
		System.out.println ();
		System.out.println ();
		System.out.println ("--stencil=<Stencil File>");
		System.out.println ("              Specify the stencil specification file for which to generate code.");
		System.out.println ();
		System.out.println ("--strategy=<Strategy File>");
		System.out.println ("              The strategy file describing the parallelization/optimization strategy.");
		System.out.println ();
		System.out.println ("--architecture=<Architecture Description File>,<Hardware Name>");
		System.out.println ("              The architecture description file and the name of the selected");
		System.out.println ("              architecture (as specified in the 'name' attribute of the");
		System.out.println ("              'architectureType' element).");
		System.out.println ();
		System.out.println ("--outdir=<Output Directory>");
		System.out.println ("              The output directory in which the generated files will be written.");
		System.out.println ("              Optional; if not specified the generated files will be created in the");
		System.out.println ("              current directory.");
		System.out.println ();
		System.out.println ("--generate=<Target>");
		System.out.println ("              The target that will be generated. <Target> can be one of:");
		System.out.println ();
		System.out.println ("              benchmark                This will generate a full benchmark harness");
		System.out.println ("                                       (default).");
		System.out.println ();
		System.out.println ("              kernel                   This will only generate the kernel file.");
		System.out.println ();
		System.out.println ("--kernel-file=<Kernel Output File Name>");
		System.out.println ("              Specify the name of the C source file to which the generated kernel");
		System.out.println ("              is written. The suffix is appended or replaced from the definition");
		System.out.println ("              in the hardware architecture description. Defaults to 'kernel'.");
		System.out.println ();
		System.out.println ("--compatibility={C|Fortran}");
		System.out.println ("              Select whether the generated code has to be compatible with Fortran");
		System.out.println ("              (creates pointer-only input types to the kernel selection function).");
		System.out.println ("              Defaults to 'C'.");
		System.out.println ();
		System.out.println ("--unroll=<UnrollFactor1,UnrollFactor2,...>");
		System.out.println ("              A list of unrolling factors applied to the inner most loop nest");
		System.out.println ("              containing the stencil computation.");
		System.out.println ();
		System.out.println ("--use-native-simd-datatypes={yes|no}]");
		System.out.println ("              Specify whether the native SSE datatype is to be used in the kernel");
		System.out.println ("              signature. This also requires that the fields are padded correctly");
		System.out.println ("              in unit stride direction. Defaults to 'no'.");
		System.out.println ();
		System.out.println ("--create-validation={yes|no}");
		System.out.println ("              Specifies whether to create code that will validate the result.");
		System.out.println ("              If <Target> is not \"benchmark\", this option will be ignored.");
		System.out.println ("              Defaults to \"yes\".");
		System.out.println ();
		System.out.println ("--validation-tolerance=<Tolerance>");
		System.out.println ("              Sets the tolerance for the relative error in the validation.");
		System.out.println ("              This option is only relevant if validation code is generated");
		System.out.println (StringUtil.concat ("              (--create-validation=yes). Defaults to ", CodeGenerationOptions.TOLERANCE_DEFAULT, "."));
		System.out.println ();
		System.out.println ("--debug=<Debug Option 1>,[<Debug Option 2>,[...,[<Debug Option N>]...]]");
		System.out.println ("              Specify debug options (as a comma-sperated list) that will influence");
		System.out.println ("              the code generator.");
		System.out.println ("              Valid debug options (for <Debug Option i>, i=1,...,N) are:");
		System.out.println ();
		System.out.println ("              print-stencil-indices    This will insert a printf statement for");
		System.out.println ("                                       every stencil calculation with the index");
		System.out.println ("                                       into the grid array at which the result");
		System.out.println ("                                       is written.");
		System.out.println ();
		System.out.println ("              print-validation-errors  Prints all values if the validation fails.");
		System.out.println ("                                       The option is ignored if no validation code");
		System.out.println ("                                       is generated.");
	}

	/**
	 * The main entry point of Patus.
	 * @param args Command line arguments
	 */
	public static void main (String[] args) throws Exception
	{
		// parse the command line
		File fileStencil = null;
		File fileStrategy = null;
		File fileArchitecture = null;
		File fileOutDir = null;
		String strArchName = null;
		CodeGenerationOptions options = new CodeGenerationOptions ();

		Matcher matcher = null;
		for (String strArg : args)
		{
			if (matcher == null)
				matcher = CodeGeneratorMain.PATTERN_ARGUMENT.matcher (strArg);
			else
				matcher.reset (strArg);

			if (!matcher.matches ())
				continue;

			String strOption = matcher.group (1);
			String strValue = matcher.group (2);

			if ("stencil".equals (strOption))
				fileStencil = new File (strValue);
			else if ("strategy".equals (strOption))
				fileStrategy = new File (strValue);
			else if ("architecture".equals (strOption))
			{
				String[] rgValues = strValue.split (",");
				fileArchitecture = new File (rgValues[0]);
				strArchName = rgValues[1];
			}
			else if ("outdir".equals (strOption))
				fileOutDir = new File (strValue);
			else if ("generate".equals (strOption))
				options.setTarget (CodeGenerationOptions.ETarget.fromString (strValue));
			else if ("kernel-file".equals (strOption))
				options.setKernelFilename (strValue);
			else if ("compatibility".equals (strOption))
				options.setCompatibility (CodeGenerationOptions.ECompatibility.fromString (strValue));
			else if ("unroll".equals (strOption))
			{
				String[] rgFactorStrings = strValue.split (",");
				int[] rgUnrollingFactors = new int[rgFactorStrings.length];
				for (int i = 0; i < rgFactorStrings.length; i++)
					rgUnrollingFactors[i] = Integer.parseInt (rgFactorStrings[i]);
				options.setUnrollingFactors (rgUnrollingFactors);
			}
			else if ("use-native-simd-datatypes".equals (strOption))
				options.setUseNativeSIMDDatatypes (strValue.equals ("yes"));
			else if ("create-validation".equals (strOption))
				options.setCreateValidation (!strValue.equals ("no"));
			else if ("validation-tolerance".equals (strOption))
				options.setValidationTolerance (Double.parseDouble (strValue));
			else if ("debug".equals (strOption))
				options.setDebugOptions (strValue.split (","));
		}

		if (fileOutDir == null)
			fileOutDir = new File (".").getAbsoluteFile ();

		if (fileStencil == null || fileStrategy == null || fileArchitecture == null || strArchName == null)
		{
			CodeGeneratorMain.printHelp ();
			return;
		}

		// find the hardware description
		ArchitectureDescriptionManager mgrHwDesc = new ArchitectureDescriptionManager (fileArchitecture);
		IArchitectureDescription hwDesc = mgrHwDesc.getHardwareDescription (strArchName);
		if (hwDesc == null)
			throw new RuntimeException (StringUtil.concat ("Could not find hardware '", strArchName, "'"));

		// create the Main object
		new CodeGeneratorMain (fileStencil, fileStrategy, hwDesc, fileOutDir, options).run ();
	}
}
