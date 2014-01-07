package ch.unibas.cs.hpwc.patus;

import java.io.File;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ch.unibas.cs.hpwc.patus.arch.ArchitectureDescriptionManager;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class CommandLineOptions
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Pattern PATTERN_CMDLINE_ARGUMENT = Pattern.compile ("^--([\\w-]+)=(.*)$");
	private final static Pattern PATTERN_INTERNAL_ARGUMENT = Pattern.compile ("^([\\w-]+)\\:\\s*(.*)$");


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private File m_fileStencil;
	private int m_nStencilDSLVersion;
	private File m_fileStrategy;
	private File m_fileArchitecture;
	private File m_fileOutDir;
	private boolean m_bIsOutDirSet;
	private String m_strArchName;
	private IArchitectureDescription m_hwDesc;
	private CodeGenerationOptions m_options;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public CommandLineOptions (String[] args, boolean bIsCommandLineArgument)
	{
		// parse the command line
		m_fileStencil = null;
		m_nStencilDSLVersion = 1;
		m_fileStrategy = null;
		m_fileArchitecture = null;
		m_fileOutDir = null;
		m_bIsOutDirSet = false;
		m_strArchName = null;
		m_hwDesc = null;
		m_options = new CodeGenerationOptions ();

		parse (args, bIsCommandLineArgument);
	}

	/**
	 * Copy constructor.
	 * @param options
	 */
	public CommandLineOptions (CommandLineOptions options)
	{
		m_fileStencil = options.getStencilFile () == null ? null : new File (options.getStencilFile ().getAbsolutePath ());
		m_nStencilDSLVersion = options.getStencilDSLVersion ();
		m_fileStrategy = options.getStrategyFile () == null ? null : new File (options.getStrategyFile ().getAbsolutePath ());
		m_fileArchitecture = options.getArchitectureDescriptionFile () == null ? null : new File (options.getArchitectureDescriptionFile ().getAbsolutePath ());
		m_fileOutDir = options.getOutputDir () == null ? null : new File (options.getOutputDir ().getAbsolutePath ());
		m_bIsOutDirSet = options.isOutputDirSet ();
		m_hwDesc = options.getHardwareDescription ();// == null ? null : options.getHardwareDescription ().clone ();
		m_options = options.getOptions () == null ? null : new CodeGenerationOptions ();
		m_options.set (options.getOptions ());
	}

	/**
	 * Parse the comandline options an set the coresponding fields.
	 * @param args Comandlinearguments
	 * @param bIsCommandLineArgument
	 */
	public void parse (String[] args, boolean bIsCommandLineArgument)
	{
		Matcher matcher = null;
		for (String strArg : args)
		{
			if (matcher == null)
				matcher = (bIsCommandLineArgument ? CommandLineOptions.PATTERN_CMDLINE_ARGUMENT : CommandLineOptions.PATTERN_INTERNAL_ARGUMENT).matcher (strArg.trim ());
			else
				matcher.reset (strArg.trim ());

			if (!matcher.matches ())
				continue;

			String strOption = matcher.group (1);
			String strValue = matcher.group (2);

			if ("stencil".equals (strOption))
				m_fileStencil = new File (strValue);
			if ("stencil2".equals (strOption))
			{
				m_fileStencil = new File (strValue);
				m_nStencilDSLVersion = 2;
			}
			else if ("strategy".equals (strOption))
				m_fileStrategy = new File (strValue);
			else if ("architecture".equals (strOption))
			{
				String[] rgValues = strValue.split (",");
				m_fileArchitecture = new File (rgValues[0]);
				m_strArchName = rgValues[1];
			}
			else if ("outdir".equals (strOption))
			{
				m_fileOutDir = new File (strValue);
				m_bIsOutDirSet = true;
			}
			else if ("generate".equals (strOption))
			{
				m_options.clearTargets ();
				String[] rgValues = strValue.split (",");
				for (String strTarget : rgValues)
					m_options.addTarget (CodeGenerationOptions.ETarget.fromString (strTarget));
			}
			else if ("kernel-file".equals (strOption))
				m_options.setKernelFilename (strValue);
			else if ("compatibility".equals (strOption))
				m_options.setCompatibility (CodeGenerationOptions.ECompatibility.fromString (strValue));
			else if ("unroll".equals (strOption))
			{
				String[] rgFactorStrings = strValue.split (",");
				UnrollConfig[] rgConfigs = new UnrollConfig[rgFactorStrings.length];
				for (int i = 0; i < rgFactorStrings.length; i++)
					rgConfigs[i] = new UnrollConfig (rgFactorStrings[i]);
				m_options.setUnrollingConfigs (rgConfigs);
			}
			else if ("use-native-simd-datatypes".equals (strOption))
				m_options.setUseNativeSIMDDatatypes (strValue.equals ("yes"));
			else if ("always-use-nonaligned-moves".equals (strOption))
				m_options.setAlwaysUseNonalignedMoves (strValue.equals ("yes"));
			else if ("optimal-instruction-scheduling".equals (strOption))
				m_options.setUseOptimalInstructionScheduling (strValue.equals ("yes"));
			else if ("create-prefetching".equals (strOption))
				m_options.setCreatePrefetching (!strValue.equals ("no"));
			else if ("balance-binary-expressions".equals (strOption))
				m_options.setBalanceBinaryExpressions (!strValue.equals ("no"));
			else if ("create-initialization".equals (strOption))
				m_options.setCreateInitialization (strValue.equals ("yes"));
			else if ("create-validation".equals (strOption))
				m_options.setCreateValidation (!strValue.equals ("no"));
			else if ("validation-tolerance".equals (strOption))
				m_options.setValidationTolerance (Double.parseDouble (strValue));
			else if ("debug".equals (strOption))
				m_options.setDebugOptions (strValue.split (","));
			
			//Option to build code which run natively on the Mic or use the Offload symantics
			else if ("build-native-Mic".equals (strOption))
				m_options.setNativeMic (strValue.equals("yes"));
		}

		if (m_fileOutDir == null)
			m_fileOutDir = new File (".").getAbsoluteFile ();

		// find the hardware description
		if (m_fileArchitecture != null && m_hwDesc == null)
		{
			ArchitectureDescriptionManager mgrHwDesc = new ArchitectureDescriptionManager (m_fileArchitecture);
			m_hwDesc = mgrHwDesc.getHardwareDescription (m_strArchName);
			if (m_hwDesc == null)
				throw new RuntimeException (StringUtil.concat ("Could not find hardware '", m_strArchName, "'"));
		}
	}
	/**
	 * Print a help message for the use of Patus and its commandlines.
	 * 
	 */
	public static void printHelp ()
	{
		System.out.println ("Usage: Patus codegen[-x]");
		System.out.println ("    --stencil=<Stencil File>  --strategy=<Strategy File>");
		System.out.println ("    --architecture=<Architecture Description File>,<Hardware Name>");
		System.out.println ("    [--outdir=<Output Directory>] [--generate=<Target>]");
		System.out.println ("    [--kernel-file=<Kernel Output File Name>] [--compatibility={C|Fortran}]");
		System.out.println ("    [--unroll=<UnrollFactor1,UnrollFactor2,...>]");
		System.out.println ("    [--use-native-simd-datatypes={yes|no}]");
		System.out.println ("    [--always-use-nonaligned-moves={yes|no}]");
		System.out.println ("    [--create-prefetching={yes|no}]");
		System.out.println ("    [--optimal-instruction-scheduling={yes|no}]");
		System.out.println ("    [--create-validation={yes|no}] [--validation-tolerance=<Tolerance>]");
		System.out.println ("    [--debug=<Debug Option 1>,[<Debug Option 2>,[...,[<Debug Option N>]...]]");
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
		System.out.println ("              The target that will be generated. <Target> can be one or a combination");
		System.out.println ("              (separated by commas) of:");
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
		System.out.println ("              Ignored if the (only) <Target> is 'benchmark'.");
		System.out.println ();
		System.out.println ("--unroll=<UnrollFactors1,UnrollFactors2,...>");
		System.out.println ("              A list of unrolling factors applied to the inner most loop nest");
		System.out.println ("              containing the stencil computation.");
		System.out.println ("              UnrollFactorsI can be either an integer, in which case the unrolling");
		System.out.println ("              factor is applied to each dimension, or it can be a list of integers");
		System.out.println ("              separated by colons, in which case a single, specific unrolling");
		System.out.println ("              configuration for this UnrollFactorsI is created. E.g.,");
		System.out.println ("                  2:1:4");
		System.out.println ("              will unroll twice in the x dimension, apply no unrolling to the");
		System.out.println ("              y dimension, and unroll four times in the z dimension. Should the");
		System.out.println ("              stencil have more than 3 dimensions, no unrolling will be applied in");
		System.out.println ("              any of the other dimensions.");
		System.out.println ();
		System.out.println ("--use-native-simd-datatypes={yes|no}");
		System.out.println ("              Specifies whether the native SSE datatype is to be used in the kernel");
		System.out.println ("              signature. This also requires that the fields are padded correctly");
		System.out.println ("              in unit stride direction. Defaults to \"no\".");
		System.out.println ();
		System.out.println ("--always-use-nonaligned-moves={yes|no}");
		System.out.println ("              Specifies whether always non-aligned instructions to transfer data");
		System.out.println ("              from/to memory to/from a SIMD register are to be used.");
		System.out.println ("              If set to 'no', it must be guaranteed that the domain size in the unit");
		System.out.println ("              stride direction, including the boundary region, is divisible by the");
		System.out.println ("              SIMD vector length. This can be achieved by array padding.");
		System.out.println ("              Padding the unit stride direction to multiples of the SIMD vector length");
		System.out.println ("              might result in increased performance.");
		System.out.println ("              Defaults to \"no\".");
		System.out.println ();
		System.out.println ("--create-prefetching={yes|no}");
		System.out.println ("              Generates prefetching code in assembly mode if set to \"yes\".");
		System.out.println ("              Defaults to \"yes\".");
		System.out.println ();
		System.out.println ("--optimal-instruction-scheduling={yes|no}]");
		System.out.println ("              Performs optimal instruction scheduling if inline assembly code is");
		System.out.println ("              generated. (The option is ignored if not in inline assembly mode.)");
		System.out.println ("              Note that if turned on, the compilation time might be substantially");
		System.out.println ("              increased.");
		System.out.println ("              Defaults to \"no\".");
		System.out.println ();
		System.out.println ("--create-initialization={yes|no}");
		System.out.println ("              Specifies whether to create initialization code.");
		System.out.println ("              Defaults to \"yes\".");
		System.out.println ("              For benchmarking kernels, the initialization is always created.");
		System.out.println ();
		System.out.println ("--create-validation={yes|no}");
		System.out.println ("              Specifies whether to create code that will validate the result.");
		System.out.println ("              If <Target> is not \"benchmark\", this option will be ignored.");
		System.out.println ("              Defaults to \"yes\".");
		System.out.println ();
		System.out.println ("--validation-tolerance=<Tolerance>");
		System.out.println ("              Sets the tolerance for the relative error in the validation.");
		System.out.println ("              This option is only relevant if validation code is generated");
		System.out.println (StringUtil.concat (
			                "              (--create-validation=yes). Defaults to ", CodeGenerationOptions.TOLERANCE_DEFAULT, "."));
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
		System.out.println ("--build_native_Mic={yes|no}");
		System.out.println ("              Specifies whether to create code that will natively run on the Mic or use the offload pragma.");
		System.out.println ("              If <Target> is not the Intel Xeon Phi, this option will be ignored.");
		System.out.println ("              Defaults to \"yes\".");
		System.out.println ();
	}

	public final File getStencilFile ()
	{
		return m_fileStencil;
	}
	
	public final int getStencilDSLVersion ()
	{
		return m_nStencilDSLVersion;
	}

	public final File getStrategyFile ()
	{
		return m_fileStrategy;
	}

	public final File getArchitectureDescriptionFile ()
	{
		return m_fileArchitecture;
	}

	public final File getOutputDir ()
	{
		return m_fileOutDir;
	}
	
	public final boolean isOutputDirSet ()
	{
		return m_bIsOutDirSet;
	}

	public final String getArchitectureName ()
	{
		return m_strArchName;
	}

	public final IArchitectureDescription getHardwareDescription ()
	{
		return m_hwDesc;
	}

	public final CodeGenerationOptions getOptions ()
	{
		return m_options;
	}
}
