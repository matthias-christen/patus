package ch.unibas.cs.hpwc.patus.preprocessor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.CodeGeneratorMain;
import ch.unibas.cs.hpwc.patus.CommandLineOptions;
import ch.unibas.cs.hpwc.patus.analysis.StrategyFix;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers;
import ch.unibas.cs.hpwc.patus.codegen.Strategy;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.symbolic.Maxima;
import ch.unibas.cs.hpwc.patus.util.FileUtil;
import ch.unibas.cs.hpwc.patus.util.IndentOutputStream;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class Preprocessor
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (Preprocessor.class);

	private final static String STENCIL_START = "begin-stencil-specification";
	private final static String STENCIL_END = "end-stencil-specification";

	private final static Pattern PATTERN_PRAGMA = Pattern.compile ("\\s*#pragma\\s+patus\\s+([A-Za-z0-9-_]+)\\s*(\\((.*)\\))*");


	///////////////////////////////////////////////////////////////////
	// Member Variables

	protected CommandLineOptions m_options;

	protected File m_fileInput;
	protected File m_fileOutput;

	protected Strategy m_strategy;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public Preprocessor (File file, CommandLineOptions options)
	{
		m_fileInput = file;
		m_fileOutput = new File (Preprocessor.getOutputFilename (file));
		m_options = options;

		m_strategy = null;
	}

	protected static String getOutputFilename (File f)
	{
		return StringUtil.concat (FileUtil.getFilenameWithoutExtension (f), "_pp.", FileUtil.getExtension (f));
	}
	
	private String getVariableName (GlobalGeneratedIdentifiers.Variable variable, boolean bIncludeTimeIndex)
	{
		if (!bIncludeTimeIndex)
			return variable.getOriginalName ();
		
		// HACK
		// variables with vector and time index are of the format
		//     <varname>_<vecidx>_<timeidx>
		// cut out the vector index
		
		String strName = variable.getName ();
		int nPos2 = strName.lastIndexOf ('_');
		if (nPos2 >= 0)
		{
			int nPos1 = strName.substring (0, nPos2).lastIndexOf ('_');
			return StringUtil.concat (strName.substring (0, nPos1), strName.substring (nPos2));
		}
		return strName;
	}

	public void start () throws IOException
	{
		// create reader and writer
		BufferedReader in = new BufferedReader (new FileReader (m_fileInput));
		PrintWriter out = new PrintWriter (new IndentOutputStream (new FileOutputStream (m_fileOutput)));

		boolean bIsExtractingStencilSpec = false;
		StringBuilder sb = new StringBuilder ();

		// process the file and replace Patus pragmas by the appropriate code
		Matcher matcherPragma = null;
		CommandLineOptions options = m_options;

		for ( ; ; )
		{
			String strLine = in.readLine ();
			if (strLine == null)
				break;

			// create or reset the #pragma matcher and the variable matcher
			if (matcherPragma == null)
 				matcherPragma = Preprocessor.PATTERN_PRAGMA.matcher (strLine);
			else
				matcherPragma.reset (strLine);

			// if a Patus pragma has been found, replace it by code generated by the backend code generator
			if (matcherPragma.matches ())
			{
				String strPragma = matcherPragma.group (1);
				String strOptions = matcherPragma.group (3);

				if (strPragma.equals (STENCIL_START))
				{
					// extract stencil specification into new file
					bIsExtractingStencilSpec = true;

					// parse the local code generation options, which override the default options
					options = m_options;
					if (strOptions != null && !"".equals (strOptions))
					{
						options = new CommandLineOptions (m_options);
						options.parse (strOptions.split (","), false);
					}
				}
				else if (strPragma.equals (STENCIL_END))
				{
					// end of stencil specification found
					bIsExtractingStencilSpec = false;

					// generate the code for the stencil and reset the stencil specification buffer
					CodeGeneratorSharedObjects data = generateCode (sb.toString (), options);
					sb.setLength (0);

					// insert the function call into the original file
					boolean bMakeFortranCompatible = options.getOptions ().getCompatibility () == CodeGenerationOptions.ECompatibility.FORTRAN;
					List<GlobalGeneratedIdentifiers.Variable> listParams = data.getData ().getGlobalGeneratedIdentifiers ().getFunctionParameterVarList (
						!bMakeFortranCompatible, false, false, bMakeFortranCompatible);
					
					Map<String, Boolean> mapIncludeTimeIndex = new HashMap<> ();
					for (GlobalGeneratedIdentifiers.Variable v : listParams)
						mapIncludeTimeIndex.put (v.getOriginalName (), mapIncludeTimeIndex.containsKey (v.getOriginalName ()));
					
					if (bMakeFortranCompatible)
					{
						// Fortran version
						out.print ("      call ");
						out.print (data.getStencilCalculation ().getName ());
						out.print (" (");
						boolean bFirst = true;
						for (GlobalGeneratedIdentifiers.Variable v : listParams)
						{
							if (!bFirst)
								out.print (", ");
							out.print (StringUtil.trimLeft (getVariableName (v, mapIncludeTimeIndex.get (v.getOriginalName ())), new char[] { '_' }));
							bFirst = false;
						}
						out.println (")");
					}
					else
					{
						// C version
						out.println ("{");
						
						// declare output variables (dummies)
						for (GlobalGeneratedIdentifiers.Variable v : listParams)
							if (v.getType ().equals (GlobalGeneratedIdentifiers.EVariableType.OUTPUT_GRID))
							{
								out.print (v.getDatatype ());
								out.print ("* ");
								out.print (getVariableName (v, mapIncludeTimeIndex.get (v.getOriginalName ())));
								out.println (';');
							}
						
						out.print (data.getStencilCalculation ().getName ());
						out.print (" (");
						boolean bFirst = true;
						for (GlobalGeneratedIdentifiers.Variable v : listParams)
						{
							if (!bFirst)
								out.print (", ");
							if (v.getType ().equals (GlobalGeneratedIdentifiers.EVariableType.OUTPUT_GRID))
								out.print ("&");
							out.print (getVariableName (v, mapIncludeTimeIndex.get (v.getOriginalName ())));
							bFirst = false;
						}
						out.println (");");
						out.println ("}");
					}
				}
			}
			else if (bIsExtractingStencilSpec)
			{
				sb.append (strLine);
				sb.append ('\n');
			}
			else
				out.println (strLine);
		}

		in.close ();
		out.close ();
	}

	protected CodeGeneratorSharedObjects generateCode (String strStencilSpecification, CommandLineOptions options)
	{
		// parse the stencil specification
		StencilCalculation stencil = StencilCalculation.parse (strStencilSpecification, options.getStencilDSLVersion (), options.getOptions ());

		// try to parse the strategy file
		if (m_strategy == null)
		{
			Preprocessor.LOGGER.info (StringUtil.concat ("Reading strategy ", options.getStrategyFile ().getName (), "..."));
			m_strategy = Strategy.load (options.getStrategyFile ().getAbsolutePath (), stencil);
			StrategyFix.fix (m_strategy, options.getHardwareDescription (), options.getOptions ());
		}

		// we want to create both the benchmarking harness and the standalone kernel source file
		options.getOptions ().addTarget (CodeGenerationOptions.ETarget.BENCHMARK_HARNESS);
		options.getOptions ().addTarget (CodeGenerationOptions.ETarget.KERNEL_ONLY);
		options.getOptions ().setCreateInitialization (false);
		options.getOptions ().setKernelFilename (StringUtil.concat ("../", stencil.getName ()));

		// create the output directory
		File fileOutputDir = options.isOutputDirSet () ? options.getOutputDir () : new File (m_fileOutput.getParentFile (), stencil.getName ());
		fileOutputDir.mkdirs ();

		return new CodeGeneratorMain (stencil, m_strategy, options.getHardwareDescription (), fileOutputDir, options.getOptions ()).generateCode ();
	}

 	/**
	 * The main entry point of Patus.
	 * @param args Command line arguments
	 */
	public static void main (String[] args)
	{
		CommandLineOptions options = new CommandLineOptions (args, true);

		if (options.getStencilFile () == null || options.getStrategyFile () == null || options.getArchitectureDescriptionFile () == null || options.getArchitectureName () == null)
		{
			CommandLineOptions.printHelp ();
			return;
		}

		boolean bHasError = false;

		try
		{
			// initialize
			Maxima.getInstance ();

			new Preprocessor (options.getStencilFile (), options).start ();

			// terminate
			Maxima.getInstance ().close ();
		}
		catch (IOException e)
		{
			bHasError = true;
			LOGGER.error (StringUtil.concat ("An error occurred during processing ", options.getStencilFile ()), e);
		}

		if (!bHasError)
			LOGGER.info ("Terminated successfully.");
	}
}
