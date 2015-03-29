package ch.unibas.cs.hpwc.patus;

import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import cetus.hir.Declaration;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.StencilSpecifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.KernelSourceFile;
import ch.unibas.cs.hpwc.patus.codegen.Strategy;
import ch.unibas.cs.hpwc.patus.codegen.TrapezoidCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Point;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.geometry.Subdomain;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class TrapezoidCodeGeneratorMain extends AbstractBaseCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (TrapezoidCodeGeneratorMain.class);

	private final static DateFormat DATE_FORMAT = new SimpleDateFormat ("yyyy/MM/dd HH:mm:ss");


	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private File m_fileOutputDirectory;

	private CodeGenerationOptions m_options;
	
	private IArchitectureDescription m_hardwareDescription;

	/**
	 * The stencil calculation (containing the specification for the stencil
	 * structure, boundary treatment, stopping criteria, etc. Parsed from the
	 * stencil description file.
	 */
	private StencilCalculation m_stencil;
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public TrapezoidCodeGeneratorMain(StencilCalculation stencil, IArchitectureDescription hardwareDescription, File fileOutputDirectory, CodeGenerationOptions options)
	{
		super ();
		
		m_stencil = stencil;
		m_hardwareDescription = hardwareDescription;
		m_fileOutputDirectory = fileOutputDirectory;
		m_options = options;
	}
	
	public void run()
	{
		// show stencil and strategy codes
		TrapezoidCodeGeneratorMain.LOGGER.debug (StringUtil.concat ("Stencil Calculation:\n", m_stencil.toString ()));

		Strategy dummyStrategy = new Strategy();
		byte nDimensionality = m_stencil.getDimensionality();
		
        Point ptMin = new Point(nDimensionality);
        Point ptMax = new Point(nDimensionality);
        for (int i = 0; i < nDimensionality; i++)
        {
        	// TODO: clean up; eg. use m_data.getData ().getGeneratedIdentifiers ().getDimensionMinIdentifier
        	ptMin.setCoord(i, new NameID(CodeGeneratorUtil.getDimensionName(i) + "_min"));
        	ptMax.setCoord(i, new NameID(CodeGeneratorUtil.getDimensionName(i) + "_max"));
        }
        
        Subdomain sg = new Subdomain (null, Subdomain.ESubdomainType.SUBDOMAIN, ptMin, new Size (ptMin, ptMax), true);
		dummyStrategy.setBaseDomain(new SubdomainIdentifier ("u", sg));
		
		List<Declaration> listParams = new ArrayList<Declaration> ();
		listParams.add (new VariableDeclaration (StencilSpecifier.STENCIL_PARAM, new VariableDeclarator (Specifier.INT, new NameID ("__t_min"))));
		dummyStrategy.setParameters (listParams);
		
		m_data = new CodeGeneratorSharedObjects (m_stencil, dummyStrategy, m_hardwareDescription, m_options);
		TrapezoidCodeGenerator cg = new TrapezoidCodeGenerator(m_data);

		// generate the code
		CodeGeneratorRuntimeOptions optionsStencil = new CodeGeneratorRuntimeOptions ();
		optionsStencil.setOption (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL);
		optionsStencil.setOption (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, false);

		TrapezoidCodeGeneratorMain.LOGGER.info ("Generating code...");
		createFunctionParameterList (true, true);
		StatementListBundle slbGenerated = cg.generate(dummyStrategy.getBody(), optionsStencil);
		addAdditionalDeclarationsAndAssignments (slbGenerated, optionsStencil);
		
		// output
		List<KernelSourceFile> listOutputs = createOutputsList ();

		// add global declarations
		for (KernelSourceFile out : listOutputs)
			addAdditionalGlobalDeclarations (out, slbGenerated.getDefault ());

		// add internal autotune parameters to the parameter list
		createFunctionInternalAutotuneParameterList (slbGenerated);

		// package the code into functions, add them to the translation unit, and write the code files
		for (KernelSourceFile out : listOutputs)
		{
			packageKernelSourceFile (out, slbGenerated, null, true);
			out.writeCode (this, m_data, m_fileOutputDirectory);
		}
		
		TrapezoidCodeGeneratorMain.LOGGER.info("Code generation completed.");
	}
	
	protected void setBaseMemoryObjectInitializers ()
	{
		// don't do anything (we don't need to initialize base memory objects)
	}

	/**
	 * Creates the list of output kernel source files.
	 * @return
	 */
	private List<KernelSourceFile> createOutputsList ()
	{
		List<KernelSourceFile> listOutputs = new ArrayList<> ();

		KernelSourceFile ksf = new KernelSourceFile (getOutputFile (m_options.getKernelFilename ()));
		ksf.setCompatibility (m_options.getCompatibility ());
		ksf.setCreateInitialization (m_options.getCreateInitialization ());
		ksf.setCreateBenchmarkingHarness (false);
		listOutputs.add (ksf);

		return listOutputs;
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
		sb.append ("c");

		return sb.toString ();
	}
	
	/**
	 *
	 * @param bIncludeAutotuneParameters
	 * @return
	 */
	public String getIncludesAndDefines (boolean bIncludeAutotuneParameters)
	{
		StringBuilder sb = new StringBuilder ();

		if (m_data.getOptions ().isDebugPrintStencilIndices ())
			sb.append ("#include <stdio.h>\n");

		// include files
		for (String strFile : m_data.getArchitectureDescription ().getIncludeFiles ())
		{
			sb.append ("#include \"");
			sb.append (strFile);
			sb.append ("\"\n");
		}

		sb.append ("#include <stdint.h>\n");
		sb.append ("#include \"patusrt.h\"\n");
		
		sb.append ("#include \"");
		sb.append (CodeGenerationOptions.DEFAULT_TUNEDPARAMS_FILENAME);
		sb.append ("\"\n");

		return sb.toString ();
	}	
	
	public static void main(String[] args)
	{
		try
		{
			CommandLineOptions options = new CommandLineOptions (args, true);
	
			if (options.getStencilFile () == null)
			{
				CommandLineOptions.printHelp ();
				return;
			}
	
			// parse the input files
	
			// try to parse the stencil file
			TrapezoidCodeGeneratorMain.LOGGER.info (StringUtil.concat ("Reading stencil specification ", options.getStencilFile ().getName (), "..."));
			StencilCalculation stencil = StencilCalculation.load (options.getStencilFile ().getAbsolutePath (), options.getStencilDSLVersion (), options.getOptions ());
			options.getOptions ().checkSettings (stencil);
	
			// create the Main object
			new TrapezoidCodeGeneratorMain (stencil, options.getHardwareDescription(), options.getOutputDir (), options.getOptions ()).run ();
			
			TrapezoidCodeGeneratorMain.LOGGER.info("complete");
		}
		catch (Exception e)
		{
			if (TrapezoidCodeGeneratorMain.LOGGER.isDebugEnabled ())
				e.printStackTrace ();
			TrapezoidCodeGeneratorMain.LOGGER.error (e.getMessage ());
		}
	}
}
