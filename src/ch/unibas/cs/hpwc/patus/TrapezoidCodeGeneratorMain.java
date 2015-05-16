package ch.unibas.cs.hpwc.patus;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.log4j.Logger;

import cetus.hir.DeclarationStatement;
import cetus.hir.Statement;
import cetus.hir.VariableDeclaration;
import ch.unibas.cs.hpwc.patus.analysis.HIRAnalyzer;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.KernelSourceFile;
import ch.unibas.cs.hpwc.patus.codegen.Strategy;
import ch.unibas.cs.hpwc.patus.codegen.TrapezoidCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class TrapezoidCodeGeneratorMain extends AbstractBaseCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (TrapezoidCodeGeneratorMain.class);


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

		String strategyCode = "strategy dummy (domain u) { for t = 1 .. stencil.t_max for point p in u(:; t) u[p; t+1] = stencil(u[p; t]); }";
		Strategy strategy = Strategy.parse(strategyCode, m_stencil);
		
		m_data = new CodeGeneratorSharedObjects (m_stencil, strategy, m_hardwareDescription, m_options);
		TrapezoidCodeGenerator cg = new TrapezoidCodeGenerator(m_data);

		// generate the code
		CodeGeneratorRuntimeOptions optionsStencil = new CodeGeneratorRuntimeOptions ();
		optionsStencil.setOption (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL);
		optionsStencil.setOption (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, false);

		TrapezoidCodeGeneratorMain.LOGGER.info ("Generating code...");
		createFunctionParameterList (true, true);
		StatementListBundle slbGenerated = cg.generate(strategy.getBody(), optionsStencil);
		addAdditionalDeclarationsAndAssignments (slbGenerated, optionsStencil);
		
		// output
		List<KernelSourceFile> listOutputs = createOutputsList ();

		// add global declarations
		for (KernelSourceFile out : listOutputs)
			addAdditionalGlobalDeclarations (out, slbGenerated.getDefault ());

		// add internal autotune parameters to the parameter list
		createFunctionInternalAutotuneParameterList (slbGenerated);
		
		// do post-generation optimizations
		optimizeCode (slbGenerated);

		// package the code into functions, add them to the translation unit, and write the code files
		for (KernelSourceFile out : listOutputs)
		{
			packageKernelSourceFile (out, slbGenerated, null, true);
			out.writeCode (this, m_data, m_fileOutputDirectory);
		}
		
		TrapezoidCodeGeneratorMain.LOGGER.info("Code generation completed.");
	}
	
	/*
	protected void setBaseMemoryObjectInitializers ()
	{
		// don't do anything (we don't need to initialize base memory objects)
	}*/
	
	/**
	 * Do post-code generation optimizations (loop unrolling, ...).
	 * @param cmpstmtBody
	 * @return
	 */
	protected void optimizeCode (StatementListBundle slbInput)
	{
		// remove declarations of unused variables
		for (ParameterAssignment pa : slbInput)
		{
			LOGGER.debug (StringUtil.concat ("Removing unused variables from ", pa.toString ()));

			StatementList sl = slbInput.getStatementList (pa);
			List<Statement> list = sl.getStatementsAsList ();
			boolean bModified = false;

			for (Iterator<Statement> it = list.iterator (); it.hasNext (); )
			{
				Statement stmt = it.next ();
				if (stmt instanceof DeclarationStatement && ((DeclarationStatement) stmt).getDeclaration () instanceof VariableDeclaration)
				{
					VariableDeclaration vdecl = (VariableDeclaration) ((DeclarationStatement) stmt).getDeclaration ();
					if (vdecl.getNumDeclarators () == 1)
					{
						if (!HIRAnalyzer.isReferenced (vdecl.getDeclarator (0).getID (), sl))
						{
							it.remove ();
							bModified = true;
						}
					}
				}
			}

			if (bModified)
				slbInput.replaceStatementList (pa, new StatementList (list));
		}
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
