package ch.unibas.cs.hpwc.patus;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.ExpressionStatement;
import cetus.hir.Identifier;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorData;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers;
import ch.unibas.cs.hpwc.patus.codegen.KernelSourceFile;
import ch.unibas.cs.hpwc.patus.codegen.Strategy;
import ch.unibas.cs.hpwc.patus.codegen.TrapezoidCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions.ETarget;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Box;
import ch.unibas.cs.hpwc.patus.geometry.Point;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.symbolic.Maxima;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
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
	
	private Box m_boxOrigDomainSize;
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public TrapezoidCodeGeneratorMain(StencilCalculation stencil, IArchitectureDescription hardwareDescription, File fileOutputDirectory, CodeGenerationOptions options)
	{
		super ();
		
		m_stencil = stencil;
		m_hardwareDescription = hardwareDescription;
		m_fileOutputDirectory = fileOutputDirectory;
		m_options = options;
		
		// set the domain size of the stencil to generic min/max expressions
		byte nDim = m_stencil.getDimensionality ();
		Point ptMin = new Point (nDim);
		Point ptMax = new Point (nDim);
		
		for (byte i = 0; i < nDim; i++)
		{
			String strCoordName = CodeGeneratorUtil.getDimensionName (i); 
			ptMin.setCoord(i, new NameID (StringUtil.concat ("__trapezoid_", strCoordName, "_min")));
			ptMax.setCoord(i, new NameID (StringUtil.concat ("__trapezoid_", strCoordName, "_max")));
		}
		
		m_boxOrigDomainSize = new Box (m_stencil.getDomainSize ());
		
		m_stencil.getDomainSize ().setMin (ptMin);
		m_stencil.getDomainSize ().setMax (ptMax);
	}
	
	public void run()
	{
		// show stencil code
		TrapezoidCodeGeneratorMain.LOGGER.debug (StringUtil.concat ("Stencil Calculation:\n", m_stencil.toString ()));

		// create the strategy
		String strategyCode =
			"strategy trapezoidal (domain u)           " +
			"{                                         " +
			"    for t = 1 .. stencil.t_max            " +
			"        for point p in u(:; t)            " +
			"            u[p; t+1] = stencil(u[p; t]); " +			
			"}                                         ";
		Strategy strategy = Strategy.parse(strategyCode, m_stencil);
		
		m_data = new CodeGeneratorSharedObjects (m_stencil, strategy, m_hardwareDescription, m_options);
		TrapezoidCodeGenerator cg = new TrapezoidCodeGenerator (m_data);

		// generate the code
		CodeGeneratorRuntimeOptions optionsStencil = new CodeGeneratorRuntimeOptions ();
		optionsStencil.setOption (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL);
		optionsStencil.setOption (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, false);
		
		TrapezoidCodeGeneratorMain.LOGGER.info ("Generating code...");
		createFunctionParameterList (true, true);
		StatementListBundle slbGenerated = cg.generate (strategy.getBody (), optionsStencil);
		addMaxPointInitializations ();
		addAdditionalDeclarationsAndAssignments (slbGenerated, optionsStencil);
		
		// output
		List<KernelSourceFile> listOutputs = createOutputsList ();

		// add global declarations
		for (KernelSourceFile out : listOutputs)
			addAdditionalGlobalDeclarations (out, slbGenerated.getDefault ());		

		// add internal autotune parameters to the parameter list
		createFunctionInternalAutotuneParameterList (slbGenerated);
		
		// add function parameters for trapezoidal code
		createFunctionTrapezoidalParameterList (cg, slbGenerated);
		
		// do post-generation optimizations
		optimizeCode (slbGenerated);
		
		m_stencil.setDomainSize (m_boxOrigDomainSize);

		// package the code into functions, add them to the translation unit, and write the code files
		for (KernelSourceFile out : listOutputs)
		{
			packageKernelSourceFile (out, slbGenerated, null, true);
			out.writeCode (this, m_data, m_fileOutputDirectory);
		}
		
		TrapezoidCodeGeneratorMain.LOGGER.info ("Code generation completed.");
	}
	
	/*
	protected void setBaseMemoryObjectInitializers ()
	{
		// don't do anything (we don't need to initialize base memory objects)
	}*/
	
	/**
	 * Add initializations for the largest dimension min/max points
	 * (p1_idx_z_min, p1_idx_z_max).
	 */
	protected void addMaxPointInitializations ()
	{		
		SubdomainIdentifier it = m_data.getCodeGenerators ().getStrategyAnalyzer ().getOuterMostSubdomainIterator ().getIterator ();
		ParameterAssignment paStencilComputation = new ParameterAssignment (CodeGeneratorData.PARAM_COMPUTATION_TYPE, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL);
	
		byte nLargestDim = (byte) (m_stencil.getDimensionality () - 1);

		// p1_idx_z_min=__trapezoid_z_min;
		m_data.getData().addInitializationStatement (
			paStencilComputation,
			new ExpressionStatement (new AssignmentExpression (
				m_data.getData ().getGeneratedIdentifiers ().getDimensionMinIdentifier (it, nLargestDim).clone (),
				AssignmentOperator.NORMAL,
				m_stencil.getDomainSize ().getMin ().getCoord (nLargestDim).clone ()
			))
		);
		
		// p1_idx_z_max=__trapezoid_z_max;
		m_data.getData ().addInitializationStatement (
			paStencilComputation,
			new ExpressionStatement (new AssignmentExpression (
				m_data.getData ().getGeneratedIdentifiers ().getDimensionMaxIdentifier (it, nLargestDim).clone (),
				AssignmentOperator.NORMAL,
				m_stencil.getDomainSize ().getMax ().getCoord (nLargestDim).clone ()
			))
		);
	}
	
	/**
	 * Adds "t_max" and the slopes to the list of function parameters.
	 * @param cg
	 * @param slb
	 */
	protected void createFunctionTrapezoidalParameterList (TrapezoidCodeGenerator cg, StatementListBundle slb)
	{
		GlobalGeneratedIdentifiers ggid = m_data.getData ().getGlobalGeneratedIdentifiers ();
		
		// add artificial domain size
		Box boxDomain = m_stencil.getDomainSize ();
		for (int i = 0; i < boxDomain.getDimensionality (); i++)
		{
			VariableDeclarator declMin = new VariableDeclarator ((NameID) boxDomain.getMin ().getCoord (i));
			ggid.addStencilFunctionArguments(new GlobalGeneratedIdentifiers.Variable (
				GlobalGeneratedIdentifiers.EVariableType.TRAPEZOIDAL_SIZE,
				new VariableDeclaration (Specifier.INT, declMin),
				declMin.getSymbolName (),
				null, null
			));

			VariableDeclarator declMax = new VariableDeclarator ((NameID) boxDomain.getMax ().getCoord (i));
			ggid.addStencilFunctionArguments (new GlobalGeneratedIdentifiers.Variable (
				GlobalGeneratedIdentifiers.EVariableType.TRAPEZOIDAL_SIZE,
				new VariableDeclaration (Specifier.INT, declMax),
				declMax.getSymbolName (),
				null, null
			));
		}
		
		// add t_max
		VariableDeclarator declTMax = (VariableDeclarator) cg.getTMax ().getSymbol ();
		ggid.addStencilFunctionArguments (new GlobalGeneratedIdentifiers.Variable (
			GlobalGeneratedIdentifiers.EVariableType.TRAPEZOIDAL_TMAX,
			new VariableDeclaration (Specifier.INT, declTMax),
			declTMax.getSymbolName (),
			null, null
		));

		// add the slopes
		Identifier[][][] slopes = cg.getSlopes ();
		for (int i = 0; i < slopes.length; i++)
		{
			for (int j = 0; j < slopes[i].length; j++)
			{
				for (int k = 0; k < slopes[i][j].length; k++)
				{
					VariableDeclarator decl = (VariableDeclarator) slopes[i][j][k].getSymbol ();
					ggid.addStencilFunctionArguments (new GlobalGeneratedIdentifiers.Variable (
						GlobalGeneratedIdentifiers.EVariableType.TRAPEZOIDAL_SLOPE,
						new VariableDeclaration (Specifier.INT, decl),
						decl.getSymbolName (),
						null, null
					));
				}
			}
		}
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
			ksf.setCreateInitialization (false);
			ksf.setCreateBenchmarkingHarness (true);

			listOutputs.add (ksf);
		}

		if (m_options.getTargets ().contains (ETarget.KERNEL_ONLY))
		{
			KernelSourceFile ksf = new KernelSourceFile (getOutputFile (m_options.getKernelFilename ()));
			ksf.setCompatibility (m_options.getCompatibility ());
			ksf.setCreateInitialization (false /*m_options.getCreateInitialization ()*/);
			ksf.setCreateBenchmarkingHarness (false);
			listOutputs.add (ksf);
		}

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
			
			Maxima.getInstance().close();
		}
		catch (Exception e)
		{
			if (TrapezoidCodeGeneratorMain.LOGGER.isDebugEnabled ())
				e.printStackTrace ();
			TrapezoidCodeGeneratorMain.LOGGER.error (e.getMessage ());
		}
	}
}
