package ch.unibas.cs.hpwc.patus.codegen;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

import org.apache.log4j.Logger;

import cetus.hir.TranslationUnit;
import ch.unibas.cs.hpwc.patus.codegen.benchmark.BenchmarkHarness;
import ch.unibas.cs.hpwc.patus.util.IndentOutputStream;

/**
 *
 * @author Matthias-M. Christen
 */
public class KernelSourceFile
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (KernelSourceFile.class);


	///////////////////////////////////////////////////////////////////
	// Member Variables

	protected File m_file;
	protected TranslationUnit m_unit;

	// configuration
	protected CodeGenerationOptions.ECompatibility m_compatibility;
	protected boolean m_bCreateInitialization;
	protected boolean m_bCreateBenchmarkingHarness;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public KernelSourceFile (File f)
	{
		m_file = f;
		m_unit = new TranslationUnit (f.getPath ());
	}

	public KernelSourceFile (String strFilename)
	{
		this (new File (strFilename));
	}

	public void writeCode (CodeGenerator cg, CodeGeneratorSharedObjects data, File fileOutputDirectory)
	{
		writeKernelSource (cg, data, fileOutputDirectory);
		writeTuningParamsSource (data, fileOutputDirectory);
	}

	/**
	 * Write the stencil kernel implementation source file.
	 */
	private void writeKernelSource (CodeGenerator cg, CodeGeneratorSharedObjects data, File fileOutputDirectory)
	{
		if (m_bCreateBenchmarkingHarness)
		{
			KernelSourceFile.LOGGER.info ("Creating benchmarking harness...");

			BenchmarkHarness bh = new BenchmarkHarness (data);
			bh.generate (fileOutputDirectory, this);
		}

		// write the code
		PrintWriter out = null;
		try
		{
			KernelSourceFile.LOGGER.info ("Writing kernel source...");

			out = new PrintWriter (new IndentOutputStream (new FileOutputStream (new File (fileOutputDirectory, m_unit.getOutputFilename ()))));
			out.println (cg.getIncludesAndDefines (true));

			String strAdditionalKernelSpecificCode = data.getCodeGenerators ().getBackendCodeGenerator ().getAdditionalKernelSpecificCode ();
			if (strAdditionalKernelSpecificCode != null)
			{
				out.println (strAdditionalKernelSpecificCode);
				out.println ();
			}

			m_unit.print (out);
			out.flush ();
			out.close ();
		}
		catch (IOException e)
		{
			e.printStackTrace ();
		}
		finally
		{
			if (out != null)
				out.close ();
		}		
	}
	
	/**
	 * Write the tuning parameters header file.
	 */
	@SuppressWarnings("static-method")
	private void writeTuningParamsSource (CodeGeneratorSharedObjects data, File fileOutputDirectory)
	{
		try
		{
			PrintWriter out = new PrintWriter (new File (fileOutputDirectory, CodeGenerationOptions.DEFAULT_TUNEDPARAMS_FILENAME));
			GlobalGeneratedIdentifiers glid = data.getData ().getGlobalGeneratedIdentifiers ();
			
			for (GlobalGeneratedIdentifiers.Variable var : glid.getAutotuneVariables ())
			{
				out.print ("#define ");
				out.print (glid.getDefinedVariableName (var));
				out.print (" 0\n");
			}
			
			out.flush ();
			out.close ();
		}
		catch (FileNotFoundException e)
		{
			e.printStackTrace ();
		}
	}

	public final CodeGenerationOptions.ECompatibility getCompatibility ()
	{
		return m_compatibility;
	}

	public final void setCompatibility (CodeGenerationOptions.ECompatibility compatibility)
	{
		m_compatibility = compatibility;
	}

	public final boolean getCreateInitialization ()
	{
		return m_bCreateInitialization;
	}

	public final void setCreateInitialization (boolean bCreateInitialization)
	{
		m_bCreateInitialization = bCreateInitialization;
	}

	public final boolean getCreateBenchmarkingHarness ()
	{
		return m_bCreateBenchmarkingHarness;
	}

	public final void setCreateBenchmarkingHarness (boolean bCreateBenchmarkingHarness)
	{
		m_bCreateBenchmarkingHarness = bCreateBenchmarkingHarness;
	}

	public final TranslationUnit getTranslationUnit ()
	{
		return m_unit;
	}
}
