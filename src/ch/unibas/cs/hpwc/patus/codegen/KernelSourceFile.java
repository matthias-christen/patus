package ch.unibas.cs.hpwc.patus.codegen;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import cetus.hir.Printable;
import cetus.hir.Procedure;
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
	protected List<Procedure> m_listExportedProcedures;

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
		m_listExportedProcedures = new ArrayList<> ();
	}

	public KernelSourceFile (String strFilename)
	{
		this (new File (strFilename));
	}
	
	public void addExportProcedure (Procedure procedure)
	{
		m_listExportedProcedures.add (procedure);
	}

	public void writeCode (IBaseCodeGenerator cg, CodeGeneratorSharedObjects data, File fileOutputDirectory)
	{
		writeKernelSource (cg, data, fileOutputDirectory);
		writeKernelHeader (fileOutputDirectory);
		writeTuningParamsSource (data, fileOutputDirectory);
	}

	/**
	 * Write the stencil kernel implementation source file.
	 */
	private void writeKernelSource (IBaseCodeGenerator cg, CodeGeneratorSharedObjects data, File fileOutputDirectory)
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
			out.println (cg.getFileHeader ());
			out.println (cg.getIncludesAndDefines (true));

			String strAdditionalKernelSpecificCode = data.getCodeGenerators ().getBackendCodeGenerator ().getAdditionalKernelSpecificCode ();
			if (strAdditionalKernelSpecificCode != null)
			{
				out.println (strAdditionalKernelSpecificCode);
				out.println ();
			}

			m_unit.print (out);
			out.flush ();
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
	
	private void writeKernelHeader (File fileOutputDirectory)
	{
		// write the kernel header file (for inclusion in application code)
		PrintWriter out = null;
		try
		{
			// create the header filename
			String strFilename = m_unit.getOutputFilename ();
			int nDotPos = strFilename.lastIndexOf ('.');
			if (nDotPos >= 0)
				strFilename = strFilename.substring (0, nDotPos) + ".h";
			else
				strFilename += ".h";
				
			out = new PrintWriter (new File (fileOutputDirectory, strFilename));
			
			// create the forward declarations
			for (Procedure proc : m_listExportedProcedures)
			{
				for (Object obj : proc.getReturnType ())
				{
					((Printable) obj).print (out);
					out.print (" ");
				}
				
				proc.getDeclarator ().print (out);
				out.println (";");
			}

			out.flush ();
		}
		catch (Exception e)
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
		PrintWriter out = null;
		try
		{
			out = new PrintWriter (new File (fileOutputDirectory, CodeGenerationOptions.DEFAULT_TUNEDPARAMS_FILENAME));
			GlobalGeneratedIdentifiers glid = data.getData ().getGlobalGeneratedIdentifiers ();
			
			for (GlobalGeneratedIdentifiers.Variable var : glid.getAutotuneVariables ())
			{
				out.print ("#define ");
				out.print (glid.getDefinedVariableName (var));
				out.print (" 0\n");
			}
			
			out.flush ();
		}
		catch (FileNotFoundException e)
		{
			e.printStackTrace ();
		}
		finally
		{
			if (out != null)
				out.close ();
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
