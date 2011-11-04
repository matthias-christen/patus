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
public class CommandLineOptionsParser
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Pattern PATTERN_ARGUMENT = Pattern.compile ("^--([\\w-]+)=(.*)$");


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private File m_fileStencil;
	private File m_fileStrategy;
	private File m_fileArchitecture;
	private File m_fileOutDir;
	private String m_strArchName;
	private IArchitectureDescription m_hwDesc;
	private CodeGenerationOptions m_options;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public CommandLineOptionsParser (String[] args)
	{
		// parse the command line
		m_fileStencil = null;
		m_fileStrategy = null;
		m_fileArchitecture = null;
		m_fileOutDir = null;
		m_strArchName = null;
		m_options = new CodeGenerationOptions ();

		Matcher matcher = null;
		for (String strArg : args)
		{
			if (matcher == null)
				matcher = CommandLineOptionsParser.PATTERN_ARGUMENT.matcher (strArg);
			else
				matcher.reset (strArg);

			if (!matcher.matches ())
				continue;

			String strOption = matcher.group (1);
			String strValue = matcher.group (2);

			if ("stencil".equals (strOption))
				m_fileStencil = new File (strValue);
			else if ("strategy".equals (strOption))
				m_fileStrategy = new File (strValue);
			else if ("architecture".equals (strOption))
			{
				String[] rgValues = strValue.split (",");
				m_fileArchitecture = new File (rgValues[0]);
				m_strArchName = rgValues[1];
			}
			else if ("outdir".equals (strOption))
				m_fileOutDir = new File (strValue);
			else if ("generate".equals (strOption))
				m_options.setTarget (CodeGenerationOptions.ETarget.fromString (strValue));
			else if ("kernel-file".equals (strOption))
				m_options.setKernelFilename (strValue);
			else if ("compatibility".equals (strOption))
				m_options.setCompatibility (CodeGenerationOptions.ECompatibility.fromString (strValue));
			else if ("unroll".equals (strOption))
			{
				String[] rgFactorStrings = strValue.split (",");
				int[] rgUnrollingFactors = new int[rgFactorStrings.length];
				for (int i = 0; i < rgFactorStrings.length; i++)
					rgUnrollingFactors[i] = Integer.parseInt (rgFactorStrings[i]);
				m_options.setUnrollingFactors (rgUnrollingFactors);
			}
			else if ("use-native-simd-datatypes".equals (strOption))
				m_options.setUseNativeSIMDDatatypes (strValue.equals ("yes"));
			else if ("create-validation".equals (strOption))
				m_options.setCreateValidation (!strValue.equals ("no"));
			else if ("validation-tolerance".equals (strOption))
				m_options.setValidationTolerance (Double.parseDouble (strValue));
			else if ("debug".equals (strOption))
				m_options.setDebugOptions (strValue.split (","));
		}

		if (m_fileOutDir == null)
			m_fileOutDir = new File (".").getAbsoluteFile ();

		// find the hardware description
		ArchitectureDescriptionManager mgrHwDesc = new ArchitectureDescriptionManager (m_fileArchitecture);
		m_hwDesc = mgrHwDesc.getHardwareDescription (m_strArchName);
		if (m_hwDesc == null)
			throw new RuntimeException (StringUtil.concat ("Could not find hardware '", m_strArchName, "'"));
	}

	public final File getStencilFile ()
	{
		return m_fileStencil;
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
