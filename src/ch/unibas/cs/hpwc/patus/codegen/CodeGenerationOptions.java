package ch.unibas.cs.hpwc.patus.codegen;

import java.util.HashSet;
import java.util.Set;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Class encapsulating the global code generation options specified on the
 * command line and in the stencil specification file.
 *
 * @author Matthias-M. Christen
 */
public class CodeGenerationOptions
{
	private final static Logger LOGGER = Logger.getLogger (CodeGenerationOptions.class);

	/**
	 * Tolerance for validation (if the absolute value of the difference
	 * is greater than this value, it is considered an error)
	 */
	public final static double TOLERANCE_DEFAULT = 1e-5;


	///////////////////////////////////////////////////////////////////
	// Inner Types

	public enum ECompatibility
	{
		C,
		FORTRAN;

		public static ECompatibility fromString (String s)
		{
			if ("C".equals (s))
				return ECompatibility.C;
			if ("Fortran".equals (s))
				return ECompatibility.FORTRAN;
			return null;
		}
	}

	public enum EDebugOption
	{
		PRINT_STENCIL_INDICES ("print-stencil-indices"),
		PRINT_VALIDATION_ERRORS ("print-validation-errors");

		private String m_strValue;
		private EDebugOption (String strValue)
		{
			m_strValue = strValue;
		}

		public static EDebugOption fromString (String s)
		{
			for (EDebugOption opt : values ())
				if (opt.m_strValue.equals (s))
					return opt;
			return null;
		}
	}

	public enum ETarget
	{
		BENCHMARK_HARNESS ("benchmark"),
		KERNEL_ONLY ("kernel");

		private String m_strValue;
		private ETarget (String strValue)
		{
			m_strValue = strValue;
		}

		public static ETarget fromString (String s)
		{
			for (ETarget target : values ())
				if (target.m_strValue.equals (s))
					return target;
			return null;
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The compatibility of the generated code (either C or Fortran)
	 */
	private ECompatibility m_compatibility;

	/**
	 * The unrolling factors to be used in the inner most loops containing the stencil computations
	 */
	private int[] m_rgUnrollingFactors;

	/**
	 * Flag indicating whether it is assumed that the native SSE datatypes are used.
	 * If so, correct padding is assumed
	 */
	private boolean m_bUseNativeSIMDDatatypes;

	private Set<EDebugOption> m_setDebugOptions;

	private ETarget m_target;

	private String m_strKernelFilename;

	private boolean m_bCreateValidation;

	private double m_fValidationTolerance;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public CodeGenerationOptions ()
	{
		// set default options
		m_compatibility = ECompatibility.C;
		m_rgUnrollingFactors = new int[] { 1, 2 };
		m_bUseNativeSIMDDatatypes = false;
		m_setDebugOptions = new HashSet<CodeGenerationOptions.EDebugOption> ();
		m_target = ETarget.BENCHMARK_HARNESS;
		m_strKernelFilename = "kernel";
		m_bCreateValidation = true;
		m_fValidationTolerance = TOLERANCE_DEFAULT;
	}

	/**
	 * Copy options.
	 * @param options
	 */
	public void set (CodeGenerationOptions options)
	{
		setCompatibility (options.getCompatibility ());
		setUnrollingFactors (options.getUnrollingFactors ());
		setUseNativeSIMDDatatypes (options.useNativeSIMDDatatypes ());
		m_setDebugOptions.addAll (options.m_setDebugOptions);
		setTarget (options.getTarget ());
		setKernelFilename (options.getKernelFilename ());
		setCreateValidation (options.createValidationCode ());
		setValidationTolerance (options.getValidationTolerance ());
	}

	public void setCompatibility (ECompatibility compatibility)
	{
		m_compatibility = compatibility;
	}

	public ECompatibility getCompatibility ()
	{
		return m_compatibility;
	}

	public void setUnrollingFactors (int... nUnrollingFactor)
	{
		if (nUnrollingFactor == null)
		{
			m_rgUnrollingFactors = new int[] { 1 };
			return;
		}

		m_rgUnrollingFactors = new int[nUnrollingFactor.length];
		System.arraycopy (nUnrollingFactor, 0, m_rgUnrollingFactors, 0, nUnrollingFactor.length);
	}

	public int[] getUnrollingFactors ()
	{
		return m_rgUnrollingFactors;
	}

	public void setUseNativeSIMDDatatypes (boolean bUseNativeSIMDDatatypes)
	{
		m_bUseNativeSIMDDatatypes = bUseNativeSIMDDatatypes;
	}

	public boolean useNativeSIMDDatatypes ()
	{
		return m_bUseNativeSIMDDatatypes;
	}

	public void setDebugOptions (String[] rgDebugOptions)
	{
		for (String strDebugOption : rgDebugOptions)
		{
			EDebugOption option = EDebugOption.fromString (strDebugOption);
			if (option != null)
				m_setDebugOptions.add (option);
			else
				LOGGER.info (StringUtil.concat ("Bad debug option: '", strDebugOption, "' is not recognized as a debug option."));
		}
	}

	public boolean isDebugOptionSet (EDebugOption option)
	{
		return m_setDebugOptions.contains (option);
	}

	public boolean isDebugPrintStencilIndices ()
	{
		return isDebugOptionSet (EDebugOption.PRINT_STENCIL_INDICES);
	}

	public ETarget getTarget ()
	{
		return m_target;
	}

	public void setTarget (ETarget target)
	{
		m_target = target;
	}

	public String getKernelFilename ()
	{
		return m_strKernelFilename;
	}

	public void setKernelFilename (String strKernelFilename)
	{
		if (strKernelFilename == null)
			m_strKernelFilename = "kernel";
		else
			m_strKernelFilename = strKernelFilename;
	}

	public boolean createValidationCode ()
	{
		// TODO: implement validation for SIMD datatypes (=> remove restriction)
		return m_bCreateValidation && !useNativeSIMDDatatypes ();
	}

	public void setCreateValidation (boolean bCreateValidation)
	{
		m_bCreateValidation = bCreateValidation;
	}

	public double getValidationTolerance ()
	{
		return m_fValidationTolerance;
	}

	public void setValidationTolerance (double fValidationTolerance)
	{
		m_fValidationTolerance = fValidationTolerance;
	}
}
