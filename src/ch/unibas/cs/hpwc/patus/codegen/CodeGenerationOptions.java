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
package ch.unibas.cs.hpwc.patus.codegen;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.log4j.Logger;

import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
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
	 * The default filename for the kernel source file
	 */
	public final static String DEFAULT_KERNEL_FILENAME = "kernel";

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

	private List<ETarget> m_listTargets;

	private String m_strKernelFilename;

	private boolean m_bCreateInitialization;

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
		m_listTargets = new ArrayList<CodeGenerationOptions.ETarget> (1);
		m_listTargets.add (ETarget.BENCHMARK_HARNESS);
		m_strKernelFilename = DEFAULT_KERNEL_FILENAME;
		m_bCreateInitialization = true;
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
		for (ETarget target : options.getTargets ())
			addTarget (target);
		setKernelFilename (options.getKernelFilename ());
		setCreateInitialization (options.getCreateInitialization ());
		setCreateValidation (options.getCreateValidationCode ());
		setValidationTolerance (options.getValidationTolerance ());
	}

	/**
	 * Check whether the code generation options are compatible.
	 */
	public void checkSettings (StencilCalculation stencil)
	{
		if (getCompatibility () == CodeGenerationOptions.ECompatibility.FORTRAN)
		{
			if (!ExpressionUtil.isValue (stencil.getMaxIterations (), 1))
				CodeGenerationOptions.LOGGER.error ("In Fortran compatiblity mode, the only permissible t_max is 1.");
		}
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

	public List<ETarget> getTargets ()
	{
		return m_listTargets;
	}

	public void clearTargets ()
	{
		m_listTargets.clear ();
	}

	public void addTarget (ETarget target)
	{
		if (!m_listTargets.contains (target))
			m_listTargets.add (target);

		// make sure that "createInitialization" is set then the target is "benchmark"
		if (target == CodeGenerationOptions.ETarget.BENCHMARK_HARNESS)
			m_bCreateInitialization = true;
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

	public void setCreateInitialization (boolean bCreateInitialization)
	{
		m_bCreateInitialization = bCreateInitialization;
	}

	public boolean getCreateInitialization ()
	{
		return m_bCreateInitialization;
	}

	public boolean getCreateValidationCode ()
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
