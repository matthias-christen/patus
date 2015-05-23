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

import ch.unibas.cs.hpwc.patus.UnrollConfig;
import ch.unibas.cs.hpwc.patus.codegen.options.StencilLoopUnrollingConfiguration;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.util.DomainPointEnumerator;
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
	private final static Logger LOGGER = Logger.getLogger(CodeGenerationOptions.class);

	/**
	 * The default filename for the kernel source file
	 */
	public final static String DEFAULT_KERNEL_FILENAME = "kernel";

	public final static String DEFAULT_TUNEDPARAMS_FILENAME = "tuned_params.h";

	/**
	 * Tolerance for validation (if the absolute value of the difference is
	 * greater than this value, it is considered an error)
	 */
	public final static double TOLERANCE_DEFAULT = 1e-5;

	
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public enum ECompatibility
	{
		C,
		FORTRAN;

		public static ECompatibility fromString(String s)
		{
			if ("C".equals(s))
				return ECompatibility.C;
			if ("Fortran".equals(s))
				return ECompatibility.FORTRAN;
			
			return null;
		}
	}

	public enum EDebugOption
	{
		PRINT_STENCIL_INDICES("print-stencil-indices"),
		PRINT_VALIDATION_ERRORS("print-validation-errors");

		private String m_strValue;

		private EDebugOption(String strValue)
		{
			m_strValue = strValue;
		}

		public static EDebugOption fromString(String s)
		{
			for (EDebugOption opt : values())
				if (opt.m_strValue.equals(s))
					return opt;
			
			return null;
		}
	}

	public enum ETarget
	{
		BENCHMARK_HARNESS("benchmark"),
		KERNEL_ONLY("kernel");

		private String m_strValue;

		private ETarget(String strValue)
		{
			m_strValue = strValue;
		}

		public static ETarget fromString(String s)
		{
			for (ETarget target : values())
				if (target.m_strValue.equals(s))
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
	 * The unrolling factors to be used in the inner most loops containing the
	 * stencil computations
	 */
	private UnrollConfig[] m_rgUnrollingConfigs;

	/**
	 * Flag indicating whether it is assumed that the native SSE datatypes are
	 * used. If so, correct padding is assumed
	 */
	private boolean m_bUseNativeSIMDDatatypes;

	private boolean m_bAlwaysUseNonalignedMoves;

	private boolean m_bBalanceBinaryExpressions;

	private Set<EDebugOption> m_setDebugOptions;

	private List<ETarget> m_listTargets;

	private String m_strKernelFilename;

	private boolean m_bCreateInitialization;

	private boolean m_bCreateValidation;

	private double m_fValidationTolerance;

	private boolean m_bUseOptimalInstructionScheduling;

	private boolean m_bCreatePrefetching;

	private boolean m_bNativeMic;

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public CodeGenerationOptions()
	{
		// set default options
		
		m_compatibility = ECompatibility.C;
		m_rgUnrollingConfigs = new UnrollConfig[] { new UnrollConfig(1), new UnrollConfig(2) };
		m_bUseNativeSIMDDatatypes = false;
		m_bAlwaysUseNonalignedMoves = false;
		m_bBalanceBinaryExpressions = true;
		m_setDebugOptions = new HashSet<>();
		m_listTargets = new ArrayList<>(1);
		m_listTargets.add(ETarget.BENCHMARK_HARNESS);
		m_strKernelFilename = DEFAULT_KERNEL_FILENAME;
		m_bCreateInitialization = true;
		m_bCreateValidation = true;
		m_fValidationTolerance = TOLERANCE_DEFAULT;
		m_bUseOptimalInstructionScheduling = false;
		m_bCreatePrefetching = true;
		m_bNativeMic = true;
	}

	/**
	 * Copy options.
	 * 
	 * @param options
	 */
	public void set(CodeGenerationOptions options)
	{
		setCompatibility(options.getCompatibility());
		setUnrollingConfigs(options.getUnrollingConfigs());
		setUseNativeSIMDDatatypes(options.useNativeSIMDDatatypes());
		setAlwaysUseNonalignedMoves(options.isAlwaysUseNonalignedMoves());
		setBalanceBinaryExpressions(options.getBalanceBinaryExpressions());
		m_setDebugOptions.addAll(options.m_setDebugOptions);
		
		for (ETarget target : options.getTargets())
			addTarget(target);
		
		setKernelFilename(options.getKernelFilename());
		setCreateInitialization(options.getCreateInitialization());
		setCreateValidation(options.getCreateValidationCode());
		setValidationTolerance(options.getValidationTolerance());
		setUseOptimalInstructionScheduling(options.getUseOptimalInstructionScheduling());
		setCreatePrefetching(options.getCreatePrefetching());
		setNativeMic(options.getNativeMic());
	}

	/**
	 * Check whether the code generation options are compatible.
	 */
	public void checkSettings(StencilCalculation stencil)
	{
		if (getCompatibility() == CodeGenerationOptions.ECompatibility.FORTRAN)
		{
			if (!ExpressionUtil.isValue(stencil.getMaxIterations(), 1))
				CodeGenerationOptions.LOGGER.error("In Fortran compatiblity mode, the only permissible t_max is 1.");
		}
	}

	public void setCompatibility(ECompatibility compatibility)
	{
		m_compatibility = compatibility;
	}

	public ECompatibility getCompatibility()
	{
		return m_compatibility;
	}

	public void setUnrollingConfigs(UnrollConfig... config)
	{
		if (config == null)
		{
			m_rgUnrollingConfigs = new UnrollConfig[] { new UnrollConfig(1) };
			return;
		}

		m_rgUnrollingConfigs = new UnrollConfig[config.length];
		for (int i = 0; i < config.length; i++)
			m_rgUnrollingConfigs[i] = config[i].clone();
	}

	public UnrollConfig[] getUnrollingConfigs()
	{
		return m_rgUnrollingConfigs;
	}

	public Set<StencilLoopUnrollingConfiguration> getStencilLoopUnrollingConfigurations(
		int nDimensionality,
		int[] rgMaxUnrollingFactorPerDimension,
		boolean bIsEligibleForStencilLoopUnrolling)
	{
		Set<StencilLoopUnrollingConfiguration> setUnrollingConfigs = new HashSet<>();
		
		if (bIsEligibleForStencilLoopUnrolling)
		{
			// create the single-unrolling configurations and count the multi-unrollings
			int nMultiUnrollingsCount = 0;
			
			for (UnrollConfig config : m_rgUnrollingConfigs)
			{
				if (config.isMultiConfig())
					nMultiUnrollingsCount++;
				else
				{
					setUnrollingConfigs.add(new StencilLoopUnrollingConfiguration(
						nDimensionality,
						config.getUnrollings(),
						rgMaxUnrollingFactorPerDimension
					));
				}
			}

			// create the multi-unrolling configurations
			DomainPointEnumerator dpe = new DomainPointEnumerator();
			for (int i = 0; i < nDimensionality; i++)
				dpe.addDimension(0, nMultiUnrollingsCount - 1);

			UnrollConfig[] rgMultiUnrollConfigs = new UnrollConfig[nMultiUnrollingsCount];
			
			int i = 0;
			for (UnrollConfig config : m_rgUnrollingConfigs)
				if (config.isMultiConfig())
					rgMultiUnrollConfigs[i++] = config;

			for (int[] rgUnrollingIndices : dpe)
			{
				StencilLoopUnrollingConfiguration config = new StencilLoopUnrollingConfiguration();
				
				for (i = 0; i < rgUnrollingIndices.length; i++)
				{
					config.setUnrollingForDimension(
						i,
						rgMultiUnrollConfigs[rgUnrollingIndices[i]].getUnrollingInDimension(0),
						rgMaxUnrollingFactorPerDimension[i]
					);
				}

				setUnrollingConfigs.add(config);
			}
		}
		else
		{
			// loop is not eligible for unrolling: add a non-unroll
			// configuration
			setUnrollingConfigs.add(new StencilLoopUnrollingConfiguration());
		}

		return setUnrollingConfigs;
	}

	public void setUseNativeSIMDDatatypes(boolean bUseNativeSIMDDatatypes)
	{
		m_bUseNativeSIMDDatatypes = bUseNativeSIMDDatatypes;
	}

	public boolean useNativeSIMDDatatypes()
	{
		return m_bUseNativeSIMDDatatypes;
	}

	public void setAlwaysUseNonalignedMoves(boolean bAlwaysUseNonalignedMoves)
	{
		m_bAlwaysUseNonalignedMoves = bAlwaysUseNonalignedMoves;
	}

	public boolean isAlwaysUseNonalignedMoves()
	{
		return m_bAlwaysUseNonalignedMoves;
	}

	public void setBalanceBinaryExpressions(boolean bBalanceBinaryExpressions)
	{
		m_bBalanceBinaryExpressions = bBalanceBinaryExpressions;
	}

	public boolean getBalanceBinaryExpressions()
	{
		return m_bBalanceBinaryExpressions;
	}

	public void setDebugOptions(String[] rgDebugOptions)
	{
		for (String strDebugOption : rgDebugOptions)
		{
			EDebugOption option = EDebugOption.fromString(strDebugOption);
			
			if (option != null)
				m_setDebugOptions.add(option);
			else
			{
				LOGGER.info(StringUtil.concat(
					"Bad debug option: '",
					strDebugOption,
					"' is not recognized as a debug option."
				));
			}
		}
	}

	public boolean isDebugOptionSet(EDebugOption option)
	{
		return m_setDebugOptions.contains(option);
	}

	public boolean isDebugPrintStencilIndices()
	{
		return isDebugOptionSet(EDebugOption.PRINT_STENCIL_INDICES);
	}

	public List<ETarget> getTargets()
	{
		return m_listTargets;
	}

	public void clearTargets()
	{
		m_listTargets.clear();
	}

	public void addTarget(ETarget target)
	{
		if (!m_listTargets.contains(target))
			m_listTargets.add(target);

		// make sure that "createInitialization" is set then the target is
		// "benchmark"
		if (target == CodeGenerationOptions.ETarget.BENCHMARK_HARNESS)
			m_bCreateInitialization = true;
	}

	public String getKernelFilename()
	{
		return m_strKernelFilename;
	}

	public void setKernelFilename(String strKernelFilename)
	{
		if (strKernelFilename == null)
			m_strKernelFilename = "kernel";
		else
			m_strKernelFilename = strKernelFilename;
	}

	public void setCreateInitialization(boolean bCreateInitialization)
	{
		m_bCreateInitialization = bCreateInitialization;
	}

	public boolean getCreateInitialization()
	{
		return m_bCreateInitialization;
	}

	public boolean getCreateValidationCode()
	{
		// TODO: implement validation for SIMD datatypes (=> remove restriction)
		return m_bCreateValidation && !useNativeSIMDDatatypes();
	}

	public void setCreateValidation(boolean bCreateValidation)
	{
		m_bCreateValidation = bCreateValidation;
	}

	public double getValidationTolerance()
	{
		return m_fValidationTolerance;
	}

	public void setValidationTolerance(double fValidationTolerance)
	{
		m_fValidationTolerance = fValidationTolerance;
	}

	public void setUseOptimalInstructionScheduling(boolean bUseOptimalInstructionScheduling)
	{
		m_bUseOptimalInstructionScheduling = bUseOptimalInstructionScheduling;
	}

	public boolean getUseOptimalInstructionScheduling()
	{
		return m_bUseOptimalInstructionScheduling;
	}

	public void setCreatePrefetching(boolean bCreatePrefetching)
	{
		m_bCreatePrefetching = bCreatePrefetching;
	}

	public boolean getCreatePrefetching()
	{
		return m_bCreatePrefetching;
	}

	public void setNativeMic(boolean bNativeMic)
	{
		m_bNativeMic = bNativeMic;
	}

	public boolean getNativeMic()
	{
		return m_bNativeMic;
	}
}
