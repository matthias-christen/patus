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

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.representation.StencilBundle;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;

/**
 * A class encapsulating the information needed by all the code generator classes.
 * @author Matthias-M. Christen
 */
public class CodeGeneratorSharedObjects
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The stencil computation object encapsulating the output of the
	 * parsed stencil specification, in particular the {@link StencilBundle} object
	 */
	private StencilCalculation m_stencil;

	/**
	 * The parallelization strategy used for the code generation
	 */
	private Strategy m_strategy;

	/**
	 * The description of the hardware for which to generate code
	 */
	private IArchitectureDescription m_hardwareDescription;

	/**
	 * Options for the stencil code generation
	 */
	private CodeGenerationOptions m_options;

	private CodeGenerators m_generators;
	private CodeGeneratorData m_data;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public CodeGeneratorSharedObjects (StencilCalculation stencil, Strategy strategy, IArchitectureDescription hardwareDescription, CodeGenerationOptions options)
	{
		m_stencil = stencil;
		m_strategy = strategy;
		m_hardwareDescription = hardwareDescription;
		m_options = options;

		if (stencil != null && strategy != null)
		{
			m_data = new CodeGeneratorData (this);
			m_generators = new CodeGenerators (this);
	
			m_data.initialize ();
			m_generators.initialize ();
		}
	}

	/**
	 * Returns the stencil calculation for which code is generated.
	 * @return The stencil calculation
	 */
	public StencilCalculation getStencilCalculation ()
	{
		return m_stencil;
	}

	/**
	 * Returns the current strategy.
	 * @return The strategy
	 */
	public Strategy getStrategy ()
	{
		return m_strategy;
	}

	/**
	 * Returns the description of the hardware for which code is being generated.
	 * @return The hardware description
	 */
	public IArchitectureDescription getArchitectureDescription ()
	{
		return m_hardwareDescription;
	}

	/**
	 * Returns the global code generation options.
	 * @return
	 */
	public CodeGenerationOptions getOptions ()
	{
		return m_options;
	}

	public CodeGenerators getCodeGenerators ()
	{
		return m_generators;
	}

	public CodeGeneratorData getData ()
	{
		return m_data;
	}
}
