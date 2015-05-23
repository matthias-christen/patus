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

import java.io.File;
import java.util.List;

import cetus.hir.CompoundStatement;
import ch.unibas.cs.hpwc.patus.AbstractBaseCodeGenerator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;

/**
 *
 * @author Matthias-M. Christen
 */
public class CodeGenerator extends AbstractBaseCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The code generator that generates the code one thread executes from the strategy code
	 */
	private ThreadCodeGenerator m_cgThreadCode;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public CodeGenerator (CodeGeneratorSharedObjects data)
	{
		super ();
		
		m_data = data;
		m_cgThreadCode = new ThreadCodeGenerator (m_data);
	}

	/**
	 * Generates the code.
	 * @param unit The translation unit in which to place the kernels
	 * @param fileOutputDirectory The directory into which the generated code is written
	 * @param bIncludeAutotuneParameters Flag specifying whether to include the autotuning parameters
	 * 	in the function signatures
	 */
	public void generate (List<KernelSourceFile> listOutputs, File fileOutputDirectory, boolean bIncludeAutotuneParameters)
	{
		createFunctionParameterList (true, bIncludeAutotuneParameters);

		// create the stencil calculation code (the code that one thread executes)
		StatementListBundle slbThreadBody = createComputeCode ();

		// create the initialization code
		StatementListBundle slbInitializationBody = createInitializationCode (listOutputs);
		
		// add global declarations
		for (KernelSourceFile out : listOutputs)
			addAdditionalGlobalDeclarations (out, slbThreadBody.getDefault ());

		// add internal autotune parameters to the parameter list
		createFunctionInternalAutotuneParameterList (slbThreadBody);

		// do post-generation optimizations
		optimizeCode (slbThreadBody);

		// package the code into functions, add them to the translation unit, and write the code files
		for (KernelSourceFile out : listOutputs)
		{
			packageKernelSourceFile (out, slbThreadBody, slbInitializationBody, bIncludeAutotuneParameters);
			out.writeCode (this, m_data, fileOutputDirectory);
		}
	}
	
	private StatementListBundle createComputeCode ()
	{
		m_data.getData ().setCreatingInitialization (false);
		CodeGeneratorRuntimeOptions optionsStencil = new CodeGeneratorRuntimeOptions ();
		optionsStencil.setOption (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL);
		optionsStencil.setOption (CodeGeneratorRuntimeOptions.OPTION_DOBOUNDARYCHECKS, m_data.getStencilCalculation ().getBoundaries () != null);

		CompoundStatement cmpstmtStrategyKernelThreadBody = m_cgThreadCode.generate (m_data.getStrategy ().getBody (), optionsStencil);
		m_data.getData ().capture ();

		StatementListBundle slbThreadBody = new SingleThreadCodeGenerator (m_data).generate (cmpstmtStrategyKernelThreadBody, optionsStencil);
		addAdditionalDeclarationsAndAssignments (slbThreadBody, optionsStencil);
		
		return slbThreadBody;
	}
	
	private StatementListBundle createInitializationCode (List<KernelSourceFile> listOutputs)
	{
		StatementListBundle slbInitializationBody = null;
		
		boolean bCreateInitialization = false;
		for (KernelSourceFile out : listOutputs)
			if (out.getCreateInitialization ())
			{
				bCreateInitialization = true;
				break;
			}
		
		if (bCreateInitialization)
		{
			m_data.getData ().setCreatingInitialization (true);
			m_data.getCodeGenerators ().reset ();
			m_data.getData ().reset ();
			CodeGeneratorRuntimeOptions optionsInitialize = new CodeGeneratorRuntimeOptions ();
			optionsInitialize.setOption (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_INITIALIZE);
			if (m_data.getArchitectureDescription ().useSIMD ())
				optionsInitialize.setOption (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, !m_data.getOptions ().useNativeSIMDDatatypes ());

			CompoundStatement cmpstmtStrategyInitThreadBody = m_cgThreadCode.generate (m_data.getStrategy ().getBody (), optionsInitialize);
			slbInitializationBody = new SingleThreadCodeGenerator (m_data).generate (cmpstmtStrategyInitThreadBody, optionsInitialize);
			addAdditionalDeclarationsAndAssignments (slbInitializationBody, optionsInitialize);
		}
		
		return slbInitializationBody;
	}
}
