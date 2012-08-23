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
package ch.unibas.cs.hpwc.patus.codegen.computation;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.log4j.Logger;

import cetus.hir.Expression;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.analysis.StencilAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorData;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.ICodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.MathUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class StencilCalculationCodeGenerator implements ICodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants and Static Types

	final static Logger LOGGER = Logger.getLogger (StencilCalculationCodeGenerator.class);
	

	///////////////////////////////////////////////////////////////////
	// Member Variables

	CodeGeneratorSharedObjects m_data;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public StencilCalculationCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
	}
	
	static ParameterAssignment createStencilCalculationParamAssignment (CodeGeneratorRuntimeOptions options)
	{
		return new ParameterAssignment (
			CodeGeneratorData.PARAM_COMPUTATION_TYPE,
			options.getIntValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL)
		);
	}

	/**
	 * Generates the code for the stencil calculation.
	 */
	@Override
	public StatementListBundle generate (Traversable trvInput, CodeGeneratorRuntimeOptions options)
	{
		if (StencilCalculationCodeGenerator.LOGGER.isDebugEnabled ())
			StencilCalculationCodeGenerator.LOGGER.debug (StringUtil.concat ("Generating code with options ", options.toString ()));

		StatementListBundle slb = new StatementListBundle (new ArrayList<Statement> ());

		if (!(trvInput instanceof Expression))
			throw new RuntimeException ("Expression as input to StencilCalculationCodeGenerator expected.");

		int nComputationType = options.getIntValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL);
		switch (nComputationType)
		{
		case CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL:
		case CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_VALIDATE:
			new StencilCodeGenerator (m_data, (Expression) trvInput, getLcmSIMDVectorLengths (), slb, options).generate ();
			break;
			
		case CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_INITIALIZE:
			new InitializeCodeGenerator (m_data, (Expression) trvInput, getLcmSIMDVectorLengths (), slb, options).generate ();
			break;
			
		default:
			throw new RuntimeException (StringUtil.concat (
				"Unknown option for ", CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION,	": ", nComputationType));
		}

		return slb;
	}
	
	public void generateSingleConstantStencilCalculation (
		Stencil stencil, Specifier specDatatype, StatementListBundle slbGenerated, CodeGeneratorRuntimeOptions options)
	{
		if (!StencilAnalyzer.isStencilConstant (stencil, m_data.getStencilCalculation ()))
			throw new RuntimeException ("This method works only for constant stencils");
		
		StencilCodeGenerator cg = new StencilCodeGenerator (m_data, null, getLcmSIMDVectorLengths (), slbGenerated, options);
		
		int[] rgDefaultOffset = new int[stencil.getDimensionality ()];
		Arrays.fill (rgDefaultOffset, 0);
		
		StatementList slInit = m_data.getData ().getInitializationStatements (
			StencilCalculationCodeGenerator.createStencilCalculationParamAssignment (options));
		cg.generateSingleCalculation (stencil, specDatatype, rgDefaultOffset, slInit, slInit);
	}
	
	/**
	 * Returns the least common multiple (LCM) of the SIMD vector lengths of
	 * all the stencil computations in the bundle.
	 * 
	 * @return The LCM of the stencil node SIMD vector lengths
	 */
	public int getLcmSIMDVectorLengths ()
	{
		int nLCM = 1;
		for (Stencil stencil : m_data.getStencilCalculation ().getStencilBundle ())
			for (StencilNode node : stencil.getOutputNodes ())
				nLCM = MathUtil.getLCM (nLCM, m_data.getArchitectureDescription ().getSIMDVectorLength (node.getSpecifier ()));
		return nLCM;
	}
}
