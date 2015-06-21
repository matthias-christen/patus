package ch.unibas.cs.hpwc.patus.codegen;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Identifier;
import cetus.hir.NameID;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;

public class TrapezoidCodeGenerator implements ICodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The shared data
	 */
	private CodeGeneratorSharedObjects m_data;
	
	/**
	 * x/y x/z x/t
	 * y/z y/t
	 * z/t
	 */
	private Identifier[][][] m_rgSlopes;
	
	private Identifier m_idTMax;
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public TrapezoidCodeGenerator(CodeGeneratorSharedObjects data)
	{
		m_data = data;
		
		m_idTMax = new Identifier(new VariableDeclarator(new NameID ("__t_max")));
		
		byte nDim = m_data.getStencilCalculation().getDimensionality();				
		
		// construct the bounds and slope variables
		m_rgSlopes = new Identifier[nDim][][];
		for (int i = 0; i < nDim; i++)
		{
			String strIdI = CodeGeneratorUtil.getDimensionName(i);
			m_rgSlopes[i] = new Identifier[nDim - i][];
			
			for (int j = 0; j < nDim - i; j++)
			{
				String strIdJ = j == nDim - i - 1 ? "t" : CodeGeneratorUtil.getDimensionName(i + j + 1);
				
				VariableDeclarator declA = new VariableDeclarator (new NameID ("__slope_" + strIdI + "_" + strIdJ + "_a"));
				VariableDeclarator declB = new VariableDeclarator (new NameID ("__slope_" + strIdI + "_" + strIdJ + "_b"));
				
				m_rgSlopes[i][j] = new Identifier[2];
				m_rgSlopes[i][j][0] = new Identifier(declA);
				m_rgSlopes[i][j][1] = new Identifier(declB);
			}
		}
	}
	
	public Identifier[][][] getSlopes()
	{
		return m_rgSlopes;
	}
	
	public Identifier getTMax()
	{
		return m_idTMax;
	}
	
	@Override
	public StatementListBundle generate(Traversable trvInput, CodeGeneratorRuntimeOptions options)
	{
		StatementListBundle slbGenerated = new StatementListBundle ();
		
		StatementListBundle slbLoopBody = new TrapezoidalSubdomainIteratorCodeGenerator(
			m_data.getCodeGenerators ().getStencilCalculationCodeGenerator (),
			m_data,
			m_rgSlopes
		).generate(
			m_data.getCodeGenerators().getStrategyAnalyzer().getOuterMostSubdomainIterator(),
			options
		);
		
		// generate the time loop
		RangeIterator itTemporal = m_data.getCodeGenerators().getStrategyAnalyzer().getMainTemporalIterator();
		itTemporal.setRange(
			Globals.ZERO.clone(),
			new BinaryExpression(m_idTMax.clone(), BinaryOperator.SUBTRACT, Globals.ONE.clone()),
			Globals.ONE.clone()
		);
		
		return m_data.getCodeGenerators ().getLoopCodeGenerator ().generate (itTemporal, itTemporal.getStart (), slbLoopBody, slbGenerated, options);
	}
}
