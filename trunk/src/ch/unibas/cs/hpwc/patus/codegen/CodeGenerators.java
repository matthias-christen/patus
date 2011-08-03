package ch.unibas.cs.hpwc.patus.codegen;

import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.backend.BackendFactory;
import ch.unibas.cs.hpwc.patus.codegen.backend.IBackend;

/**
 *
 * @author Matthias-M. Christen
 */
public class CodeGenerators
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The code generator that generates backend-specific code
	 */
	private IBackend m_backendCodeGenerator;

	/**
	 * The code generator responsible for expanding loops ({@link RangeIterator}s, {@link SubdomainIterator}s)
	 */
	private LoopCodeGenerator m_loopCodeGenerator;

	/**
	 * The code generator that calculates indices
	 */
	private IndexCalculatorCodeGenerator m_indexCalculator;

	/**
	 * The code generator that generates the stencil expressions
	 */
	private StencilCalculationCodeGenerator m_stencilCodeGenerator;

	private ConstantGeneratedIdentifiers m_constCodeGenerator;

	private SIMDScalarGeneratedIdentifiers m_SIMDScalarCodeGenerator;

	private UnrollGeneratedIdentifiers m_unrolledIdsCodeGenerator;

	private DatatransferCodeGenerator m_datatransferCodeGenerator;

	/**
	 * The code generator that creates FMA calls
	 */
	private FuseMultiplyAddCodeGenerator m_fmaCodeGenerator;

	/**
	 * The strategy analyzer
	 */
	private StrategyAnalyzer m_analyzer;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public CodeGenerators (CodeGeneratorSharedObjects objects)
	{
		m_backendCodeGenerator = BackendFactory.create (objects.getArchitectureDescription ().getBackend (), objects);
		m_loopCodeGenerator = new LoopCodeGenerator (objects);
		m_indexCalculator = new IndexCalculatorCodeGenerator (objects);
		m_stencilCodeGenerator = new StencilCalculationCodeGenerator (objects);
		m_constCodeGenerator = new ConstantGeneratedIdentifiers (objects);
		m_SIMDScalarCodeGenerator = new SIMDScalarGeneratedIdentifiers (objects);
		m_unrolledIdsCodeGenerator = new UnrollGeneratedIdentifiers (objects);
		m_datatransferCodeGenerator = new DatatransferCodeGenerator (objects);
		m_fmaCodeGenerator = new FuseMultiplyAddCodeGenerator (objects);

		m_analyzer = new StrategyAnalyzer (objects);
	}

	public void initialize ()
	{
	}

	/**
	 * Returns the code generator that generates backend-specific code.
	 * @return the code generator that generates backend-specific code
	 */
	public IBackend getBackendCodeGenerator ()
	{
		return m_backendCodeGenerator;
	}

	/**
	 * Returns the loop code generator, which is responsible for expanding loops
	 * {@link RangeIterator}s and {@link SubdomainIterator}s.
	 * @return The code generator responsible for expanding loops
	 */
	public LoopCodeGenerator getLoopCodeGenerator ()
	{
		return m_loopCodeGenerator;
	}

	/**
	 * Returns the code generator that calculates indices and maps between different index dimensionalities.
	 * @return The index calculator
	 */
	public IndexCalculatorCodeGenerator getIndexCalculator ()
	{
		return m_indexCalculator;
	}

	/**
	 * Returns the code generator that generates the code for stencil expressions.
	 * @return The code generator for stencil expressions
	 */
	public StencilCalculationCodeGenerator getStencilCalculationCodeGenerator ()
	{
		return m_stencilCodeGenerator;
	}

	/**
	 *
	 * @return
	 */
	public ConstantGeneratedIdentifiers getConstantGeneratedIdentifiers ()
	{
		return m_constCodeGenerator;
	}

	/**
	 *
	 * @return
	 */
	public SIMDScalarGeneratedIdentifiers getSIMDScalarGeneratedIdentifiers ()
	{
		return m_SIMDScalarCodeGenerator;
	}

	/**
	 *
	 * @return
	 */
	public UnrollGeneratedIdentifiers getUnrollGeneratedIdentifiers ()
	{
		return m_unrolledIdsCodeGenerator;
	}

	/**
	 *
	 * @return
	 */
	public DatatransferCodeGenerator getDatatransferCodeGenerator ()
	{
		return m_datatransferCodeGenerator;
	}

	/**
	 * Returns the code generator that creates Fused Multiply Add (FMA) expressions.
	 * @return The code generator that creates FMA expressions
	 */
	public FuseMultiplyAddCodeGenerator getFMACodeGenerator ()
	{
		return m_fmaCodeGenerator;
	}

	/**
	 * Returns a strategy analyzer object.
	 * @return A strategy analyzer
	 */
	public StrategyAnalyzer getStrategyAnalyzer ()
	{
		return m_analyzer;
	}

}
