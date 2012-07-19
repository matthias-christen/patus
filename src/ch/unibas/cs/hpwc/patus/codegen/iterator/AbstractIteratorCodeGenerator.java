package ch.unibas.cs.hpwc.patus.codegen.iterator;

import cetus.hir.AccessExpression;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.Loop;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;

public abstract class AbstractIteratorCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;
	
	private Loop m_loop;
	private CompoundStatement m_cmpstmtLoopBody;
	private CompoundStatement m_cmpstmtOutput;
	private boolean m_bContainsStencilCall;
	private CodeGeneratorRuntimeOptions m_options;

	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public AbstractIteratorCodeGenerator (CodeGeneratorSharedObjects data,
		Loop loop, CompoundStatement cmpstmtLoopBody, CompoundStatement cmpstmtOutput,
		boolean bContainsStencilCall, CodeGeneratorRuntimeOptions options)
	{
		m_data = data;
		m_loop = loop;
		m_cmpstmtLoopBody = cmpstmtLoopBody;
		m_cmpstmtOutput = cmpstmtOutput;
		m_bContainsStencilCall = bContainsStencilCall;
		m_options = options;
	}

	protected Expression getIdentifier (Expression exprOrig, String strIdentifier, StatementListBundle slb, CodeGeneratorRuntimeOptions options)
	{
		if (exprOrig instanceof IDExpression || exprOrig instanceof Literal || exprOrig instanceof AccessExpression)
			return exprOrig;
		
		return m_data.getCodeGenerators ().getConstantGeneratedIdentifiers ().getConstantIdentifier (
			exprOrig, strIdentifier, Globals.SPECIFIER_SIZE, slb, null, options);
	}
	
	/**
	 * Creates a new identifier named <code>strIdentifier</code>, declares it
	 * locally in the function, and initializes it with
	 * <code>exprInitializer</code>.
	 * 
	 * @param strIdentifier
	 *            The name of the identifier
	 * @param exprInitializer
	 *            The initial value, or <code>null</code> if the variable isn't
	 *            to be initialized
	 * @return The newly created identifier
	 */
	protected Identifier createIdentifier (String strIdentifier, Expression exprInitializer)
	{
		VariableDeclarator decl = new VariableDeclarator (new NameID (strIdentifier));
		m_data.getData ().addDeclaration (new VariableDeclaration (Globals.SPECIFIER_INDEX, decl));

		if (exprInitializer != null)
			decl.setInitializer (new ValueInitializer (exprInitializer));
		
		return new Identifier (decl);
	}
	
	public abstract void generate ();

	public CodeGeneratorSharedObjects getData ()
	{
		return m_data;
	}

	public RangeIterator getRangeIterator ()
	{
		if (!(m_loop instanceof RangeIterator))
			throw new RuntimeException ("The iterator is not a RangeIterator.");
		return (RangeIterator) m_loop;
	}

	public SubdomainIterator getSubdomainIterator ()
	{
		if (!(m_loop instanceof SubdomainIterator))
			throw new RuntimeException ("The iterator is not a SubdomainIterator.");
		return (SubdomainIterator) m_loop;
	}

	public CompoundStatement getLoopBody ()
	{
		return m_cmpstmtLoopBody;
	}

	public CompoundStatement getOutputStatement ()
	{
		return m_cmpstmtOutput;
	}

	public boolean containsStencilCall ()
	{
		return m_bContainsStencilCall;
	}

	public CodeGeneratorRuntimeOptions getOptions ()
	{
		return m_options;
	}	
}
