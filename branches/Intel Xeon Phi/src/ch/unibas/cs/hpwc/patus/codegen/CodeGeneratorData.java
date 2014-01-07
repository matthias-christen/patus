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
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;

import cetus.hir.Declaration;
import cetus.hir.IDExpression;
import cetus.hir.Statement;
import ch.unibas.cs.hpwc.patus.ast.Parameter;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.ast.UniqueStatementList;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;

/**
 *
 * @author Matthias-M. Christen
 */
public class CodeGeneratorData
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Object OBJ_DUMMY = new Object ();

	public final static Parameter PARAM_COMPUTATION_TYPE = new Parameter ("__computation_type__");
	public final static int CALCTYPE_STENCIL = 0;
	public final static int CALCTYPE_INITIALIZATION = 1;
	
	static
	{
		PARAM_COMPUTATION_TYPE.addValue (CALCTYPE_STENCIL);
		PARAM_COMPUTATION_TYPE.addValue (CALCTYPE_INITIALIZATION);
	}



	///////////////////////////////////////////////////////////////////
	// Member Variables

	private GlobalGeneratedIdentifiers m_generatedGlobalIdentifiers;

	/**
	 * The list of internal autotuning parameters, e.g., the loop unrolling factors,
	 * padding, etc. (everything that must be known during code generation)
	 */
	private List<String> m_listInternalAutotuningParameters;

	/**
	 * List of declarations that need to be added to the generated code
	 */
	private Map<Declaration, Object> m_setDeclarationsToAdd;

	/**
	 * List of declarations for global variables
	 */
	private Set<Declaration> m_setGlobalDeclarationsToAdd;

	/**
	 * Stack of declaration sets, captures the current declarations when
	 * {@link CodeGeneratorSharedObjects#capture()} is invoked
	 */
	private Stack<Map<Declaration, Object>> m_stackDeclarations;

	/**
	 * List of statements to which initialization code can be appended.
	 * The statements in this list will be added immediately below the variable declarations.
	 */
	private StatementListBundle m_slbInitializationStatements;

	/**
	 * Manages the identifiers created from {@link SubdomainIterator}s
	 */
	private SubdomainGeneratedIdentifiers m_generatedIdentifiers;

	/**
	 * The memory object manager that determines the memory objects to create and determines which
	 * memory object to use for a given {@link SubdomainIdentifier} and {@link StencilNode}
	 */
	private MemoryObjectManager m_moManager;

	private boolean m_bIsCreatingInitialization;

	/**
	 * A comparator for declarations
	 */
	private Comparator<Declaration> m_comparatorDeclarations = new Comparator<Declaration> ()
	{
		@Override
		public int compare (Declaration decl1, Declaration decl2)
		{
			IDExpression id1 = decl1.getDeclaredIDs ().size () > 0 ? (IDExpression) decl1.getDeclaredIDs ().get (0) : null;
			IDExpression id2 = decl2.getDeclaredIDs ().size () > 0 ? (IDExpression) decl2.getDeclaredIDs ().get (0) : null;
			if (id1 == null)
				return id2 == null ? 0 : 1;
			if (id2 == null)
				return -1;
			return id1.getName ().compareTo (id2.getName ());

			//return decl1.toString ().compareTo (decl2.toString ());
		}
	};


	///////////////////////////////////////////////////////////////////
	// Implementation

	public CodeGeneratorData (CodeGeneratorSharedObjects objects)
	{
		// create the list of internal autotuning parameters
		m_listInternalAutotuningParameters = new ArrayList<> ();
		m_listInternalAutotuningParameters.add (IInternalAutotuningParameters.LOOP_UNROLLING);
		m_listInternalAutotuningParameters.add (IInternalAutotuningParameters.PADDING);

		m_setDeclarationsToAdd = new TreeMap<> (m_comparatorDeclarations);
		m_setGlobalDeclarationsToAdd = new TreeSet<> (m_comparatorDeclarations);		
		m_slbInitializationStatements = new StatementListBundle (new UniqueStatementList ());
		m_stackDeclarations = new Stack<> ();

		m_generatedGlobalIdentifiers = new GlobalGeneratedIdentifiers (objects);
		m_generatedIdentifiers = new SubdomainGeneratedIdentifiers (objects);
		m_moManager = new MemoryObjectManager (objects);

		m_bIsCreatingInitialization = false;
	}

	public void initialize ()
	{
		m_moManager.initialize ();
	}

	/**
	 * Returns the global identifier generator.
	 * @return
	 */
	public GlobalGeneratedIdentifiers getGlobalGeneratedIdentifiers ()
	{
		return m_generatedGlobalIdentifiers;
	}

	/**
	 * Returns the memory object manager.
	 * @return The memory object manager
	 */
	public MemoryObjectManager getMemoryObjectManager ()
	{
		return m_moManager;
	}

	/**
	 * Returns the list of internal autotuning parameters.
	 * @return The list of internal autotuning parameters
	 */
	public List<String> getInternalAutotuningParameters ()
	{
		return m_listInternalAutotuningParameters;
	}

	/**
	 * Returns the number of internal autotuning parameters.
	 * @return
	 */
	public int getInternalAutotuningParametersCount ()
	{
		return m_listInternalAutotuningParameters.size ();
	}

	/**
	 * Adds a declaration that will be added at the top of the generated code.
	 * @param declaration The declaration to add
	 */
	public void addDeclaration (Declaration declaration)
	{
		// overwrite existing declarations
		if (m_setDeclarationsToAdd.containsKey (declaration))
			m_setDeclarationsToAdd.remove (declaration);
		m_setDeclarationsToAdd.put (declaration, CodeGeneratorData.OBJ_DUMMY);
	}

	/**
	 * Adds a global declaration to the generated code.
	 * @param declaration
	 */
	public void addGlobalDeclaration (Declaration declaration)
	{
		m_setGlobalDeclarationsToAdd.add (declaration);
	}

	/**
	 * Returns an iterable over the declarations that need to be added to the final generated code.
	 * @return An iterable over the per-kernel declarations to add
	 */
	public Iterable<Declaration> getDeclarationsToAdd ()
	{
		return m_setDeclarationsToAdd.keySet ();
	}

	/**
	 * Returns an iterable over the declarations that need to be added to the global scope of the
	 * generated code.
	 * @return An iterable over the global declarations to add
	 */
	public Iterable<Declaration> getGlobalDeclarationsToAdd ()
	{
		return m_setGlobalDeclarationsToAdd;
	}

	/**
	 * Returns the number of declarations that are added to the generated kernels.
	 * @return The number of declarations to add to a kernel
	 */
	public int getNumberOfDeclarationsToAdd ()
	{
		return m_setDeclarationsToAdd.size ();
	}

	/**
	 * Returns the number of global declarations that are added to the generated code.
	 * @return The number of global declarations to add
	 */
	public int getNumberOfGlobalDeclarationsToAdd ()
	{
		return m_setGlobalDeclarationsToAdd.size ();
	}

	public void addInitializationStatement (Statement stmtInit)
	{
		// add the statement
		m_slbInitializationStatements.addStatement (stmtInit);
	}	

	public void addInitializationStatement (ParameterAssignment pa, Statement stmtInit)
	{
		// make sure the statement list exists (we don't want the default statement list to be created)
		getInitializationStatements (pa);

		// add the statement
		for (Parameter param : pa)
			m_slbInitializationStatements.addStatement (stmtInit, param, pa.getParameterValue (param));
	}

	public void addInitializationStatements (Statement... rgStatements)
	{
		// add the statements
		for (Statement stmt : rgStatements)
			addInitializationStatement (stmt);
	}

	public void addInitializationStatements (ParameterAssignment pa, Statement... rgStatements)
	{
		for (Statement stmt : rgStatements)
			addInitializationStatement (pa, stmt);
	}

	public StatementList getInitializationStatements (ParameterAssignment pa)
	{
		StatementList sl = m_slbInitializationStatements.getStatementList (pa);
		if (sl == null)
			m_slbInitializationStatements.replaceStatementList (pa, sl = new UniqueStatementList ());
		return sl;
	}
	
	/**
	 *
	 * @return
	 */
	public SubdomainGeneratedIdentifiers getGeneratedIdentifiers ()
	{
		return m_generatedIdentifiers;
	}

	/**
	 *
	 */
	public void capture ()
	{
		Map<Declaration, Object> setDecls = new TreeMap<> (m_comparatorDeclarations);
		for (Declaration decl : m_setDeclarationsToAdd.keySet ())
			setDecls.put (decl.clone (), CodeGeneratorData.OBJ_DUMMY);
		m_stackDeclarations.push (setDecls);
	}

	public void release ()
	{
		if (!m_stackDeclarations.isEmpty ())
			m_setDeclarationsToAdd = m_stackDeclarations.pop ();
	}

	/**
	 * Resets the declarations lists.
	 */
	public void reset ()
	{
		m_setDeclarationsToAdd.clear ();
		if (!m_stackDeclarations.isEmpty ())
			for (Declaration decl : m_stackDeclarations.peek ().keySet ())
				m_setDeclarationsToAdd.put (decl.clone (), CodeGeneratorData.OBJ_DUMMY);

		m_moManager.clear ();
		m_generatedIdentifiers.reset ();
	}

	public void setCreatingInitialization (boolean bIsCreatingInitialization)
	{
		m_bIsCreatingInitialization = bIsCreatingInitialization;
	}

	public boolean isCreatingInitialization ()
	{
		return m_bIsCreatingInitialization;
	}
}
