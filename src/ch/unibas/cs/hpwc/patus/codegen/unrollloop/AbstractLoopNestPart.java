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
package ch.unibas.cs.hpwc.patus.codegen.unrollloop;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cetus.hir.AnnotationStatement;
import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.DeclarationStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FloatLiteral;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.NullStatement;
import cetus.hir.Statement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.codegen.Operators;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public abstract class AbstractLoopNestPart implements ILoopNestPart
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public static class IndexVariable
	{
		private Identifier m_idIndexVariable;
		private Statement m_stmtInitialization;

		public IndexVariable (Identifier idIndexVariable, Statement stmtInitialization)
		{
			m_idIndexVariable = idIndexVariable;
			m_stmtInitialization = stmtInitialization;
		}
	}

	public static class LoopStep
	{
		private BinaryOperator m_oprStep;
		private Expression m_exprStep;

		public LoopStep (BinaryOperator oprStep, Expression exprStep)
		{
			m_oprStep = oprStep;
			m_exprStep = exprStep;
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The loop to analyze
	 */
	private ForLoop m_loop;

	/**
	 * The number of the loop within the loop nest, couting from the outermost to the innermost loop
	 */
	private int m_nLoopNumber;

	/**
	 * Shared data
	 */
	private UnrollLoopSharedObjects m_data;

	/**
	 * The loop index variable
	 */
	private Identifier m_idLoopIndex;

	/**
	 * The loop index variable initialization statement
	 */
	private Statement m_stmtInitialization;

	/**
	 * The operator used to modify the loop index variable
	 */
	private BinaryOperator m_oprStep;

	/**
	 * The step expression
	 */
	private Expression m_exprStep;

	/**
	 * The map of new variables that are declared within the loop body and
	 * need to be replaced when unrolling.
	 * Each entry in the list contains the identifier for the corresponding unroll index.
	 */
	private Map<Integer, Map<IDExpression, List<IDExpression>>> m_mapNewVariablesMaps;
	private Map<Integer, Set<IDExpression>> m_mapDuplicatedVariablesSets;

	/**
	 * The data of the parent loop
	 */
	private ILoopNestPart m_lnpPrevLoop;

	/**
	 * The data of the next deeper nested loop in the nest
	 */
	private ILoopNestPart m_lnpNextLoop;


	private Map<Integer, LoopNest> m_mapUnrolledLoopNest;
	private Map<Integer, LoopNest> m_mapCleanupLoopNest;


	///////////////////////////////////////////////////////////////////
	// Implementation: Initialization

	/**
	 * Creates the loop nest part object.
	 */
	protected AbstractLoopNestPart ()
	{
		m_mapNewVariablesMaps = null;
		m_mapNewVariablesMaps = null;

		m_mapUnrolledLoopNest = null;
		m_mapCleanupLoopNest = null;
	}

	@Override
	public void init (UnrollLoopSharedObjects data, ForLoop loop, int nLoopNumber)
	{
		m_data = data;
		m_loop = loop;
		m_nLoopNumber = nLoopNumber;

		// initialize the map lists for new and duplicated variables
		m_mapNewVariablesMaps = new HashMap<Integer, Map<IDExpression, List<IDExpression>>> ();
		m_mapDuplicatedVariablesSets = new HashMap<Integer, Set<IDExpression>> ();

		// extract the loop index  variable
		IndexVariable idx = extractIndexVariable ();
		m_idLoopIndex = idx.m_idIndexVariable;
		m_stmtInitialization = idx.m_stmtInitialization;
		m_data.getLoopIndices ().put (m_idLoopIndex, true);

		// extract the loop increment operator and expression
		LoopStep step = extractStep ();
		m_oprStep = step.m_oprStep;
		m_exprStep = step.m_exprStep;
	}


	///////////////////////////////////////////////////////////////////
	// Implementation: Structure Information

	@Override
	public ForLoop getLoop ()
	{
		return m_loop;
	}

	@Override
	public ILoopNestPart getParent ()
	{
		return m_lnpPrevLoop;
	}

	@Override
	public void setParent (ILoopNestPart loop)
	{
		m_lnpPrevLoop = loop;
	}

	@Override
	public boolean hasChildLoops ()
	{
		return m_lnpNextLoop != null;
	}

	@Override
	public ILoopNestPart getChild ()
	{
		return m_lnpNextLoop;
	}

	@Override
	public void setChild (ILoopNestPart loop)
	{
		m_lnpNextLoop = loop;
	}

	/**
	 * Returns the data object that allows to share data with all the objects
	 * involved in the loop unrolling process.
	 * @return The shared data object
	 */
	protected UnrollLoopSharedObjects getSharedData ()
	{
		return m_data;
	}

	/**
	 * Returns the unroll factor for this loop nest part.
	 * @return The loop nest part's unroll factor
	 */
	public int[] getUnrollFactor ()
	{
		return m_data.getUnrollingFactorsForLoop (m_nLoopNumber);
	}

	/**
	 * Restricts all the unrolling factors of the loop with number <code>nLoopNumber</code> to a
	 * single value, <code>nUnrollingFactor</code>.
	 * (Handle complete unrolling.)
	 * @param nUnrollingFactor The unrolling factor to which to restrict to in the unrolling configurations
	 * @see UnrollLoopSharedObjects#restrictUnrollingFactorTo(int, int)
	 */
	public void restrictUnrollingFactorTo (int nUnrollingFactor)
	{
		m_data.restrictUnrollingFactorTo (m_nLoopNumber, nUnrollingFactor);
	}


	///////////////////////////////////////////////////////////////////
	// Implementation: Loop Information

	@Override
	public IDExpression getLoopIndex ()
	{
		return m_idLoopIndex;
	}

	@Override
	public IDExpression getEndValueIdentifier ()
	{
		return new NameID (m_idLoopIndex.getName () + "__end__");
	}

	/**
	 *
	 * @return
	 */
	public BinaryOperator getStepOperator ()
	{
		return m_oprStep;
	}

	/**
	 *
	 * @return
	 */
	public Expression getStepExpression ()
	{
		return m_exprStep;
	}


	///////////////////////////////////////////////////////////////////
	// Implementation: Loop Information Extraction

	/**
	 * Extracts the index variable and the statement initializing the index variable
	 * from the loop structure and returns it.
	 * If the index variable is declared in the loop head, the declaration is extracted
	 * and added to the list of unrolled statements such that the declaration occurs
	 * before the loop.
	 * If no index variable can be found, an {@link IllegalArgumentException}
	 * is thrown, and the loop is marked non-unrollable.
	 * @return The loop index variable
	 */
	protected abstract IndexVariable extractIndexVariable ();

	/**
	 * Extracts the step operator and the step expression (i.e. the increment by which the loop index variable
	 * is modified in each iteration via the operator {@link AbstractLoopNestPart#getStepOperator()}.
	 * @return The a {@link LoopStep} object containing the step operator and the step expression
	 */
	protected abstract LoopStep extractStep ();

	/**
	 * Derives a condition expression that is suitable for the unrolled loop.
	 * @return A condition expression for the unrolled loop
	 */
	protected List<Expression> getUnrolledLoopConditionExpression ()
	{
		List<Expression> listConditions = new ArrayList<Expression> (m_data.getUnrollFactorsCount ());

		if (m_loop.getCondition () instanceof BinaryExpression)
		{
			BinaryExpression bexprCondition = (BinaryExpression) m_loop.getCondition ();

			// check whether the loop index occurs in the expression
			boolean bHasLoopIndex = false;
			for (DepthFirstIterator it = new DepthFirstIterator (bexprCondition); it.hasNext (); )
			{
				Object obj = it.next ();
				if (obj instanceof Identifier && m_idLoopIndex.equals (obj))
				{
					bHasLoopIndex = true;
					break;
				}
			}

			// modify the RHS of the expression
			if (bHasLoopIndex)
			{
				for (int nUnrollFactor : m_data.getUnrollingFactorsForLoop (m_nLoopNumber))
				{
					listConditions.add (new BinaryExpression (
						Symbolic.simplify (new BinaryExpression (
							bexprCondition.getLHS ().clone (),
							m_oprStep,
							getMultiStepInternal (nUnrollFactor - 1))),
						bexprCondition.getOperator (),
						bexprCondition.getRHS ().clone ()));
				}
			}
			else
			{
				for (int k = 0; k < m_data.getUnrollFactorsCount (); k++)
					listConditions.add (m_loop.getCondition ().clone ());
			}
		}
		else
		{
			// if none of the above holds, just return the original condition
			for (int k = 0; k < m_data.getUnrollFactorsCount (); k++)
				listConditions.add (m_loop.getCondition ().clone ());
		}

		return listConditions;
	}

	/**
	 * Returns an expression in which the step has been applied <code>nExponent</code> times.
	 * @param nExponent The expression counting how many times the step is applied
	 * @return An expression in which the step has been applied <code>nExponent</code> times
	 */
	abstract protected Expression getMultiStep (int nExponent);

	private Expression getMultiStepInternal (int nExponent)
	{
		Expression exprMultiStep = getSharedData ().getStepCache ().get (nExponent);
		if (exprMultiStep == null)
		{
			exprMultiStep = getMultiStep (nExponent);
			getSharedData ().getStepCache ().put (nExponent, exprMultiStep);
		}

		return exprMultiStep.clone ();
	}

	/**
	 * Returns the step expression for the unrolled loop.
	 * @return The step expression for the unrolled loop
	 */
	private List<Expression> getUnrolledStepExpression ()
	{
		List<Expression> listSteps = new ArrayList<Expression> (m_data.getUnrollFactorsCount ());
		for (int nUnrollFactor : m_data.getUnrollingFactorsForLoop (m_nLoopNumber))
		{
			listSteps.add (new AssignmentExpression (
				m_idLoopIndex.clone (),
				Operators.getAssignmentOperatorFromBinaryOperator (m_oprStep),
				getMultiStepInternal (nUnrollFactor)));
		}

		return listSteps;
	}

	/**
	 *
	 * @param id
	 * @param nUnrollFactor
	 */
	protected void addDuplicatedVariable (IDExpression id, int nUnrollFactor)
	{
		// find the set to which to add the identifier
		Set<IDExpression> set = m_mapDuplicatedVariablesSets.get (nUnrollFactor);
		if (set == null)
			m_mapDuplicatedVariablesSets.put (nUnrollFactor, set = new HashSet<IDExpression> ());

		// add the identifier to the set
		set.add (id);
	}

	/**
	 *
	 * @param id
	 * @param listReplacementIdentifiers
	 * @param nUnrollFactor
	 */
	protected void addNewVariable (IDExpression id, List<IDExpression> listReplacementIdentifiers, int nUnrollFactor)
	{
		// find the map to which to add the identifier
		Map<IDExpression, List<IDExpression>> map = m_mapNewVariablesMaps.get (nUnrollFactor);
		if (map == null)
			m_mapNewVariablesMaps.put (nUnrollFactor, map = new HashMap<IDExpression, List<IDExpression>> ());

		// add the (id, listReplacementIdentifiers) key-value pair to the map
		map.put (id, listReplacementIdentifiers);
	}

	/**
	 * Determines whether there exists a replacement for the variable <code>id</code> for the unroll factor <code>nUnrollFactor</code>.
	 * @param id The variable identifier to check
	 * @param nUnrollFactor The unroll factor
	 * @return <code>true</code> iff <code>id</code> is replaced in an <code>nUnrollFactor</code>-fold unrolled version of the loop
	 */
	protected boolean hasNewVariableFor (IDExpression id, int nUnrollFactor)
	{
		Map<IDExpression, List<IDExpression>> map = m_mapNewVariablesMaps.get (nUnrollFactor);
		if (map == null)
			return false;
		return map.containsKey (id);
	}

	/**
	 * Returns the variable identifier
	 * @param id
	 * @param nUnrollIndex
	 * @param nUnrollFactor
	 * @return
	 */
	protected IDExpression getNewVariableFor (IDExpression id, int nUnrollIndex, int nUnrollFactor)
	{
		return m_mapNewVariablesMaps.get (nUnrollFactor).get (id).get (nUnrollIndex).clone ();
	}

	/**
	 * Determines whether the variable <code>exprVariable</code> has been duplicated for the unrolling
	 * factor <code>nUnrollFactor</code>.
	 * @param exprVariable The variable to test as an expression
	 * @param nUnrollFactor The unrolling factor to look at
	 * @return <code>true</code> iff the variable <code>exprVariable</code> has been duplicated for the
	 * 	unrolling factor <code>nUnrollFactor</code>
	 */
	protected boolean isVariableDuplicated (Expression exprVariable, int nUnrollFactor)
	{
		Set<IDExpression> set = m_mapDuplicatedVariablesSets.get (nUnrollFactor);
		if (set == null)
			return false;
		return set.contains (exprVariable);
	}

	/**
	 * Finds variable declarations in the original block and duplicates them.
	 * The variables are added to the variable set <code>m_setNewVariables</code>
	 * that will be used by {@link UnrollLoop#duplicateBlock(int)} to substitute
	 * the variables accordingly.
	 */
	protected void findVariableDeclarations (CompoundStatement cmpstmtOrig, CompoundStatement cmpstmtUnrolled, int nUnrollFactor)
	{
		// duplicate the index variable
		if (m_data.isCreatingTemporariesForLoopIndices ())
		{
			List<IDExpression> listTemps = new ArrayList<IDExpression> ();
			for (int i = 1; i < nUnrollFactor; i++)
			{
				Identifier idTmp = SymbolTools.getTemp (cmpstmtUnrolled, m_idLoopIndex);
				listTemps.add (idTmp);	//???
				addDuplicatedVariable (idTmp, nUnrollFactor);
				///m_mapDuplicatedVariablesMaps.get (nUnrollFactor).put (idTmp, true);
			}
			///m_listNewVariablesMaps.get (nUnrollIndex).put (m_idLoopIndex, listTemps);
			addNewVariable (m_idLoopIndex, listTemps, nUnrollFactor);

			for (int i = 1; i < nUnrollFactor; i++)
			{
				cmpstmtUnrolled.addStatement (new ExpressionStatement (new AssignmentExpression (
					listTemps.get (i),
					AssignmentOperator.NORMAL,
					Symbolic.simplify (new BinaryExpression (m_idLoopIndex, m_oprStep, getMultiStepInternal (i))))));
			}
		}

		// find variable declarations and duplicate them
		for (Traversable t : cmpstmtOrig.getChildren ())
		{
			if (t instanceof DeclarationStatement)
			{
				DeclarationStatement stmt = (DeclarationStatement) t;
				if (stmt.getDeclaration () instanceof VariableDeclaration)
				{
					// we found a variable declaration...
					VariableDeclaration declaration = (VariableDeclaration) stmt.getDeclaration ();

					for (int j = 0; j < declaration.getNumDeclarators (); j++)
					{
						VariableDeclarator declarator = (VariableDeclarator) declaration.getDeclarator (j);

						// check if the variable is assigned a value that depends on the loop index
						if (dependsOnLoopIndex (declarator, cmpstmtOrig))
						{
							// if yes, duplicate the variable declaration
							List<IDExpression> listTemps = new ArrayList<IDExpression> ();
							for (int i = 0; i < nUnrollFactor; i++)
							{
								Identifier idNew = SymbolTools.getTemp (cmpstmtUnrolled, declaration.getSpecifiers (), declarator.getSymbolName ());
								listTemps.add (idNew);
								///m_listDuplicatedVariablesMaps.get (nUnrollIndex).put (idNew, true);
								addDuplicatedVariable (idNew, nUnrollFactor);

								if (declarator.getInitializer () != null)
								{
									Expression exprInit = ((Expression) declarator.getInitializer ().getChildren ().get (0)).clone ();
									replaceIdentifiers (exprInit, i, nUnrollFactor);
									cmpstmtUnrolled.addStatement (new ExpressionStatement (new AssignmentExpression (idNew, AssignmentOperator.NORMAL, exprInit)));
								}
							}

							///m_listNewVariablesMaps.get (nUnrollIndex).put (id, listTemps);
							addNewVariable (declarator.getID (), listTemps, nUnrollFactor);
						}
						else
							cmpstmtUnrolled.addDeclaration (declaration.clone ());
					}
				}
			}
		}
	}

	/**
	 * Checks whether the identifier <code>id</code> is assigned a value somewhere
	 * within the loop body that depends on the loop index.
	 * @param id The identifier to check
	 * @return <code>true</code> iff <code>id</code> is assigned a value that depends
	 * 	on the loop index
	 */
	protected boolean dependsOnLoopIndex (Symbol id, CompoundStatement cmpstmtBody)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (cmpstmtBody); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof DeclarationStatement)
			{
				if (((DeclarationStatement) obj).getDeclaration () instanceof VariableDeclaration)
				{
					VariableDeclaration decl = (VariableDeclaration) ((DeclarationStatement) obj).getDeclaration ();
					for (int i = 0; i < decl.getNumDeclarators (); i++)
					{
						if (decl.getDeclarator (i) instanceof VariableDeclarator)
						{
							VariableDeclarator d = (VariableDeclarator) decl.getDeclarator (i);
							if (d.equals (id) && d.getInitializer () != null && (d.getInitializer () instanceof ValueInitializer))
								return dependsOnLoopIndex (((ValueInitializer) d.getInitializer ()).getValue ());
						}
					}
				}
			}
			else if (obj instanceof AssignmentExpression)
			{
				AssignmentExpression aexpr = (AssignmentExpression) obj;
				Expression exprLHS = aexpr.getLHS ();
				if (exprLHS instanceof Identifier)
				{
					if (((Identifier) exprLHS).getSymbol ().equals (id))
						return dependsOnLoopIndex (aexpr.getRHS ());
				}
				else if (aexpr.getLHS ().equals (id))
					return dependsOnLoopIndex (aexpr.getRHS ());
			}
		}

		return false;
	}

	/* (non-Javadoc)
	 * @see ch.unibas.cs.hpwc.patus.codegen.ILoopNestPart#getNestedLoop()
	 */
	@Override
	public ForLoop getNestedLoop ()
	{
		// check whether the loop has a body
		if (m_loop == null)
			return null;
		if (m_loop.getBody () == null)
			return null;

		// check whether the loop body is a ForLoop
		if (m_loop.getBody () instanceof ForLoop)
			return (ForLoop) m_loop.getBody ();

		// not a ForLoop... check whether it is a ForLoop nested somewhere within CompoundStatements
		if (m_loop.getBody () instanceof CompoundStatement)
		{
			for (DepthFirstIterator it = new DepthFirstIterator (m_loop.getBody ()); it.hasNext (); )
			{
				Object obj = it.next ();
				if (obj instanceof CompoundStatement)
					;
				else if (obj instanceof ForLoop)
					return (ForLoop) obj;
				else
					return null;
			}
		}

		// nothing found...
		return null;
	}

	@Override
	public boolean dependsOnLoopIndex (Expression expr)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof Identifier)
				if (m_idLoopIndex.equals (obj))
					return true;
		}

		return false;
	}

	@Override
	public boolean dependsOnlyOnLoopIndices (Expression expr)
	{
		boolean bDependsOnLoopIndex = false;
		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof Identifier)
			{
				if (!m_data.getLoopIndices ().containsKey (obj))
					return false;
				bDependsOnLoopIndex = true;
			}
		}

		return bDependsOnLoopIndex;
	}


	///////////////////////////////////////////////////////////////////
	// Implementation: Operations

	/**
	 * Replaces identifiers (the loop index variable, the variables that have been
	 * declared within the loop body that depend on the loop index variable), with
	 * the corresponding variables that have been created for the unrolled loop.
	 * @param trvParent The parent traversable in which the replacement takes place
	 * @param nUnrollFactor The unroll factor for which the identifiers are replaced
	 */
	protected void replaceIdentifiers (Traversable trvParent, int nUnrollIndex, int nUnrollFactor)
	{
		for (Traversable trvChild : trvParent.getChildren ())
		{
			if (trvChild instanceof VariableDeclaration)
				break;
			else if (trvChild instanceof Identifier)
			{
				Identifier id = (Identifier) trvChild;
				if (id.equals (m_idLoopIndex))
				{
					Expression exprReplaceIndex = getUnrolledIndex (nUnrollIndex, nUnrollFactor);
					if (!exprReplaceIndex.equals (m_idLoopIndex))
						id.swapWith (exprReplaceIndex);
				}
				else if (hasNewVariableFor (id, nUnrollFactor))
				{
					//id.swapWith (new Identifier (id.getName () + nUnrollIndex));
					id.swapWith (getNewVariableFor (id, nUnrollIndex, nUnrollFactor));
				}
			}
			else if (trvChild instanceof ArrayAccess)
				replaceIdentifiers (trvChild, nUnrollIndex, nUnrollFactor);
			else if (trvChild instanceof Expression)
			{
				replaceIdentifiers (trvChild, nUnrollIndex, nUnrollFactor);

				// we don't want to simplify literals
				boolean bIsLiteral = (trvChild instanceof IntegerLiteral) || (trvChild instanceof FloatLiteral);

				// we don't want to simplify assignment expressions either
				boolean bIsAssignmentExpression = trvChild instanceof AssignmentExpression;

				// no point in simplifying function calls (simplify the arguments)
				boolean bIsFunctionCall = trvChild instanceof FunctionCall;

				// we don't want to simplify if the parent is a binary expression: in that case we try to simplify the parent directly
				boolean bIsParentExpression = (trvChild.getParent () instanceof BinaryExpression) || (trvChild.getParent () instanceof UnaryExpression);

				// ... however, if the parent is an assignment expression and this child is its RHS, then simplify
				boolean bIsRHSOfAssignment = trvChild.getParent () instanceof AssignmentExpression && ((AssignmentExpression) trvChild.getParent ()).getRHS () == trvChild;

				// simplify...
				if (!bIsLiteral && !bIsAssignmentExpression && !bIsFunctionCall && (!bIsParentExpression || (bIsParentExpression && bIsRHSOfAssignment)))
					((Expression) trvChild).swapWith (Symbolic.optimizeExpression ((Expression) trvChild));
			}
		}
	}

	/**
	 * Returns the expression with which the loop index variable has to be replace for
	 * a particular unrolling index, <code>nUnrollIndex</code>
	 * @param nUnrollIndex The loop unrolling index
	 * @return The expression replacing the loop index variable
	 */
	protected Expression getUnrolledIndex (int nUnrollIndex, int nUnrollFactor)
	{
		if (m_data.isCreatingTemporariesForLoopIndices ())
			return m_mapNewVariablesMaps.get (nUnrollFactor).get (m_idLoopIndex).get (nUnrollIndex).clone ();

		return new BinaryExpression (m_idLoopIndex.clone (), m_oprStep, getMultiStepInternal (nUnrollIndex));
	}

	private void createUnrolledLoopHeads ()
	{
		// create a list that will contain the unrolled loop nests for each unrolling configurations (contained in m_listUnrollFactor)
		m_mapUnrolledLoopNest = new HashMap<Integer, LoopNest> ();
		m_mapCleanupLoopNest = new HashMap<Integer, LoopNest> ();

		// get condition and step expressions of the unrolled loop nests
		List<Expression> listConditions = getUnrolledLoopConditionExpression ();
		List<Expression> listSteps = getUnrolledStepExpression ();

		// create a new loop nest with the unrolled code for each of the unrolling configurations
		int i = 0;
		for (int j : m_data.getUnrollingFactorsForLoop (m_nLoopNumber))
		{
			m_mapUnrolledLoopNest.put (j, new LoopNest (m_stmtInitialization.clone (), listConditions.get (i), listSteps.get (i), new NullStatement ()));
			m_mapCleanupLoopNest.put (j,
				j <= 1 ?
					null :
					new LoopNest (
						new ExpressionStatement (new AssignmentExpression (
							m_idLoopIndex.clone (),
							AssignmentOperator.NORMAL,
							getEndValueIdentifier ())),
						m_loop.getCondition ().clone (),
						m_loop.getStep ().clone (),
						new NullStatement ()));

			i++;
		}
	}

	@Override
	public LoopNest getUnrolledLoopHead (int nUnrollFactor)
	{
		if (m_mapUnrolledLoopNest == null)
			createUnrolledLoopHeads ();
		return m_mapUnrolledLoopNest.get (nUnrollFactor);
	}

	@Override
	public LoopNest getCleanupLoopHead (int nUnrollFactor)
	{
		if (m_mapCleanupLoopNest == null)
			createUnrolledLoopHeads ();
		return m_mapCleanupLoopNest.get (nUnrollFactor);
	}

	@Override
	public CompoundStatement unrollBody (CompoundStatement cmpstmtBody, int nUnrollFactor)
	{
		// nothing to do if the unroll factor is 1
		if (nUnrollFactor == 1)
			return cmpstmtBody;

		CompoundStatement cmpstmtInput = new CompoundStatement ();
		removeNestedCompoundStatements (cmpstmtBody, cmpstmtInput);

		CompoundStatement cmpstmtNewBody = new CompoundStatement ();

		// generate temporary variable declarations if required and adds them to cmpstmtUnrolled
		findVariableDeclarations (cmpstmtInput, cmpstmtNewBody, nUnrollFactor);

		for (int i = 0; i < nUnrollFactor; i++)
			unrollStatements (cmpstmtInput, cmpstmtNewBody, i, nUnrollFactor);

		return cmpstmtNewBody;
	}

	private void removeNestedCompoundStatements (CompoundStatement cmpstmtOrig, CompoundStatement cmpstmtNew)
	{
		for (Traversable t : cmpstmtOrig.getChildren ())
		{
			if (t instanceof CompoundStatement)
				removeNestedCompoundStatements ((CompoundStatement) t, cmpstmtNew);
			else if (t instanceof Statement)
				cmpstmtNew.addStatement (((Statement) t).clone ());
		}
	}

	/**
	 *
	 * @param cmpstmtIn
	 * @param cmpstmtAddCode
	 * @param nCurrentFactor
	 * @param nUnrollFactor
	 */
	private void unrollStatements (CompoundStatement cmpstmtIn, CompoundStatement cmpstmtAddCode, int nCurrentFactor, int nUnrollFactor)
	{
		Map<Expression, Expression> mapAssignments = new HashMap<Expression, Expression> ();

		for (Traversable t : (cmpstmtIn.clone ()).getChildren ())
		{
			if (t instanceof AnnotationStatement)
				cmpstmtAddCode.addStatement (((Statement) t).clone ());
			else if (t instanceof CompoundStatement)
			{
				CompoundStatement cmpstmtNew = new CompoundStatement ();
				unrollStatements ((CompoundStatement) t, cmpstmtNew, nCurrentFactor, nUnrollFactor);
				cmpstmtAddCode.addStatement (cmpstmtNew);
			}
			else if ((t instanceof Statement) && !(t instanceof DeclarationStatement))
			{
				// check whether this is an assignment to a duplicated variable
				boolean bIsConstAssignmentToDuplicate = false;

				if (t instanceof ExpressionStatement)
				{
					// get the expression; if it is an assignment expression, decide whether we can omit it
					// (whether the same value has been assigned to it in a previous unroll pass)
					Expression expr = ((ExpressionStatement) t).getExpression ();
					if (expr instanceof AssignmentExpression && AssignmentOperator.NORMAL.equals (((AssignmentExpression) expr).getOperator ()))
					{
						Expression exprLHS = ((AssignmentExpression) expr).getLHS ();
						Expression exprRHS = ((AssignmentExpression) expr).getRHS ();

						// if the LHS is a variable that has been duplicated, check whether the same expression
						// (which by definition is directly dependent of the loop index) has been assigned to it.
						// If this is the case and the value depends _only_ on the loop index, omit the assignment.
						// Otherwise, if it also depends on other variables, include it in the unrolled body
						// for safety (we should check whether the other variables depend on the loop index/whether
						// they have been altered somewhere in the loop body in order to determine whether this
						// assignment statement can be actually left out).
						if (isVariableDuplicated (exprLHS, nUnrollFactor))
						{
							if (dependsOnlyOnLoopIndices (exprRHS))
							{
								Expression exprRHSCached = mapAssignments.get (exprLHS);
								if (exprRHSCached != null && exprRHSCached.equals (exprRHS))
									bIsConstAssignmentToDuplicate = true;
								else
									mapAssignments.put (exprLHS, exprRHS);
							}
						}
					}
				}

				// if the statement isn't an assignment to a duplicated variable that has been assigned the same
				// value before, add the statement to the unrolled body (replacing loop and duplicated identifiers)
				if (!bIsConstAssignmentToDuplicate)
				{
					replaceIdentifiers (t, nCurrentFactor, nUnrollFactor);
					cmpstmtAddCode.addStatement (((Statement) t).clone ());
				}
			}
		}
	}


	///////////////////////////////////////////////////////////////////
	// Object Overrides

	@Override
	public String toString ()
	{
		return StringUtil.concat (
			"Loop index: ", m_idLoopIndex,
			"\nLoop step: ", m_oprStep, ", ", getUnrolledStepExpression (),
			"\nUnroll factor: ", Arrays.toString (m_data.getUnrollingFactorsForLoop (m_nLoopNumber)),
			"\n\nChild loop:\n", m_lnpNextLoop == null ? "(none)" : m_lnpNextLoop);
	}
}
