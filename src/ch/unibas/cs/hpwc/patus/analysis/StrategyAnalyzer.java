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
package ch.unibas.cs.hpwc.patus.analysis;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cetus.hir.AnnotationStatement;
import cetus.hir.AssignmentExpression;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.DeclarationStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.IntegerLiteral;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.ast.BoundaryCheck;
import ch.unibas.cs.hpwc.patus.ast.Loop;
import ch.unibas.cs.hpwc.patus.ast.RangeIterator;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.Strategy;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.ASTUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class StrategyAnalyzer
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	private class GridHierarchyNode implements Iterable<GridHierarchyNode>
	{
		private SubdomainIdentifier m_sdidNode;
		private SubdomainIterator m_sdIterator;
		private GridHierarchyNode m_nodeParent;
		private List<GridHierarchyNode> m_listChildren;

		public GridHierarchyNode (SubdomainIdentifier sdid, SubdomainIterator sdit, GridHierarchyNode nodeParent)
		{
			m_sdidNode = sdid;
			m_sdIterator = sdit;
			m_nodeParent = nodeParent;

			m_listChildren = new LinkedList<> ();

			// add this node as a child to the parent node
			if (nodeParent != null)
				nodeParent.m_listChildren.add (this);

			// add the node to the global map
			if (m_mapSubdomainIdentifiers.containsKey (sdid))
				throw new RuntimeException (StringUtil.concat ("The subdomain identifier ", sdid.getName (), " occurs in more than one context in the strategy."));
			m_mapSubdomainIdentifiers.put (sdid, this);
		}

		public SubdomainIdentifier getIdentifier ()
		{
			return m_sdidNode;
		}

		public SubdomainIterator getIterator ()
		{
			return m_sdIterator;
		}

		public SubdomainIdentifier getParentIdentifier ()
		{
			return m_nodeParent == null ? null : m_nodeParent.m_sdidNode;
		}

		@Override
		public Iterator<GridHierarchyNode> iterator ()
		{
			return m_listChildren.iterator ();
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;

	/**
	 * The strategy to analyze
	 */
	private Strategy m_strategy;

	/**
	 * The root node of the grid hierarchy tree
	 */
	private GridHierarchyNode m_nodeGridHierarchyRoot;

	/**
	 * Maps subdomain identifiers to their corresponding node in the grid hierarchy.
	 * Note that theoretically a subdomain identifier can map to multiple nodes depending
	 * on the context, c.f. this example:
	 * <pre>
	 * 	for v in u
	 * 		for p in v
	 * 			for q in p
	 * 		for q in v
	 * </pre>
	 * We do not allow this in strategies, so we have a 1-to-1 correspondence.
	 */
	private Map<SubdomainIdentifier, GridHierarchyNode> m_mapSubdomainIdentifiers;

	/**
	 * Sizes of subdomains (value) on a specific parallelism level (key)
	 */
	private Map<Integer, Size> m_mapParallelismLevelDomainSizes;
	private Map<Integer, Size> m_mapParallelismLevelIteratorSizes;

	/**
	 * List of number of parallel units per loop level
	 */
	private List<Expression> m_listNumParallelUnitsPerLevel;

	/**
	 * List of lists of maximum values per level. The list {@link StrategyAnalyzer#m_listNumParallelUnitsPerLevel}
	 * contains element-wise max reductions of the entries in this list.
	 * (Calculation is done at the end of {@link StrategyAnalyzer#analyze()}.)
	 */
	private List<List<Expression>> m_listParallelUnitNumbersPerLevel;

	/**
	 * Map mapping each loop to its level within the loop nest
	 */
	private Map<Loop, Integer> m_mapLoopLevels;

	/**
	 * Loop indices of temporal iterators
	 */
	private Set<IDExpression> m_setTimeIndices;
	private Map<RangeIterator, Expression> m_mapTimeBlockingFactors;
	private RangeIterator m_itMainTemporalIterator;

	private Map<Loop, Expression> m_mapMaximumTotalTimesteps;

	private Map<Traversable, List<SubdomainIterator>> m_mapRelativeInnerMostSubdomainIterators;

	private Loop m_loopOuterMost;
	private SubdomainIterator m_sgitOuterMost;


	/**
	 * Total number of parallel units that theoretically can execute the code.
	 */
	private Expression m_exprParallelUnitsCount;


	/**
	 * The maximum loop nest depth discovered in the loop nest
	 */
	private int m_nMaxLoopNestDepth;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates the strategy analyzer object.
	 * 
	 * @param strategy
	 *            The strategy to analyze
	 */
	public StrategyAnalyzer (CodeGeneratorSharedObjects data)
	{
		m_data = data;
		m_strategy = m_data.getStrategy ();

		// initialize internal variables
		m_listNumParallelUnitsPerLevel = null;
		m_mapLoopLevels = null;
		m_nMaxLoopNestDepth = 0;

		m_loopOuterMost = null;
		m_sgitOuterMost = null;

		m_mapMaximumTotalTimesteps = new HashMap<> ();
		m_mapRelativeInnerMostSubdomainIterators = new HashMap<> ();

		// analyze the hierarchical structure of the grids
		m_mapSubdomainIdentifiers = new HashMap<> ();
		m_mapParallelismLevelDomainSizes = new HashMap<> ();
		m_mapParallelismLevelIteratorSizes = new HashMap<> ();
		m_mapParallelismLevelDomainSizes.put (0, new Size (m_data.getStencilCalculation ().getDomainSize ().getSize ()));
		
		if (m_strategy.getBody () != null)
			buildGridHierarchy (m_strategy.getBody (), null);

		// analyze the temporal structure
		m_setTimeIndices = new HashSet<> ();
		m_mapTimeBlockingFactors = new HashMap<> ();
		m_itMainTemporalIterator = null;
		findTemporalIterators ();
	}

	/**
	 * Builds the subdomain tree structure.
	 * 
	 * @param trvParent
	 *            The parent object in the strategy AST
	 * @param nodeParent
	 *            The parent subdomain node
	 */
	protected void buildGridHierarchy (Traversable trvParent, GridHierarchyNode nodeParent)
	{
		GridHierarchyNode nodeParentLocal = nodeParent;
		
		for (Traversable trvChild : trvParent.getChildren ())
		{
			if (trvChild instanceof SubdomainIterator)
			{
				SubdomainIterator it = (SubdomainIterator) trvChild;

				// if the parent node is null, create the root node of the hierarchy
				if (nodeParentLocal == null)
				{
					nodeParentLocal = new GridHierarchyNode (it.getDomainIdentifier (), null, null);
					if (m_nodeGridHierarchyRoot == null)
						m_nodeGridHierarchyRoot = nodeParentLocal;
					else
					{
						// the root has already been created
						// check whether the node created now and the node created before have the same subdomain identifier
						if (!m_nodeGridHierarchyRoot.getIdentifier ().equals (it))
							throw new RuntimeException ("The root grid in the strategy is ambiguous.");
					}
				}

				// record subdomain sizes on parallelism levels
				if (it.isParallelLevel ())
				{
					// check for existing subdomain size
					Size sizeSubdomain = m_mapParallelismLevelDomainSizes.get (it.getParallelismLevel ());
					if (sizeSubdomain != null)
					{
						// parallelism level already has a subdomain:
						// sizes must match, otherwise an exception is thrown
						if (!sizeSubdomain.equals (it.getDomainSubdomain ().getSize ()))
							throw new RuntimeException (StringUtil.concat ("Subdomain size mismatch on parallelism level", it.getParallelismLevel ()));
					}
					else
						m_mapParallelismLevelDomainSizes.put (it.getParallelismLevel (), sizeSubdomain = it.getDomainSubdomain ().getSize ());

					Size sizeIterator = m_mapParallelismLevelIteratorSizes.get (it.getParallelismLevel ());
					if (sizeIterator != null)
					{
						// parallelism level already has a subdomain:
						// sizes must match, otherwise an exception is thrown
						if (!sizeSubdomain.equals (it.getIteratorSubdomain ().getSize ()))
							throw new RuntimeException (StringUtil.concat ("Subdomain size mismatch on parallelism level", it.getParallelismLevel ()));
					}
					else
						m_mapParallelismLevelIteratorSizes.put (it.getParallelismLevel (), it.getIteratorSubdomain ().getSize ());
				}
				else if (m_mapParallelismLevelIteratorSizes.get (0) == null)
					m_mapParallelismLevelIteratorSizes.put (0, it.getIteratorSubdomain ().getSize ());					

				// create the iterator node and recursively build the structure
				buildGridHierarchy (trvChild, new GridHierarchyNode (it.getIterator (), it, nodeParentLocal));
			}
			else
				buildGridHierarchy (trvChild, nodeParentLocal);
		}
	}

	/**
	 * Returns the subdomain that is the root subdomain of the strategy, i.e.
	 * the &quot;outer most&quot; subdomain that is iterated over in the
	 * strategy. This is also the base from which the arguments to the stencil
	 * kernel are constructed.
	 * 
	 * @return The root subdomain
	 */
	public SubdomainIdentifier getRootSubdomain ()
	{
		return m_nodeGridHierarchyRoot == null ? null : m_nodeGridHierarchyRoot.getIdentifier ();
	}

	/**
	 * Returns the parent subdomain, given a {@link SubdomainIdentifier}
	 * <code>sdid</code>.
	 * 
	 * @param sdid
	 *            The subdomain identifier for which the parent subdomain is
	 *            sought
	 * @return The parent grid of <code>sgid</code>
	 */
	public SubdomainIdentifier getParentGrid (SubdomainIdentifier sdid)
	{
		GridHierarchyNode node = m_mapSubdomainIdentifiers.get (sdid);
		return node == null ? null : node.getParentIdentifier ();
	}

	/**
	 * Returns the {@link SubdomainIterator} in which the {@link SubdomainIdentifier} <code>sdid</code>
	 * occurs as the iterator.
	 * @param sdid
	 * @return
	 */
	public SubdomainIterator getIteratorForSubdomainIdentifier (SubdomainIdentifier sdid)
	{
		GridHierarchyNode node = m_mapSubdomainIdentifiers.get (sdid);
		return node == null ? null : node.getIterator ();
	}

	/**
	 * Returns a list of all the child subdomains of <code>sdidParent</code>.
	 * 
	 * @param sdidParent
	 *            The subdomain identifier whose children are sought
	 * @return A list of all the children of <code>sdidParent</code>
	 */
	public List<SubdomainIdentifier> getChildGrids (SubdomainIdentifier sdidParent)
	{
		List<SubdomainIdentifier> listChildren = new LinkedList<> ();
		for (GridHierarchyNode nodeChild : m_mapSubdomainIdentifiers.get (sdidParent))
			listChildren.add (nodeChild.getIdentifier ());
		return listChildren;
	}

	/**
	 * Returns the outermost loop of the strategy.
	 * @return The outermost loop
	 */
	public Loop getOuterMostLoop ()
	{
		if (m_loopOuterMost == null)
		{
			for (DepthFirstIterator it = new DepthFirstIterator (m_strategy.getBody ()); it.hasNext (); )
			{
				Object obj = it.next ();
				if (obj instanceof Loop)
					return m_loopOuterMost = (Loop) obj;
			}
		}

		return m_loopOuterMost;
	}

	public SubdomainIterator getOuterMostSubdomainIterator ()
	{
		if (m_sgitOuterMost == null)
		{
			for (DepthFirstIterator it = new DepthFirstIterator (m_strategy.getBody ()); it.hasNext (); )
			{
				Object obj = it.next ();
				if (obj instanceof SubdomainIterator)
					return m_sgitOuterMost = (SubdomainIterator) obj;
			}
		}

		return m_sgitOuterMost;
	}

	/**
	 * Finds temporal iterators and associates them with their time blocking factor.
	 */
	protected void findTemporalIterators ()
	{
		if (m_strategy.getBody () == null)
			return;
		
		for (DepthFirstIterator it = new DepthFirstIterator (m_strategy.getBody ()); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof RangeIterator)
			{
				RangeIterator loop = (RangeIterator) obj;

				// check whether this is a temporal loop
				boolean bIsTemporal = false;
				if (loop.isMainTemporalIterator ())
				{
					bIsTemporal = true;
					m_itMainTemporalIterator = loop;
				}
				else
				{
					for (IDExpression idIndex : m_setTimeIndices)
					{
						if (ASTUtil.dependsOn (loop.getStart (), idIndex))
						{
							bIsTemporal = true;
							break;
						}
					}
				}

				if (bIsTemporal)
				{
					m_setTimeIndices.add (loop.getLoopIndex ());
					m_mapTimeBlockingFactors.put (loop, LoopAnalyzer.getTripCount (loop));
				}
			}
		}
	}

	/**
	 * Determines whether the strategy implements temporal blocking.
	 * 
	 * @return <code>true</code> iff the strategy implements some form of time
	 *         blocking, i.e. whether within a block data is overwritten before
	 *         proceeding to the next block
	 */
	public boolean isTimeblocked ()
	{
		/*
		// find the "time" loop
		for (DepthFirstIterator it = new DepthFirstIterator (m_strategy.getBody ()); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof RangeIterator)
			{
				RangeIterator loop = (RangeIterator) obj;
				Expression exprEnd = loop.getEnd ();
				if ((exprEnd instanceof StencilProperty) && StencilProperty.T_MAX.equals (exprEnd))
				{
					// we found a loop whose "end" expression is T_MAX
					// check the step

					if (ExpressionUtil.isValue (loop.getStep (), 1))
						 return false;
					return true;
				}
			}
		}
		*/

		if (m_itMainTemporalIterator == null)
			return false;
		if (ExpressionUtil.isValue (m_itMainTemporalIterator.getStep (), 1))
			 return false;


		// if we're in doubt, assume the strategy is time-blocked
		return true;
	}

	/**
	 * Finds the loop index variable of the loop over the time dimension.
	 * 
	 * @return The time loop index variable identifier
	 */
	public IDExpression getTimeIndexVariable ()
	{
		/*
		// find the "time" loop
		for (DepthFirstIterator it = new DepthFirstIterator (m_strategy.getBody ()); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof RangeIterator)
			{
				RangeIterator loop = (RangeIterator) obj;
				Expression exprEnd = loop.getEnd ();
				if ((exprEnd instanceof StencilProperty) && StencilProperty.T_MAX.equals (exprEnd))
				{
					// we found a loop whose "end" expression is T_MAX
					// assume this is the temporal loop
					return loop.getLoopIndex ();
				}
			}
		}

		// index variable could not be found
		return null;
		*/

		if (m_itMainTemporalIterator == null)
			return null;
		return m_itMainTemporalIterator.getLoopIndex ();
	}
	
	public RangeIterator getMainTemporalIterator ()
	{
		return m_itMainTemporalIterator;
	}

	/**
	 * Returns the total number of virtual parallel units that are used
	 * to parallelize the code
	 * @return The total number of virtual parallel units
	 */
	public Expression getTotalParallelUnits ()
	{
		// make sure that the information has been gathered
		analyze ();

		// return the number of virtual parallel units
		return m_exprParallelUnitsCount;
	}

	/**
	 * Returns the maximum number of parallel units for a given loop level
	 * 
	 * @param nLevel
	 *            The loop level
	 * @return The number of parallel units for level <code>nLevel</code>
	 */
	public Expression getParallelUnitsForLevel (int nLevel)
	{
		// make sure that the information has been gathered
		analyze ();

		if (nLevel < 0 || nLevel >= m_listNumParallelUnitsPerLevel.size ())
			return new IntegerLiteral (1);
		return m_listNumParallelUnitsPerLevel.get (nLevel);
	}

	/**
	 * Returns the maximum depth of the loop nest.
	 * 
	 * @return The loop nest depth
	 */
	public int getMaximumLoopNestDepth ()
	{
		analyze ();
		return m_nMaxLoopNestDepth;
	}

	/**
	 * Analyzes the strategy.
	 */
	protected void analyze ()
	{
		if (m_mapLoopLevels != null)
			return;

		m_mapLoopLevels = new HashMap<> ();
		m_listParallelUnitNumbersPerLevel = new ArrayList<> ();

		// get the number of parallel units
		m_exprParallelUnitsCount = Symbolic.simplify (getParallelUnits (m_strategy.getBody (), m_nMaxLoopNestDepth), Symbolic.ALL_VARIABLES_INTEGER);

		// find the maximum number of units for each level
		m_listNumParallelUnitsPerLevel = new ArrayList<> ();
		for (List<Expression> listNumbersPerLevel : m_listParallelUnitNumbersPerLevel)
			m_listNumParallelUnitsPerLevel.add (ExpressionUtil.max (listNumbersPerLevel));
	}

	/**
	 * Build a structure of numbers of threads executing each parallel loop in the loop nest.
	 * @param stmtInput
	 */
	protected Expression getParallelUnits (Traversable trvParent, int nLevel)
	{
		// calculate the maximum number of threads (parallel units)
		//                  -----
		// #ParallelUnits =  | |  max  #Threads
		//                    i    j           i,j
		//
		// i is the level, j the number of the loop in that level

		List<Expression> listChildParallelUnitsCount = new LinkedList<> ();

		for (Traversable t : trvParent.getChildren ())
		{
			if (t instanceof Loop)
			{
				// calculate the maximum loop nest depth
				if (nLevel > m_nMaxLoopNestDepth)
					m_nMaxLoopNestDepth = nLevel;

				// recursively gather information
				Loop loop = (Loop) t;
				m_mapLoopLevels.put (loop, nLevel);
				listChildParallelUnitsCount.add (getParallelUnits (loop, nLevel + 1));
			}
			else if (t instanceof CompoundStatement)
				listChildParallelUnitsCount.add (getParallelUnits (t, nLevel));
		}

		// if the loop doesn't have any child loops just return the number of threads of this loop
		if (listChildParallelUnitsCount.size () == 0)
			return (trvParent instanceof Loop) ? ((Loop) trvParent).getNumberOfThreads () : new IntegerLiteral (1);

		// otherwise return the maximum of the thread numbers of the child loops
		long nMax = 0;
		Expression exprMax = null;

		for (Expression exprNumThreads : listChildParallelUnitsCount)
		{
			if (exprNumThreads instanceof IntegerLiteral)
				nMax = Math.max (nMax, ((IntegerLiteral) exprNumThreads).getValue ());
			else
			{
				if (exprMax == null)
					exprMax = exprNumThreads;
				else
					exprMax = ExpressionUtil.max (exprMax, exprNumThreads);
			}
		}

		// save the per-level max value
		for (int i = m_listParallelUnitNumbersPerLevel.size (); i <= nLevel; i++)
			m_listParallelUnitNumbersPerLevel.add (new LinkedList<Expression> ());
		List<Expression> listMax = m_listParallelUnitNumbersPerLevel.get (nLevel);
		listMax.add (exprMax == null ? new IntegerLiteral (nMax) : exprMax);

		// return #thds(loopParent) * max
		// merge nMax and exprMax
		Expression exprThisNumThreads = (trvParent instanceof Loop) ? ((Loop) trvParent).getNumberOfThreads () : new IntegerLiteral (1);
		if (exprMax == null)
		{
			if (exprThisNumThreads instanceof IntegerLiteral)
				return new IntegerLiteral (nMax * ((IntegerLiteral) exprThisNumThreads).getValue ());
			return new BinaryExpression (new IntegerLiteral (nMax), BinaryOperator.MULTIPLY, exprThisNumThreads);
		}
		if (nMax > 0)
			return new BinaryExpression (ExpressionUtil.max (exprMax, new IntegerLiteral (nMax)), BinaryOperator.MULTIPLY, exprThisNumThreads);
		return new BinaryExpression (exprMax, BinaryOperator.MULTIPLY, exprThisNumThreads);
	}

	/**
	 * Determines whether the statement <code>stmt</code> is a declaration of a
	 * strategy argument.
	 * 
	 * @param stmt
	 *            The statement to check
	 * @return <code>true</code> if <code>stmt</code> declares a strategy argument
	 */
	public boolean isStrategyArgumentDeclaration (Statement stmt)
	{
		if (stmt instanceof DeclarationStatement)
		{
			DeclarationStatement declstmt = (DeclarationStatement) stmt;
			if (declstmt.getDeclaration () instanceof VariableDeclaration)
			{
				VariableDeclaration decl = (VariableDeclaration) declstmt.getDeclaration ();
				return m_strategy.isParameter ((VariableDeclarator) decl.getDeclarator (0));
			}
		}

		return false;
	}

	/**
	 * <p>
	 * Determines whether data must be loaded within the subdomain iterator
	 * <code>it</code>.
	 * </p>
	 * <p>
	 * Data is loaded if:
	 * <ul>
	 * 	<li>the iterator is the last iterator in its parallelism level</li>
	 * 	<li><code>it</code> is the iterator above the point iterator containing a
	 * 		stencil call</li>
	 * 	<li>the iterator is no point iterator and contains a stencil call</li>
	 * </ul>
	 * </p>
	 * 
	 * @param it
	 *            The iterator to test
	 * @return <code>true</code> iff data is loaded in the iterator
	 *         <code>it</code>
	 */
	public boolean isDataLoadedInIterator (SubdomainIterator it, IArchitectureDescription desc)
	{
		// check whether the hardware requires data transfers on this level
		if (!desc.hasExplicitLocalDataCopies (it.getParallelismLevel ()))
			return false;

		// data is loaded
		// - if the iterator is the last iterator in its parallelism level (i.e., the nested iterator belongs to a new parallelism level)
		// - if it is the iterator above the point iterator containing a stencil call
		// - if the iterator is no point iterator and contains a stencil call

		int nMaxChildParallelismLevel = 0;
		for (GridHierarchyNode nodeChild : m_mapSubdomainIdentifiers.get (it.getIterator ()))
			nMaxChildParallelismLevel = Math.max (nodeChild.getIterator ().getParallelismLevel (), nMaxChildParallelismLevel);
		if (nMaxChildParallelismLevel > it.getParallelismLevel ())
			return true;

		if (isIteratorImmediatelyAbovePointIteratorWithStencilCall (it))
			return true;

		if (!it.getIteratorSubdomain ().getBox ().isPoint () && StrategyAnalyzer.directlyContainsStencilCall (it))
			return true;

		//	Remark: check whether that data has been transferred already in the parent
		//		    subdomain iterator (data volume might be to large for the local memory
		//          of the hardware -- how to handle this case?

		return false;
	}

	/**
	 * Determines recursively whether the traversable <code>trvParent</code> is
	 * an
	 * immediate parent of a point iterator.
	 * 
	 * @param trvParent
	 *            The traversable to check
	 * @return <code>true</code> if an iterator is found immediately below
	 *         <code>trvParent</code> that is a point iterator
	 */
	private boolean isIteratorImmediatelyAbovePointIteratorWithStencilCall (Traversable trvParent)
	{
		for (Traversable trvChild : trvParent.getChildren ())
		{
			if (trvChild instanceof SubdomainIterator)
			{
				// check whether this is a point iterator
				// if it is, we immediately know that trvParent is a parent of a point iterator
				// if not, proceed to the other siblings and continue to check
				if (((SubdomainIterator) trvChild).getIteratorSubdomain ().getBox ().isPoint ())
					return StrategyAnalyzer.directlyContainsStencilCall (trvChild);
			}
			else if (trvChild instanceof RangeIterator)
			{
				// a range iterator was found; the parent iterator trvParent is not
				// the immediate parent of a point iterator...
				// proceed to the next sibling
			}
			else
			{
				// statement other than an iterator:
				// check the subtree
				if (isIteratorImmediatelyAbovePointIteratorWithStencilCall (trvChild))
					return StrategyAnalyzer.directlyContainsStencilCall (trvChild);
			}
		}

		// no children...
		// this is the last node, and it is obviously not an iterator...
		return false;
	}

	/**
	 * Returns <code>true</code> iff <code>trv</code> is a stencil call (an
	 * expression statement that is a call to the formal stencil function).
	 * 
	 * @param trv
	 *            The traversable to test
	 * @return <code>true</code> iff <code>trv</code> is a stencil call
	 */
	public static boolean isStencilCall (Traversable trv)
	{
		return trv instanceof ExpressionStatement && StrategyAnalyzer.isStencilCall (((ExpressionStatement) trv).getExpression ());
	}

	/**
	 * Determines whether the {@link SubdomainIterator} it contains a stencil
	 * call as an immediate child.
	 * 
	 * @param it
	 *            The subdomain iterator to test
	 * @return <code>true</code> iff the subdomain iterator <code>it</code>
	 *         contains a stencil call immediately below it
	 */
	//	public static boolean directlyContainsStencilCall (SubdomainIterator it)
//	{
//		for (Traversable trv : it.getChildren ())
//		{
//			if (trv instanceof CompoundStatement && !(trv instanceof Loop))
//			{
//				if (StrategyAnalyzer.directlyContainsStencilCall ((CompoundStatement) trv))
//					return true;
//			}
//
//			if (StrategyAnalyzer.isStencilCall (trv))
//				return true;
//		}
//		return false;
//	}
//
	public static boolean directlyContainsStencilCall (Traversable trv)
	{
		for (Traversable trvChild : trv.getChildren ())
		{
			if (trvChild instanceof CompoundStatement && !(trvChild instanceof Loop))
			{
				if (StrategyAnalyzer.directlyContainsStencilCall (trvChild))
					return true;
			}
			else if (trvChild instanceof BoundaryCheck)
			{
				if (StrategyAnalyzer.directlyContainsStencilCall (((BoundaryCheck) trvChild).getWithChecks ()))
					return true;
				if (StrategyAnalyzer.directlyContainsStencilCall (((BoundaryCheck) trvChild).getWithoutChecks ()))
					return true;
			}
			else if (StrategyAnalyzer.isStencilCall (trvChild))
				return true;
		}

		return false;
	}

	/**
	 * Determines whether the traversable <code>trv</code> contains a stencil call somewhere in its subtree.
	 * @param trv
	 * @return
	 */
	public static boolean containsStencilCall (Traversable trv)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (trv); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof Traversable && StrategyAnalyzer.isStencilCall ((Traversable) obj))
				return true;
		}

		return false;
	}
	
	//private static boolean is

	/**
	 * Determines whether the {@link SubdomainIterator} it is eligible for stencil loop unrolling, i.e.
	 * if the loop only contains a stencil call.
	 * @param it
	 * @return
	 */
	public static boolean isEligibleForStencilLoopUnrolling (SubdomainIterator it)
	{
		boolean bContainsStencilCall = false;
		
		// if the loop body is a single statement it has to be a stencil call to be eligible for loop unrolling
		if (it.getLoopBody () instanceof ExpressionStatement)
			return StrategyAnalyzer.isStencilCall (it.getLoopBody ());

		for (Traversable trv : it.getLoopBody ().getChildren ())
		{
			if (trv instanceof ExpressionStatement)
			{
				if (StrategyAnalyzer.isStencilCall (((ExpressionStatement) trv).getExpression ()))
					bContainsStencilCall = true;
				else
				{
					// found something that is an expression statement, but isn't a stencil call
					// the loop is not eligible
					return false;
				}
			}
			else if (trv instanceof AnnotationStatement || trv instanceof CompoundStatement)
			{
				// ignore annotation statements
				continue;
			}
			else
			{
				// found something that isn't an expression statement or an annotation statement
				// assume that the loop is not eligible
				return false;
			}
		}

		return bContainsStencilCall;
	}

	/**
	 * Determines whether the loop <code>loop</code> contains a loop nested
	 * within <code>loop</code>.
	 * 
	 * @param loop
	 *            The loop to test
	 * @return <code>true</code> iff <code>loop</code> contains another loop
	 */
	public static boolean hasNestedLoops (Loop loop)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (loop); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj != loop && obj instanceof Loop)
				return true;
		}
		return false;
	}


	///////////////////////////////////////////////////////////////////
	// Static Functions

	/**
	 * Determines whether the expression <code>expr</code> is a call to the
	 * stencil function.
	 * Asserts that the expression <code>expr</code> is an assignment expression
	 * with a function call
	 * to {@link Globals#FNX_STENCIL} as right hand side.
	 * 
	 * @param expr
	 *            The expression to check
	 * @return <code>true</code> iff the expression <code>expr</code> is a call
	 *         to the stencil function
	 */
	public static boolean isStencilCall (Expression expr)
	{
		if (expr instanceof AssignmentExpression)
		{
			AssignmentExpression aexpr = (AssignmentExpression) expr;
			if (aexpr.getRHS () instanceof FunctionCall)
				return Globals.FNX_STENCIL.equals (((FunctionCall) aexpr.getRHS ()).getName ());
		}

		return false;
	}

	/**
	 * Assumes that the expression <code>expr</code> is a stencil call (can be verified by
	 * {@link StrategyAnalyzer#isStencilCall(Expression)}) and returns the argument to the
	 * stencil function.
	 * @param expr The expression which to interpret as stencil call and from which to extract
	 * 	the argument to the function
	 * @return The argument to the stencil function
	 */
	public static Expression getStencilArgument (Expression expr)
	{
		return ((FunctionCall) ((AssignmentExpression) expr).getRHS ()).getArgument (0);
	}

	/**
	 * Tries to find the maximum local timestep that is performed in the <code>loop</code> if it is a
	 * {@link RangeIterator} or in the temporal loop immediately below it if <code>loop</code> is a
	 * {@link SubdomainIterator}. If there is no temporal loop, 1 is returned. If the maximum timestep
	 * can't be determined, {@link Integer#MAX_VALUE} is returned.
	 * @param it
	 * @return
	 */
//	public int getMaximumTimstepOfTemporalIterator (Loop loop)
//	{
//		if (loop instanceof SubdomainIterator)
//		{
//			// find a temporal loop below this subdomain iterator
//			for (DepthFirstIterator it = new DepthFirstIterator (loop); it.hasNext (); )
//			{
//				Object obj = it.next ();
//				if (obj instanceof RangeIterator)
//					return getMaximumTimstepOfTemporalIterator ((RangeIterator) obj);
//				if (obj instanceof SubdomainIterator)
//					return 1;
//			}
//		}
//
//		if (!(loop instanceof RangeIterator))
//			return 1;
//
//		Expression exprTripCount = LoopAnalyzer.getConstantTripCount ((RangeIterator) loop);
//		return exprTripCount == null ? Integer.MAX_VALUE : (int) ((IntegerLiteral) exprTripCount).getValue ();
//	}

	public Expression getMaximumTimstepOfTemporalIterator (Loop loop)
	{
		RangeIterator itTemporalLoop = null;
		if (!(loop instanceof RangeIterator) || (loop instanceof RangeIterator && !isTemporalLoop ((RangeIterator) loop)))
		{
			for (DepthFirstIterator it = new DepthFirstIterator (loop); it.hasNext (); )
			{
				Object obj = it.next ();
				if (obj instanceof RangeIterator && isTemporalLoop ((RangeIterator) obj))
				{
					itTemporalLoop = (RangeIterator) obj;
					break;
				}
			}
		}

		return itTemporalLoop == null ? null : LoopAnalyzer.getTripCount (itTemporalLoop);
	}

	/**
	 * Returns the total number of timesteps in the strategy in non-synchronized temporal loops.
	 * @return The total number of timesteps in the strategy
	 */
	public Expression getMaximumTotalTimestepsCount ()
	{
		return getMaximumTotalTimestepsCount (null);
	}

	/**
	 * Returns the total number of timesteps in the strategy below the loop <code>loop</code> in non-synchronized loops
	 * (&quot;global timeblocking factor&quot;)
	 * @param loop The loop below which to search for and accumulate timesteps or <code>null</code> if the total number
	 * 	in the entire strategy is desired
	 * @return The total number of timesteps below <code>loop</code>
	 */
	public Expression getMaximumTotalTimestepsCount (Loop loop)
	{
		Expression exprMaximumTotalTimestepsCount = m_mapMaximumTotalTimesteps.get (loop);
		if (exprMaximumTotalTimestepsCount != null)
			return exprMaximumTotalTimestepsCount;

		Expression exprTotalTimesteps = null;
		for (DepthFirstIterator it = new DepthFirstIterator (loop == null ? m_strategy.getBody () : loop); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof RangeIterator && obj != m_itMainTemporalIterator /* (should be: "non-synchronized")*/ && isTemporalLoop ((RangeIterator) obj))
			{
				Expression exprTripCount = LoopAnalyzer.getTripCount ((RangeIterator) obj);
				if (exprTripCount instanceof IntegerLiteral)
				{
					long nValue = ((IntegerLiteral) exprTripCount).getValue ();
					if (nValue == 0)
						return new IntegerLiteral (0);
					if (nValue == 1)
						continue;
				}

				if (exprTotalTimesteps == null)
					exprTotalTimesteps = exprTripCount;
				else
					exprTotalTimesteps = new BinaryExpression (exprTotalTimesteps, BinaryOperator.MULTIPLY, exprTripCount.clone ());
			}
		}

		m_mapMaximumTotalTimesteps.put (
			loop,
			exprTotalTimesteps = exprTotalTimesteps == null ?
				new IntegerLiteral (1) :
				Symbolic.simplify (exprTotalTimesteps, Symbolic.ALL_VARIABLES_INTEGER)
		);
		return exprTotalTimesteps;
	}

//	/**
//	 * Finds the timeblocking factor for the loop below the spatial iterator <code>sgit</code>, but above the
//	 * AST object <code>trvStatement</code>. If there is no such loop, 1 is returned.
//	 * Parallel temporal iterators are assumed to have timeblocking factor 1.
//	 * For sequential temporal loops, the timeblocking factor is the loop step.
//	 * @param sgit
//	 * @param trvStatement
//	 * @return
//	 */
//	public Expression getTimeBlockingFactor (SubdomainIterator sgit, Traversable trvStatement)
//	{
//		if (sgit == null)
//			return new IntegerLiteral (1);
//
//		// find a temporal loop below the subdomain iterator
//		RangeIterator itTime = findTemporalLoopBetween (sgit, trvStatement);
//		if (itTime == null)
//			return new IntegerLiteral (1);
//
//		// if the loop is parallel, assume it isn't timeblocked
//		if (itTime.isParallel ())
//			return new IntegerLiteral (1);
//
//		// the loop step is the time blocking factor
//		return itTime.getStep ();
//	}
//
//	/**
//	 * Returns the time block size of the first timeblock that is found.
//	 * XXX !! only first time block !!
//	 * @return
//	 */
//	public Expression getTimeBlockSize ()
//	{
//		IDExpression idTimeIdx = getTimeIndexVariable ();
//		if (idTimeIdx == null)
//			return new IntegerLiteral (1);
//
//		for (DepthFirstIterator it = new DepthFirstIterator (m_strategy.getBody ()); it.hasNext (); )
//		{
//			Object obj = it.next ();
//			if (obj instanceof RangeIterator)
//			{
//				RangeIterator loop = (RangeIterator) obj;
//
//				// check whether both start and end indices depend on the time index
//				if (ASTUtil.dependsOn (loop.getStart (), idTimeIdx) && ASTUtil.dependsOn (loop.getEnd (), idTimeIdx))
//				{
//					if (!loop.getStep ().equals (new IntegerLiteral (1)))
//						throw new RuntimeException ("Time loops must have 1 increments");
//
//					// calculate the number of iterations
//					return Symbolic.simplify (new BinaryExpression (
//						new BinaryExpression (loop.getEnd (), BinaryOperator.SUBTRACT, loop.getStart ()),
//						BinaryOperator.ADD,
//						new IntegerLiteral (1)));
//				}
//			}
//		}
//
//		return new IntegerLiteral (1);
//	}

	/**
	 * Determines whether <code>loop</code> is a temporal iterator.
	 * @param loop The loop to check
	 * @return <code>true</code> iff <code>loop</code> is a temporal iterator
	 */
	public boolean isTemporalLoop (RangeIterator loop)
	{
		return m_setTimeIndices.contains (loop.getLoopIndex ());
	}

	/**
	 * Determines whether the iterator <code>loop</code> is the innermost temporal iterator.
	 * @param loop The loop to check
	 * @return <code>true</code> iff <code>loop</code> is the innermost temporal iterator
	 */
	public boolean isInnerMostTemporalLoop (RangeIterator loop)
	{
		if (!isTemporalLoop (loop))
			return false;

		for (DepthFirstIterator it = new DepthFirstIterator (loop); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof RangeIterator && obj != loop)
				if (isTemporalLoop ((RangeIterator) obj))
					return false;
		}

		return true;
	}
	
	public static boolean isInnerMostParallelLoop (Loop loop)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (loop); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof Loop && obj != loop)
				if (((Loop) obj).isParallel ())
					return false;
		}

		return true;
	}

	/**
	 * Finds the temporal loop enclosing the HIR object <code>trv</code>.
	 * @param trv
	 * @return
	 */
	public RangeIterator getEnclosingTemporalLoop (Traversable trv)
	{
		Traversable trvParent = trv;
		while (trvParent != null)
		{
			trvParent = trvParent.getParent ();
			if (trvParent instanceof RangeIterator && isTemporalLoop ((RangeIterator) trvParent))
				return ((RangeIterator) trvParent);
		}

		return null;
	}

	/**
	 * Finds the inner most subdomain iterator relative to <code>trv</code> that isn't
	 * &quot;obscured&quot; by a temporal loop, i.e., finds the inner most subdomain
	 * iterator immediately above a containing temporal loop if there is any.
	 * @param
	 * @return
	 */
	public List<SubdomainIterator> getRelativeInnerMostSubdomainIterators (Traversable trv)
	{
		List<SubdomainIterator> list = m_mapRelativeInnerMostSubdomainIterators.get (trv);
		if (list == null)
			list = getRelativeInnerMostSubdomainIteratorsRecursive (trv, trv);
		return list;
	}

	private List<SubdomainIterator> getRelativeInnerMostSubdomainIteratorsRecursive (Traversable trv, Traversable trvOrig)
	{
		List<SubdomainIterator> list = new ArrayList<> ();

		for (Traversable trvChild : trv.getChildren ())
		{
			if (trv instanceof SubdomainIterator)
			{
				List<SubdomainIterator> listChild = getRelativeInnerMostSubdomainIteratorsRecursive (trvChild, trvOrig);
				if (listChild.isEmpty ())
					list.add ((SubdomainIterator) trv);
				else
					list.addAll (listChild);
			}
			else if (trv instanceof RangeIterator && isTemporalLoop ((RangeIterator) trv) && trv != trvOrig)
				return list;
			else
				list.addAll (getRelativeInnerMostSubdomainIteratorsRecursive (trvChild, trvOrig));
		}

		return list;
	}

	/**
	 * Finds the time blocking factor of the HIR object <code>trv</code>, i.e.,
	 * searches the enclosing temporal loop and returns the time blocking factor
	 * defined in that loop.
	 * @param trv
	 * @return
	 */
	public Expression getTimeBlockingFactor (Traversable trv)
	{
		//////XXX
		// only returns the TBF within the outer most temporal loop for now...
		
		if (m_strategy.getBody () == null)
			return new IntegerLiteral (1);

		for (DepthFirstIterator it = new DepthFirstIterator (m_strategy.getBody ()); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof RangeIterator && isTemporalLoop ((RangeIterator) obj))
				return ((RangeIterator) obj).getStep ();
		}
		return new IntegerLiteral (1);

		//////XXX
	}

	/**
	 * Finds the local timestep for the statement (typically a stencil call)
	 * <code>trvStatement</code> below the subdomain iterator <code>sgit</code>.
	 * If no temporal iterator is found between <code>sgit</code> and
	 * <code>trvStatement</code>,
	 * the local timestep defaults to <code>0</code>.
	 * 
	 * @param sgit
	 * @param trvStatement
	 * @param exprTemporalExpression
	 *            The temporal expression that is used as basis to calculate the
	 *            local timestep:
	 *            If a range iterator is found, the local timestep is computed
	 *            as <code>exprTemporalExpression - loopStartValue</code>.
	 *            If <code>exprTemporalExpression</code> is <code>null</code>,
	 *            the loop's index identifier is used instead as
	 *            temporal expression.
	 * @return the local timestep for the statement <code>trvStatement</code>
	 */
	public Expression getLocalTimestep (SubdomainIterator sgit, Traversable trvStatement, Expression exprTemporalExpression)
	{
		// find a temporal loop below the subdomain iterator
		if (sgit == null)
			return new IntegerLiteral (0);
		//RangeIterator itTime = findTemporalLoopBetween (sgit, trvStatement);
		RangeIterator itTime = getEnclosingTemporalLoop (sgit);
		if (itTime == null || itTime == m_itMainTemporalIterator)
			return new IntegerLiteral (0);

		return Symbolic.simplify (
			new BinaryExpression (
				(exprTemporalExpression == null ? itTime.getLoopIndex () : exprTemporalExpression).clone (),
				BinaryOperator.SUBTRACT,
				itTime.getStart ()),
			Symbolic.ALL_VARIABLES_INTEGER);
	}

//	private RangeIterator findTemporalLoopBetween (Traversable trvUpperBound, Traversable trvLowerBound)
//	{
//		for (Traversable trvChild : trvUpperBound.getChildren ())
//		{
//			if (trvChild instanceof RangeIterator)
//				return (RangeIterator) trvChild;
//			if (trvChild == trvLowerBound)
//				return null;
//
//			RangeIterator itResult = findTemporalLoopBetween (trvChild, trvLowerBound);
//			if (itResult != null)
//				return itResult;
//		}
//
//		return null;
//	}

	/**
	 * Returns the number of parallelism levels used in the strategy.
	 * @return
	 */
	public int getParallelismLevelsCount ()
	{
		return m_mapParallelismLevelDomainSizes.size ();
	}

	/**
	 * Returns the size of the subdomain on parallelism level
	 * <code>nParallelismLevel</code> or <code>null</code> if there is no such
	 * parallelism level.
	 * 
	 * @param nParallelismLevel
	 *            The parallelism level for which to retrive the subdomain size
	 * @return The subdomain size on parallelism level
	 *         <code>nParallelismLevel</code>
	 */
	public Size getDomainSizeForParallelismLevel (int nParallelismLevel)
	{
		return m_mapParallelismLevelDomainSizes.get (nParallelismLevel);
	}
	
	public Size getIteratorSizeForParallismLevel (int nParallelismLevel)
	{
		return m_mapParallelismLevelIteratorSizes.get (nParallelismLevel);		
	}
}
