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
package ch.unibas.cs.hpwc.patus.representation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import cetus.hir.Expression;
import cetus.hir.FloatLiteral;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.PointerSpecifier;
import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.ast.StencilProperty;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObjectManager;
import ch.unibas.cs.hpwc.patus.codegen.ProjectionMask;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.geometry.Box;
import ch.unibas.cs.hpwc.patus.geometry.Vector;
import ch.unibas.cs.hpwc.patus.grammar.stencil.Parser;
import ch.unibas.cs.hpwc.patus.grammar.stencil.Scanner;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * The StencilCalculation encapsulates the stencil structure ({@link StencilBundle}),
 * the boundary treatment, and the stopping criteria.
 *
 * @author Matthias-M. Christen
 */
public class StencilCalculation
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public enum EArgumentType
	{
		INPUT_GRID,
		OUTPUT_GRID,
		PARAMETER
	}

	public static class ArgumentType
	{
		protected EArgumentType m_type;
		protected Specifier m_specType;

		@SuppressWarnings("unused")
		private ArgumentType ()
		{
			// class can't be instantiated
		}

		protected ArgumentType (EArgumentType type, Specifier specType)
		{
			m_type = type;
			m_specType = specType;
		}

		public EArgumentType getType ()
		{
			return m_type;
		}

		public Specifier getSpecifier ()
		{
			return m_specType;
		}

		public List<Specifier> getSpecifiers ()
		{
			return CodeGeneratorUtil.specifiers (m_specType, m_type != EArgumentType.PARAMETER ? PointerSpecifier.UNQUALIFIED : null);
		}
	}

	public static class GridType extends ArgumentType
	{
		private int m_nVectorIdx;
		private Box m_boxDimension;

		public GridType (EArgumentType type, Specifier specType, int nVectorIdx, Box boxDimension)
		{
			super (type, specType);
			if (type != EArgumentType.INPUT_GRID && type != EArgumentType.OUTPUT_GRID)
				throw new RuntimeException ("The argument type must be one of INPUT_GRID, OUTPUT_GRID.");

			m_specType = specType;
			m_nVectorIdx = nVectorIdx;
			m_boxDimension = boxDimension;
		}

		public int getVectorIndex ()
		{
			return m_nVectorIdx;
		}

		public Box getBoxDimension ()
		{
			return m_boxDimension;
		}
	}

	public static class ParamType extends ArgumentType
	{
		private int[] m_rgDimensions;

		public ParamType (Specifier specType, List<Integer> listDimensions)
		{
			super (EArgumentType.PARAMETER, specType);

			// copy the dimensions
			m_rgDimensions = new int[listDimensions.size ()];
			int i = 0;
			for (int nDim : listDimensions)
				m_rgDimensions[i++] = nDim;
		}

		public int[] getDimensions ()
		{
			return m_rgDimensions;
		}

		public int getDimensionsCount ()
		{
			return m_rgDimensions.length;
		}
	}

	public enum EGridSizeOptions
	{
		/**
		 * Chooses grid sizes automatically depending on the references in the stencil specification
		 */
		AUTOMATIC,

		/**
		 * Makes all base grids of equal size
		 */
		EQUAL_BASE;
	}


	///////////////////////////////////////////////////////////////////
	// Static Members

	/**
	 * Loads and parses the stencil calculation from file <code>strFilename</code>.
	 * @param strFilename The file to load
	 * @return The stencil calculation object described in <code>strFilename</code>
	 */
	public static StencilCalculation load (String strFilename, CodeGenerationOptions options)
	{
		Parser parser = new Parser (new Scanner (strFilename));
		parser.setOptions (options);
		parser.Parse ();
		if (parser.hasErrors ())
			throw new RuntimeException ("Parsing the stencil specification failed.");

		return parser.getStencilCalculation ();
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The name of the stencil calculation. This name will be used as name of the kernel
	 * in the generated code.
	 */
	private String m_strName;

	/**
	 * The stencil operation arguments in the order they are specified in the stencil specification
	 */
	private List<String> m_listArguments;

	/**
	 * Maps argument names to the argument types
	 */
	private Map<String, ArgumentType> m_mapArguments;

	/**
	 * Maps argument names to reference stencil nodes
	 */
	private Map<String, StencilNode> m_mapReferenceNodes;

	/**
	 * The set of input stencil nodes that correspond to the base memory objects
	 */
	private StencilNodeSet m_stencilNodeSetBaseInput;

	/**
	 * The set of output stencil nodes that correspond to the base memory objects
	 */
	private StencilNodeSet m_stencilNodeSetBaseOutput;

	/**
	 * The actual stencil
	 */
	private StencilBundle m_stencil;

	/**
	 * The maximum number of iterations
	 */
	private Expression m_exprMaxIter;

	/**
	 * The domain filter
	 */
	private DomainFilter m_filter;

	/**
	 * The symbolic size of the grid on which the stencil is executed
	 */
	private Box m_boxDomain;

	/**
	 * The grid size options, defaults to {@link EGridSizeOptions#AUTOMATIC}.
	 */
	private EGridSizeOptions m_optGridSize;

	/**
	 * A list of size parameters to the stencil definition (contained in the
	 * domain size definition and the optional size parameters to the grids)
	 */
	private List<NameID> m_listSizeParameters;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Creates the stencil calculation object.
	 * @param strName The name of the stencil calculation which will be used as kernel
	 * 	name in the generated code
	 */
	public StencilCalculation (String strName)
	{
		m_strName = strName;

		m_stencil = null;
		m_boxDomain = null;
		m_exprMaxIter = StencilProperty.getMaxTime ().clone ();

		// Create the maps for grids and parameters
		// Note that the grid is initialized from the StencilNodes and created lazily
		// when the data is first required
		m_listArguments = new ArrayList<String> ();
		m_mapArguments = new HashMap<String, ArgumentType> ();
		m_mapReferenceNodes = new HashMap<String, StencilNode> ();

		m_listSizeParameters = new LinkedList<NameID> ();

		// initialize defaults
		m_optGridSize = EGridSizeOptions.AUTOMATIC;
	}

	/**
	 * Returns the name of the stencil calculation, i.e. the name of the kernel in the
	 * generated code.
	 * @return The name of the stencil calculation
	 */
	public String getName ()
	{
		return m_strName;
	}

	/**
	 *
	 * @return
	 */
	public StencilBundle getStencilBundle ()
	{
		return m_stencil;
	}

	/**
	 *
	 * @param stencil
	 */
	public void setStencil (StencilBundle stencil)
	{
		m_stencil = stencil;

		ProjectionMask mask = new ProjectionMask (Vector.getZeroVector (getDimensionality ()));
		m_stencilNodeSetBaseInput = new StencilNodeSet (m_stencil.getFusedStencil (), StencilNodeSet.ENodeTypes.INPUT_NODES).applyMask (mask);
		m_stencilNodeSetBaseOutput = new StencilNodeSet (m_stencil.getFusedStencil (), StencilNodeSet.ENodeTypes.OUTPUT_NODES).applyMask (mask);
	}

	/**
	 *
	 * @param filter
	 */
	public void setDomainFilter (DomainFilter filter)
	{
		m_filter = filter;
	}

	public DomainFilter getDomainFilter ()
	{
		return m_filter;
	}

	public void setBoundaryTreatment ()
	{

	}

//	public void setStoppingCriteria ()
//	{
//
//	}

	public void setMaxIterations (Expression exprMaxIter)
	{
		if (exprMaxIter == null)
			m_exprMaxIter = StencilProperty.getMaxTime ().clone ();
		else
		{
			if (exprMaxIter instanceof FloatLiteral)
				m_exprMaxIter = new IntegerLiteral ((long) Math.ceil (((FloatLiteral) exprMaxIter).getValue ()));
			else
				m_exprMaxIter = exprMaxIter.clone ();
		}
	}

	/**
	 * Sets the (symbolic) grid box on which the stencil is executed.
	 * @param boxGrid The grid box
	 */
	public void setDomainSize (Box boxGrid)
	{
		m_boxDomain = boxGrid;
	}

	/**
	 * Returns the (symbolic) size of the grid on which the stencil is executed.
	 * @return The grid size
	 */
	public Box getDomainSize ()
	{
		return m_boxDomain;
	}

	public void setGridSizeOptions (EGridSizeOptions opt)
	{
		m_optGridSize = opt;
	}

	public EGridSizeOptions getGridSizeOptions ()
	{
		return m_optGridSize;
	}

	/**
	 *
	 * @param strArgumentName
	 * @param nodeset
	 * @param type
	 */
	private void addStencilNodes (String strArgumentName, StencilNodeSet nodeset, Box boxDimension, EArgumentType type)
	{
		for (StencilNode node : nodeset)
		{
			if (strArgumentName.equals (node.getName ()))
			{
				String strNodeArgName = MemoryObjectManager.createMemoryObjectName (null, node, null, true);
				ArgumentType atNodeType = new GridType (type, node.getSpecifier (), node.getIndex ().getVectorIndex (), boxDimension);

				m_listArguments.add (strNodeArgName);
				m_mapArguments.put (strNodeArgName, atNodeType);
				m_mapReferenceNodes.put (strNodeArgName, node);
			}
		}
	}

	/**
	 * Adds a stencil operation argument.
	 * @param strArgumentName The name of the argument
	 * @param type Specifies the type of the parameter if the argument is a parameter
	 * 	or <code>null</code> if the type is a grid
	 */
	public void addStencilOperationArgument (String strArgumentName, Box boxDimension, ArgumentType type)
	{
		if (type == null)
		{
			addStencilNodes (strArgumentName, m_stencilNodeSetBaseInput, boxDimension, EArgumentType.INPUT_GRID);
			addStencilNodes (strArgumentName, m_stencilNodeSetBaseOutput, boxDimension, EArgumentType.OUTPUT_GRID);
		}
		else
		{
			m_listArguments.add (strArgumentName);
			m_mapArguments.put (strArgumentName, type);
		}
	}

	public void addSizeParameter (NameID nidSizeParam)
	{
		m_listSizeParameters.add (nidSizeParam);
	}

	public Iterable<NameID> getSizeParameters ()
	{
		return m_listSizeParameters;
	}

	/**
	 * Returns an iterable over all the stencil operation arguments.
	 * @return An iterable over the stencil arguments
	 */
	public Iterable<String> getArguments ()
	{
		return m_listArguments;
	}

	/**
	 * Returns all the arguments of type <code>type</code>.
	 * @param type The type of the arguments that are sought
	 * @return An iterable over all the arguments that match <code>type</code>
	 */
	public Iterable<String> getArguments (EArgumentType type)
	{
		List<String> listArgs = new ArrayList<String> ();
		for (String strArg : m_listArguments)
			if (m_mapArguments.get (strArg).getType ().equals (type))
				listArgs.add (strArg);
		return listArgs;
	}

	/**
	 * Returns the argument type for a given argument name.
	 * @param strArgumentName The argument name
	 * @return The argument for the name <code>strArgumentName</code> or <code>null</code> if
	 * 	no argument with name <code>strArgumentName</code> exists
	 */
	public ArgumentType getArgumentType (String strArgumentName)
	{
		return m_mapArguments.get (strArgumentName);
	}

	public GridType getArgumentType (int nVecIdx)
	{
		for (ArgumentType argType : m_mapArguments.values ())
			if ((argType instanceof GridType) && ((GridType) argType).getVectorIndex () == nVecIdx)
				return (GridType) argType;
		return null;
	}

	public boolean isArgument (String strArgumentName)
	{
		return m_mapArguments.containsKey (strArgumentName);
	}

	/**
	 * Returns a reference node for a given argument name.
	 * @param strArgumentName The argument name
	 * @return A reference stencil node corresponding to the argument
	 */
	public StencilNode getReferenceStencilNode (String strArgumentName)
	{
		return m_mapReferenceNodes.get (strArgumentName);
	}

	/**
	 *
	 * @return
	 */
	public StencilNodeSet getInputBaseNodeSet ()
	{
		return m_stencilNodeSetBaseInput;
	}

	/**
	 *
	 * @return
	 */
	public StencilNodeSet getOutputBaseNodeSet ()
	{
		return m_stencilNodeSetBaseOutput;
	}

	public byte getDimensionality ()
	{
		return m_stencil.getDimensionality ();
	}

	public Expression getMaxIterations ()
	{
		return m_exprMaxIter;
	}

	@Override
	public String toString ()
	{
		return StringUtil.concat (
			"> Stencil \"", m_strName, "\":\n",
			m_stencil,
			"\n\n> Filter:\n",
			"\n\n> Boundary Treatment:\n",
			"\n\n> Stopping Criteria:\n");
	}

	public String getStencilExpressions ()
	{
		StringBuilder sb = new StringBuilder ();
		for (Stencil stencil : m_stencil)
		{
			sb.append (stencil.getStencilExpression ());
			sb.append ("\n");
		}
		return sb.toString ();
	}
}
