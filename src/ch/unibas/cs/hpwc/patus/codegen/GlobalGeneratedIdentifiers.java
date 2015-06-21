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
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Declaration;
import cetus.hir.Expression;
import cetus.hir.NameID;
import cetus.hir.PointerSpecifier;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.SizeofExpression;
import cetus.hir.Specifier;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.arch.TypeDeclspec;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.grammar.strategy.IAutotunerParam;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Class encapsulating all the generated global identifiers, i.e.,
 * the stencil function,
 *
 * @author Matthias-M. Christen
 */
public class GlobalGeneratedIdentifiers
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public enum EVariableType
	{
		OUTPUT_GRID (0x01),
		INPUT_GRID (0x02),
		SIZE_PARAMETER (0x04),
		KERNEL_PARAMETER (0x08),
		AUTOTUNE_PARAMETER (0x10),
		INTERNAL_AUTOTUNE_PARAMETER (0x20),
		INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER (0x40),
		INTERNAL_ADDITIONAL_KERNEL_PARAMETER (0x80),
		TRAPEZOIDAL_SIZE (0x100),
		TRAPEZOIDAL_TMAX (0x200),
		TRAPEZOIDAL_SLOPE (0x400);


		private int m_nMaskValue;

		private EVariableType (int nMaskValue)
		{
			m_nMaskValue = nMaskValue;
		}

		public int mask ()
		{
			return m_nMaskValue;
		}

		public boolean isGrid ()
		{
			return INPUT_GRID.equals (this) || OUTPUT_GRID.equals (this);
		}
		
		public boolean isAutotuningParameter ()
		{
			return AUTOTUNE_PARAMETER.equals (this) || INTERNAL_AUTOTUNE_PARAMETER.equals (this) || INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER.equals (this);
		}
	}

	/**
	 *
	 */
	public static class Variable
	{
		private EVariableType m_type;
		private List<Specifier> m_listSpecifiers;
		private Specifier m_specType;
		private String m_strName;
		private String m_strOriginalName;

		private Expression m_exprDefaultValue;

		/**
		 * Size of the array
		 */
		private Expression m_exprSize;

		/**
		 * Memory object box. Only valid for grid type variables.
		 */
		private Size m_sizeBox;
		
		/**
		 * The values this variable can take (for auto-tuning)
		 */
		private IAutotunerParam m_autoparam;

		private VariableDeclaration m_declaration;


		public Variable (EVariableType type, List<Specifier> listSpecifiers, String strName, String strOriginalName, Expression exprSize)
		{
			m_type = type;
			m_listSpecifiers = listSpecifiers;
			m_specType = findDatatype ();
			m_strName = strName;
			m_strName = strOriginalName;
			m_exprSize = exprSize;
			m_sizeBox = null;
			m_autoparam = null;
			m_exprDefaultValue = null;

			m_declaration = new VariableDeclaration (listSpecifiers, new VariableDeclarator (new NameID (strName)));
		}

		public Variable (EVariableType type, VariableDeclaration decl, String strOriginalName, Expression exprDefaultValue, Expression exprSize, Size sizeBox)
		{
			m_type = type;
			m_declaration = decl;
			m_listSpecifiers = decl.getSpecifiers ();
			m_specType = findDatatype ();
			m_strName = decl.getDeclarator (0).getID ().getName ();
			m_strOriginalName = strOriginalName;
			m_exprSize = exprSize;
			m_sizeBox = sizeBox;
			m_autoparam = null;
			m_exprDefaultValue = exprDefaultValue;
		}
		
		public Variable (EVariableType type, VariableDeclaration decl, String strOriginalName, Expression exprSize, IAutotunerParam autoparam)
		{
			m_type = type;
			m_declaration = decl;
			m_listSpecifiers = decl.getSpecifiers ();
			m_specType = findDatatype ();
			m_strName = decl.getDeclarator (0).getID ().getName ();
			m_strOriginalName = strOriginalName;
			m_exprSize = exprSize;
			m_sizeBox = null;
			m_autoparam = autoparam;
			m_exprDefaultValue = null;
		}

		/**
		 * Creates a new variable.
		 * @param type The variable type
		 * @param decl The variable declaration
		 * @param specOrig The specifier of the original type (of the underlying stencil node)
		 * @param strArgumentName The name of the variable
		 * @param data
		 */
		public Variable (EVariableType type, VariableDeclaration decl, String strArgumentName, String strOriginalName, Expression exprDefaultValue, CodeGeneratorSharedObjects data)
		{
			this (
				type,
				decl,
				strOriginalName,
				exprDefaultValue,
				type.isGrid () ?
					GlobalGeneratedIdentifiers.getGridMemorySize (decl, strArgumentName, data) :
					new SizeofExpression (decl.getSpecifiers ()),
				type.isGrid () ? GlobalGeneratedIdentifiers.getGridMemoryBox (decl, strArgumentName, data) : null
			);
		}

		public final EVariableType getType ()
		{
			return m_type;
		}

		public final boolean isGrid ()
		{
			return m_type.isGrid ();
		}
		
		public final boolean isAutotuningParameter ()
		{
			return m_type.isAutotuningParameter ();
		}

		public final List<Specifier> getSpecifiers ()
		{
			return m_listSpecifiers;
		}
		
		public final Specifier getDatatype ()
		{
			return m_specType;
		}
		
		private Specifier findDatatype ()
		{
			for (Specifier s : m_listSpecifiers)
				if (Globals.isBaseDatatype (s))
					return s;
			return null;	/* TODO: what should happen if no data type is found? */
		}

		public final String getName ()
		{
			return m_strName;
		}

		public final String getOriginalName ()
		{
			return m_strOriginalName;
		}
		
		public final Expression getDefaultValue ()
		{
			return m_exprDefaultValue;
		}

		public final Expression getSize ()
		{
			return m_exprSize;
		}

		public final Size getBoxSize ()
		{
			return m_sizeBox;
		}
		
		public final IAutotunerParam getAutotuneParam ()
		{
			return m_autoparam;
		}

		public final VariableDeclaration getDeclaration ()
		{
			return m_declaration;
		}

		@Override
		public String toString()
		{
			return StringUtil.concat (
				"<", m_type.toString (), "> ",
				StringUtil.join (m_listSpecifiers, ""), " ",
				m_strName, " [", m_exprSize.toString (), "]");
		}

		@Override
		public boolean equals (Object obj)
		{
			if (!(obj instanceof Variable))
				return false;

			Variable var = (Variable) obj;

			if (!m_strName.equals (var.getName ()))
				return false;
			if (m_type != var.getType ())
				return false;
			if (!m_listSpecifiers.equals (var.getSpecifiers ()))
				return false;
			if (!m_exprSize.equals (var.getSize ()))
				return false;

			return true;
		}

		@Override
		public int hashCode ()
		{
			return m_strName.hashCode () + m_type.hashCode () * 11;
		}
	}

	private static Size getGridMemoryBox (VariableDeclaration decl, String strArgumentName, CodeGeneratorSharedObjects data)
	{
		MemoryObject mo = data.getData ().getMemoryObjectManager ().getMemoryObject (
			data.getCodeGenerators ().getStrategyAnalyzer ().getRootSubdomain (),
			data.getStencilCalculation ().getReferenceStencilNode (strArgumentName),
			false);
		return mo.getSize ();
	}

	private static Expression getGridMemorySize (VariableDeclaration decl, String strArgumentName, CodeGeneratorSharedObjects data)
	{
		MemoryObject mo = data.getData ().getMemoryObjectManager ().getMemoryObject (
			data.getCodeGenerators ().getStrategyAnalyzer ().getRootSubdomain (),
			data.getStencilCalculation ().getReferenceStencilNode (strArgumentName),
			false);

		// get size of the original specifier (double / float)
		StencilCalculation.ArgumentType type = data.getStencilCalculation ().getArgumentType (strArgumentName);

		return new BinaryExpression (
			mo.getSize (/*new IntegerLiteral (0), data.getCodeGenerators ().getStrategyAnalyzer ().getMaximumTotalTimestepsCount ()*/).getVolume (),
			BinaryOperator.MULTIPLY,
			new SizeofExpression (CodeGeneratorUtil.specifiers (type.getSpecifier ())));
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;

	private NameID m_nidInitializeFunction;

	private NameID m_nidStencilFunction;
	
	private List<Variable> m_listVariables;
	
	private Map<String, Variable> m_mapVariables;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 *
	 * @param data
	 */
	public GlobalGeneratedIdentifiers (CodeGeneratorSharedObjects data)
	{
		m_data = data;
		m_listVariables = new LinkedList<> ();
		m_mapVariables = new HashMap<> ();
	}

	/**
	 * Returns the list of all variables.
	 * @return All the variables
	 */
	public List<Variable> getVariables ()
	{
		List<Variable> list = new ArrayList<> (m_listVariables.size ());
		list.addAll (m_listVariables);
		return list;
	}

	/**
	 * Returns a list of all variables of the type <code>type</code>.
	 * @param type The variable type
	 * @return All the variables of a certain type, <code>type</code>
	 */
	public List<Variable> getVariables (EVariableType type)
	{
		return getVariables (type.mask ());
	}

	/**
	 * Returns a list of all variables satisfying a type mask, <code>nVariableTypeMask</code>.
	 * E.g., to retrieve all input grids, use
	 * <pre>getVariables (EVariableType.INPUT_GRID.mask ()),</pre>
	 * to retrieve all input or output grids, use
	 * <pre>getVariables (EVariableType.INPUT_GRID.mask () | EVariableType.OUTPUT_GRID.mask ()),</pre>
	 * to get all variables but output grids, use
	 * <pre>getVariables (~EVariableType.OUTPUT_GRID.mask ())</pre>
	 * @param m_strType The variable type
	 * @return All the variables of a certain type, <code>type</code>
	 */
	public List<Variable> getVariables (int nVariableTypeMask)
	{
		List<Variable> list = new ArrayList<> (m_listVariables.size ());
		for (Variable v : m_listVariables)
			if ((v.getType ().mask () & nVariableTypeMask) != 0)
				list.add (v);
		return list;
	}

	/**
	 * Returns all the variables of type {@link EVariableType#INPUT_GRID}.
	 * @return All the input grid variables
	 */
	public List<Variable> getInputGrids ()
	{
		return getVariables (EVariableType.INPUT_GRID);
	}

	/**
	 * Returns all the variables of type {@link EVariableType#OUTPUT_GRID}.
	 * @return All the output grid variables
	 */
	public List<Variable> getOutputGrids ()
	{
		return getVariables (EVariableType.OUTPUT_GRID);
	}
	
	public List<Variable> getAutotuneVariables ()
	{
		return getVariables (
			EVariableType.AUTOTUNE_PARAMETER.mask () |
			EVariableType.INTERNAL_AUTOTUNE_PARAMETER.mask () |
			EVariableType.INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER.mask ()
		);
	}
	
	/**
	 * Finds and returns a variable by its name, or returns <code>null</code> if
	 * there is no variable with that name.
	 * 
	 * @param strVarName
	 *            The name of the variable to retrieve
	 * @return The variable named <code>strVarName</code> or <code>null</code>
	 *         if there is no such variable
	 */
	public Variable getVariableByName (String strVarName)
	{
		return m_mapVariables.get (strVarName);
	}

	public NameID getInitializeFunctionName ()
	{
		return m_nidInitializeFunction;
	}

	/**
	 * Returns a list of {@link NameID}s derived from the list of variables, <code>itVars</code>.
	 * @param itVars An iterable over the variables for which to create a list of declarations
	 * @param bOriginalTypes If set to <code>true</code>, the variable declarations use the original data types
	 * 	(as defined in the stencil specification, i.e. <code>float</code> and <code>double</code> (for grids and stencil
	 * 	params). If set to <code>false</code>, the corresponding derived types (e.g., SIMD types) are used
	 * @return
	 */
	private static List<Declaration> getDeclarations (Iterable<Variable> itVars, CodeGenerationOptions.ECompatibility compatibility)
	{
		boolean bIsFortranCompatible = compatibility == CodeGenerationOptions.ECompatibility.FORTRAN;

		List<Declaration> listDecls = new ArrayList<> ();
		for (Variable v : itVars)
		{
			List<Specifier> listSpecs = new ArrayList<> (v.getSpecifiers ().size () + 1);
			listSpecs.addAll (v.getSpecifiers ());
			if (bIsFortranCompatible && !v.isGrid ())
				listSpecs.add (PointerSpecifier.UNQUALIFIED);

			listDecls.add (new VariableDeclaration (listSpecs, new VariableDeclarator (new NameID (v.getName ()))));
		}

		return listDecls;
	}

	/**
	 *
	 * @return
	 */
	public VariableDeclaration getInitializeFunctionDeclaration (CodeGenerationOptions.ECompatibility compatibility)
	{
		if (m_nidInitializeFunction == null)
			return null;

		List<Specifier> listSpecs = new ArrayList<> ();
		listSpecs.addAll (m_data.getArchitectureDescription ().getDeclspecs (TypeDeclspec.KERNEL));
		listSpecs.add (Specifier.VOID);

		return new VariableDeclaration (
			listSpecs,
			new ProcedureDeclarator (
				m_nidInitializeFunction.clone (),
				GlobalGeneratedIdentifiers.getDeclarations (getVariables (
					~EVariableType.OUTPUT_GRID.mask () &
					~EVariableType.INTERNAL_AUTOTUNE_PARAMETER.mask () &
					~EVariableType.INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER.mask ()),
				compatibility)
			)
		);
	}

	/**
	 *
	 * @param nidInitializeFunction
	 */
	public void setInitializeFunctionName (NameID nidInitializeFunction)
	{
		m_nidInitializeFunction = nidInitializeFunction;
	}

	/**
	 * Returns the name of the (generic, parametrized) stencil function.
	 * @return
	 */
	public NameID getStencilFunctionName ()
	{
		return m_nidStencilFunction;
	}

	/**
	 *
	 * @return
	 */
	public VariableDeclaration getStencilFunctionDeclaration (CodeGenerationOptions.ECompatibility compatibility)
	{
		List<Specifier> listSpecs = new ArrayList<> ();
		listSpecs.addAll (m_data.getArchitectureDescription ().getDeclspecs (TypeDeclspec.KERNEL));
		listSpecs.add (Specifier.VOID);

		// if Fortran-compatibility mode is turned on, don't include the output grids in the stencil kernel declaration
		boolean bIsFortranCompatible = compatibility == CodeGenerationOptions.ECompatibility.FORTRAN;
		List<Variable> listVariables = getVariables (
			~EVariableType.INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER.mask () & (bIsFortranCompatible ? ~EVariableType.OUTPUT_GRID.mask () : ~0));

		return new VariableDeclaration (
			listSpecs,
			new ProcedureDeclarator (m_nidStencilFunction.clone (), getDeclarations (listVariables, compatibility)));
	}

	/**
	 *
	 * @param nidStencilFunction
	 */
	public void setStencilFunctionName (NameID nidStencilFunction)
	{
		m_nidStencilFunction = nidStencilFunction;
	}

	/**
	 *
	 * @param varArgument
	 */
	public void addStencilFunctionArguments (Variable varArgument)
	{
		if (!m_mapVariables.containsKey (varArgument.getName ()))
		{
			m_listVariables.add (varArgument);
			m_mapVariables.put (varArgument.getName (), varArgument);
		}
	}

	/**
	 * Returns a list of declarations.
	 * If <code>bIncludeOutputGrids</code> is set to <code>false</code>, no output grid variable declarations
	 * are contained in the result list.
	 * If <code>bIncludeAutotuneParameters</code> is set to <code>false</code>, no autotuner parameter variable declarations
	 * are contained in the result list.
	 * @param bIncludeOutputGrids
	 * @param bIncludeAutotuneParameters
	 * @param bMakeCompatibleWithFortran Flag indicating whether to make the parameter declarations compatible with Fortran, i.e., create pointer
	 * 	parameters (<code>bMakeCompatibleWithFortran == true</code>) or use the default C convention
	 * @return
	 */
	public List<Declaration> getFunctionParameterList (
		boolean bIncludeOutputGrids, boolean bIncludeAutotuneParameters, boolean bIncludeInternalAutotuneParameters, boolean bMakeCompatibleWithFortran)
	{
		List<Declaration> listDecls = new ArrayList<> (m_listVariables.size ());
		for (Variable v : getFunctionParameterVarList (bIncludeOutputGrids, bIncludeAutotuneParameters, bIncludeInternalAutotuneParameters, bMakeCompatibleWithFortran))
		{
			if (bMakeCompatibleWithFortran)
			{
				// add the pointer specifier
				List<Specifier> listSpecOrig = v.getDeclaration ().getSpecifiers ();
				if (listSpecOrig.get (listSpecOrig.size () - 1).equals (PointerSpecifier.UNQUALIFIED))
					listDecls.add (v.getDeclaration ());
				else
				{
					// add a pointer specifier if the parameter is not a pointer
					List<Specifier> listSpecs = new ArrayList<> (listSpecOrig.size () + 1);
					listSpecs.addAll (listSpecOrig);
					listSpecs.add (PointerSpecifier.UNQUALIFIED);

					listDecls.add (new VariableDeclaration (listSpecs, v.getDeclaration ().getDeclarator (0)));
				}
			}
			else
				listDecls.add (v.getDeclaration ());
		}

		return listDecls;
	}

	public List<Variable> getFunctionParameterVarList (
		boolean bIncludeOutputGrids, boolean bIncludeAutotuneParameters, boolean bIncludeInternalAutotuneParameters, boolean bMakeCompatibleWithFortran)
	{
		List<Variable> listVars = new ArrayList<> (m_listVariables.size ());
		for (Variable v : m_listVariables)
		{
			if (v.getType ().equals (EVariableType.OUTPUT_GRID) && (!bIncludeOutputGrids || bMakeCompatibleWithFortran))
				continue;
			if (v.getType ().equals (EVariableType.AUTOTUNE_PARAMETER) && !bIncludeAutotuneParameters)
				continue;
			if (v.getType ().equals (EVariableType.INTERNAL_AUTOTUNE_PARAMETER) && !bIncludeInternalAutotuneParameters)
				continue;
			if (v.getType ().equals (EVariableType.INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER))
				continue;

			listVars.add (v);
		}

		return listVars;
	}
	
	public String getDefinedVariableName (Variable var)
	{
		StringBuilder sb = new StringBuilder (m_data.getStencilCalculation ().getName ().toUpperCase ());
		sb.append ('_');
		sb.append (var.getName ().toUpperCase ());
		
		return sb.toString ();
	}
}
