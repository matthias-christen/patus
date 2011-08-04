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

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

import cetus.hir.AnnotationStatement;
import cetus.hir.ArrayAccess;
import cetus.hir.ArraySpecifier;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.BreakStatement;
import cetus.hir.Case;
import cetus.hir.CommentAnnotation;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Declarator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.Identifier;
import cetus.hir.Initializer;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.NestedDeclarator;
import cetus.hir.PointerSpecifier;
import cetus.hir.Procedure;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.SizeofExpression;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.SwitchStatement;
import cetus.hir.TranslationUnit;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.analysis.HIRAnalyzer;
import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.arch.TypeDeclspec;
import ch.unibas.cs.hpwc.patus.ast.Parameter;
import ch.unibas.cs.hpwc.patus.ast.ParameterAssignment;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.StencilSpecifier;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.ASTUtil;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class CodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static boolean SINGLE_ASSIGNMENT = false;

	private final static Logger LOGGER = Logger.getLogger (CodeGenerator.class);

	private final static DateFormat DATE_FORMAT = new SimpleDateFormat ("yyyy/MM/dd HH:mm:ss");


	///////////////////////////////////////////////////////////////////
	// Inner Types

	private abstract class GeneratedProcedure
	{
		private NameID m_nidFunctionName;
		private List<Declaration> m_listParams;
		private StatementListBundle m_slbBody;
		private TranslationUnit m_unit;

		public GeneratedProcedure (NameID nidFunctionName, List<Declaration> listParams, StatementListBundle slbBody, TranslationUnit unit)
		{
			m_nidFunctionName = nidFunctionName;
			m_listParams = listParams;
			m_slbBody = slbBody;
			m_unit = unit;
		}

		public final NameID getFunctionName ()
		{
			return m_nidFunctionName;
		}

		public final List<Declaration> getParams ()
		{
			return m_listParams;
		}

		public final StatementListBundle getBodyCodes ()
		{
			return m_slbBody;
		}

		public final TranslationUnit getTranslationUnit ()
		{
			return m_unit;
		}

		/**
		 *
		 * @param listAdditionalDeclSpecs List of additional declspecs for the function. Can be <code>null</code> if no
		 * 	additional specifiers are required.
		 * @param listParams
		 * @param cmpstmtBody
		 * @param unit
		 */
		public void addProcedureDeclaration (List<Specifier> listAdditionalDeclSpecs, CompoundStatement cmpstmtBody, boolean bIncludeStencilCommentAnnotation)
		{
			addProcedureDeclaration (listAdditionalDeclSpecs, m_listParams, cmpstmtBody, bIncludeStencilCommentAnnotation);
		}

		public void addProcedureDeclaration (List<Specifier> listAdditionalDeclSpecs, List<Declaration> listParams, CompoundStatement cmpstmtBody, boolean bIncludeStencilCommentAnnotation)
		{
			setGlobalGeneratedIdentifiersFunctionName (m_nidFunctionName);
			addProcedureDeclaration (listAdditionalDeclSpecs, m_nidFunctionName.getName (), listParams, cmpstmtBody, bIncludeStencilCommentAnnotation);
		}

		public void addProcedureDeclaration (List<Specifier> listAdditionalDeclSpecs, String strFunctionName, CompoundStatement cmpstmtBody, boolean bIncludeStencilCommentAnnotation)
		{
			addProcedureDeclaration (listAdditionalDeclSpecs, strFunctionName, m_listParams, cmpstmtBody, bIncludeStencilCommentAnnotation);
		}

		@SuppressWarnings("unchecked")
		public void addProcedureDeclaration (List<Specifier> listAdditionalDeclSpecs, String strFunctionName, List<Declaration> listParams, CompoundStatement cmpstmtBody, boolean bIncludeStencilCommentAnnotation)
		{
			List<Specifier> listSpecs = new ArrayList<Specifier> (listAdditionalDeclSpecs == null ? 1 : listAdditionalDeclSpecs.size () + 1);
			if (listAdditionalDeclSpecs != null)
				listSpecs.addAll (listAdditionalDeclSpecs);
			listSpecs.add (Specifier.VOID);

			Procedure procedure = new Procedure (
				listSpecs,
				new ProcedureDeclarator (new NameID (strFunctionName), (List<Declaration>) CodeGeneratorUtil.clone (listParams)),
				cmpstmtBody);

			if (bIncludeStencilCommentAnnotation)
				procedure.annotate (new CommentAnnotation (m_data.getStencilCalculation ().getStencilExpressions ()));

			m_unit.addDeclaration (procedure);
		}

		protected abstract void setGlobalGeneratedIdentifiersFunctionName (NameID nidFnxName);
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The data shared by the code generators
	 */
	private CodeGeneratorSharedObjects m_data;

	/**
	 * The code generator that generates the code one thread executes from the strategy code
	 */
	private ThreadCodeGenerator m_cgThreadCode;

	/**
	 * Map of identifiers that are arguments to the stencil kernel.
	 * The identifiers are created when the function signature is created.
	 */
	private Map<String, Identifier> m_mapStencilOperationInputArgumentIdentifiers;

	/**
	 * Map of identifiers that are arguments to the stencil kernel.
	 * The identifiers are created when the function signature is created.
	 */
	private Map<String, Identifier> m_mapStencilOperationOutputArgumentIdentifiers;

	private int m_nCodeVariantsCount;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public CodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
		m_cgThreadCode = new ThreadCodeGenerator (m_data);

		m_mapStencilOperationInputArgumentIdentifiers = new HashMap<String, Identifier> ();
		m_mapStencilOperationOutputArgumentIdentifiers = new HashMap<String, Identifier> ();

		m_nCodeVariantsCount = 0;
	}

	/**
	 * Generates the code.
	 * @param unit The translation unit in which to place the kernels
	 * @param bIncludeAutotuneParameters Flag specifying whether to include the autotuning parameters
	 * 	in the function signatures
	 */
	public void generate (TranslationUnit unit, boolean bIncludeAutotuneParameters)
	{
		createFunctionParameterList (true, bIncludeAutotuneParameters);


		// create the stencil calculation code

		m_data.getData ().setCreatingInitialization (false);
		CodeGeneratorRuntimeOptions optionsStencil = new CodeGeneratorRuntimeOptions ();

		// create the code that one thread executes
		CompoundStatement cmpstmtStrategyKernelThreadBody = m_cgThreadCode.generate (m_data.getStrategy ().getBody (), optionsStencil);
		m_data.getData ().capture ();

		StatementListBundle slbThreadBody = new SingleThreadCodeGenerator (m_data).generate (cmpstmtStrategyKernelThreadBody, optionsStencil);
		addAdditionalDeclarationsAndAssignments (slbThreadBody);


		// create the initialization code

		m_data.getData ().setCreatingInitialization (true);
		m_data.getData ().reset ();
		CodeGeneratorRuntimeOptions optionsInitialize = new CodeGeneratorRuntimeOptions ();
		optionsInitialize.setOption (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_INITIALIZE);
		if (m_data.getArchitectureDescription ().useSIMD ())
			optionsInitialize.setOption (CodeGeneratorRuntimeOptions.OPTION_NOVECTORIZE, !m_data.getOptions ().useNativeSIMDDatatypes ());

		CompoundStatement cmpstmtStrategyInitThreadBody = m_cgThreadCode.generate (m_data.getStrategy ().getBody (), optionsInitialize);
		StatementListBundle slbInitializationBody = new SingleThreadCodeGenerator (m_data).generate (cmpstmtStrategyInitThreadBody, optionsInitialize);
		addAdditionalDeclarationsAndAssignments (slbInitializationBody);


		// add global declarations
		addAdditionalGlobalDeclarations (unit, slbThreadBody.getDefault ());

		// add internal autotune parameters to the parameter list
		createFunctionInternalAutotuneParameterList (slbThreadBody);

		// do post-generation optimizations
		optimizeCode (slbThreadBody);

		// package the code into functions and add them to the translation unit

		// stencil function(s)
		boolean bMakeFortranCompatible = m_data.getOptions ().getCompatibility () == CodeGenerationOptions.ECompatibility.FORTRAN;
		String strStencilKernelName = m_data.getStencilCalculation ().getName ();
		packageCode (new GeneratedProcedure (
			new NameID (bMakeFortranCompatible ? Globals.createFortranName (strStencilKernelName) : strStencilKernelName),
			m_data.getData ().getGlobalGeneratedIdentifiers ().getFunctionParameterList (true, bIncludeAutotuneParameters, false, false),
			slbThreadBody,
			unit)
		{
			@Override
			protected void setGlobalGeneratedIdentifiersFunctionName (NameID nidFnxName)
			{
				m_data.getData ().getGlobalGeneratedIdentifiers ().setStencilFunctionName (nidFnxName);
			}
		}, true, bIncludeAutotuneParameters);

		// initialization function
		packageCode (new GeneratedProcedure (
			Globals.getInitializeFunction (bMakeFortranCompatible),
			m_data.getData ().getGlobalGeneratedIdentifiers ().getFunctionParameterList (false, bIncludeAutotuneParameters, false, false),
			slbInitializationBody,
			unit)
		{
			@Override
			protected void setGlobalGeneratedIdentifiersFunctionName (NameID nidFnxName)
			{
				m_data.getData ().getGlobalGeneratedIdentifiers ().setInitializeFunctionName (nidFnxName);
			}
		}, false, bIncludeAutotuneParameters);
	}

	/**
	 * Adds additional global declarations.
	 * @param unit The translation unit in which the declarations are placed
	 */
	private void addAdditionalGlobalDeclarations (TranslationUnit unit, Traversable trvContext)
	{
		for (Declaration decl : m_data.getData ().getGlobalDeclarationsToAdd ())
		{
			VariableDeclarator declarator = (VariableDeclarator) decl.getChildren ().get (0);
			if (HIRAnalyzer.isReferenced (new Identifier (declarator), trvContext))
				unit.addDeclaration (decl);
		}

		// add the stencil counter
//		VariableDeclarator declStencilCounter = new Variabled
//		unit.addDeclaration (new VariableDeclaration (declStencilCounter));
	}

	/**
	 * Create initializers for the base memory objects if required.
	 * The base memory objects are initialized with the input arguments to the stencil kernel.
	 */
	private void setBaseMemoryObjectInitializers ()
	{
		MemoryObjectManager mom = m_data.getData ().getMemoryObjectManager ();
		StrategyAnalyzer analyzer = m_data.getCodeGenerators ().getStrategyAnalyzer ();

		// create the initializers only if we can't use pointer swapping
		if (mom.canUsePointerSwapping (analyzer.getOuterMostSubdomainIterator ()))
			return;

		SubdomainIdentifier sdidBase = analyzer.getRootGrid ();
		boolean bAreBaseMemoryObjectsReferenced = mom.areMemoryObjectsReferenced (sdidBase);
		if (bAreBaseMemoryObjectsReferenced)
		{
			// create an initializer per memory object per vector index
			StencilNodeSet setAll = m_data.getStencilCalculation ().getInputBaseNodeSet ().union (m_data.getStencilCalculation ().getOutputBaseNodeSet ());
			for (int nVecIdx : setAll.getVectorIndices ())
			{
				// get only the memory objects (= classes of stencil nodes) that have the vector index nVecIdx
				StencilNodeSet setVecIdx = setAll.restrict (null, nVecIdx);
				List<Expression> listPointers = new ArrayList<Expression> (setVecIdx.size ());
				StencilNode nodeRepresentant = null;

				// add the stencil nodes in the per-vector index set to the list of pointers
				// (the stencil nodes (= memory objects) are ordered by time index implicitly
				for (StencilNode node : setVecIdx)
				{
					listPointers.add (m_mapStencilOperationInputArgumentIdentifiers.get (MemoryObjectManager.createMemoryObjectName (null, node, null, true)).clone ());
					nodeRepresentant = node;
				}

				// create and set the initializer
				if (nodeRepresentant != null)
				{
					MemoryObject mo = m_data.getData ().getMemoryObjectManager ().getMemoryObject (sdidBase, nodeRepresentant, true);
					((VariableDeclarator) mo.getIdentifier ().getSymbol ()).setInitializer (new Initializer (listPointers));
				}
			}
		}
	}

	/**
	 * Adds additional declarations and assignments to the internal pointers from the kernel references to the
	 * function body.
	 * @param cmpstmt The kernel body
	 */
	private void addAdditionalDeclarationsAndAssignments (StatementListBundle slbCode)
	{
		// if necessary, add the pointer initializers
		setBaseMemoryObjectInitializers ();

		// add the additional declarations to the code
		List<Statement> listDeclarationsAndAssignments = new ArrayList<Statement> (m_data.getData ().getNumberOfDeclarationsToAdd ());
		for (Declaration decl : m_data.getData ().getDeclarationsToAdd ())
			listDeclarationsAndAssignments.add (new DeclarationStatement (decl));

		/*
		// if necessary, allocate space for local memory objects
		for (Memoryobj m_data.getMemoryObjectManager ().g)
		{

		}*/

		// add the initialization code
		listDeclarationsAndAssignments.add (new AnnotationStatement (new CommentAnnotation ("Initializations")));
		for (Statement stmt : m_data.getData ().getInitializationStatements ())
			listDeclarationsAndAssignments.add (stmt);
		listDeclarationsAndAssignments.add (new AnnotationStatement (new CommentAnnotation ("Implementation")));

		// add the statements at the top
		slbCode.addStatementsAtTop (listDeclarationsAndAssignments);

		// add statements at bottom: assign the output grid to the kernel's output argument
		/*
		if (bAreBaseMemoryObjectsReferenced)
		{
			for (String strGrid : m_data.getStencilCalculation ().getOutputGrids ().keySet ())
			{
				StencilCalculation.GridType grid = m_data.getStencilCalculation ().getGrid (strGrid);
				cmpstmt.addStatement (new ExpressionStatement (new AssignmentExpression (
					new UnaryExpression (UnaryOperator.DEREFERENCE, null)
					CodeGeneratorUtil.specifiers (grid.getSpecifier (), PointerSpecifier.UNQUALIFIED, PointerSpecifier.UNQUALIFIED),
					new NameID (strGrid))));
			}
			cmpstmt.addStatement (new ExpressionStatement (new AssignmentExpression (null, null, null)));
		}
		*/
	}

	/**
	 *
	 * @param mapCodes
	 * @param unit
	 */
	@SuppressWarnings("unchecked")
	private void packageCode (GeneratedProcedure proc, boolean bIncludeStencilCommentAnnotation, boolean bIncludeAutotuneParameters)
	{
		int nCodesCount = proc.getBodyCodes ().size ();
		boolean bIsFortranCompatible = m_data.getOptions ().getCompatibility () == CodeGenerationOptions.ECompatibility.FORTRAN;

		if (nCodesCount == 0)
		{
			// no codes: add an empty function
			proc.addProcedureDeclaration (null, new CompoundStatement (), true);
		}
		else
		{
			// there is at least one code
			List<String> listStencilFunctions = new ArrayList<String> (nCodesCount);

			for (ParameterAssignment pa : proc.getBodyCodes ())
			{
				// build a name for the function
				StringBuilder sbFunctionName = new StringBuilder (proc.getFunctionName ().getName ());
				for (Parameter param : pa)
				{
					// skip the default parameter
					if (param == StatementListBundle.DEFAULT_PARAM)
						continue;

					sbFunctionName.append ("__");
					sbFunctionName.append (param.getName ());
					sbFunctionName.append ("_");
					sbFunctionName.append (pa.getParameterValue (param));
				}
				String strDecoratedFunctionName = sbFunctionName.toString ();

				// create the body compound statement
				Statement stmtBody = proc.getBodyCodes ().getStatementList (pa).getCompoundStatement ();
				CompoundStatement cmpstmtBody = null;
				if (stmtBody instanceof CompoundStatement)
					cmpstmtBody = (CompoundStatement) stmtBody;
				else
				{
					cmpstmtBody = new CompoundStatement ();
					cmpstmtBody.addStatement (stmtBody);
				}

				// add the code to the translation unit
				if (nCodesCount == 1 && !bIsFortranCompatible)
					proc.addProcedureDeclaration (m_data.getArchitectureDescription ().getDeclspecs (TypeDeclspec.KERNEL), cmpstmtBody, bIncludeStencilCommentAnnotation);
				else
					proc.addProcedureDeclaration (m_data.getArchitectureDescription ().getDeclspecs (TypeDeclspec.LOCALFUNCTION), strDecoratedFunctionName, cmpstmtBody, false);

				listStencilFunctions.add (strDecoratedFunctionName);
			}

			// add a function call to the kernel (if there is more than one)
			if (nCodesCount > 1 || bIsFortranCompatible)
			{
				// add a function that selects the right unrolling configuration based on a command line parameter
				NameID nidCodeVariants = new NameID (StringUtil.concat ("g_rgCodeVariants", m_nCodeVariantsCount++));
				if (m_data.getArchitectureDescription ().useFunctionPointers ())
					proc.getTranslationUnit ().addDeclaration (createCodeVariantFnxPtrArray (nidCodeVariants, listStencilFunctions, proc.getParams ()));

				// build the function parameter list: same as for the stencil functions, but with additional parameters for the code selection
				List<Identifier> listSelectors = new ArrayList<Identifier> ();
				List<Integer> listSelectorsCount = new ArrayList<Integer> ();
				for (Parameter param : proc.getBodyCodes ().getParameters ())
				{
					VariableDeclarator decl = new VariableDeclarator (new NameID (param.getName ()));
					listSelectors.add (new Identifier (decl));
					listSelectorsCount.add (param.getValues ().length);
				}

				int[] rgSelectorsCount = new int[listSelectorsCount.size ()];
				int i = 0;
				for (int nValue : listSelectorsCount)
					rgSelectorsCount[i++] = nValue;

				// add the function call
				proc.addProcedureDeclaration (
					m_data.getArchitectureDescription ().getDeclspecs (TypeDeclspec.KERNEL),
					m_data.getData ().getGlobalGeneratedIdentifiers ().getFunctionParameterList (
						true, bIncludeAutotuneParameters, bIncludeAutotuneParameters, bIsFortranCompatible),
					getFunctionSelector (
						nidCodeVariants,
						listStencilFunctions,
						(List<Declaration>) CodeGeneratorUtil.clone (proc.getParams ()),
						listSelectors,
						rgSelectorsCount,
						bIsFortranCompatible),
					bIncludeStencilCommentAnnotation);
			}
		}
	}

	/**
	 * Creates a variable declaration for the argument named <code>strArgName</code>.
	 * @param strArgName
	 * @return
	 */
	protected VariableDeclaration createArgumentDeclaration (String strArgName, boolean bCreateOutputArgument)
	{
		StencilCalculation.ArgumentType type = m_data.getStencilCalculation ().getArgumentType (strArgName);

		// create the variable name and the variable declarator
		String strName = StringUtil.concat (strArgName, bCreateOutputArgument ? MemoryObjectManager.SUFFIX_OUTPUTGRID : null);
		VariableDeclarator decl = new VariableDeclarator (new NameID (strName));

		// add to the identifier map
		(bCreateOutputArgument ? m_mapStencilOperationOutputArgumentIdentifiers : m_mapStencilOperationInputArgumentIdentifiers).
			put (strArgName, new Identifier (decl));

		// create and return the actual variable declaration
		List<Specifier> listSpecifiers = type.getSpecifiers ();
		int i = 0;
		Specifier specType = null;
		if (!type.getType ().equals (StencilCalculation.EArgumentType.PARAMETER))
		{
			for (Iterator<Specifier> it = listSpecifiers.iterator (); it.hasNext (); i++)
			{
				Specifier spec = it.next ();
				if (spec.equals (Specifier.FLOAT) || spec.equals (Specifier.DOUBLE))
				{
					it.remove ();
					specType = spec;
					break;
				}
			}
		}

		if (specType != null)
		{
			if (m_data.getOptions ().useNativeSIMDDatatypes () && m_data.getArchitectureDescription ().useSIMD ())
				listSpecifiers.addAll (i, m_data.getArchitectureDescription ().getType (specType));
			else
				listSpecifiers.add (i, specType);
		}

		if (bCreateOutputArgument)
			listSpecifiers.add (PointerSpecifier.UNQUALIFIED);

		return new VariableDeclaration (listSpecifiers, decl);
	}

	/**
	 * Creates a list of variable declarations containing the variables for
	 * <ul>
	 * 	<li>stencil grids</li>
	 * 	<li>stencil parameters</li>
	 * 	<li>strategy autotuner parameters</li>
	 * </ul>
	 * @return A list of variable declarations for the stencil kernel function
	 */
	protected void createFunctionParameterList (boolean bIncludeOutputGrids, boolean bIncludeAutotuneParameters)
	{
		for (Declaration declParam : m_data.getStrategy ().getParameters ())
		{
			if (declParam instanceof VariableDeclaration)
			{
				VariableDeclaration decl = (VariableDeclaration) declParam;
				List<Specifier> listSpecifiers = decl.getSpecifiers ();
				for (int i = 0; i < decl.getNumDeclarators (); i++)
				{
					VariableDeclarator declarator = (VariableDeclarator) decl.getDeclarator (i);
					String strParamName = declarator.getSymbolName ();

					if (listSpecifiers.size () == 1 && StencilSpecifier.STENCIL_GRID.equals (listSpecifiers.get (0)))
					{
						// the strategy argument is a grid identifier

						// add output arguments
						for (String strArgOutput : m_data.getStencilCalculation ().getArguments (StencilCalculation.EArgumentType.OUTPUT_GRID))
						{
							// create the variable declaration
							VariableDeclaration declArg = createArgumentDeclaration (strArgOutput, true);

							// set the variable as stencil function argument in the global generated identifiers
							m_data.getData ().getGlobalGeneratedIdentifiers ().addStencilFunctionArguments (new GlobalGeneratedIdentifiers.Variable (
								GlobalGeneratedIdentifiers.EVariableType.OUTPUT_GRID, declArg, strArgOutput, m_data));
						}

						// add arguments (grids and parameters)
						for (String strArgument : m_data.getStencilCalculation ().getArguments ())
						{
							VariableDeclaration declArg = createArgumentDeclaration (strArgument, false);

							StencilCalculation.EArgumentType typeArg = m_data.getStencilCalculation ().getArgumentType (strArgument).getType ();
							GlobalGeneratedIdentifiers.EVariableType typeVar = StencilCalculation.EArgumentType.PARAMETER.equals (typeArg) ?
								GlobalGeneratedIdentifiers.EVariableType.KERNEL_PARAMETER : GlobalGeneratedIdentifiers.EVariableType.INPUT_GRID;

							m_data.getData ().getGlobalGeneratedIdentifiers ().addStencilFunctionArguments (
								new GlobalGeneratedIdentifiers.Variable (typeVar, declArg, strArgument, m_data));
						}

						// add size parameters (all variables used to specify domain size and the grid size arguments to the operation)
						for (NameID nidSizeParam : m_data.getStencilCalculation ().getSizeParameters ())
						{
							VariableDeclaration declSize = (VariableDeclaration) CodeGeneratorUtil.createVariableDeclaration (Globals.SPECIFIER_SIZE, nidSizeParam.clone (), null);
							m_data.getData ().getGlobalGeneratedIdentifiers ().addStencilFunctionArguments (
								new GlobalGeneratedIdentifiers.Variable (GlobalGeneratedIdentifiers.EVariableType.SIZE_PARAMETER, declSize, new SizeofExpression (CodeGeneratorUtil.specifiers (Globals.SPECIFIER_SIZE)), null));
						}
					}
					else if (listSpecifiers.size () == 1 && StencilSpecifier.STRATEGY_AUTO.equals (listSpecifiers.get (0)))
					{
						// the strategy argument is an autotuner parameter

						// add autotuner parameters
						VariableDeclaration declAutoParam = (VariableDeclaration) CodeGeneratorUtil.createVariableDeclaration (Globals.SPECIFIER_SIZE, strParamName, null);

						m_data.getData ().getGlobalGeneratedIdentifiers ().addStencilFunctionArguments (
							new GlobalGeneratedIdentifiers.Variable (GlobalGeneratedIdentifiers.EVariableType.AUTOTUNE_PARAMETER, declAutoParam, new SizeofExpression (CodeGeneratorUtil.specifiers (Specifier.INT)), null));
					}
				}
			}
		}
	}

	protected void createFunctionInternalAutotuneParameterList (StatementListBundle slbCode)
	{
		if (slbCode.size () <= 1)
			return;

		// build the function parameter list: same as for the stencil functions, but with additional parameters for the code selection
		for (Parameter param : slbCode.getParameters ())
		{
			if (param.getValues ().length == 0)
				continue;

			VariableDeclarator decl = new VariableDeclarator (new NameID (param.getName ()));

			m_data.getData ().getGlobalGeneratedIdentifiers ().addStencilFunctionArguments (
				new GlobalGeneratedIdentifiers.Variable (
					GlobalGeneratedIdentifiers.EVariableType.INTERNAL_AUTOTUNE_PARAMETER,
					new VariableDeclaration (Specifier.INT, decl),
					new SizeofExpression (CodeGeneratorUtil.specifiers (Specifier.INT)),
					null)
			);
		}
	}

	/**
	 * Creates a variable declaration for an array of function pointers with parameters defined in
	 * <code>listParams</code> an initializes it with the functions named as the entries of the list
	 * <code>listFunctionNames</code>.
	 * @param listFunctionNames The list of function names, one for each of the function variants
	 * @param listParams The list of parameters for the stencil function
	 * @return
	 */
	protected VariableDeclaration createCodeVariantFnxPtrArray (NameID nidCodeVariants, List<String> listFunctionNames, List<Declaration> listParams)
	{
		// we want something like this:
		//    void (*rgFunctions[]) (float, char) = { a, b, c };
		// where a, b, c are functions with the defined signatures

		// create the list of types for the declaration of the function pointer array
		List<VariableDeclaration> listTypeList = new ArrayList<VariableDeclaration> (listParams.size ());
		for (Declaration declParam : listParams)
		{
			List<Specifier> listSpecifiers = ((VariableDeclaration) declParam).getSpecifiers ();
			if (listSpecifiers.size () > 0)
				listTypeList.add (new VariableDeclaration (listSpecifiers));
			else
			{
				Declarator decl = ((VariableDeclaration) declParam).getDeclarator (0);
				listTypeList.add (new VariableDeclaration (decl.getSpecifiers ()));
			}
		}

		// declare the array of function pointers
		NestedDeclarator declFunctionTable = new NestedDeclarator (
			new VariableDeclarator (
				CodeGeneratorUtil.specifiers (PointerSpecifier.UNQUALIFIED),
				nidCodeVariants,
				CodeGeneratorUtil.specifiers (new ArraySpecifier ())),
			listTypeList);

		// build a list of function identifiers
		List<NameID> listFunctionIdentifiers = new ArrayList<NameID> (listFunctionNames.size ());
		for (String strFunctionName : listFunctionNames)
			listFunctionIdentifiers.add (new NameID (strFunctionName));

		// set the initializer
		declFunctionTable.setInitializer (new Initializer (listFunctionIdentifiers));

		// add the declaration to the function body
		return new VariableDeclaration (CodeGeneratorUtil.specifiers (Specifier.STATIC, Specifier.VOID), declFunctionTable);
	}

	/**
	 * Returns the function body of a function selecting the function for a specific unrolling determined by
	 * a command line parameter.
	 * @param listFunctionNames The list of function names, one for each of the function variants
	 * @param listParams The list of parameters for the stencil function
	 * @return
	 */
	@SuppressWarnings("unchecked")
	protected CompoundStatement getFunctionSelector (
		NameID nidFunctionSelector,
		List<String> listFunctionNames, List<Declaration> listParams, List<Identifier> listCodeSelectors,
		int[] rgCodeSelectorsCount, boolean bMakeCompatibleWithFortran)
	{
		CompoundStatement cmpstmtBody = new CompoundStatement ();

		// create a list with the function parameters as identifiers
		List<Expression> listFnxParams = new ArrayList<Expression> (listParams.size ());
		for (Declaration declaration : listParams)
		{
			VariableDeclaration vardecl = (VariableDeclaration) declaration;
			if (vardecl.getNumDeclarators () > 1)
				throw new RuntimeException ("NotImpl: multiple variable declarators");

			// check for double pointer; this is not compatible with fortran
			if (bMakeCompatibleWithFortran && HIRAnalyzer.isDoublePointer (vardecl))
			{
				// one timestep is ok, just pass NULL as output pointer
				// throw runtime exception if more than one timestep
				if (!ExpressionUtil.isValue (m_data.getStencilCalculation ().getMaxIterations (), 1))
					throw new RuntimeException ("Fortran is not supported with multiple timesteps.");
				else
				{
					// create a new pointer variable
					VariableDeclarator decl = new VariableDeclarator (vardecl.getDeclarator (0).getID ().clone ());
					cmpstmtBody.addDeclaration (new VariableDeclaration (ASTUtil.dereference (vardecl.getSpecifiers ()), decl));
					Identifier idPointer = new Identifier (decl);

					listFnxParams.add (new UnaryExpression (UnaryOperator.ADDRESS_OF, idPointer));
				}
			}
			else if (bMakeCompatibleWithFortran && HIRAnalyzer.isNoPointer (vardecl))
			{
				// the original parameter is no pointer; an indirection was added to the kernel function declaration
				// add a dereference here where the C kernel is called
				listFnxParams.add (new UnaryExpression (
					UnaryOperator.DEREFERENCE,
					(((VariableDeclarator) vardecl.getDeclarator (0)).getID ()).clone ()));
			}
			else
				listFnxParams.add ((((VariableDeclarator) vardecl.getDeclarator (0)).getID ()).clone ());
		}

		// calculate the code selection expression
		Expression exprCodeSelector = null;
		int i = 0;
		for (Identifier idCodeSelector : listCodeSelectors)
		{
			if (rgCodeSelectorsCount[i] == 1)
			{
				// add a '0' if there is only one selector possibility
				if (exprCodeSelector == null)
					exprCodeSelector = Globals.ZERO.clone ();
			}
			else if (rgCodeSelectorsCount[i] != 0)
			{
				// add a dereference if necessary
				Expression exprCodeSel = bMakeCompatibleWithFortran ?
					new UnaryExpression (UnaryOperator.DEREFERENCE, idCodeSelector.clone ()) :
					idCodeSelector.clone ();

				// calculate the selector
				if (exprCodeSelector == null)
					exprCodeSelector = exprCodeSel;
				else
				{
					exprCodeSelector = new BinaryExpression (
						new BinaryExpression (exprCodeSelector, BinaryOperator.MULTIPLY, new IntegerLiteral (rgCodeSelectorsCount[i - 1])),
						BinaryOperator.ADD,
						exprCodeSel);
				}
			}

			i++;
		}

		if (m_data.getArchitectureDescription ().useFunctionPointers ())
		{
			// function pointers can be used: call the actual kernel function by calling the corresponding
			// function pointer in the array

			cmpstmtBody.addStatement (new ExpressionStatement (new FunctionCall (
				new ArrayAccess (nidFunctionSelector.clone (), exprCodeSelector),
				(List<Expression>) CodeGeneratorUtil.clone (listFnxParams))));
		}
		else
		{
			// no function pointers can be used: create a "switch" statement selecting the actual kernel

			// build the switch statement
			CompoundStatement cmpstmtSwitch = new CompoundStatement ();
			int nIdx = 0;
			for (String strFnxName : listFunctionNames)
			{
				cmpstmtSwitch.addStatement (new Case (new IntegerLiteral (nIdx)));
				cmpstmtSwitch.addStatement (new ExpressionStatement (new FunctionCall (
					new NameID (strFnxName),
					(List<Expression>) CodeGeneratorUtil.clone (listFnxParams))));
				cmpstmtSwitch.addStatement (new BreakStatement ());

				nIdx++;
			}

			// add the "switch" to the function body and return it
			cmpstmtBody.addStatement (new SwitchStatement (exprCodeSelector, cmpstmtSwitch));
		}

		return cmpstmtBody;
	}

	////////////////

	private static int m_nTempCount = 0;
	private Expression substituteBinaryExpressionRecursive (List<Specifier> listSpecs, Expression expr, CompoundStatement cmpstmt)
	{
		if (expr instanceof BinaryExpression)
		{
			Expression exprLHS = substituteBinaryExpressionRecursive (listSpecs, ((BinaryExpression) expr).getLHS (), cmpstmt);
			Expression exprRHS = substituteBinaryExpressionRecursive (listSpecs, ((BinaryExpression) expr).getRHS (), cmpstmt);

			VariableDeclarator decl = new VariableDeclarator (new NameID (StringUtil.concat ("__tmp", CodeGenerator.m_nTempCount++)));
			decl.setInitializer (new ValueInitializer (new BinaryExpression (exprLHS, ((BinaryExpression) expr).getOperator (), exprRHS)));
			cmpstmt.addDeclaration (new VariableDeclaration (listSpecs, decl));
			return new Identifier (decl);
		}

		return expr.clone ();
	}

	@SuppressWarnings("unchecked")
	private CompoundStatement substituteBinaryExpression (Identifier idLHS, AssignmentOperator op, BinaryExpression expr)
	{
		CompoundStatement cmpstmt = new CompoundStatement ();
		cmpstmt.addStatement (new ExpressionStatement (new AssignmentExpression (
			idLHS.clone (),
			op,
			substituteBinaryExpressionRecursive (idLHS.getSymbol ().getTypeSpecifiers (), expr, cmpstmt))));
		return cmpstmt;
	}

	private void substituteBinaryExpressions (Traversable trv)
	{
		if (trv instanceof ExpressionStatement)
		{
			Expression expr = ((ExpressionStatement) trv).getExpression ();
			if (expr instanceof AssignmentExpression)
			{
				AssignmentExpression aexpr = (AssignmentExpression) expr;
				if (aexpr.getLHS () instanceof Identifier && aexpr.getRHS () instanceof BinaryExpression)
					((Statement) trv).swapWith (substituteBinaryExpression ((Identifier) aexpr.getLHS (), aexpr.getOperator (), (BinaryExpression) ((AssignmentExpression) expr).getRHS ()));
			}
		}
		else
		{
			for (Traversable trvChild : trv.getChildren ())
				substituteBinaryExpressions (trvChild);
		}
	}

	////////////////

	/**
	 * Do post-code generation optimizations (loop unrolling, ...).
	 * @param cmpstmtBody
	 * @return
	 */
	protected void optimizeCode (StatementListBundle slbInput)
	{
		// create one assignment for each subexpression
		if (CodeGenerator.SINGLE_ASSIGNMENT)
		{
			for (ParameterAssignment pa : slbInput)
			{
				StatementList sl = slbInput.getStatementList (pa);
				for (Statement stmt : sl.getStatementsAsList ())
					substituteBinaryExpressions (stmt);
			}
		}

		// remove declarations of unused variables
		for (ParameterAssignment pa : slbInput)
		{
			LOGGER.info (StringUtil.concat ("Removing unused variables from ", pa.toString ()));

			StatementList sl = slbInput.getStatementList (pa);
			List<Statement> list = sl.getStatementsAsList ();
			boolean bModified = false;

			for (Iterator<Statement> it = list.iterator (); it.hasNext (); )
			{
				Statement stmt = it.next ();
				if (stmt instanceof DeclarationStatement && ((DeclarationStatement) stmt).getDeclaration () instanceof VariableDeclaration)
				{
					VariableDeclaration vdecl = (VariableDeclaration) ((DeclarationStatement) stmt).getDeclaration ();
					if (vdecl.getNumDeclarators () == 1)
					{
						if (!HIRAnalyzer.isReferenced (vdecl.getDeclarator (0).getID (), sl))
						{
							it.remove ();
							bModified = true;
						}
					}
				}
			}

			if (bModified)
				slbInput.replaceStatementList (pa, new StatementList (list));
		}

		// remove
	}

	/**
	 *
	 * @param bIncludeAutotuneParameters
	 * @return
	 */
	public String getIncludesAndDefines (boolean bIncludeAutotuneParameters)
	{
		/*
		return StringUtil.concat (
			"#include <stdio.h>\n#include <stdlib.h>\n\n",
			bIncludeAutotuneParameters ? "#include \"kerneltest.h\"\n\n" : null);
		*/

		//return "#define t_max 1\n#define THREAD_NUMBER 0\n#define NUMBER_OF_THREADS 1\n\n";
		//return "#define t_max 1";

		StringBuilder sb = new StringBuilder ();

		// print header
		sb.append ("/**\n * Kernel and initialization code for the stencil\n *");
		for (Stencil stencil : m_data.getStencilCalculation ().getStencilBundle ())
		{
			sb.append ("\n * ");
			sb.append (stencil.getStencilExpression ());
		}
		
		sb.append ("\n * \n * Strategy: ");
		sb.append (m_data.getStrategy ().getFilename ());
		
		sb.append ("\n * \n * This code was generated by Patus on ");
		sb.append (DATE_FORMAT.format (new Date ()));
		sb.append ("\n */\n\n");

		if (m_data.getOptions ().isDebugPrintStencilIndices ())
			sb.append ("#include <stdio.h>\n");

		// include files
		for (String strFile : m_data.getArchitectureDescription ().getIncludeFiles ())
		{
			sb.append ("#include \"");
			sb.append (strFile);
			sb.append ("\"\n");
		}

		sb.append ("#include <stdint.h>\n");
		//sb.append ("#include \"patusrt.h\"\n");

		////////
		//sb.append ("#define t_max 1");
		////////

		return sb.toString ();
	}
}