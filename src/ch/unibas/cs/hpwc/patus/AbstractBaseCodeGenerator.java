package ch.unibas.cs.hpwc.patus;

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
import cetus.hir.IDExpression;
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
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorData;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers.Variable;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.IBaseCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.KernelSourceFile;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObject;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObjectManager;
import ch.unibas.cs.hpwc.patus.codegen.StencilNodeSet;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.grammar.strategy.IAutotunerParam;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilCalculation;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.ASTUtil;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public abstract class AbstractBaseCodeGenerator implements IBaseCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static boolean SINGLE_ASSIGNMENT = false;

	private final static DateFormat DATE_FORMAT = new SimpleDateFormat ("yyyy/MM/dd HH:mm:ss");

	private final static Logger LOGGER = Logger.getLogger (AbstractBaseCodeGenerator.class);
	
	
	///////////////////////////////////////////////////////////////////
	// Inner Types

	protected abstract class GeneratedProcedure
	{
		private String m_strBaseFunctionName;
		private List<Declaration> m_listParams;
		private StatementListBundle m_slbBody;
		private KernelSourceFile m_kernelSourceFile;
		private boolean m_bMakeFortranCompatible;

		public GeneratedProcedure (String strBaseFunctionName, List<Declaration> listParams, StatementListBundle slbBody,
			KernelSourceFile kernelSourceFile, boolean bMakeFortranCompatible)
		{
			m_strBaseFunctionName = strBaseFunctionName;
			m_listParams = listParams;
			m_slbBody = slbBody;
			m_kernelSourceFile = kernelSourceFile;
			m_bMakeFortranCompatible = bMakeFortranCompatible;
		}

		public final String getBaseFunctionName ()
		{
			return m_strBaseFunctionName;
		}
		
		public final String getFunctionName (boolean bIsParametrizedVersion)
		{
			String strFnxName = bIsParametrizedVersion ? StringUtil.concat (m_strBaseFunctionName, Globals.PARAMETRIZED_FUNCTION_SUFFIX) : m_strBaseFunctionName;
			if (m_bMakeFortranCompatible)
				strFnxName = Globals.createFortranName (strFnxName);
			return strFnxName;
		}

		public final List<Declaration> getParams ()
		{
			return m_listParams;
		}

		public final StatementListBundle getBodyCodes ()
		{
			return m_slbBody;
		}

		public final KernelSourceFile getKernelSourceFile ()
		{
			return m_kernelSourceFile;
		}

		/**
		 *
		 * @param listAdditionalDeclSpecs List of additional declspecs for the function. Can be <code>null</code> if no
		 * 	additional specifiers are required.
		 * @param listParams
		 * @param cmpstmtBody
		 * @param unit
		 */
		public Procedure addProcedureDeclaration (List<Specifier> listAdditionalDeclSpecs, CompoundStatement cmpstmtBody,
			boolean bIsParametrizedVersion, boolean bIncludeStencilCommentAnnotation)
		{
			return addProcedureDeclaration (listAdditionalDeclSpecs, m_listParams, cmpstmtBody, bIsParametrizedVersion, bIncludeStencilCommentAnnotation);
		}

		public Procedure addProcedureDeclaration (List<Specifier> listAdditionalDeclSpecs, List<Declaration> listParams, CompoundStatement cmpstmtBody,
			boolean bIsParametrizedVersion, boolean bIncludeStencilCommentAnnotation)
		{
			// set the name of the stencil/initialization function, which is called in the benchmarking harness
			// i.e., always set the parametrized version
			setGlobalGeneratedIdentifiersFunctionName (new NameID (getFunctionName (true)));
			
			return addProcedureDeclaration (listAdditionalDeclSpecs, getFunctionName (bIsParametrizedVersion), listParams, cmpstmtBody, bIncludeStencilCommentAnnotation);
		}

		public Procedure addProcedureDeclaration (List<Specifier> listAdditionalDeclSpecs, String strFunctionName,
			CompoundStatement cmpstmtBody, boolean bIncludeStencilCommentAnnotation)
		{
			return addProcedureDeclaration (listAdditionalDeclSpecs, strFunctionName, m_listParams, cmpstmtBody, bIncludeStencilCommentAnnotation);
		}

		@SuppressWarnings("unchecked")
		public Procedure addProcedureDeclaration (List<Specifier> listAdditionalDeclSpecs, String strFunctionName,
			List<Declaration> listParams, CompoundStatement cmpstmtBody, boolean bIncludeStencilCommentAnnotation)
		{
			List<Specifier> listSpecs = new ArrayList<> (listAdditionalDeclSpecs == null ? 1 : listAdditionalDeclSpecs.size () + 1);
			if (listAdditionalDeclSpecs != null)
				listSpecs.addAll (listAdditionalDeclSpecs);
			listSpecs.add (Specifier.VOID);

			Procedure procedure = new Procedure (
				listSpecs,
				new ProcedureDeclarator (new NameID (strFunctionName), (List<Declaration>) CodeGeneratorUtil.clone (listParams)),
				cmpstmtBody
			);

			if (bIncludeStencilCommentAnnotation)
				procedure.annotate (new CommentAnnotation (m_data.getStencilCalculation ().getStencilExpressions ()));

			m_kernelSourceFile.getTranslationUnit ().addDeclaration (procedure);
			
			return procedure;
		}

		protected abstract void setGlobalGeneratedIdentifiersFunctionName (NameID nidFnxName);
	}
	
	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * The data shared by the code generators
	 */
	protected CodeGeneratorSharedObjects m_data;

	/**
	 * Map of identifiers that are arguments to the stencil kernel.
	 * The identifiers are created when the function signature is created.
	 */
	protected Map<String, Identifier> m_mapStencilOperationInputArgumentIdentifiers;

	/**
	 * Map of identifiers that are arguments to the stencil kernel.
	 * The identifiers are created when the function signature is created.
	 */
	protected Map<String, Identifier> m_mapStencilOperationOutputArgumentIdentifiers;

	protected int m_nCodeVariantsCount;

	
	public AbstractBaseCodeGenerator ()
	{
		m_mapStencilOperationInputArgumentIdentifiers = new HashMap<> ();
		m_mapStencilOperationOutputArgumentIdentifiers = new HashMap<> ();

		m_nCodeVariantsCount = 0;
	}
	
	//abstract public void generate (List<KernelSourceFile> listOutputs, File fileOutputDirectory, boolean bIncludeAutotuneParameters);
	
	protected void packageKernelSourceFile (KernelSourceFile out, StatementListBundle slbThreadBody, StatementListBundle slbInitializationBody, boolean bIncludeAutotuneParameters)
	{
		// stencil function(s)
		boolean bMakeFortranCompatible = out.getCompatibility () == CodeGenerationOptions.ECompatibility.FORTRAN;
		
		packageCode (new GeneratedProcedure (
			m_data.getStencilCalculation ().getName (),
			m_data.getData ().getGlobalGeneratedIdentifiers ().getFunctionParameterList (true, bIncludeAutotuneParameters, false, false),
			slbThreadBody,
			out,
			bMakeFortranCompatible)
		{
			@Override
			protected void setGlobalGeneratedIdentifiersFunctionName (NameID nidFnxName)
			{
				m_data.getData ().getGlobalGeneratedIdentifiers ().setStencilFunctionName (nidFnxName);
			}
		}, true, true, bIncludeAutotuneParameters, bIncludeAutotuneParameters, out.getCompatibility ());

		// initialization function
		if (slbInitializationBody != null && out.getCreateInitialization ())
		{
			packageCode (new GeneratedProcedure (
				Globals.getInitializeFunctionName (m_data.getStencilCalculation ().getName ()),
				m_data.getData ().getGlobalGeneratedIdentifiers ().getFunctionParameterList (false, bIncludeAutotuneParameters, false, false),
				slbInitializationBody,
				out,
				bMakeFortranCompatible)
			{
				@Override
				protected void setGlobalGeneratedIdentifiersFunctionName (NameID nidFnxName)
				{
					m_data.getData ().getGlobalGeneratedIdentifiers ().setInitializeFunctionName (nidFnxName);
				}
			}, false, false, bIncludeAutotuneParameters, false, out.getCompatibility ());
		}
	}

	/**
	 * Adds additional global declarations.
	 * @param unit The translation unit in which the declarations are placed
	 */
	protected void addAdditionalGlobalDeclarations (KernelSourceFile out, Traversable trvContext)
	{
		TranslationUnit unit = out.getTranslationUnit ();
		for (Declaration decl : m_data.getData ().getGlobalDeclarationsToAdd ())
		{
			VariableDeclarator declarator = (VariableDeclarator) decl.getChildren ().get (0);
			if (HIRAnalyzer.isReferenced (new Identifier (declarator), trvContext))
				unit.addDeclaration (decl);
		}
	}

	/**
	 *
	 * @param mapCodes
	 * @param unit
	 */
	protected void packageCode (GeneratedProcedure proc,
		boolean bIncludeStencilCommentAnnotation, boolean bIncludeOutputGrids, boolean bIncludeAutotuneParameters, boolean bIncludeInternalAutotuneParameters,
		CodeGenerationOptions.ECompatibility compatibility)
	{
		int nCodesCount = proc.getBodyCodes ().size ();
		boolean bIsFortranCompatible = compatibility == CodeGenerationOptions.ECompatibility.FORTRAN;

		if (nCodesCount == 0)
		{
			// no codes: add an empty function
			proc.addProcedureDeclaration (null, new CompoundStatement (), false, true);
		}
		else
		{
			// there is at least one code
			List<String> listStencilFunctions = new ArrayList<> (nCodesCount);
			Procedure procedureLast = null;

			for (ParameterAssignment pa : proc.getBodyCodes ())
			{
				// build a name for the function
				StringBuilder sbFunctionName = new StringBuilder (proc.getBaseFunctionName ());
				for (Parameter param : pa)
				{
					// skip the default parameter
					if (param.equals (StatementListBundle.DEFAULT_PARAM))
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
					procedureLast = proc.addProcedureDeclaration (m_data.getArchitectureDescription ().getDeclspecs (TypeDeclspec.KERNEL), cmpstmtBody, true, bIncludeStencilCommentAnnotation);
				else
					procedureLast = proc.addProcedureDeclaration (m_data.getArchitectureDescription ().getDeclspecs (TypeDeclspec.LOCALFUNCTION), strDecoratedFunctionName, cmpstmtBody, false);

				listStencilFunctions.add (strDecoratedFunctionName);
			}

			// add a function call to the kernel (if there is more than one)
			if (nCodesCount > 1 || bIsFortranCompatible)
				createParametrizedProxyFunction (proc, listStencilFunctions, bIncludeStencilCommentAnnotation, bIncludeAutotuneParameters, bIsFortranCompatible);
			else if (nCodesCount == 1 && procedureLast != null)
				proc.getKernelSourceFile ().addExportProcedure (procedureLast);
			
			// add a function calling the kernel function with the parameters from the tuned_params.h
			if (m_data.getData ().getGlobalGeneratedIdentifiers ().getAutotuneVariables ().size () > 0)
			{
				createUnparametrizedProxyFunction (
					proc, listStencilFunctions,
					bIncludeStencilCommentAnnotation, bIncludeOutputGrids,
					bIncludeAutotuneParameters, bIncludeInternalAutotuneParameters,
					bIsFortranCompatible
				);
			}
		}
	}
	
	@SuppressWarnings("unchecked")
	protected void createParametrizedProxyFunction (GeneratedProcedure proc, List<String> listStencilFunctions,
		boolean bIncludeStencilCommentAnnotation, boolean bIncludeAutotuneParameters, boolean bIsFortranCompatible)
	{
		// add a function that selects the right unrolling configuration based on a command line parameter
		NameID nidCodeVariants = new NameID (StringUtil.concat ("g_rgCodeVariants", m_nCodeVariantsCount++));
		if (m_data.getArchitectureDescription ().useFunctionPointers ())
			proc.getKernelSourceFile ().getTranslationUnit ().addDeclaration (AbstractBaseCodeGenerator.createCodeVariantFnxPtrArray (nidCodeVariants, listStencilFunctions, proc.getParams ()));

		// build the function parameter list: same as for the stencil functions, but with additional parameters for the code selection
		List<Identifier> listSelectors = new ArrayList<> ();
		List<Integer> listSelectorsCount = new ArrayList<> ();
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
		Procedure procedureExported = proc.addProcedureDeclaration (
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
			true,
			bIncludeStencilCommentAnnotation
		);
		
		proc.getKernelSourceFile ().addExportProcedure (procedureExported);
	}
	
	protected void createUnparametrizedProxyFunction (GeneratedProcedure proc, List<String> listStencilFunctions,
		boolean bIncludeStencilCommentAnnotation, boolean bIncludeOutputGrids,
		boolean bIncludeAutotuneParameters, boolean bIncludeInternalAutotuneParameters,
		boolean bIsFortranCompatible)
	{
		GlobalGeneratedIdentifiers glid = m_data.getData ().getGlobalGeneratedIdentifiers ();
		
		// the function body containing the call to the parametrized kernel function
		CompoundStatement cmpstmtFnxCall = new CompoundStatement ();
		
		// create temporary variables for auto-tuning parameters so that we can pass pointers
		if (bIsFortranCompatible)
		{
			for (Variable var : glid.getAutotuneVariables ())
			{
				VariableDeclarator decl = new VariableDeclarator (new NameID (var.getName ()));
				decl.setInitializer (new ValueInitializer (new NameID (glid.getDefinedVariableName (var))));
				cmpstmtFnxCall.addDeclaration (new VariableDeclaration (Globals.SPECIFIER_INDEX, decl));				
			}
		}
		
		List<Declaration> listArgs = m_data.getData ().getGlobalGeneratedIdentifiers ().getFunctionParameterList (
			bIncludeOutputGrids, bIncludeAutotuneParameters, bIncludeInternalAutotuneParameters, bIsFortranCompatible);
		List<Expression> listFnxCallArgs = new ArrayList<> ();

		for (Declaration decl : listArgs)
		{
			IDExpression id = ((VariableDeclarator) ((VariableDeclaration) decl).getDeclarator (0)).getID ();
			Variable var = glid.getVariableByName (id.getName ());
			
			if (var != null)
			{
				if (!bIsFortranCompatible && var.isAutotuningParameter ())
					listFnxCallArgs.add (new NameID (glid.getDefinedVariableName (var)));
				else
					listFnxCallArgs.add (id.clone ());
			}
			else
				listFnxCallArgs.add (id.clone ());
		}
		
		// create the function call
		cmpstmtFnxCall.addStatement (new ExpressionStatement (new FunctionCall (
			new NameID (proc.getFunctionName (true)),
			listFnxCallArgs
		)));
		
		// create the function
		List<Specifier> listSpecs = new ArrayList<> ();
		listSpecs.addAll (m_data.getArchitectureDescription ().getDeclspecs (TypeDeclspec.KERNEL));
		//listSpecs.add (Specifier.INLINE);
		
		Procedure procedureExported = proc.addProcedureDeclaration (
			listSpecs,
			m_data.getData ().getGlobalGeneratedIdentifiers ().getFunctionParameterList (bIncludeOutputGrids, false, false, bIsFortranCompatible),
			cmpstmtFnxCall,
			false,
			bIncludeStencilCommentAnnotation
		);
		
		proc.getKernelSourceFile ().addExportProcedure (procedureExported);
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
		List<Expression> listFnxParams = new ArrayList<> (listParams.size ());
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
		for (int i = listCodeSelectors.size () - 1; i >= 0; i--)
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
				Identifier idCodeSelector = listCodeSelectors.get (i);
				Expression exprCodeSel = bMakeCompatibleWithFortran ?
					new UnaryExpression (UnaryOperator.DEREFERENCE, idCodeSelector.clone ()) :
					idCodeSelector.clone ();

				// calculate the selector
				if (exprCodeSelector == null)
					exprCodeSelector = exprCodeSel;
				else
				{
					exprCodeSelector = new BinaryExpression (
						exprCodeSel,
						BinaryOperator.ADD,
						new BinaryExpression (new IntegerLiteral (rgCodeSelectorsCount[i]), BinaryOperator.MULTIPLY, exprCodeSelector)
					);
				}
			}
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
	
	/**
	 * Creates a variable declaration for an array of function pointers with parameters defined in
	 * <code>listParams</code> an initializes it with the functions named as the entries of the list
	 * <code>listFunctionNames</code>.
	 * @param listFunctionNames The list of function names, one for each of the function variants
	 * @param listParams The list of parameters for the stencil function
	 * @return
	 */
	public static VariableDeclaration createCodeVariantFnxPtrArray (NameID nidCodeVariants, List<String> listFunctionNames, List<Declaration> listParams)
	{
		// we want something like this:
		//    void (*rgFunctions[]) (float, char) = { a, b, c };
		// where a, b, c are functions with the defined signatures

		// create the list of types for the declaration of the function pointer array
		List<VariableDeclaration> listTypeList = new ArrayList<> (listParams.size ());
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
		List<NameID> listFunctionIdentifiers = new ArrayList<> (listFunctionNames.size ());
		for (String strFunctionName : listFunctionNames)
			listFunctionIdentifiers.add (new NameID (strFunctionName));

		// set the initializer
		declFunctionTable.setInitializer (new Initializer (listFunctionIdentifiers));

		// add the declaration to the function body
		return new VariableDeclaration (CodeGeneratorUtil.specifiers (Specifier.STATIC, Specifier.VOID), declFunctionTable);
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
				if (Globals.isBaseDatatype (spec))
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
								GlobalGeneratedIdentifiers.EVariableType.OUTPUT_GRID, declArg, strArgOutput, strArgOutput, null, m_data));
						}

						// add arguments (grids and parameters)
						for (String strArgument : m_data.getStencilCalculation ().getArguments ())
						{
							VariableDeclaration declArg = createArgumentDeclaration (strArgument, false);

							StencilCalculation.ArgumentType type = m_data.getStencilCalculation ().getArgumentType (strArgument);
							StencilCalculation.EArgumentType typeArg = type.getType ();
							GlobalGeneratedIdentifiers.EVariableType typeVar = StencilCalculation.EArgumentType.PARAMETER.equals (typeArg) ?
								GlobalGeneratedIdentifiers.EVariableType.KERNEL_PARAMETER : GlobalGeneratedIdentifiers.EVariableType.INPUT_GRID;

							StencilNode node = m_data.getStencilCalculation ().getReferenceStencilNode (strArgument);
							String strOrigName = node == null ? strArgument : node.getName ();

							m_data.getData ().getGlobalGeneratedIdentifiers ().addStencilFunctionArguments (
								new GlobalGeneratedIdentifiers.Variable (
									typeVar, declArg, strArgument, strOrigName,
									type instanceof StencilCalculation.ParamType ? ((StencilCalculation.ParamType) type).getDefaultValue () : null,
									m_data)
							);
						}

						// add size parameters (all variables used to specify domain size and the grid size arguments to the operation)
						for (NameID nidSizeParam : m_data.getStencilCalculation ().getSizeParameters ())
						{
							VariableDeclaration declSize = (VariableDeclaration) CodeGeneratorUtil.createVariableDeclaration (Globals.SPECIFIER_SIZE, nidSizeParam.clone (), null);
							m_data.getData ().getGlobalGeneratedIdentifiers ().addStencilFunctionArguments (
								new GlobalGeneratedIdentifiers.Variable (
									GlobalGeneratedIdentifiers.EVariableType.SIZE_PARAMETER,
									declSize,
									nidSizeParam.getName (),
									null,
									new SizeofExpression (CodeGeneratorUtil.specifiers (Globals.SPECIFIER_SIZE)),
									(Size) null
								)
							);
						}
					}
					else if (listSpecifiers.size () == 1 && StencilSpecifier.STRATEGY_AUTO.equals (listSpecifiers.get (0)))
					{
						// the strategy argument is an autotuner parameter

						// add autotuner parameters
						VariableDeclaration declAutoParam = (VariableDeclaration) CodeGeneratorUtil.createVariableDeclaration (Globals.SPECIFIER_SIZE, strParamName, null);

						m_data.getData ().getGlobalGeneratedIdentifiers ().addStencilFunctionArguments (
							new GlobalGeneratedIdentifiers.Variable (
								GlobalGeneratedIdentifiers.EVariableType.AUTOTUNE_PARAMETER,
								declAutoParam,
								strParamName,
								null,
								new SizeofExpression (CodeGeneratorUtil.specifiers (Specifier.INT)),
								(Size) null
							)
						);
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
					param.getName (),
					new SizeofExpression (CodeGeneratorUtil.specifiers (Specifier.INT)),
					new IAutotunerParam.AutotunerRangeParam (Globals.ZERO.clone (), new IntegerLiteral (param.getValues ().length - 1))
				)
			);
		}
	}
	
	/**
	 * Adds additional declarations and assignments to the internal pointers from the kernel references to the
	 * function body.
	 * @param cmpstmt The kernel body
	 */
	protected void addAdditionalDeclarationsAndAssignments (StatementListBundle slbCode, CodeGeneratorRuntimeOptions options)
	{
		// if necessary, add the pointer initializers
		setBaseMemoryObjectInitializers ();

		// add the additional declarations to the code
		List<Statement> listDeclarationsAndAssignments = new ArrayList<> (m_data.getData ().getNumberOfDeclarationsToAdd ());
		for (Declaration decl : m_data.getData ().getDeclarationsToAdd ())
			listDeclarationsAndAssignments.add (new DeclarationStatement (decl));

		/*
		// if necessary, allocate space for local memory objects
		for (Memoryobj m_data.getMemoryObjectManager ().g)
		{

		}*/

		// add the initialization code		
		boolean bIsFirst = true;
		for (Statement stmt : m_data.getData ().getInitializationStatements (
			new ParameterAssignment (
				CodeGeneratorData.PARAM_COMPUTATION_TYPE,
				options.getIntValue (CodeGeneratorRuntimeOptions.OPTION_STENCILCALCULATION, CodeGeneratorRuntimeOptions.VALUE_STENCILCALCULATION_STENCIL)
			)))
		{
			if (bIsFirst)
				listDeclarationsAndAssignments.add (new AnnotationStatement (new CommentAnnotation ("Initializations")));
			bIsFirst = false;

			listDeclarationsAndAssignments.add (stmt);
		}
		
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
	 * Create initializers for the base memory objects if required.
	 * The base memory objects are initialized with the input arguments to the stencil kernel.
	 */
	protected void setBaseMemoryObjectInitializers ()
	{
		MemoryObjectManager mom = m_data.getData ().getMemoryObjectManager ();
		StrategyAnalyzer analyzer = m_data.getCodeGenerators ().getStrategyAnalyzer ();

		// create the initializers only if we can't use pointer swapping
		if (mom.canUsePointerSwapping (analyzer.getOuterMostSubdomainIterator ()))
			return;

		SubdomainIdentifier sdidBase = analyzer.getRootSubdomain ();
		boolean bAreBaseMemoryObjectsReferenced = mom.areMemoryObjectsReferenced (sdidBase);
		if (bAreBaseMemoryObjectsReferenced)
		{
			// create an initializer per memory object per vector index
			StencilNodeSet setAll = m_data.getStencilCalculation ().getInputBaseNodeSet ().union (m_data.getStencilCalculation ().getOutputBaseNodeSet ());
			for (int nVecIdx : setAll.getVectorIndices ())
			{
				// get only the memory objects (= classes of stencil nodes) that have the vector index nVecIdx
				StencilNodeSet setVecIdx = setAll.restrict (null, nVecIdx);
				List<Expression> listPointers = new ArrayList<> (setVecIdx.size ());
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
	
	public String getFileHeader ()
	{
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
		
		return sb.toString ();
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
		sb.append ("#include \"patusrt.h\"\n");
		
		sb.append ("#include \"");
		sb.append (CodeGenerationOptions.DEFAULT_TUNEDPARAMS_FILENAME);
		sb.append ("\"\n");

		////////
		//sb.append ("#define t_max 1");
		////////

		return sb.toString ();
	}
	
	
	///////////////////////////////////////////////////////////////////
	// Post-Code Generation Code Optimization

	protected static int m_nTempCount = 0;
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
	protected CompoundStatement substituteBinaryExpression (Identifier idLHS, AssignmentOperator op, BinaryExpression expr)
	{
		CompoundStatement cmpstmt = new CompoundStatement ();
		cmpstmt.addStatement (new ExpressionStatement (new AssignmentExpression (
			idLHS.clone (),
			op,
			substituteBinaryExpressionRecursive (idLHS.getSymbol ().getTypeSpecifiers (), expr, cmpstmt))));
		return cmpstmt;
	}

	protected void substituteBinaryExpressions (Traversable trv)
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

	/**
	 * Do post-code generation optimizations (loop unrolling, ...).
	 * @param cmpstmtBody
	 * @return
	 */
	protected void optimizeCode (StatementListBundle slbInput)
	{
		// create one assignment for each subexpression
		if (AbstractBaseCodeGenerator.SINGLE_ASSIGNMENT)
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
			LOGGER.debug (StringUtil.concat ("Removing unused variables from ", pa.toString ()));

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
	}
}
