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
package ch.unibas.cs.hpwc.patus.codegen.backend;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CharLiteral;
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
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.PointerSpecifier;
import cetus.hir.ReturnStatement;
import cetus.hir.SizeofExpression;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.StringLiteral;
import cetus.hir.Typecast;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.UserSpecifier;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.CodeGenerationOptions;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers.EVariableType;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers.Variable;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.KernelSourceFile;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObject;
import ch.unibas.cs.hpwc.patus.codegen.ValidationCodeGenerator;
import ch.unibas.cs.hpwc.patus.grammar.strategy.IAutotunerParam;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.ASTUtil;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.FileUtil;
import ch.unibas.cs.hpwc.patus.util.MathUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public abstract class AbstractNonKernelFunctionsImpl implements INonKernelFunctions
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static Logger LOGGER = Logger.getLogger (AbstractNonKernelFunctionsImpl.class);


	///////////////////////////////////////////////////////////////////
	// Inner Types

	/**
	 * Specifies in which form output grids are to be returned by calls to
	 * {@link AbstractNonKernelFunctionsImpl#getExpressionForVariable(Variable, EOutputGridType)}
	 * and {@link AbstractNonKernelFunctionsImpl#getExpressionsForVariables(List, EOutputGridType)}.
	 */
	public enum EOutputGridType
	{
		/**
		 * Denotes that the output grid variable name is to be returned
		 */
		OUTPUTGRID_DEFAULT,

		/**
		 * Denotes that a pointer to an output grid is desired
		 */
		OUTPUTGRID_POINTER,

		/**
		 * Type-casts the output grid argument to the desired type
		 */
		OUTPUTGRID_TYPECAST
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;

	private KernelSourceFile m_kernelSourceFile;

	/**
	 * Maps variables (input and output grids) to their identifiers in the code
	 */
	private Map<Variable, Identifier> m_mapVariableIdentifiers;

	/**
	 * Maps variables (strategy autotuner parameters and size parameters) to the
	 * indices in the command line value array
	 */
	private Map<Variable, Integer> m_mapCommandLineParamIndices;

	/**
	 * Maps command line parameter indices to variables (inversion of
	 * {@link AbstractNonKernelFunctionsImpl#m_mapCommandLineParamIndices})
	 */
	private List<Variable> m_listCommandLineParams;

	//private Identifier m_idParams;

	/**
	 * Samples values for stencil parameters
	 */
	private double m_fSampleValue;

	private int m_nTmpArgCount;

	/**
	 * Value assignments to stencil parameters
	 */
	private Map<String, Double> m_mapParamValues;

	private ValidationCodeGenerator m_cgValidate;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public AbstractNonKernelFunctionsImpl (CodeGeneratorSharedObjects data)
	{
		m_data = data;

		m_mapVariableIdentifiers = new HashMap<> ();
		m_mapParamValues = new HashMap<> ();

		m_fSampleValue = 0.0;
		m_nTmpArgCount = 0;

		m_cgValidate = null;
	}

	@Override
	public void initializeNonKernelFunctionCG ()
	{
	}

	public void initialize ()
	{
		if (m_mapCommandLineParamIndices != null)
			return;

		m_mapCommandLineParamIndices = new HashMap<> ();
		m_listCommandLineParams = new ArrayList<> ();

		int nIdx = 0;
		for (GlobalGeneratedIdentifiers.Variable var : m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables ())
		{
			switch (var.getType ())
			{
			case INPUT_GRID:
			case OUTPUT_GRID:
				VariableDeclarator decl = new VariableDeclarator (new NameID (var.getName ()));
				m_mapVariableIdentifiers.put (var, new Identifier (decl));
				break;

			case AUTOTUNE_PARAMETER:
			case INTERNAL_AUTOTUNE_PARAMETER:
			case INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER:
			case SIZE_PARAMETER:
				m_mapCommandLineParamIndices.put (var, nIdx++);
				m_listCommandLineParams.add (var);
				break;
			}
		}
	}

	public void setKernelSourceFile (KernelSourceFile ksf)
	{
		m_kernelSourceFile = ksf;
	}
	
	public KernelSourceFile getKernelSourceFile ()
	{
		return m_kernelSourceFile;
	}

	/**
	 * Returns the expression that substitutes a {@link Variable} in the driver code.
	 * @param variable The variable for which to retrieve the Cetus HIR expression.
	 * @param typeOutputGrid The form in which the expressions corresponding to {@link EVariableType#OUTPUT_GRID}
	 * 	type variables are desired: as regular identifiers or as pointers
	 * @return The HIR expression corresponding to <code>variable</code>
	 */
	public Expression getExpressionForVariable (GlobalGeneratedIdentifiers.Variable variable, EOutputGridType typeOutputGrid)
	{
		initialize ();

		switch (variable.getType ())
		{
		case INPUT_GRID:
			{
				Expression expr = m_mapVariableIdentifiers.get (variable);
				return expr == null ? null : expr.clone ();
			}

		case OUTPUT_GRID:
			{
				Expression expr = m_mapVariableIdentifiers.get (variable);
				if (expr == null)
					return null;
				
				switch (typeOutputGrid)
				{
				case OUTPUTGRID_DEFAULT:
					return expr.clone ();
				case OUTPUTGRID_POINTER:
					return new UnaryExpression (UnaryOperator.ADDRESS_OF, expr.clone ());
				default:
					return null;
				}
			}

		case KERNEL_PARAMETER:
			return getParameterSampleValue (variable);

		default:
			//return getCommandLineParameterForVariable (variable);
			return new NameID (variable.getName ());
		}
	}

	/**
	 * Returns the expression that substitutes a {@link Variable} in the driver code.
	 * @param variable The variable for which to retrieve the Cetus HIR expression.
	 * @return The HIR expression corresponding to <code>variable</code>. {@link EVariableType#OUTPUT_GRID}
	 * 	type variables are returned as regular identifiers.
	 */
	public Expression getExpressionForVariable (GlobalGeneratedIdentifiers.Variable variable)
	{
		return getExpressionForVariable (variable, EOutputGridType.OUTPUTGRID_DEFAULT);
	}

	/**
	 *
	 * @param listVariables
	 * @param typeOutputGrid The form in which the expressions corresponding to {@link EVariableType#OUTPUT_GRID}
	 * 	type variables are desired: as regular identifiers or as pointers
	 * @return
	 */
	public List<Expression> getExpressionsForVariables (List<Variable> listVariables, EOutputGridType typeOutputGrid)
	{
		List<Expression> listExprs = new ArrayList<> (listVariables.size ());
		for (Variable v : listVariables)
			listExprs.add (getExpressionForVariable (v, typeOutputGrid));
		return listExprs;
	}

	/**
	 *
	 * @param listVariables
	 * @return
	 */
	public List<Expression> getExpressionsForVariables (List<Variable> listVariables)
	{
		return getExpressionsForVariables (listVariables, EOutputGridType.OUTPUTGRID_DEFAULT);
	}

	public Expression getParameterSampleValue (Variable variable)
	{
		Double fValue = m_mapParamValues.get (variable.getName ());
		if (fValue == null)
		{
			Expression exprDefault = variable.getDefaultValue ();
			if (exprDefault != null)
				return exprDefault;
			
			m_fSampleValue += 0.1;
			m_mapParamValues.put (variable.getName (), fValue = m_fSampleValue);
		}

		return new FloatLiteral (fValue);
	}

	@Override
	public StatementList forwardDecls ()
	{
		initialize ();

		GlobalGeneratedIdentifiers ids = m_data.getData ().getGlobalGeneratedIdentifiers ();
		VariableDeclaration declInit = ids.getInitializeFunctionDeclaration (m_kernelSourceFile.getCompatibility ());
		VariableDeclaration declStencil = ids.getStencilFunctionDeclaration (m_kernelSourceFile.getCompatibility ());

		StatementList sl = new StatementList (new ArrayList<Statement> ());
		if (declInit != null)
			sl.addDeclaration (declInit.clone ());
		if (declStencil != null)
			sl.addDeclaration (declStencil.clone ());

		return sl;
	}

	@Override
	public StatementList declareGrids ()
	{
		initialize ();

		StatementList sl = new StatementList (new ArrayList<Statement> ());
		for (GlobalGeneratedIdentifiers.Variable var : m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables ())
		{
			if (var.isGrid ())
			{
				Expression exprId = getExpressionForVariable (var);
				if (exprId instanceof Identifier)
				{
					VariableDeclaration decl = null;
					if (var.getType () == GlobalGeneratedIdentifiers.EVariableType.INPUT_GRID)
						decl = new VariableDeclaration (var.getSpecifiers (), (VariableDeclarator) ((Identifier) exprId).getSymbol ());
					else if (var.getType () == GlobalGeneratedIdentifiers.EVariableType.OUTPUT_GRID)
						decl = new VariableDeclaration (ASTUtil.dereference (var.getSpecifiers ()), (VariableDeclarator) ((Identifier) exprId).getSymbol ());

					// add the variable declaration
					if (decl != null)
					{
						sl.addDeclaration (decl);

						// declare additional grids for validation
						if (m_data.getOptions ().getCreateValidationCode ())
						{
							sl.addDeclaration (new VariableDeclaration (
								decl.getSpecifiers (),
								new VariableDeclarator (new NameID (StringUtil.concat (((Identifier) exprId).getName (), ValidationCodeGenerator.SUFFIX_REFERENCE)))));
						}
					}
				}
			}
		}

		// parse the command line
		StatementList slParseCmdLine = parseCommandLine ();
		if (slParseCmdLine != null)
			sl.addStatements (slParseCmdLine);

		return sl;
	}
	
	public StatementList parseCommandLine ()
	{
		initialize ();
		
		// Generate this code:
		//
		// int nParamArgsCount = 0;
		// char** rgArgs = (char**) malloc (argc * sizeof (char*));
		// for (int i = 1; i < argc; i++)
		//     if (*(argv[i]) != '-')
		//     {
		//         rgArgs[nParamArgsCount] = argv[i];
		//         nParamArgsCount++;
		//     }
		// if (nParamArgsCount != #args)
		// {
		//     puts ("Syntax: [progname] {{ <arg[i]> : i=0..#args-1 }}");
		//     exit (-1);
		// }
		// <paramname> = atoi (rgArgs[i]); : i=0..#args-1
		// free (rgArgs);
	
		StatementList sl = new StatementList (new ArrayList<Statement> ());

		// int nParamArgsCount = 0;
		VariableDeclarator declParamArgsCount = new VariableDeclarator (new NameID ("nParamArgsCount"));
		declParamArgsCount.setInitializer (new ValueInitializer (Globals.ZERO.clone ()));
		sl.addStatement (new DeclarationStatement (new VariableDeclaration (Specifier.INT, declParamArgsCount)));
		
		// char** rgArgs = (char**) malloc (argc * sizeof (char*));
		VariableDeclarator declArgs = new VariableDeclarator (new NameID ("rgArgs"));
		declArgs.setInitializer (new ValueInitializer (new Typecast (
			CodeGeneratorUtil.specifiers (Specifier.CHAR, PointerSpecifier.UNQUALIFIED, PointerSpecifier.UNQUALIFIED),
			new FunctionCall (
				new NameID ("malloc"),
				CodeGeneratorUtil.expressions (new BinaryExpression (
					new NameID ("argc"),
					BinaryOperator.MULTIPLY,
					new SizeofExpression (CodeGeneratorUtil.specifiers (Specifier.CHAR, PointerSpecifier.UNQUALIFIED))
				))
			)
		)));
		sl.addStatement (new DeclarationStatement (new VariableDeclaration (
			CodeGeneratorUtil.specifiers (Specifier.CHAR, PointerSpecifier.UNQUALIFIED, PointerSpecifier.UNQUALIFIED),
			declArgs
		)));

		// create the "if" within the "for", then the "for"
		//         rgArgs[nParamArgsCount] = argv[i];
		//         nParamArgsCount++;
		VariableDeclarator declI = new VariableDeclarator (new NameID ("ii"));
		sl.addStatement (new DeclarationStatement (new VariableDeclaration (Specifier.INT, declI)));
		CompoundStatement cmpstmtIfBody1 = new CompoundStatement ();
		cmpstmtIfBody1.addStatement (new ExpressionStatement (new AssignmentExpression (
			new ArrayAccess (new Identifier (declArgs), new Identifier (declParamArgsCount)),
			AssignmentOperator.NORMAL,
			new ArrayAccess (new NameID ("argv"), new Identifier (declI))
		)));
		// for (int i = 1; i < argc; i++)
		//     if (*(argv[i]) != '-') ...
		cmpstmtIfBody1.addStatement (new ExpressionStatement (new UnaryExpression (UnaryOperator.PRE_INCREMENT, new Identifier (declParamArgsCount))));
		sl.addStatement (new ForLoop (
			new ExpressionStatement (new AssignmentExpression (new Identifier (declI), AssignmentOperator.NORMAL, Globals.ONE.clone ())),
			new BinaryExpression (new Identifier (declI), BinaryOperator.COMPARE_LT, new NameID ("argc")),
			new UnaryExpression (UnaryOperator.PRE_INCREMENT, new Identifier (declI)),
			new IfStatement (
				new BinaryExpression (
					new UnaryExpression (UnaryOperator.DEREFERENCE, new ArrayAccess (new NameID ("argv"), new Identifier (declI))),
					BinaryOperator.COMPARE_NE,
					new CharLiteral ('-')
				),
				cmpstmtIfBody1
			)
		));
		
		// if (nParamArgsCount != #args)
		// {
		//     puts ("Syntax: [progname] {{ <arg[i]> : i=1..#args }}");
		//     exit (-1);
		// }
		CompoundStatement cmpstmtIfBody2 = new CompoundStatement ();

		StringBuilder sbMsg = new StringBuilder ("Wrong number of parameters. Syntax:\\n%s");
		for (Variable v : m_listCommandLineParams)
		{
			sbMsg.append (" <");
			sbMsg.append (v.getName ());
			sbMsg.append (">");
		}
		sbMsg.append ("\\n");

		cmpstmtIfBody2.addStatement (new ExpressionStatement (new FunctionCall (
			new NameID ("printf"),
			CodeGeneratorUtil.expressions (new StringLiteral (sbMsg.toString ()), new ArrayAccess (new NameID ("argv"), new IntegerLiteral (0))))));
		cmpstmtIfBody2.addStatement (new ExpressionStatement (new FunctionCall (
			new NameID ("exit"),
			CodeGeneratorUtil.expressions (new IntegerLiteral (-1)))));

		sl.addStatement (new IfStatement (
			new BinaryExpression (new Identifier (declParamArgsCount), BinaryOperator.COMPARE_NE, new IntegerLiteral (m_mapCommandLineParamIndices.size ())),
			cmpstmtIfBody2
		));

		// <paramname> = atoi (rgArgs[i]); : i=1..#args
		for (Variable var : m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables (
			GlobalGeneratedIdentifiers.EVariableType.SIZE_PARAMETER.mask () |
			GlobalGeneratedIdentifiers.EVariableType.AUTOTUNE_PARAMETER.mask () |
			GlobalGeneratedIdentifiers.EVariableType.INTERNAL_AUTOTUNE_PARAMETER.mask () |
			GlobalGeneratedIdentifiers.EVariableType.INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER.mask ()))
		{
			VariableDeclarator decl = new VariableDeclarator (new NameID (var.getName ()));
			decl.setInitializer (new ValueInitializer (new FunctionCall (
				new NameID ("atoi"),
				CodeGeneratorUtil.expressions (new ArrayAccess (new Identifier (declArgs), new IntegerLiteral (m_mapCommandLineParamIndices.get (var)))))));
			sl.addStatement (new DeclarationStatement (new VariableDeclaration (Globals.SPECIFIER_SIZE, decl)));
		}
		
		// free (rgArgs);
		sl.addStatement (new ExpressionStatement (new FunctionCall (new NameID ("free"), CodeGeneratorUtil.expressions (new Identifier (declArgs)))));

		return sl;
	}

	/**
	 * Parses the command line parameters and stores the arguments in an array.
	 * Subclasses can invoke/override if necessary.
	 * By default, this method is called in {@link AbstractNonKernelFunctionsImpl#declareGrids()}.
	 * @return A statement list that provides the code for parsing the command line or <code>null</code>
	 * 	if no parsing is required
	 */
	@Deprecated
	public StatementList parseCommandLine_Old ()
	{
		initialize ();

		// if (argc != #args)
		// {
		//     puts ("Syntax: [progname] {{ <arg[i]> : i=1..#args }}");
		//     exit (-1);
		// }
		// ////int rgParams[] = { {{ atoi (<arg[i]>) : i=1..#args }} };
		// <paramname> = atoi (<arg[i]>);  : i=1..#args

		StatementList sl = new StatementList (new ArrayList<Statement> ());

		CompoundStatement cmpstmtIfBody = new CompoundStatement ();

		StringBuilder sbMsg = new StringBuilder ("Wrong number of parameters. Syntax:\\n%s");
		for (Variable v : m_listCommandLineParams)
		{
			sbMsg.append (" <");
			sbMsg.append (v.getName ());
			sbMsg.append (">");
		}
		sbMsg.append ("\\n");

		cmpstmtIfBody.addStatement (new ExpressionStatement (new FunctionCall (
			new NameID ("printf"),
			CodeGeneratorUtil.expressions (new StringLiteral (sbMsg.toString ()), new ArrayAccess (new NameID ("argv"), new IntegerLiteral (0))))));
		cmpstmtIfBody.addStatement (new ExpressionStatement (new FunctionCall (
			new NameID ("exit"),
			CodeGeneratorUtil.expressions (new IntegerLiteral (-1)))));

		sl.addStatement (new IfStatement (
			new BinaryExpression (new NameID ("argc"), BinaryOperator.COMPARE_NE, new IntegerLiteral (m_mapCommandLineParamIndices.size () + 1)),
			cmpstmtIfBody));

		/*
		VariableDeclarator declParams = new VariableDeclarator (new NameID ("rgParams"), ArraySpecifier.UNBOUNDED);
		m_idParams = new Identifier (declParams);
		List<Expression> listExprs = new ArrayList<Expression> (m_mapCommandLineParamIndices.size ());
		for (int i = 0; i < m_mapCommandLineParamIndices.size (); i++)
			listExprs.add (new FunctionCall (new NameID ("atoi"), CodeGeneratorUtil.expressions (new ArrayAccess (new NameID ("argv"), new IntegerLiteral (i + 1)))));
		declParams.setInitializer (new Initializer (listExprs));
		sl.addStatement (new DeclarationStatement (new VariableDeclaration (Specifier.INT, declParams)));

		// declare and assign size variables
		for (Variable varSize : m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables (GlobalGeneratedIdentifiers.EVariableType.SIZE_PARAMETER))
		{
			VariableDeclarator decl = new VariableDeclarator (new NameID (varSize.getName ()));
			decl.setInitializer (new ValueInitializer (getCommandLineParameterForVariable (varSize)));
			sl.addStatement (new DeclarationStatement (new VariableDeclaration (Globals.SPECIFIER_SIZE, decl)));
		}
		*/

		// declare and assign size variables and strategy parameters
		for (Variable var : m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables (
			GlobalGeneratedIdentifiers.EVariableType.SIZE_PARAMETER.mask () |
			GlobalGeneratedIdentifiers.EVariableType.AUTOTUNE_PARAMETER.mask () |
			GlobalGeneratedIdentifiers.EVariableType.INTERNAL_AUTOTUNE_PARAMETER.mask () |
			GlobalGeneratedIdentifiers.EVariableType.INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER.mask ()))
		{
			VariableDeclarator decl = new VariableDeclarator (new NameID (var.getName ()));
			decl.setInitializer (new ValueInitializer (new FunctionCall (
				new NameID ("atoi"),
				CodeGeneratorUtil.expressions (new ArrayAccess (new NameID ("argv"), new IntegerLiteral (m_mapCommandLineParamIndices.get (var)))))));
			sl.addStatement (new DeclarationStatement (new VariableDeclaration (Globals.SPECIFIER_SIZE, decl)));
		}

		return sl;
	}

	@Override
	public StatementList allocateGrids ()
	{
		initialize ();

		// default alignment; will be adjusted in the loop
		int nAlignRestrict = 1;

		StatementList sl = new StatementList (new ArrayList<Statement> ());
		for (GlobalGeneratedIdentifiers.Variable varGrid : m_data.getData ().getGlobalGeneratedIdentifiers ().getInputGrids ())
		{
			// check whether size is compatible with SIMD vector length
			if (m_data.getArchitectureDescription ().useSIMD () && !m_data.getOptions ().useNativeSIMDDatatypes ())
			{
				Specifier specType = varGrid.getSpecifiers ().get (0);
				int nSIMDVectorLength = m_data.getArchitectureDescription ().getSIMDVectorLength (specType);
				
				int nAlignRestrictForType = m_data.getArchitectureDescription ().getAlignmentRestriction (specType);
				nAlignRestrict = nAlignRestrict == 1 ? nAlignRestrictForType : MathUtil.getLCM (nAlignRestrict, nAlignRestrictForType);
				
				if (nSIMDVectorLength > 1 && nAlignRestrict > 1)
				{
					Expression exprUnitStrideSize = varGrid.getBoxSize ().getCoord (0);

					// create the abort statement (print error message and return)
					CompoundStatement cmpstmtAbort = new CompoundStatement ();
					cmpstmtAbort.addStatement (new ExpressionStatement (new FunctionCall (
						new NameID ("printf"),
						CodeGeneratorUtil.expressions (
							new StringLiteral (StringUtil.concat (
								"Non-native, aligned SIMD type mode requires that ", exprUnitStrideSize.toString (),
								" is divisible by ", nSIMDVectorLength, " [", exprUnitStrideSize.toString (), " = %d].\\n")),
							exprUnitStrideSize.clone ())
						)
					));
					cmpstmtAbort.addStatement (new ReturnStatement (new IntegerLiteral (-1)));

					sl.addStatement (new IfStatement (
						new BinaryExpression (
							new BinaryExpression (exprUnitStrideSize.clone (), BinaryOperator.MODULUS, new IntegerLiteral (nSIMDVectorLength)),
							BinaryOperator.COMPARE_NE,
							Globals.ZERO.clone ()),
						cmpstmtAbort));
				}
			}

			// get the size of the grid to alloc; extend it if there are alignment restrictions
			Expression exprSize = varGrid.getSize ();
			if (nAlignRestrict > 1)
				exprSize = new BinaryExpression (exprSize, BinaryOperator.ADD, new IntegerLiteral (nAlignRestrict - 1));

			// add the malloc statement
			Identifier idGrid = m_mapVariableIdentifiers.get (varGrid).clone ();
			sl.addStatement (new ExpressionStatement (new AssignmentExpression (
				idGrid,
				AssignmentOperator.NORMAL,
				new Typecast (
					varGrid.getSpecifiers (),
					new FunctionCall (new NameID ("malloc"), CodeGeneratorUtil.expressions (exprSize)))
				)
			));

			// create the malloc statement for the reference grid (if validation code is to be generated)
			if (m_data.getOptions ().getCreateValidationCode ())
			{
				sl.addStatement (new ExpressionStatement (new AssignmentExpression (
					new NameID (StringUtil.concat (idGrid.getName (), ValidationCodeGenerator.SUFFIX_REFERENCE)),
					AssignmentOperator.NORMAL,
					new Typecast (
						varGrid.getSpecifiers (),
						new FunctionCall (new NameID ("malloc"), CodeGeneratorUtil.expressions (exprSize.clone ())))
					)
				));
			}
		}

		return sl;
	}

	@Override
	public StatementList initializeGrids ()
	{
		initialize ();

		NameID nidInitialize = m_data.getData ().getGlobalGeneratedIdentifiers ().getInitializeFunctionName ();
		if (nidInitialize == null)
			return null;

		StatementList sl = new StatementList ();

		List<Variable> listArgVars =
			m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables (
				~EVariableType.OUTPUT_GRID.mask () & ~EVariableType.INTERNAL_AUTOTUNE_PARAMETER.mask ());

		sl.addStatement (new ExpressionStatement (new FunctionCall (nidInitialize.clone (), getFunctionArguments (listArgVars, sl, true))));

		// initialize the reference grids
		if (m_data.getOptions ().getCreateValidationCode ())
		{
			Expression exprInit = new FunctionCall (nidInitialize.clone (), getFunctionArguments (listArgVars, sl, false));
			exprInit = (Expression) ASTUtil.addSuffixToIdentifiers (exprInit, ValidationCodeGenerator.SUFFIX_REFERENCE, getGridSet ());
			sl.addStatement (new ExpressionStatement (exprInit));
		}

		return sl;
	}

	@Override
	public StatementList sendData ()
	{
		initialize ();

		// don't need to copy data
		return null;
	}

	@Override
	public StatementList receiveData ()
	{
		initialize ();

		// don't need to copy data
		return null;
	}

	@Override
	public StatementList computeStencil ()
	{
		initialize ();

		// if the code is compatible with Fortran, don't include the output grids
		boolean bIsFortran = m_kernelSourceFile.getCompatibility () == CodeGenerationOptions.ECompatibility.FORTRAN;
		int nMask = bIsFortran ? ~EVariableType.OUTPUT_GRID.mask () : ~0;

		StatementList sl = new StatementList ();
		List<Expression> listArgs = getFunctionArguments (
			m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables (~EVariableType.INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER.mask () & nMask),
			sl, true
		);

		sl.addStatement (new ExpressionStatement (new FunctionCall (
			m_data.getData ().getGlobalGeneratedIdentifiers ().getStencilFunctionName ().clone (),
			listArgs
		)));

		return sl;
	}

	@Override
	public StatementList validateComputation ()
	{
		if (m_data.getOptions ().useNativeSIMDDatatypes ())
		{
			LOGGER.error ("Validation not implemented for native SIMD datatypes");
			return new StatementList ();

			// TODO: implement for native SIMD types...
			// (e.g. cast to scalar base types)
		}

		initialize ();
		m_cgValidate = new ValidationCodeGenerator (m_data);

		// create the validation code
		StatementList sl = m_cgValidate.generate (getGridSet ());

		// replace parameter variables by values
		Map<String, Expression> mapVariables = new HashMap<> ();
		for (Variable v : m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables ())
			if (v.getType () == EVariableType.KERNEL_PARAMETER)
				mapVariables.put (v.getName (), getParameterSampleValue (v));

		for (Statement stmt : sl)
		{
			for (DepthFirstIterator it = new DepthFirstIterator (stmt); it.hasNext (); )
			{
				Object o = it.next ();
				if (o instanceof IDExpression)
				{
					Expression exprValue = mapVariables.get (((IDExpression) o).getName ());
					if (exprValue != null)
						((IDExpression) o).swapWith (exprValue.clone ());
				}
			}
		}

		return sl;
	}
	
	/**
	 * Returns a global alignment restriction that is valid for all types of the used variables
	 * in <code>listVariables</code>.
	 * @param listVariables The list of variables for which to determine the common alignment restriction
	 * @return The alignment restriction common to the types of the variables in <code>listVariables</code>
	 */
	public int getGlobalAlignmentRestriction (List<Variable> listVariables)
	{
		Set<Specifier> setTypes = new HashSet<> ();
		for (Variable v : listVariables)
		{
			for (Specifier s : v.getSpecifiers ())
				if (Globals.isBaseDatatype (s))
				{
					setTypes.add (s);
					break;
				}
		}

		int nAlignRestrict = 1;
		for (Specifier s : setTypes)
		{
			int nAlignRestrForType = m_data.getArchitectureDescription ().getAlignmentRestriction (s);
			nAlignRestrict = nAlignRestrict == 1 ? nAlignRestrForType : MathUtil.getLCM (nAlignRestrict, nAlignRestrForType);
		}
		
		return nAlignRestrict;
	}
	
	public Expression getArgumentExpressionFromVariable (Variable v, StatementList slAuxStatements, int nAlignRestrict, boolean bForceAlign)
	{
		boolean bIsPowerOfTwo = MathUtil.isPowerOfTwo (nAlignRestrict);
		boolean bIsFortranCompatible = m_kernelSourceFile.getCompatibility () == CodeGenerationOptions.ECompatibility.FORTRAN;

		Expression exprArg = getExpressionForVariable (v, EOutputGridType.OUTPUTGRID_POINTER);
		if (v.getType () == EVariableType.INPUT_GRID && nAlignRestrict > 1)
		{
			if (bForceAlign)
			{
				// force the array to be aligned at nAlignRestrict boundaries

				if (bIsPowerOfTwo)
				{
					// (type*) (((uint_ptr) <ptr> + (align_restrict - 1)) & ~(align_restrict - 1))
					return new Typecast (
						v.getSpecifiers (),
						new BinaryExpression (
							new BinaryExpression (
								new Typecast (CodeGeneratorUtil.specifiers (new UserSpecifier (new NameID ("uintptr_t"))), exprArg.clone ()),
								BinaryOperator.ADD,
								new IntegerLiteral (nAlignRestrict - 1)
							),
							BinaryOperator.BITWISE_AND,
							new UnaryExpression (
								UnaryOperator.BITWISE_COMPLEMENT,
								new Typecast (CodeGeneratorUtil.specifiers (new UserSpecifier (new NameID ("uintptr_t"))), new IntegerLiteral (nAlignRestrict - 1))
							)
						)
					);
				}
				else
				{
					// (type*) (((((uint_ptr) <ptr>) + (align_restrict - 1)) / align_restrict) * align_restrict)
					//             \--------------- ceil (ptr/align_restrict) ---------------/

					return new Typecast (
						v.getSpecifiers (),
						new BinaryExpression (
							ExpressionUtil.ceil (
								new Typecast (CodeGeneratorUtil.specifiers (new UserSpecifier (new NameID ("uintptr_t"))), exprArg.clone ()),
								new IntegerLiteral (nAlignRestrict),
								false),
							BinaryOperator.MULTIPLY,
							new IntegerLiteral (nAlignRestrict)
						)
					);
				}
			}
			else
			{
				// no forcing alignment...
				return exprArg.clone ();
			}
		}
		else
		{
			if (bIsFortranCompatible)
			{
				if (!(exprArg instanceof IDExpression))
				{
					VariableDeclarator decl = new VariableDeclarator (new NameID (StringUtil.concat ("__tmparg_", m_nTmpArgCount++)));
					decl.setInitializer (new ValueInitializer (exprArg.clone ()));
					slAuxStatements.addStatement (new DeclarationStatement (new VariableDeclaration (v.getSpecifiers (), decl)));
					exprArg = new Identifier (decl);
				}

				return new UnaryExpression (UnaryOperator.ADDRESS_OF, exprArg.clone ());
			}
			else
				return exprArg.clone ();
		}
	}

	/**
	 * Creates a list of arguments to the <code>initialize</code> and the <code>kernel</code>
	 * functions. Takes care of alignment restrictions.
	 * @return List of function arguments to <code>initialize</code> and <code>kernel</code>
	 */
	public List<Expression> getFunctionArguments (List<Variable> listVariables, StatementList sl, boolean bForceAlign)
	{
		// get alignment restrictions
		int nAlignRestrict = getGlobalAlignmentRestriction (listVariables);		

		// build the list of arguments; adjust the pointers so that the alignment restrictions are satisfied
		List<Expression> listArgs = new ArrayList<> ();
		for (Variable v : listVariables)
			listArgs.add (getArgumentExpressionFromVariable (v, sl, nAlignRestrict, bForceAlign));

		return listArgs;
	}

	public Set<IDExpression> getGridSet ()
	{
		List<Variable> listVars = m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables (
			GlobalGeneratedIdentifiers.EVariableType.INPUT_GRID.mask () |
			GlobalGeneratedIdentifiers.EVariableType.OUTPUT_GRID.mask ());

		Set<IDExpression> set = new HashSet<> ();
		for (Variable v : listVars)
			set.add (new NameID (v.getName ()));

		return set;
	}

	@Override
	public StatementList deallocateGrids ()
	{
		initialize ();

		Set<IDExpression> setGrids = null;

		StatementList sl = new StatementList (new ArrayList<Statement> ());
		for (Variable v : m_data.getData ().getGlobalGeneratedIdentifiers ().getInputGrids ())
		{
			Expression exprVar = getExpressionForVariable (v);
			sl.addStatement (new ExpressionStatement (new FunctionCall (
				new NameID ("free"), CodeGeneratorUtil.expressions (exprVar))));

			if (m_data.getOptions ().getCreateValidationCode ())
			{
				if (setGrids == null)
					setGrids = getGridSet ();

				Expression exprVarRef = exprVar.clone ();
				exprVarRef = (Expression) ASTUtil.addSuffixToIdentifiers (exprVarRef, ValidationCodeGenerator.SUFFIX_REFERENCE, setGrids);
				sl.addStatement (new ExpressionStatement (new FunctionCall (
					new NameID ("free"), CodeGeneratorUtil.expressions (exprVarRef))));
			}
		}

		return sl;
	}
	
	@Override
	public StatementList writeGrids (String strFilenameFormat, String strType)
	{
		initialize ();

		StatementList sl = new StatementList (new ArrayList<Statement> ());
		
		byte nDimensionality = m_data.getStencilCalculation ().getDimensionality ();
		
		// determine the type of grid to write
		GlobalGeneratedIdentifiers.EVariableType type = GlobalGeneratedIdentifiers.EVariableType.OUTPUT_GRID;
		boolean bIsInput = "input".equals (strType);
		if (bIsInput)
			type = GlobalGeneratedIdentifiers.EVariableType.INPUT_GRID;
		
		boolean bIsFortran = m_kernelSourceFile.getCompatibility () == CodeGenerationOptions.ECompatibility.FORTRAN;
		int nMask = bIsFortran ? ~EVariableType.OUTPUT_GRID.mask () : ~0;
		int nAlignRestrict = getGlobalAlignmentRestriction (m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables (~EVariableType.INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER.mask () & nMask));
		
		// add a function call for each grid
		for (GlobalGeneratedIdentifiers.Variable varGrid : m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables (type))
		{
			Specifier specType = varGrid.getSpecifiers ().get (0);
			NameID nidWriteFnx = new NameID (specType.equals (Specifier.FLOAT) ? "write_data_f" : "write_data_d");
			
			String strFilename = strFilenameFormat.replace ("%s", varGrid.getName ());
			
			// build the list of arguments
			List<Expression> listArgs = new ArrayList<> (3 + nDimensionality);
			listArgs.add (new StringLiteral (strFilename));
			listArgs.add (new IntegerLiteral (nDimensionality));
			listArgs.add (bIsInput ? getArgumentExpressionFromVariable (varGrid, sl, nAlignRestrict, true) : new NameID (varGrid.getName ()));
			
			for (int i = 0; i < nDimensionality; i++)
				listArgs.add (varGrid.getBoxSize ().getCoord (i).clone ());
			
			// add the function call
			sl.addStatement (new ExpressionStatement (new FunctionCall (nidWriteFnx, listArgs)));
		}

		return sl;
	}

	@Override
	public Expression getFlopsPerStencil ()
	{
		return new IntegerLiteral (m_data.getStencilCalculation ().getStencilBundle ().getFlopsCount ());
	}

	@Override
	public Expression getGridPointsCount ()
	{
		byte nDim = m_data.getStencilCalculation ().getDimensionality ();
		Expression[] rgExprPointsCount = new Expression[nDim + 1];
		rgExprPointsCount[0] = m_data.getStencilCalculation ().getMaxIterations ().clone ();
		for (int i = 0; i < nDim; i++)
		{
			rgExprPointsCount[i + 1] = ExpressionUtil.increment (
				new BinaryExpression (
					m_data.getStencilCalculation ().getDomainSize ().getMax ().getCoord (i).clone (),
					BinaryOperator.SUBTRACT,
					m_data.getStencilCalculation ().getDomainSize ().getMin ().getCoord (i).clone ()
				)
			);
		}
		return ExpressionUtil.product (rgExprPointsCount);
	}

	/**
	 * Returns the number of bytes in the memory objects corresponding to the stencil nodes <code>nodes</code>.
	 * @param nodes The nodes for which to look for the memory objects and in which to count bytes
	 * @return The number of bytes in the memory objects corresponding to the stencil nodes in <code>nodes</code>
	 */
	public Expression getBytes (Iterable<StencilNode> nodes)
	{
		Expression exprTotalBytes = null;

		SubdomainIdentifier sdidRoot = m_data.getCodeGenerators ().getStrategyAnalyzer ().getRootSubdomain ();
		Map<MemoryObject, StencilNode> mapMemoryObjects = new HashMap<> ();

		for (StencilNode node : nodes)
			mapMemoryObjects.put (m_data.getData ().getMemoryObjectManager ().getMemoryObject (sdidRoot, node, true), node);

		for (MemoryObject mo : mapMemoryObjects.keySet ())
		{
			Expression exprMemoryObjectBytes = new BinaryExpression (
				mo.getSize ().getVolume (),
				BinaryOperator.MULTIPLY,
				new SizeofExpression (CodeGeneratorUtil.specifiers (mapMemoryObjects.get (mo).getSpecifier ())));

			if (exprTotalBytes == null)
				exprTotalBytes = exprMemoryObjectBytes;
			else
				exprTotalBytes = new BinaryExpression (exprTotalBytes.clone (), BinaryOperator.ADD, exprMemoryObjectBytes);
		}

		return exprTotalBytes;
	}

	@Override
	public Expression getBytesTransferred ()
	{
		Expression exprReadBytes = getBytes (
			m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ()/*,
			new IntegerLiteral (0)*/); // !!! XXX !!! timeblocking!!???
		Expression exprWriteBytes = getBytes (
			m_data.getStencilCalculation ().getStencilBundle ().getFusedStencil ().getOutputNodes ()/*,
			//new IntegerLiteral (-1)); // without boundary
			m_data.getCodeGenerators ().getStrategyAnalyzer ().getMaximumTotalTimestepsCount ()*/);

		if (exprReadBytes == null)
		{
			if (exprWriteBytes == null)
				return new IntegerLiteral (0);
			return exprWriteBytes;
		}

		Expression exprBytes = exprWriteBytes == null ? exprReadBytes : new BinaryExpression (exprReadBytes, BinaryOperator.ADD, exprWriteBytes);
		return new BinaryExpression (m_data.getStencilCalculation ().getMaxIterations ().clone (), BinaryOperator.MULTIPLY, exprBytes);
	}

	@Override
	public Expression getDoValidation ()
	{
		return (m_data.getOptions ().getCreateValidationCode () ? Globals.ONE : Globals.ZERO).clone ();
	}

	@Override
	public Expression getValidates ()
	{
		// assume the code is OK if no validation code has been added
		if (m_cgValidate == null)
			return Globals.ONE.clone ();

		Expression exprHasValidationErrors = m_cgValidate.getHasValidationErrors ();
		if (exprHasValidationErrors == null)
			return Globals.ONE.clone ();

		return new UnaryExpression (UnaryOperator.LOGICAL_NEGATION, exprHasValidationErrors.clone ());
	}
	
	
	// ----------------------------------------------------------------
	// Makefile Variable Processors

	
	@Override
	public String getTestNonautotuneExeParams ()
	{
		StringBuilder sb = new StringBuilder ();
		for (GlobalGeneratedIdentifiers.Variable var : m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables ())
		{
			if (var.getType ().equals (GlobalGeneratedIdentifiers.EVariableType.SIZE_PARAMETER))
			{
				sb.append ("ifndef ");
				sb.append (var.getName ());
				sb.append ("\n\t@echo For tuning, the value of ");
				sb.append (var.getName ());
				sb.append (" must be specified. E.g. 'make tune ");
				sb.append (var.getName ());
				sb.append ("=100'\n\t@false\nendif\n");
			}
		}
		
		return sb.toString ();
	}
	
	@Override
	public String getAutotuner ()
	{
		StringBuilder sb = new StringBuilder ("java -jar ");
		sb.append (FileUtil.getJarFilePath ());
		sb.append (" autotune");
				
		return sb.toString ();
	}
	
	@Override
	public String getExeParams ()
	{
		List<Variable> listVariables = m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables (
			GlobalGeneratedIdentifiers.EVariableType.SIZE_PARAMETER.mask () |
			GlobalGeneratedIdentifiers.EVariableType.AUTOTUNE_PARAMETER.mask () |
			GlobalGeneratedIdentifiers.EVariableType.INTERNAL_AUTOTUNE_PARAMETER.mask () |
			GlobalGeneratedIdentifiers.EVariableType.INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER.mask ()
		);
		
		// build a map variable name => variable index
		Map<String, Integer> mapArgName2ArgIdx = new HashMap<> ();
		int nIdx = 0;
		for (Variable var : listVariables)
			mapArgName2ArgIdx.put (var.getName (), nIdx++);
		
		// build the command line
		StringBuilder sb = new StringBuilder ();
		for (Variable var : listVariables)
		{
			if (var.getType ().equals (GlobalGeneratedIdentifiers.EVariableType.SIZE_PARAMETER))
			{
				sb.append ("$(");
				sb.append (var.getName ());
				sb.append (')');
			}
			else
			{
				IAutotunerParam param = m_data.getStrategy ().getAutotuneSpecification (var.getName ());
				if (param == null)
					param = var.getAutotuneParam ();
				
				if (param != null)
				{
					// append the auto-tune specification
					// (any size parameters replaced by references to the corresponding command line argument)
					sb.append ('@');
					sb.append (m_data.getData ().getGlobalGeneratedIdentifiers ().getDefinedVariableName (var));
					sb.append ('=');
					sb.append (param.replaceIdentifiers (mapArgName2ArgIdx).replaceAll ("\\(", "\\\\\\(").replaceAll ("\\)", "\\\\\\)"));
				}
				else
				{
					// no auto-tune specification found; just add the variable to the command line
					sb.append ("$(");
					sb.append (var.getName ());
					sb.append (')');
				}
			}
						
			sb.append (' ');
		}

		return sb.toString ();
	}
}
