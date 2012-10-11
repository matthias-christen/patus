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
package ch.unibas.cs.hpwc.patus.util;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import cetus.hir.Annotation;
import cetus.hir.AnnotationStatement;
import cetus.hir.ArraySpecifier;
import cetus.hir.BinaryOperator;
import cetus.hir.CommentAnnotation;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Declarator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Initializer;
import cetus.hir.NameID;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;

/**
 * Common Cetus patterns that occur during code generation.
 *
 * @author Matthias-M. Christen
 */
public class CodeGeneratorUtil
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private final static String[] DIMENSION_NAMES = new String[] { "x", "y", "z", "u", "v", "w" };


	///////////////////////////////////////////////////////////////////
	// Implementation

	/////////////
	// TODO: parameter declaration => identifiers

	/**
	 * Creates a variable declaration.
	 * 
	 * @param spec
	 *            The specifier
	 * @param strVarName
	 *            The name of the variable
	 * @param exprInit
	 *            The expression with which the variable is initialized. Can be
	 *            <code>null</code> if no initialization is desired
	 * @return A {@link VariableDeclaration} object
	 */
	public static Declaration createVariableDeclaration (Specifier spec, String strVarName, Expression exprInit)
	{
		return CodeGeneratorUtil.createVariableDeclaration (spec, new NameID (strVarName), exprInit);
	}

	/**
	 * Creates a variable declaration.
	 * 
	 * @param spec
	 *            The specifier
	 * @param idVarName
	 *            The identifier
	 * @param exprInit
	 *            The expression with which the variable is initialized. Can be
	 *            <code>null</code> if no initialization is desired
	 * @return A {@link VariableDeclaration} object
	 */
	public static Declaration createVariableDeclaration (Specifier spec, IDExpression idVarName, Expression exprInit)
	{
		VariableDeclarator declVar = new VariableDeclarator (idVarName);
		if (exprInit != null)
			declVar.setInitializer (new ValueInitializer (exprInit));
		return new VariableDeclaration (spec, declVar);
	}
	///////////

	private static String createName (Object... rgParts)
	{
		StringBuilder sb = new StringBuilder ();
		for (Object objPart : rgParts)
		{
			if (objPart instanceof String)
				sb.append ((String) objPart);
			else if (objPart instanceof IDExpression)
				sb.append (((IDExpression) objPart).getName ());
			else
				sb.append (objPart.toString ());
		}

		return sb.toString ();
	}

	/**
	 *
	 * @param rgParts
	 * @return
	 */
	public static IDExpression createNameID (Object... rgParts)
	{
		return new NameID (CodeGeneratorUtil.createName (rgParts));
	}

//	public static Identifier createIdentifier (Specifier specifier, Object... rgParts)
//	{
//		VariableDeclarator decl = new VariableDeclarator (specifier, CodeGeneratorUtil.createNameID (rgParts));
//		return new Identifier (decl);
//	}

	public static Statement createComment (String s, boolean bOneLiner)
	{
		CommentAnnotation comment = new CommentAnnotation (s);
		comment.setOneLiner (bOneLiner);
		return new AnnotationStatement (comment);
	}

	/**
	 * Returns the dimension name for the numbered dimension <code>nDim</code>.
	 * 
	 * @param nDim
	 *            The number of the dimension
	 * @return A name ("x", "y", ...) corresponding to <code>nDim</code>
	 */
	public static String getDimensionName (int nDim)
	{
		if (nDim < 0)
			throw new RuntimeException ("Dimensions < 0 are not supported");

		if (nDim >= CodeGeneratorUtil.DIMENSION_NAMES.length)
		{
			StringBuilder sb = new StringBuilder ("x");
			sb.append (nDim);
			return sb.toString ();
		}

		return CodeGeneratorUtil.DIMENSION_NAMES[nDim];
	}

	/**
	 * Returns the alternate dimension name for the numbered dimension
	 * <code>nDim</code>, i.e., one of x0, x1, x2, ...
	 * 
	 * @param nDim
	 *            The number of the dimension
	 * @return A alternate name for the dimension <code>nDim</code>: one of x0,
	 *         x1, x2, ...
	 */
	public static String getAltDimensionName (int nDim)
	{
		if (nDim < 0)
			throw new RuntimeException ("Dimensions < 0 are not supported");
		return StringUtil.concat ("x", nDim);
	}

	/**
	 * Gets the dimension from a dimension name, such as <code>x</code>,
	 * <code>y</code>, <code>x0</code>, <code>x4</code>, ...
	 * If <code>strDim</code> doesn't represent a valid dimension, -1 is
	 * returned.
	 * 
	 * @param strDim
	 *            The name of the dimension
	 * @return The dimension of -1 if <code>strDim</code> doesn't represent a
	 *         valid dimension
	 */
	public static int getDimensionFromName (String strDim)
	{
		if (strDim == null)
			return -1;

		// check whether the dimension string is one of the named dimensions
		for (int i = 0; i < CodeGeneratorUtil.DIMENSION_NAMES.length; i++)
			if (CodeGeneratorUtil.DIMENSION_NAMES[i].equals (strDim))
				return i;

		if (strDim.charAt (0) == 'x')
		{
			try
			{
				return Integer.parseInt (strDim.substring (1));
			}
			catch (NumberFormatException e)
			{
				return -1;
			}
		}

		return -1;
	}
	
	public static int getDimensionFromIdentifier (Traversable trv)
	{
		if ((trv instanceof IDExpression) && !(trv instanceof StencilNode))
			return CodeGeneratorUtil.getDimensionFromName (((IDExpression) trv).getName ());
		return -1;
	}
	
	/**
	 * Determines whether <code>strDim</code> is a identifier name used to
	 * identify a dimension.
	 * 
	 * @param strDim
	 *            The name to check
	 * @return <code>true</code> iff <code>strDim</code> is the name of a
	 *         dimension identifier
	 */
	public static boolean isDimensionIdentifier (String strDim)
	{
		return CodeGeneratorUtil.getDimensionFromName (strDim) >= 0;
	}

	/**
	 *
	 * @param idArrayName
	 * @param rgDimensions
	 * @return
	 */
	public static Declaration createArrayDeclaration (IDExpression idArrayName, Expression... rgDimensions)
	{
		List<Expression> listArrayDims = new ArrayList<> (Math.max (1, rgDimensions.length));
		if (rgDimensions.length == 0)
			listArrayDims.add (null);
		else
			for (Expression exprDim : rgDimensions)
				listArrayDims.add (exprDim);

		return new VariableDeclaration (new VariableDeclarator (idArrayName, new ArraySpecifier (listArrayDims)));
	}

	/**
	 * Creates a forward declaration for a function named
	 * <code>idFunctionName</code> with parameters <code>rgParams</code> and
	 * return value <code>specReturn</code>
	 * 
	 * @param specReturn
	 *            The return value specifier
	 * @param idFunctionName
	 *            The name of the function
	 * @param rgParams
	 *            The function parameters
	 * @return A forward declaration for the function specified by
	 *         <code>specReturn</code>, <code>idFunctionName</code>,
	 *         <code>rgParams</code>
	 */
	public static Declaration createForwardDeclaration (Specifier specReturn, IDExpression idFunctionName, Expression... rgParams)
	{
		// create a list of specifiers
		List<Specifier> listSpecifiersReturn = new ArrayList<> (1);
		listSpecifiersReturn.add (specReturn);

		// return the declaration
		return CodeGeneratorUtil.createForwardDeclaration (listSpecifiersReturn, idFunctionName, rgParams);
	}

	/**
	 * Creates a forward declaration for a function named
	 * <code>idFunctionName</code> with parameters <code>rgParams</code> and
	 * return value <code>listSpecifiersReturn</code>
	 * 
	 * @param listSpecifiersReturn
	 *            The return value specifiers
	 * @param idFunctionName
	 *            The name of the function
	 * @param rgParams
	 *            The function parameters
	 * @return A forward declaration for the function specified by
	 *         <code>listSpecifiersReturn</code>, <code>idFunctionName</code>,
	 *         <code>rgParams</code>
	 */
	public static Declaration createForwardDeclaration (List<Specifier> listSpecifiersReturn, IDExpression idFunctionName, Expression... rgParams)
	{
		// create a list of parameters
		List<Expression> listParams = new ArrayList<> (rgParams.length);
		for (Expression expr : rgParams)
			listParams.add (expr);

		// create the declaration
		return new VariableDeclaration (listSpecifiersReturn, new ProcedureDeclarator (idFunctionName, listParams));
	}

	/**
	 * Creates a list of specifiers from <code>rgSpecifiers</code>.
	 * 
	 * @param rgSpecifiers
	 *            The specifiers
	 * @return A list of specifiers
	 */
	public static List<Specifier> specifiers (Specifier... rgSpecifiers)
	{
		List<Specifier> list = new ArrayList<> (rgSpecifiers.length);
		for (Specifier specifier : rgSpecifiers)
			if (specifier != null)
				list.add (specifier);
		return list;
	}

	/**
	 * Creates a list of expressions from <code>rgExpressions</code>.
	 * 
	 * @param rgSpecifiers
	 *            The expressions
	 * @return A list of expressions
	 */
	public static List<Expression> expressions (Expression... rgExpressions)
	{
		List<Expression> list = new ArrayList<> (rgExpressions.length);
		for (Expression expr : rgExpressions)
			if (expr != null)
				list.add (expr);
		return list;
	}

	@SuppressWarnings("unchecked")
	public static <T extends Expression> Set<T> set (Expression... rgExpression)
	{
		Set<T> set = new HashSet<> ();
		for (Expression expr : rgExpression)
		{
			if (expr != null)
			{
				try
				{
					set.add ((T) expr);
				}
				catch (ClassCastException e)
				{
				}
			}
		}

		return set;
	}

	/**
	 * Finds the first statement in a compound statement.
	 * 
	 * @param cmpstmt
	 *            The compound statement in which to look for statements
	 * @return The first statement within <code>cmpstmt</code> or
	 *         <code>null</code> if there are no statements
	 */
	public static Statement getFirstStatement (CompoundStatement cmpstmt)
	{
		// find the first statement
		for (Traversable trvChild : cmpstmt.getChildren ())
			if (trvChild instanceof Statement && !(trvChild instanceof AnnotationStatement) && !(trvChild instanceof DeclarationStatement))
				return (Statement) trvChild;
		return null;
	}

	/**
	 * Appends the statements in <code>rgStatementsToAdd</code> to the compound
	 * statement <code>cmpstmt</code>.
	 * 
	 * @param cmpstmt
	 *            The compound statement to which to add the statements in
	 *            <code>rgStatementsToAdd</code>
	 * @param rgStatementsToAdd
	 *            The statements to add to <code>cmpstmt</code>
	 */
	public static void addStatements (CompoundStatement cmpstmt, Statement... rgStatementsToAdd)
	{
		for (Statement stmt : rgStatementsToAdd)
			cmpstmt.addStatement (stmt);
	}

	/**
	 * Appends the statements in <code>listStatementsToAdd</code> to the
	 * compound statement <code>cmpstmt</code>.
	 * 
	 * @param cmpstmt
	 *            The compound statement to which to add the statements in
	 *            <code>rgStatementsToAdd</code>
	 * @param listStatementsToAdd
	 *            The list of statements to add to <code>cmpstmt</code>
	 */
	public static void addStatements (CompoundStatement cmpstmt, List<Statement> listStatementsToAdd)
	{
		for (Statement stmt : listStatementsToAdd)
			cmpstmt.addStatement (stmt);
	}

	/**
	 * Adds the statements in <code>slToAdd</code> to the compound statement
	 * <code>cmpstmt</code>.
	 * 
	 * @param cmpstmt
	 *            The compound statement to which to add the statements in
	 *            <code>slToAdd</code>
	 * @param slToAdd
	 *            The list of statements to add to <code>cmpstmt</code>
	 */
	public static void addStatements (CompoundStatement cmpstmt, StatementList slToAdd)
	{
		for (Statement stmt : slToAdd)
			cmpstmt.addStatement (stmt.getParent () == null ? stmt : stmt.clone ());
	}

	/**
	 * Adds the statement <code>stmtToAdd</code> at the top of
	 * <code>cmpstmt</code>.
	 * 
	 * @param cmpstmt
	 *            The compound statement to which the statement
	 *            <code>stmtToAdd</code> is added
	 * @param stmtToAdd
	 *            The statement to add
	 */
	public static void addStatementAtTop (CompoundStatement cmpstmt, Statement stmtToAdd)
	{
		// find the first statement
		Statement stmtFirst = CodeGeneratorUtil.getFirstStatement (cmpstmt);

		// add the statement
		if (stmtFirst == null)
			cmpstmt.addStatement (stmtToAdd);
		else
			cmpstmt.addStatementBefore (stmtFirst, stmtToAdd);
	}

	/**
	 * Adds the statements in <code>rgStatementsToAdd</code> at the top of
	 * <code>cmpstmt</code>.
	 * 
	 * @param cmpstmt
	 *            The compound statement to which to add the statements in
	 *            <code>rgStatementsToAdd</code>
	 * @param rgStatementsToAdd
	 *            The statements to add at the top of <code>cmpstmt</code>
	 */
	public static void addStatementsAtTop (CompoundStatement cmpstmt, Statement... rgStatementsToAdd)
	{
		// find the first statement
		Statement stmtFirst = CodeGeneratorUtil.getFirstStatement (cmpstmt);

		// add the statements
		if (stmtFirst == null)
		{
			for (Statement stmt : rgStatementsToAdd)
			{
				if (stmt instanceof DeclarationStatement)
				{
					Declaration decl = ((DeclarationStatement) stmt).getDeclaration ();
					decl.setParent (null);
					cmpstmt.addDeclaration (decl);
				}
				else
					cmpstmt.addStatement (stmt);
			}
		}
		else
		{
			for (Statement stmt : rgStatementsToAdd)
			{
				if (stmt instanceof DeclarationStatement)
				{
					Declaration decl = ((DeclarationStatement) stmt).getDeclaration ();
					decl.setParent (null);
					cmpstmt.addDeclaration (decl);
				}
				else
					cmpstmt.addStatementBefore (stmtFirst, stmt);
			}
		}
	}

	/**
	 * Adds the statements in the list <code>listStatementsToAdd</code> to the
	 * compound statement <code>cmpstmt</code>.
	 * 
	 * @param cmpstmt
	 *            The compound statement to which to add the list of statements
	 * @param listStatementsToAdd
	 *            The list of statements to add to <code>cmpstmt</code>
	 */
	public static void addStatementsAtTop (CompoundStatement cmpstmt, List<Statement> listStatementsToAdd)
	{
		Statement[] rgStatement = new Statement[listStatementsToAdd.size ()];
		listStatementsToAdd.toArray (rgStatement);
		CodeGeneratorUtil.addStatementsAtTop (cmpstmt, rgStatement);
	}

	/**
	 * Adds the comment <code>strComment</code> to the compound statement
	 * <code>cmpstmt</code>.
	 * 
	 * @param cmpstmt
	 *            The compound statement to which to add the comment
	 * @param strComment
	 *            The comment to add to <code>cmpstmt</code>
	 */
	public static void addComment (CompoundStatement cmpstmt, String strComment)
	{
		CommentAnnotation comment = new CommentAnnotation (strComment);
		comment.setOneLiner (true);
		cmpstmt.annotate (comment);
	}

	/**
	 * Clones a Cetus HIR object or a list of HIR objects.
	 * 
	 * @param obj
	 *            The HIR object of list of HIR objects
	 * @return A copy of <code>obj</code>
	 */
	public static Object clone (Object obj)
	{
		if (obj instanceof List<?>)
		{
			List<?> listOrig = (List<?>) obj;
			List<Object> listCopy = new ArrayList<> (listOrig.size ());
			for (Object o : listOrig)
				listCopy.add (CodeGeneratorUtil.clone (o));
			return listCopy;
		}

		if (obj instanceof Expression)
			return ((Expression) obj).clone ();
		if (obj instanceof Declarator)
			return ((Declarator) obj).clone ();
		if (obj instanceof Declaration)
			return ((Declaration) obj).clone ();
		if (obj instanceof Annotation)
			return ((Annotation) obj).clone ();
		if (obj instanceof Statement)
			return ((Statement) obj).clone ();
		if (obj instanceof Specifier)
			return obj;
		if (obj instanceof BinaryOperator)
			return obj;
		if (obj instanceof Initializer)
			return ((Initializer) obj).clone ();

		// not cloned: Symbolic, Program, TranslationUnit

		return null;
	}

	/**
	 *
	 * @param exprArgument
	 * @return
	 */
	public static FunctionCall createStencilFunctionCall (Expression exprArgument)
	{
		return new FunctionCall (Globals.FNX_STENCIL.clone (), CodeGeneratorUtil.expressions (exprArgument));
	}

	/**
	 * Returns the statement <code>stmt</code> wrapped in a
	 * {@link CompoundStatement} or cast to a {@link CompoundStatement} if it is
	 * already a compound statement.
	 * 
	 * @param stmt
	 * @return
	 */
	public static CompoundStatement getCompoundStatementOrphan (Statement stmt)
	{
		boolean bIsOrphan = stmt.getParent () == null;

		if (stmt instanceof CompoundStatement)
			return bIsOrphan ? (CompoundStatement) stmt : ((CompoundStatement) stmt).clone ();

		CompoundStatement cmpstmt = new CompoundStatement ();
		cmpstmt.addStatement (bIsOrphan ? stmt : stmt.clone ());
		return cmpstmt;
	}

	public static Specifier getDominantType (Specifier spec1, Specifier spec2)
	{
		if (spec1 == null)
			return spec2;
		if (spec1.equals (Specifier.FLOAT))
			return spec2;
		return Specifier.DOUBLE;
	}
}
