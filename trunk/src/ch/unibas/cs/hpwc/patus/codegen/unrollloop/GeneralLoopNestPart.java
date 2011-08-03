package ch.unibas.cs.hpwc.patus.codegen.unrollloop;

import java.util.ArrayList;
import java.util.List;

import cetus.analysis.LoopTools;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Statement;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;

/**
 * This class represents a single loop within a loop nest and allows
 * to manipulate this loop.
 */
class GeneralLoopNestPart extends AbstractLoopNestPart
{
	///////////////////////////////////////////////////////////////////
	// Implementation

	public GeneralLoopNestPart ()
	{
	}


	///////////////////////////////////////////////////////////////////
	// Gather Loop Information

	@Override
	protected AbstractLoopNestPart.IndexVariable extractIndexVariable ()
	{
		//////////////////////////////////////
		// get the loop index

		Expression exprLoopIndex = LoopTools.getIndexVariable (getLoop ());
		if (exprLoopIndex == null || !(exprLoopIndex instanceof Identifier))
			throw new IllegalArgumentException ("The loop index variable can't be extracted");

		// check whether the loop index is an identifier
		if (!(exprLoopIndex instanceof Identifier))
			throw new IllegalArgumentException ("The loop index isn't a single identifier");

		Identifier idLoopIndex = (Identifier) exprLoopIndex;

		// make sure that the loop index isn't modified within the loop body
		for (DepthFirstIterator it = new DepthFirstIterator (getLoop ().getBody ()); it.hasNext (); )
		{
			Object obj = it.next ();
			if (obj instanceof AssignmentExpression)
			{
				AssignmentExpression ae = (AssignmentExpression) obj;
				if (ae.getLHS () instanceof Identifier && ((Identifier) ae.getLHS ()).equals (idLoopIndex))
					throw new IllegalArgumentException ("The loop index must not be modified within the loop to unroll");
			}
			else if (obj instanceof UnaryExpression)
			{
				UnaryExpression ue = (UnaryExpression) obj;
				if (ue.getExpression () instanceof Identifier &&
					((Identifier) ue.getExpression ()).equals (idLoopIndex) &&
					(ue.getOperator ().equals (UnaryOperator.PRE_INCREMENT) || ue.getOperator ().equals (UnaryOperator.POST_INCREMENT) ||
					ue.getOperator ().equals (UnaryOperator.PRE_DECREMENT) || ue.getOperator ().equals (UnaryOperator.POST_DECREMENT)))
				{
					throw new IllegalArgumentException ("The loop index must not be modified within the loop to unroll");
				}
			}
		}


		//////////////////////////////////////
		// find the initialization statement

		// take the declaration of the out of the loop
		Statement stmtInitialization = null;
		if (getLoop ().getInitialStatement () instanceof DeclarationStatement)
		{
			Declaration declaration = ((DeclarationStatement) getLoop ().getInitialStatement ()).getDeclaration ();
			if (declaration instanceof VariableDeclaration)
			{
				// if the declarators in the init statement don't include the loop index, just keep them as they are
				int nNumDeclarators = ((VariableDeclaration) declaration).getNumDeclarators ();
				boolean bLoopIdxContainedInInit = false;
				for (int i = 0; i < nNumDeclarators; i++)
				{
					VariableDeclarator declarator = (VariableDeclarator) ((VariableDeclaration) declaration).getDeclarator (i);
					if (declarator == idLoopIndex.getSymbol ())
					{
						bLoopIdxContainedInInit = true;
						if (nNumDeclarators > 1)
							throw new IllegalArgumentException ("for loops with more than one declarator in the init statement are not supported");
					}
				}

				if (!bLoopIdxContainedInInit)
				{
					// the loop index isn't contained in the init statement; just copy the one from the original loop
					stmtInitialization = getLoop ().getInitialStatement ().clone ();
				}
				else
				{
					// the loop index is referenced
					// add the declaration outside of the loop, keep the initializer there
					VariableDeclarator declarator = (VariableDeclarator) ((VariableDeclaration) declaration).getDeclarator (0);
					idLoopIndex = (Identifier) declarator.getID ().clone (); /* !!CHECK!! */

					for (CompoundStatement cmpstmt : getSharedData ().getUnrolledStatements ())
					{
						cmpstmt.addDeclaration (new VariableDeclaration (
							((VariableDeclaration) declaration).getSpecifiers (),
							new VariableDeclarator (idLoopIndex, declarator.getTrailingSpecifiers ())));
					}

					stmtInitialization = new ExpressionStatement (new AssignmentExpression (
						idLoopIndex,
						AssignmentOperator.NORMAL,
						(Expression) declarator.getInitializer ().getChildren ().get (0)));
				}
			}
			else
			{
				// some other declaration... (?)
				stmtInitialization = getLoop ().getInitialStatement ().clone ();
			}
		}
		else
			stmtInitialization = getLoop ().getInitialStatement ().clone ();

		return new IndexVariable (idLoopIndex, stmtInitialization);
	}

	@Override
	protected AbstractLoopNestPart.LoopStep extractStep ()
	{
		Expression exprStep = getLoop ().getStep ();

		BinaryOperator oprStep = null;
		Expression exprIncrement = null;

		if (exprStep instanceof UnaryExpression)
		{
			// handle unary expressions: post/pre increments/decrements
			UnaryExpression uexprStep = (UnaryExpression) exprStep;
			if (UnaryOperator.PRE_INCREMENT.equals (uexprStep.getOperator ()) || UnaryOperator.POST_INCREMENT.equals (uexprStep.getOperator ()))
				oprStep = BinaryOperator.ADD;
			else if (UnaryOperator.PRE_DECREMENT.equals (uexprStep.getOperator ()) || UnaryOperator.POST_DECREMENT.equals (uexprStep.getOperator ()))
				oprStep = BinaryOperator.SUBTRACT;
			else
				throw new IllegalArgumentException ("The operator in the step is not valid");

			exprIncrement = new IntegerLiteral (1);
		}
		else if (exprStep instanceof AssignmentExpression)
		{
			// handle assignment expressions
			AssignmentExpression assexprStep = (AssignmentExpression) exprStep;
			if (assexprStep.getOperator ().equals (AssignmentOperator.ADD))
				oprStep = BinaryOperator.ADD;
			else if (assexprStep.getOperator ().equals (AssignmentOperator.SUBTRACT))
				oprStep = BinaryOperator.SUBTRACT;
			else if (assexprStep.getOperator ().equals (AssignmentOperator.MULTIPLY))
				oprStep = BinaryOperator.MULTIPLY;
			else if (assexprStep.getOperator ().equals (AssignmentOperator.DIVIDE))
				oprStep = BinaryOperator.DIVIDE;
			else if (assexprStep.getOperator ().equals (AssignmentOperator.SHIFT_LEFT))
				oprStep = BinaryOperator.SHIFT_LEFT;
			else if (assexprStep.getOperator ().equals (AssignmentOperator.SHIFT_RIGHT))
				oprStep = BinaryOperator.SHIFT_RIGHT;

			if (oprStep != null)
				exprIncrement = assexprStep.getRHS ().clone ();
			else
			{
				if (assexprStep.getOperator ().equals (AssignmentOperator.NORMAL))
				{
					if (assexprStep.getRHS () instanceof BinaryExpression)
					{
						BinaryExpression bexprStep = (BinaryExpression) assexprStep.getRHS ();
						if (bexprStep.getLHS () instanceof Identifier && bexprStep.getLHS ().equals (getLoopIndex ()))
						{
							oprStep = bexprStep.getOperator ();
							exprIncrement = bexprStep.getRHS ().clone ();
						}
						else
							throw new IllegalArgumentException ("Only assignments of values to the loop index variable are supported in the loop step expression");
					}
					else
						throw new IllegalArgumentException ("The assignment to the loop index variable in the step expression is not supported");
				}
				else
					throw new IllegalArgumentException ("The operation in the loop index stepping is not supported");
			}
		}

		return new AbstractLoopNestPart.LoopStep (oprStep, exprIncrement);
	}

	@Override
	protected Expression getMultiStep (int nExponent)
	{
		BinaryOperator oprStep = getStepOperator ();
		Expression exprStep = getStepExpression ();

		if (BinaryOperator.ADD.equals (oprStep) || BinaryOperator.SUBTRACT.equals (exprStep) ||
			BinaryOperator.SHIFT_LEFT.equals (oprStep) || BinaryOperator.SHIFT_RIGHT.equals (oprStep))
		{
			return Symbolic.simplify (new BinaryExpression (exprStep, BinaryOperator.MULTIPLY, new IntegerLiteral (nExponent)));
		}

		if (BinaryOperator.MULTIPLY.equals (oprStep) || BinaryOperator.DIVIDE.equals (oprStep))
		{
			if (exprStep instanceof IntegerLiteral)
				return new IntegerLiteral ((long) Math.pow (((IntegerLiteral) exprStep).getValue (), nExponent));

			List<Expression> listArgs = new ArrayList<Expression> ();
			listArgs.add (exprStep.clone ());
			listArgs.add (new IntegerLiteral (nExponent));
			return new FunctionCall (new NameID ("pow"), listArgs);
		}

		throw new IllegalArgumentException ("Can't compute a multi step");
	}
}