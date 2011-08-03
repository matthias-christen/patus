package ch.unibas.cs.hpwc.patus.codegen.unrollloop;

import java.util.List;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FloatLiteral;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.ValueInitializer;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.analysis.LoopAnalyzer;
import ch.unibas.cs.hpwc.patus.symbolic.Symbolic;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public class UniformlyIncrementingLoopNestPart extends AbstractLoopNestPart
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private Expression m_exprStart;
	private Expression m_exprEnd;
	private Expression m_exprStride;

	private boolean m_bIsConstantTripCountLoop;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public UniformlyIncrementingLoopNestPart ()
	{
		super ();

		m_exprStart = null;
		m_exprEnd = null;
		m_bIsConstantTripCountLoop = false;
	}

	/**
	 * Checks the initial statement and returns the loop index variable.
	 * Also sets the {@link UniformlyIncrementingLoopNestPart#m_exprStart} expression.
	 */
	private Identifier getLoopIndexVariable ()
	{
		Statement stmtInit = getLoop ().getInitialStatement ();

		// initialization must not be empty
		if (stmtInit == null)
			throw new IllegalArgumentException ("The loop must have an initial statement");

		// initialization must be an expression statement
		if (!(stmtInit instanceof ExpressionStatement) && !(stmtInit instanceof DeclarationStatement))
			throw new IllegalArgumentException ("The initial statement must be an expression statement");

		// more precisely: an assignment expression
		Identifier idLoopIndex = null;
		Declaration declLoopIndex = null;

		if (stmtInit instanceof ExpressionStatement)
		{
			// initial statement is an expression statement

			Expression exprInitial = ((ExpressionStatement) stmtInit).getExpression ();
			if (!(exprInitial instanceof AssignmentExpression))
				throw new IllegalArgumentException ("The initial expression must be an assignment");
			AssignmentExpression aexprInitial = (AssignmentExpression) exprInitial;

			// the left hand side must be an integer (int / long), the assignment operator must be a "="
			if (!(aexprInitial.getLHS () instanceof IDExpression))
				throw new IllegalArgumentException ("The initial statement must be an assignment to an identifier");

			if (!AssignmentOperator.NORMAL.equals (aexprInitial.getOperator ()))
				throw new IllegalArgumentException ("The assignment operator of the initial assignment to the loop index variable is not valid");

			// get the loop index variable and find the variable declaration
			idLoopIndex = (Identifier) aexprInitial.getLHS ();
			declLoopIndex = idLoopIndex.findDeclaration ();

			// get the initial expression
			m_exprStart = aexprInitial.getRHS ();
			if (m_exprStart == null)
				throw new IllegalArgumentException ("The right hand side of the initial assignment to the loop index variable is not valid");
		}
		else if (stmtInit instanceof DeclarationStatement)
		{
			// initial statement is a declaration
			declLoopIndex = ((DeclarationStatement) stmtInit).getDeclaration ();
			if (!(declLoopIndex instanceof VariableDeclaration))
				throw new IllegalArgumentException ("The declaration must be a variable declaration");
			VariableDeclaration vardecl = (VariableDeclaration) ((DeclarationStatement) stmtInit).getDeclaration ();

			// find the declaration of the loop index variable
			if (vardecl.getNumDeclarators () == 0)
				throw new IllegalArgumentException ("No variables declared");
			else if (vardecl.getNumDeclarators () > 1)
				throw new IllegalArgumentException ("Too many variables declared");

			// get the variable declarator and check whether there is an initializer
			if (!(vardecl.getDeclarator (0) instanceof VariableDeclarator))
				throw new IllegalArgumentException ("Declarator must be an variable declarator");

			// get the loop index variable
			VariableDeclarator declarator = ((VariableDeclarator) vardecl.getDeclarator (0));
			idLoopIndex = new Identifier (declarator);
			if (declarator.getInitializer () == null)
				throw new IllegalArgumentException ("The loop index variable must be initialized");

			// get the initial value
			if (declarator.getInitializer () instanceof ValueInitializer)
				m_exprStart = ((ValueInitializer) declarator.getInitializer ()).getValue ();
		}

		// check the type of the loop index variable
		if (declLoopIndex == null)
			throw new IllegalArgumentException ("Declaration of the loop index variable can't be found");
		if (!(declLoopIndex instanceof VariableDeclaration))
			throw new IllegalArgumentException ("Declaration of the loop index isn't a variable declaration");
		List<Specifier> listSpecifiers = ((VariableDeclaration) declLoopIndex).getSpecifiers ();
		if (listSpecifiers == null)
			throw new IllegalArgumentException ("Declaration of the loop index doesn't have any specifiers");
		if (!listSpecifiers.contains (Specifier.INT) && !listSpecifiers.contains (Specifier.LONG))
			throw new IllegalArgumentException ("The loop index variable must be an integer (int or long)");

		return idLoopIndex;
	}

	/**
	 * Check the loop condition.
	 * Also sets the {@link UniformlyIncrementingLoopNestPart#m_exprEnd} expression.
	 */
	private void checkCondition (Identifier idLoopIndex)
	{
		// TODO: end {>|>=} i could also be possible

		// condition must be a binary expression
		if (!(getLoop ().getCondition () instanceof BinaryExpression))
			throw new IllegalArgumentException ("The loop condition must be a binary expression");

		// the left hand side must be the loop index
		BinaryExpression bexprCond = (BinaryExpression) getLoop ().getCondition ();
		if (!(bexprCond.getLHS ().equals (idLoopIndex)))
			throw new IllegalArgumentException ("The left hand side of the loop condition must be the loop index variable");

		// get the end value; the operator must be "<" or "<="
		m_exprEnd = null;
		if (BinaryOperator.COMPARE_LT.equals (bexprCond.getOperator ()))
			m_exprEnd = ExpressionUtil.decrement (bexprCond.getRHS ().clone ());
		else if (BinaryOperator.COMPARE_LE.equals (bexprCond.getOperator ()))
			m_exprEnd = bexprCond.getRHS ();
		else
			throw new IllegalArgumentException ("The comparison operator must be one of '<', '<='");
	}

	/**
	 * Determines whether the loop is a constant trip count-loop and if so, set the corresponding
	 * parameters so it will be unrolled completely.
	 */
	private void analyzeLoop ()
	{
		// try to determine whether the loop is executed at all
		if (Symbolic.ELogicalValue.TRUE.equals (Symbolic.isTrue (
			new BinaryExpression (m_exprStart.clone (), BinaryOperator.COMPARE_GT, m_exprEnd.clone ()),
			Symbolic.ALL_VARIABLES_POSITIVE)))
		{
			// start is greater than end => loop is not executed
			m_bIsConstantTripCountLoop = true;
			restrictUnrollingFactorTo (0);
		}
		else if (Symbolic.ELogicalValue.TRUE.equals (Symbolic.isTrue (
			new BinaryExpression (m_exprStart.clone (), BinaryOperator.COMPARE_EQ, m_exprEnd.clone ()),
			Symbolic.ALL_VARIABLES_POSITIVE)))
		{
			// start == end, i.e. loop is executed exactly once
			m_bIsConstantTripCountLoop = true;
			restrictUnrollingFactorTo (1);
		}
		else
		{
			Expression exprTripCount = LoopAnalyzer.getConstantTripCount (m_exprStart, m_exprEnd, m_exprStride);
			if (exprTripCount != null)
			{
				// the trip count evaluates to a number => unroll the loop
				m_bIsConstantTripCountLoop = true;
				restrictUnrollingFactorTo ((exprTripCount instanceof IntegerLiteral) ?
					(int) ((IntegerLiteral) exprTripCount).getValue () :
					(int) ((FloatLiteral) exprTripCount).getValue ());
			}
		}
	}

	@Override
	protected AbstractLoopNestPart.IndexVariable extractIndexVariable ()
	{
		// check the initialization statement
		Identifier idLoopIndex = getLoopIndexVariable ();

		// check the condition
		checkCondition (idLoopIndex);

		return new AbstractLoopNestPart.IndexVariable (idLoopIndex, getLoop ().getInitialStatement ().clone ());
	}

	@Override
	protected LoopStep extractStep ()
	{
		Expression exprStep = getLoop ().getStep ();

		if (exprStep instanceof UnaryExpression)
		{
			// unary expression: must be an increment expression of the loop index variable
			UnaryExpression uexprStep = (UnaryExpression) exprStep;
			if (!getLoopIndex ().equals (uexprStep.getExpression ()))
				throw new IllegalArgumentException ("The loop's step expression must modify the loop index variable");
			if (UnaryOperator.POST_INCREMENT.equals (uexprStep.getOperator ()) || UnaryOperator.PRE_INCREMENT.equals (uexprStep.getOperator ()))
				m_exprStride = new IntegerLiteral (1);
			else
				throw new IllegalArgumentException ("Unary operators other than pre and post increment are not allowed");
		}
		else if (exprStep instanceof AssignmentExpression)
		{
			// assignment: must assign a new value to the loop index variable
			AssignmentExpression aexprStep = (AssignmentExpression) exprStep;
			if (!getLoopIndex ().equals (aexprStep.getLHS ()))
				throw new IllegalArgumentException ("The loop's step expression must modify the loop index variable");

			// operator must be "+" or "+="
			if (AssignmentOperator.ADD.equals (aexprStep.getOperator ()))
				m_exprStride = aexprStep.getRHS ();
			else if (AssignmentOperator.SUBTRACT.equals (aexprStep.getOperator ()))
				m_exprStride = new UnaryExpression (UnaryOperator.MINUS, aexprStep.getRHS ());
			else if (AssignmentOperator.NORMAL.equals (aexprStep.getOperator ()))
			{
				// the right hand side must be a binary expression being an increment of the loop index variable
				if (!(aexprStep.getRHS () instanceof BinaryExpression))
					throw new IllegalArgumentException ("The right hand side must be a binary expression");

				// we want that the right hand side is something of the form e := i + c
				// (c independent of i), hence we check whether (e-i) is independent of i
				// need also to check that c is (strictly) positive
				m_exprStride = new BinaryExpression (aexprStep.getRHS ().clone (), BinaryOperator.SUBTRACT, getLoopIndex ().clone ());
				if (Symbolic.ELogicalValue.FALSE.equals (Symbolic.isIndependentOf (m_exprStride, getLoopIndex ())))
					throw new IllegalArgumentException ("The stride must be independent of the loop index variable");
				if (Symbolic.ELogicalValue.FALSE.equals (Symbolic.isPositive (m_exprStride, Symbolic.ALL_VARIABLES_POSITIVE)))
					throw new IllegalArgumentException ("The stride must be strictly positive");
			}
			else
				throw new IllegalArgumentException ("Assignment operators other than '+=', '-=', '=' are not permitted");
		}
		else
			throw new IllegalArgumentException ("The step expression must be either a unary expression or an assignment expression");

		// analyze the loop:
		// determine whether this is a constant trip count loop
		analyzeLoop ();

		return new AbstractLoopNestPart.LoopStep (BinaryOperator.ADD, m_exprStride.clone ());
	}

	@Override
	protected Expression getUnrolledIndex (int nUnrollIndex, int nUnrollListIdx)
	{
		if (m_bIsConstantTripCountLoop)
			return Symbolic.simplify (new BinaryExpression (m_exprStart.clone (), BinaryOperator.ADD, new BinaryExpression (m_exprStride.clone (), BinaryOperator.MULTIPLY, new IntegerLiteral (nUnrollIndex))));

		return super.getUnrolledIndex (nUnrollIndex, nUnrollListIdx);
	}

	@Override
	protected Expression getMultiStep (int nExponent)
	{
		return Symbolic.simplify (new BinaryExpression (getStepExpression ().clone (), BinaryOperator.MULTIPLY, new IntegerLiteral (nExponent)));
	}

	@Override
	public LoopNest getUnrolledLoopHead (int nUnrollFactor)
	{
		// if this is a constant trip count-loop (i.e. it will be unrolled completely)
		// don't create a new loop header (just return an empty loop nest since the loop won't be needed
		// in the final code (setting the body of an empty loop nest or appending a loop to an empty loop
		// nest results in the empty, surrounding loop nest object being discarded))
		if (m_bIsConstantTripCountLoop)
			return new LoopNest ();

		// otherwise just do the standard stuff
		return super.getUnrolledLoopHead (nUnrollFactor);
	}

	@Override
	public LoopNest getCleanupLoopHead (int nUnrollFactor)
	{
		// no cleanup loop if the trip count is constant
		if (m_bIsConstantTripCountLoop)
			return null;

		return super.getCleanupLoopHead (nUnrollFactor);
	}

	/**
	 *
	 * @return
	 */
	public boolean isConstantTripCount ()
	{
		return m_bIsConstantTripCountLoop;
	}
}
