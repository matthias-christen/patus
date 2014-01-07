package ch.unibas.cs.hpwc.patus.codegen;

import java.util.HashMap;
import java.util.Map;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.ast.BoundaryCheck;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIterator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Box;
import ch.unibas.cs.hpwc.patus.geometry.Point;
import ch.unibas.cs.hpwc.patus.representation.Stencil;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.AnalyzeTools;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;

public class BoundaryCheckCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants
	
	private final static Expression INFINITY = null;

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	private CodeGeneratorSharedObjects m_data;
	
	private static Map<Expression, Box> m_mapBoundingBoxes = new HashMap<> ();
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public BoundaryCheckCodeGenerator (CodeGeneratorSharedObjects data)
	{
		m_data = data;
	}

	/**
	 * Generates an expression that checks whether the current position of the
	 * {@link BoundaryCheck}'s subdomain iterator intersects with any boundary
	 * specifications.
	 * 
	 * @param bc
	 *            The {@link BoundaryCheck} object
	 * @param options
	 *            code generation options
	 * @return An expression checking whether <code>bc</code>'s subdomain
	 *         iterator is currently at a boundary position
	 */
	public Expression generate (BoundaryCheck bc, CodeGeneratorRuntimeOptions options)
	{
		// nothing to do if there are no boundaries
		if (m_data.getStencilCalculation ().getBoundaries () == null)
			return null;
		
		SubdomainIterator sdit = bc.getSubdomainIterator ();
		Box boxIterator = new Box (sdit.getIteratorSubdomain ().getBox ());
		
		// nothing to do for point iterators; the checks will be done immediately before the boundary computations 
		if (boxIterator.isPoint ())
			return null;

		boxIterator.offset (m_data.getData ().getGeneratedIdentifiers ().getIndexPoint (sdit.getIterator ()));		
		
		// the result expression
		Expression exprCheck = null;
		
		// check each node in each boundary stencil
		for (Stencil stcBoundary : m_data.getStencilCalculation ().getBoundaries ())
		{
			for (StencilNode node : stcBoundary.getAllNodes ())
			{
				Expression exprConstraint = node.getConstraint ();
				if (exprConstraint != null)
				{
					Box boxBoundary = BoundaryCheckCodeGenerator.getBoundingBox (exprConstraint, boxIterator.getDimensionality ());
					
					// if no boundary box could be determined, treat everything as boundary, i.e., return TRUE
					if (boxBoundary == null)
						return Globals.ONE.clone ();
					
					exprCheck = BoundaryCheckCodeGenerator.addExpression (exprCheck, BoundaryCheckCodeGenerator.generateBoxCheck (boxIterator, boxBoundary));
				}
			}
		}
		
		return exprCheck;
	}
		
	/**
	 * Tries to determine the bounding box of the expression <code>expr</code>.
	 * @param expr
	 * @return
	 */
	protected static Box getBoundingBox (Expression expr, byte nDimensionality)
	{
		Box box = m_mapBoundingBoxes.get (expr);
		if (box != null)
			return box;
		
		Expression[] rgNoBounds = new Expression[nDimensionality];
		for (int i = 0; i < nDimensionality; i++)
			rgNoBounds[i] = INFINITY;
		Point ptMin = new Point (rgNoBounds);
		Point ptMax = new Point (rgNoBounds);
		box = new Box (ptMin, ptMax);
		
		if (!analyzeExpressionRecursive (expr, box))
			box = null;
		
		m_mapBoundingBoxes.put (expr, box);
		return box;
	}
	
	private static boolean analyzeExpressionRecursive (Expression expr, Box box)
	{
		if (expr instanceof BinaryExpression)
		{
			BinaryExpression bexpr = (BinaryExpression) expr;
			
			if (bexpr.getOperator ().equals (BinaryOperator.LOGICAL_AND))
			{
				// both LHS and RHS are expected to be comparison expressions,
				// otherwise we say we can't interpret the expression				
				return analyzeExpressionRecursive (bexpr.getLHS (), box) && analyzeExpressionRecursive (bexpr.getRHS (), box);
			}
			else if (AnalyzeTools.isComparisonOperator (bexpr.getOperator ()) && !bexpr.getOperator ().equals (BinaryOperator.COMPARE_NE))
			{
				int nDim = 0;
				Expression exprConstraint = null;
				BinaryOperator op = null;
				
				int nDimLHS = CodeGeneratorUtil.getDimensionFromIdentifier (bexpr.getLHS ());
				int nDimRHS = CodeGeneratorUtil.getDimensionFromIdentifier (bexpr.getRHS ());
				
				if (nDimLHS >= 0 && nDimRHS < 0)
				{
					nDim = nDimLHS;
					exprConstraint = bexpr.getRHS ();
					op = bexpr.getOperator ();
					
					if (op.equals (BinaryOperator.COMPARE_LT))
					{
						op = BinaryOperator.COMPARE_LE;
						exprConstraint = new BinaryExpression (exprConstraint, BinaryOperator.SUBTRACT, Globals.ONE.clone ());
					}
					else if (op.equals (BinaryOperator.COMPARE_GT))
					{
						op = BinaryOperator.COMPARE_GE;
						exprConstraint = new BinaryExpression (exprConstraint, BinaryOperator.ADD, Globals.ONE.clone ());
					}
				}
				else if (nDimLHS < 0 && nDimRHS >= 0)
				{
					// normalize so that the dimension identifier appears on the LHS
					
					nDim = nDimRHS;
					exprConstraint = bexpr.getLHS ();

					if (bexpr.getOperator ().equals (BinaryOperator.COMPARE_EQ))
						op = BinaryOperator.COMPARE_EQ;
					else if (bexpr.getOperator ().equals (BinaryOperator.COMPARE_LE))
						op = BinaryOperator.COMPARE_GE;
					else if (bexpr.getOperator ().equals (BinaryOperator.COMPARE_LT))
					{
						op = BinaryOperator.COMPARE_GE;
						exprConstraint = new BinaryExpression (exprConstraint, BinaryOperator.ADD, Globals.ONE.clone ());
					}
					else if (bexpr.getOperator ().equals (BinaryOperator.COMPARE_GE))
						op = BinaryOperator.COMPARE_LE;
					else if (bexpr.getOperator ().equals (BinaryOperator.COMPARE_GT))
					{
						op = BinaryOperator.COMPARE_LE;
						exprConstraint = new BinaryExpression (exprConstraint, BinaryOperator.SUBTRACT, Globals.ONE.clone ());
					}
					else
						return false;					
				}
				else
					return false;
				
				// set a box coordinate
				if (op == null)
					return false;				
				if (op.equals (BinaryOperator.COMPARE_EQ))
				{
					box.getMin ().setCoord (nDim, exprConstraint);
					box.getMax ().setCoord (nDim, exprConstraint);					
				}
				else if (op.equals (BinaryOperator.COMPARE_LE))
					box.getMax ().setCoord (nDim, exprConstraint);
				else if (op.equals (BinaryOperator.COMPARE_GE))
					box.getMin ().setCoord (nDim, exprConstraint);
				
				return true;
			}
			else
				return false;	// not an iterpretable box
		}

		return false;
	}
		
	/**
	 * Creates an expression that checks whether the boxes <code>box1</code> and
	 * <code>box2</code> intersect.
	 * 
	 * @param box1
	 *            One box
	 * @param box2
	 *            Another box
	 * @return <code>An expression</code> evaluating to <code>true</code> if the
	 *         boxes <code>box1</code> and <code>box2</code> intersect
	 */
	protected static Expression generateBoxCheck (Box box1, Box box2)
	{
		if (box1.getDimensionality () != box2.getDimensionality ())
			throw new RuntimeException ("Boxes must have the same dimensionalities.");
		
		Point ptLeft1 = box1.getMin ();
		Point ptRight1 = box1.getMax ();
		Point ptLeft2 = box2.getMin ();
		Point ptRight2 = box2.getMax ();
		
		Expression exprCheck = null;
		for (int i = 0; i < box1.getDimensionality (); i++)
		{
			if (ptLeft1.getCoord (i) != INFINITY && ptRight2.getCoord (i) != INFINITY)
				exprCheck = BoundaryCheckCodeGenerator.addExpression (exprCheck, new BinaryExpression (ptLeft1.getCoord (i).clone (), BinaryOperator.COMPARE_LE, ptRight2.getCoord (i).clone ()));
			if (ptLeft2.getCoord (i) != INFINITY && ptRight1.getCoord (i) != INFINITY)
				exprCheck = BoundaryCheckCodeGenerator.addExpression (exprCheck, new BinaryExpression (ptLeft2.getCoord (i).clone (), BinaryOperator.COMPARE_LE, ptRight1.getCoord (i).clone ()));			
		}
		
		return exprCheck;
	}
	
	public static Expression addExpression (Expression exprOrig, Expression exprNew)
	{
		if (exprNew == null)
			return exprOrig;
		if (exprOrig == null)
			return exprNew;
		return new BinaryExpression (exprOrig, BinaryOperator.LOGICAL_AND, exprNew);
	}
	
//	public Expression generate ()
//	{
//		
//	}
}
