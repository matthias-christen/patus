package cetus.transforms;

import cetus.analysis.*;
import cetus.exec.*;
import cetus.hir.*;

import java.util.*;

/**
 * Transforms loops so they start with a lower bound of 0
 * and run to some upper bound with a stride of 1.
 */
public class LoopNormalization extends TransformPass
{
  /* private static int num_normalized = 0; */

  /** Constructs a loop normalization pass */
  public LoopNormalization(Program program)
  {
    super(program);
  }

  /** Returns the pass name */
  public String getPassName()
  {
    return new String("[LoopNormalization]");
  }

  /** Starts the transformation */
  public void start()
  {
    DepthFirstIterator iter = new DepthFirstIterator(program);
    while ( iter.hasNext() )
    {
      Object o = iter.next();
      if ( o instanceof ForLoop )
      {
        ForLoop loop = (ForLoop)o;
        if ( LoopTools.isCanonical(loop) )
          normalizeLoop(loop);
      }
    }
  }

  // Expression types not handled by loop normalizer.
  private static Set<Class> avoid_set = new HashSet<Class>();
  static
  {
    avoid_set.add(AccessExpression.class);
    avoid_set.add(Typecast.class);
    avoid_set.add(UnaryExpression.class);
    avoid_set.add(AssignmentExpression.class);
    avoid_set.add(FunctionCall.class);
  }

  // Prevent the normalizer from performing unsafe transformation.
  private boolean isEligible(Expression e)
  {
    if ( IRTools.containsClasses(e, avoid_set) )
      return false;
    List type_list = SymbolTools.getExpressionType(e);
    return (
      type_list != null &&
      type_list.size() == 1 &&
      (type_list.get(0) == Specifier.INT || type_list.get(0) == Specifier.LONG)
    );
  }

  // Make the initial statement start at zero.
  private void normalizeLoop(ForLoop loop)
  {
    Identifier index = (Identifier)LoopTools.getIndexVariable(loop);
    Expression init_val = LoopTools.getLowerBoundExpression(loop);
    Expression last_val = LoopTools.getUpperBoundExpression(loop);
    Expression step_val = LoopTools.getIncrementExpression(loop);

    // Strict type checking for this transformation.
    if ( !isEligible(index) || !isEligible(init_val) ||
      !isEligible(last_val) || !isEligible(step_val) ||
      !(step_val instanceof IntegerLiteral) )
      return;

    // No need for normalization
    if ( init_val.toString().equals("0") && step_val.toString().equals("1") )
      return;

    PrintTools.printStatus(getPassName()+" "+loop.getProcedure().getName()+": ",1);
    PrintTools.printStatus(LoopTools.toControlString(loop)+" --> ", 1);

    /*
    CodeAnnotation orig_code = 
      new CodeAnnotation("#ifndef LOOPNORM"+(num_normalized++)+"\n"+
      loop.toString()+"\n"+"#else\n");
    */

    // Extract nzt condition for last assignment to the original index.
    Expression nzt_condition = loop.getCondition().clone();

    // Create a temporary index.
    Identifier new_index = SymbolTools.getTemp(loop.getParent(), index);

    // Grab the old initial statement
    Statement old_assign = loop.getInitialStatement();

    // Set the new initial statement
    ExpressionStatement new_init = new ExpressionStatement(
      new AssignmentExpression(
        (Expression)new_index.clone(),
        AssignmentOperator.NORMAL,
        new IntegerLiteral(0)));
    loop.setInitialStatement(new_init);

    Expression subst = Symbolic.multiply(step_val, new_index);
    subst = Symbolic.add(subst, init_val);

    // Modify condition
    // First, try to keep the form as simple as possible (e.g., i_0 < expr).
    RangeDomain rd = new RangeDomain();
    BinaryExpression condition = (BinaryExpression)loop.getCondition();
    BinaryOperator op = condition.getOperator();
    Expression new_rhs = Symbolic.subtract(condition.getRHS(), init_val);
    new_rhs = Symbolic.divide(new_rhs, step_val);
    BinaryExpression new_condition = new BinaryExpression(
      (Expression)new_index.clone(), op, new_rhs);
    if ( (rd.compare(step_val, new IntegerLiteral(0))).isLE() )
    {
      if ( op == BinaryOperator.COMPARE_GT )
        new_condition.setOperator(BinaryOperator.COMPARE_LT);
      else if ( op == BinaryOperator.COMPARE_GE )
        new_condition.setOperator(BinaryOperator.COMPARE_LE);
      else if ( op == BinaryOperator.COMPARE_LT )
        new_condition.setOperator(BinaryOperator.COMPARE_GT);
      else if ( op == BinaryOperator.COMPARE_LE )
        new_condition.setOperator(BinaryOperator.COMPARE_GE);
    }
    loop.setCondition(new_condition);

    // Modify step
    UnaryExpression new_step = new UnaryExpression(
      UnaryOperator.POST_INCREMENT, (Expression)new_index.clone());
    loop.setStep(new_step);

    PrintTools.printlnStatus(LoopTools.toControlString(loop), 1);

    // Modify the original index variable in terms of the new index variable.
    IRTools.replaceSymbolIn(loop.getBody(), index.getSymbol(), subst);

    // Insert the old initial statement before the loop; the first assignment
    // should always occur.
    ((CompoundStatement)loop.getParent()).addStatementBefore(loop, old_assign);

    /* old_assign.annotateBefore(orig_code); */

    // Inserts last value assignment after the loop.
    Statement last_assign = new IfStatement(
      nzt_condition,
      new ExpressionStatement(
        new AssignmentExpression(
          (Expression)index.clone(),
          AssignmentOperator.NORMAL,
          subst.clone()
        )
      )
    );
    ((CompoundStatement)loop.getParent()).addStatementAfter(loop, last_assign);

    /* last_assign.annotateAfter(new CodeAnnotation("#endif\n")); */
  }

}
