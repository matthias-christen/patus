package cetus.transforms;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

import cetus.exec.*;
import cetus.hir.*;

/**
 * Transforms a program such that every statement contains
 * at most one function call.  In the case of nested calls,
 * the innermost calls will be called first.  Temporaries
 * are introduced to hold the results of function calls.
 * Remember that this normalization does not guarantee the intended program
 * structure (single call per statement) since there may be a case that is
 * not normalized to provide code correctness.
 */
public class SingleCall extends ProcedureTransformPass
{
  private static final String pass_name = "[SingleCall]";

  /** Constructs a new SingleCall transformation pass */
  public SingleCall(Program program)
  {
    super(program);
  }

  // Returns list of function calls in stmt that can be precomputed before stmt
  // The returned list should contain the function calls in post order for
  // equivalent evaluation.
  private static List<FunctionCall> getEligibleFunctionCalls(Statement stmt)
  {
    List<FunctionCall> ret = new LinkedList<FunctionCall>();
    PostOrderIterator iter = new PostOrderIterator(stmt,
        CompoundStatement.class,     // handled separately
        DeclarationStatement.class,  // won't be handled
        ConditionalExpression.class, // won't be handled
        StatementExpression.class    // won't be handled
    );
    while (iter.hasNext()) {
      Object o = iter.next();
      if (o instanceof FunctionCall) {
        FunctionCall fc = (FunctionCall)o;
        if (isUnsafe(stmt, fc)) {
          ret.clear();
          break;
        } else if (isUnnecessary(stmt, fc)) {
          continue;
        }
        ret.add(fc);
      }
    }
    return ret;
  }

  // Test for unsafe logic
  private static boolean isUnsafeLogic(Statement stmt, FunctionCall fc)
  {
    Traversable t = fc.getParent();
    while (t != stmt) {
      if (t instanceof BinaryExpression) {
        BinaryOperator bop = ((BinaryExpression)t).getOperator();
        if (bop == BinaryOperator.LOGICAL_AND ||
            bop == BinaryOperator.LOGICAL_OR)
          return true;
      }
      t = t.getParent();
    }
    return false;
  }

  // Test for possible unsafe scenario.
  // 1. BinaryOperation that possibly short-circuited
  // 2. Function pointers
  // 3. Within loop-controlling constructs
  // 4. No type information is extracted
  private static boolean isUnsafe(Statement stmt, FunctionCall fc)
  {
    BinaryExpression be = null;
    List types = null;
    return (
      isUnsafeLogic(stmt, fc) ||
      !(fc.getName() instanceof Identifier) ||
      (types=getSpecifiers(fc)) == null ||
      (types.size() == 1 && types.get(0) == Specifier.VOID) ||
      IRTools.getAncestorOfType(fc, Statement.class) instanceof Loop
    );
  }

  // Test for unnecessary transformation.
  // 1. Function call is the only expression in the statement
  // 2. Already in a simple form: lhs = foo();
  private static boolean isUnnecessary(Statement stmt, FunctionCall fc)
  {
    Traversable parent = fc.getParent();
    return (
      parent.equals(stmt) ||
      parent.getParent().equals(stmt) && parent instanceof AssignmentExpression
    );
  }

  /** Performs transformation for the specified procedure */
  public void transformProcedure(Procedure proc)
  {
    List<CompoundStatement> comp_stmts =
        IRTools.getDescendentsOfType(proc, CompoundStatement.class);
    for (CompoundStatement comp_stmt : comp_stmts)
      transformCompound(comp_stmt);
  }

  // Check if it is possible to get valid specifiers from the function call.
  private static List getSpecifiers(FunctionCall fc)
  {
    Symbol symbol = SymbolTools.getSymbolOf(fc.getName());
    if (symbol == null)
      return null;
    List ret = new LinkedList(symbol.getTypeSpecifiers());
    // Remove specifiers not for types.
    ret.remove(Specifier.EXTERN);
    ret.remove(Specifier.STATIC);
    ret.remove(Specifier.INLINE);
    return ret;
  }

  private static void transformCompound(CompoundStatement cs)
  {
    for (Traversable child : new LinkedList<Traversable>(cs.getChildren())) {
      Statement stmt = (Statement)child; // should be guaranteed
      for (FunctionCall fcall : getEligibleFunctionCalls(stmt)) {
        // Actions being performed here for stmt: s = ..... foo() ....;
        // - get a new variable -> foo_0
        // - create a new assignment foo_0 = foo_0;
        // - swap the LHS of the assignment with the original function call.
        // - inserts the new assignment
        // Result:
        // {
        //   <types> foo_0;
        //   foo_0 = foo();
        //   s = ..... foo_0 .....;
        // }
        List types = getSpecifiers(fcall);
        Identifier id =
            SymbolTools.getTemp(stmt, types, fcall.getName().toString());
        Statement assign = new ExpressionStatement(new AssignmentExpression(
            id.clone(), AssignmentOperator.NORMAL, id));
        id.swapWith(fcall);
        cs.addStatementBefore(stmt, assign);
        CommentAnnotation info =
          new CommentAnnotation("Normalized Call: " + fcall);
        info.setOneLiner(true);
        assign.annotateBefore(info);
      }
    }
  }

  public String getPassName()
  {
    return pass_name;
  }
}
