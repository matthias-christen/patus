package cetus.transforms;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

import cetus.exec.*;
import cetus.hir.*;

/**
 * Transforms a program such that every procedure has at
 * most one return statement.  A new variable is introduced
 * to store the return value.  Each return statement is
 * replaced with a store of the return value followed by
 * a jump to a final return statement that returns the
 * variable.
 */
public class SingleReturn extends ProcedureTransformPass
{
  /** Constructs a new SingleReturn transformation pass. */
  public SingleReturn(Program program)
  {
    super(program);
  }

  public String getPassName()
  {
    return new String("[SingleReturn]");
  }

  public void transformProcedure(Procedure proc)
  {
    List<ReturnStatement> ret_stmts =
        IRTools.getDescendentsOfType(proc, ReturnStatement.class);

    CompoundStatement body = proc.getBody();

    List ret_type = new LinkedList(proc.getReturnType());

    /* Remove scope qualifiers not necessary in type resolution */
    while (ret_type.remove(Specifier.INLINE));
    while (ret_type.remove(Specifier.STATIC));
    while (ret_type.remove(Specifier.EXTERN));

    boolean ret_type_void =
        (ret_type.contains(Specifier.VOID) && ret_type.size() == 1);

    Identifier ret_id = null;
    /* add a variable _ret_val of the same type as the procedure return type */
    if (!ret_type_void)
    {
      /* check for implicit int (TODO - differentiate this from constructors) */
      if (ret_type.size() == 0)
        ret_type = (new ChainedList<Specifier>()).addLink(Specifier.INT);
      ret_id = SymbolTools.getTemp(body, ret_type, "_ret_val");
    }

    /* add a labeled return statement to the end of the procedure */
    String done_label = "_done";
    body.addStatement(new Label(new NameID(done_label)));

    if (!ret_type_void)
      body.addStatement(new ReturnStatement(ret_id.clone()));
    else
      body.addStatement(new ReturnStatement());

    /* redirect the preexisting return statements to the labeled return */
    for (ReturnStatement ret_stmt : ret_stmts) {
      // Identify the parent compound statement.
      CompoundStatement comp_stmt =
          IRTools.getAncestorOfType(ret_stmt, CompoundStatement.class);
      // Add goto statement.
      Statement goto_stmt = new GotoStatement(new NameID(done_label));
      comp_stmt.addStatementAfter(ret_stmt, goto_stmt);
      // Add temporary assignments.
      if (!ret_type_void) {
        Statement new_stmt = new ExpressionStatement(new AssignmentExpression(
            ret_id.clone(),
            AssignmentOperator.NORMAL,
            ret_stmt.getExpression().clone()
        ));
        comp_stmt.addStatementBefore(goto_stmt, new_stmt);
      }
      // Remove the original return statement.
      comp_stmt.removeStatement(ret_stmt);
      // Add comments.
      CommentAnnotation info =
          new CommentAnnotation("Normalized Return: " + ret_stmt);
      info.setOneLiner(true);
      goto_stmt.annotateBefore(info);
    }
  }
}
