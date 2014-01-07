package cetus.transforms;

import java.util.*;
import cetus.hir.*;

/**
 * This normalization pass is intended to simplify
 * return statements for the Procedures in the
 * program. Return statements with no return expressions 
 * are added at the end of the procedure if a return 
 * statement doesn't exist already. For all return 
 * statements in the procedure, the return 
 * expressions are assigned to a standard return variable 
 * and are replaced in the return statement with 
 * this standard return variable.
 * Especially useful for statement based analyses and 
 * simplification of interprocedural analyses
 */
public class NormalizeReturn extends ProcedureTransformPass {
	
	public NormalizeReturn(Program program)
	{
		super(program);
	}
	
	public String getPassName()
	{
		return "[NORMALIZE-RETURN]";
	}
	
	public void transformProcedure(Procedure proc)
	{
		Identifier return_id = null;
		
		// Get the last statement of the procedure
		List children = proc.getBody().getChildren();

    // Empty procedure body will have "null" last_statement.
    Statement last_statement = null;
    if ( !children.isEmpty() )
		  last_statement = (Statement)children.get(children.size()-1);

		// Irrespective of the return type, if the procedure 
		// doesn't already contain a return statement, insert one.
		// This does not change the semantics as the return value 
		// for a procedure with a valid return type but no return 
		// statement is undefined in the first place
		if ( !(last_statement instanceof ReturnStatement) )
		{
			ReturnStatement return_statement = new ReturnStatement();
      if ( children.isEmpty() )
        (proc.getBody()).addStatement(return_statement);
      else
			  (proc.getBody()).addStatementAfter(last_statement, return_statement);
		}
		
		// If procedure has a return type other than void, then 
		// create a standard return variable of return type for the
		// procedure and insert a declaration for it in the procedure body

		// Now get all return statements for the procedure and simplify
		DepthFirstIterator iter = new DepthFirstIterator(proc.getBody());
		while (iter.hasNext())
		{
			Object o = iter.next();
			if (o instanceof ReturnStatement)
			{
				ReturnStatement ret_stmt = (ReturnStatement)o;
				if (ret_stmt.getExpression() != null)
				{
					// Create return variable if it hasn't already been created
					// The return expression will be non-null only if there 
					// exists a return type on the procedure
					if (return_id == null)
					{
            List return_type = proc.getReturnType();
            // static type is not necessary for the temporary variable.
            return_type.remove(Specifier.STATIC);
						return_id = SymbolTools.getTemp(
							proc.getBody(), 
              return_type,
							"normalizeReturn_val");
					}

					// Use clone
					AssignmentExpression new_assign = 
						new AssignmentExpression(return_id.clone(),
								AssignmentOperator.NORMAL,
								ret_stmt.getExpression().clone());
					ExpressionStatement assign_stmt = 
						new ExpressionStatement(new_assign);
					
					// Insert the new assignment statement right before
					// the return statement
					CompoundStatement parent = (CompoundStatement)ret_stmt.getParent();
					parent.addStatementBefore(ret_stmt, assign_stmt);
					
					// Replace the return expression in the return statement
					// with the new return var
					(ret_stmt.getExpression()).swapWith(return_id.clone());
				}
			}
		}
	}
}
