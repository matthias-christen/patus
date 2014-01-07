package cetus.treewalker;

import java.lang.*;
import java.util.*;

import cetus.hir.*;

/** Base class for all tree walkers.
 */
public abstract class TreeWalker
{
  protected int anonymous_count = 0;

  protected Declaration saved_decl = null;

  protected TranslationUnit tunit;

  protected TreeWalker(String filename)
  {
    tunit = new TranslationUnit(filename);
  }

  protected Expression action_assignment_expression(TreeNode root)
  {
    if (root.getChildCount() == 1)
      return (Expression)doAction(root.getChild(0));
    else
      return new AssignmentExpression((Expression)doAction(root.getChild(0)),
        AssignmentOperator.fromString(root.getChild(1).getChild(0).getText()),
        (Expression)doAction(root.getChild(2))); 
  }

  protected CompoundStatement action_compound_statement(TreeNode root)
  {
    CompoundStatement stmt = new CompoundStatement();
    List list = defaultListAction(root);

    Iterator iter = list.iterator();
    while (iter.hasNext())
    {
      Object o = iter.next();

      if (o instanceof Statement)
        stmt.addStatement((Statement)o);
      else
        stmt.addDeclaration((Declaration)o);
    }

    return stmt;
  }

  protected Expression action_conditional_expression(TreeNode root)
  {
    if (root.getChildCount() == 1)
      return (Expression)doAction(root.getChild(0));
    else
      return new ConditionalExpression((Expression)doAction(root.getChild(0)),
        (Expression)doAction(root.getChild(2)),
        (Expression)doAction(root.getChild(4)));
  }

  protected Declaration action_declaration(TreeNode root)
  {
    return (Declaration)doAction(root.getChild(0));
  }

  protected List<Declaration> action_declaration_list(TreeNode root)
  {
    List list = new LinkedList();

    if (root.hasChildren())
    {
      for (TreeNode n : root)
      {
        Object o = doAction(n);
        if (o != null)
        {
          if (saved_decl != null)
          {
            list.add(saved_decl);
            saved_decl = null;
          }
          list.add(o);

/*          try {
            list = (List)o;
          } catch (ClassCastException e) {
            if (saved_decl != null)
            {
              list.add(saved_decl);
              saved_decl = null;
            }
            list.add(o);
          } */
        }
      }
    }

    return list;
  }

  protected List<Specifier> action_declaration_specifiers(TreeNode root)
  {
    return defaultListAction(root);
  }

  protected Declarator action_enumerator(TreeNode root)
  {
    VariableDeclarator decl = new VariableDeclarator(SymbolTools.getOrphanID(root.getChild(0).getText()));
    if (root.getChildCount() > 1)
    {
      Initializer init = new Initializer((Expression)doAction(root.getChild(2)));
      decl.setInitializer(init);
    }
    return decl;
  }

  protected Expression action_expression(TreeNode root)
  {
    if (root.getChildCount() == 1)
      return (Expression)doAction(root.getChild(0));
    else
    {
      Expression expr  = (Expression)doAction(root.getChild(0));
      Expression expr2 = (Expression)doAction(root.getChild(2));

      if (expr instanceof CommaExpression)
      {
        ((CommaExpression)expr).addExpression(expr2);
        return expr;
      }
      else
      {
        List list = new LinkedList();
        list.add(expr);
        list.add(expr2);
        return new CommaExpression(list);
      }
    }
  }

  protected Statement action_jump_statement(TreeNode root)
  {
    String s = root.getChild(0).getText();

    if (s.equals("return"))
    {
      if (root.getChildCount() == 3)
      {
        Expression expr = (Expression)doAction(root.getChild(1));
        return new ReturnStatement(expr);
      }
      else
        return new ReturnStatement();
    }
    else if (s.equals("break"))
      return new BreakStatement();
    else if (s.equals("continue"))
      return new ContinueStatement();
    else if (s.equals("goto"))
      return new GotoStatement(SymbolTools.getOrphanID(root.getChild(1).getText()));
    else
      throw new RuntimeException("unknown jump statement");
  }

  protected CompoundStatement action_labeled_statement(TreeNode root)
  {
    String s = root.getChild(0).getText();

    CompoundStatement stmt = new CompoundStatement();

    if (s.equals("case"))
    {
      stmt.addStatement(new Case((Expression)doAction(root.getChild(1))));
      stmt.addStatement((Statement)doAction(root.getChild(3)));
    }
    else if (s.equals("default"))
    {
      stmt.addStatement(new Default());
      stmt.addStatement((Statement)doAction(root.getChild(2)));
    }
    else
    {
      stmt.addStatement(new Label(SymbolTools.getOrphanID(s)));
      stmt.addStatement((Statement)doAction(root.getChild(2)));
    }

    return stmt;
  }

  protected Statement action_selection_statement(TreeNode root)
  {
    Expression expr = (Expression)doAction(root.getChild(2));
    Statement stmt = (Statement)doAction(root.getChild(4));

    if (root.getChild(0).getText().equals("if"))
    {
      if (root.getChildCount() > 5)
        return new IfStatement(expr, stmt, (Statement)doAction(root.getChild(6)));
      else
        return new IfStatement(expr, stmt);
    }
    else
    {
      if (stmt instanceof CompoundStatement)
        return new SwitchStatement(expr, (CompoundStatement)stmt);
      else
      {
        SwitchStatement s_stmt = new SwitchStatement(expr);
        s_stmt.getBody().addStatement(stmt);
        return s_stmt;
      }
    }
  }

  protected Specifier action_specifier(TreeNode root)
  {
    return Specifier.fromString(root.getChild(0).getText());
  }

  protected Statement action_statement(TreeNode root)
  {
    Object o = doAction(root.getChild(0));
    if (o instanceof Declaration)
      return new DeclarationStatement((Declaration)o);
    else
      return (Statement)o;
  }

  protected List<String> action_string_literal_list(TreeNode root)
  {
    List list = null;

    if (root.getChildCount() == 1)
    {
      list = new LinkedList();
      list.add(root.getChild(0).getText());
    }
    else
    {
      list = action_string_literal_list(root.getChild(0));
      list.add(root.getChild(1).getText());
    }

    return list;
  }

  protected TranslationUnit action_translation_unit(TreeNode root)
  {
    if (root.hasChildren())
    {
      List<Declaration> list = (List<Declaration>)doAction(root.getChild(0));

      for (Declaration d : list)
        tunit.addDeclaration(d);
    }

    return tunit;
  }

  protected Expression defaultBinaryExpressionAction(TreeNode root)
  {
    if (root.getChildCount() == 1)
      return (Expression)doAction(root.getChild(0));
    else
    {
      Expression lhs = (Expression)doAction(root.getChild(0));
      Expression rhs = (Expression)doAction(root.getChild(2));

      if (lhs == null || rhs == null)
      {
        root.printTree(0);
        System.exit(1);
      }

      return new BinaryExpression(lhs, BinaryOperator.fromString(root.getChild(1).getText()), rhs);
    }
  }

  protected List defaultCompactListAction(TreeNode root)
  {
    List list = new LinkedList();

    if (root.hasChildren())
    {
      for (TreeNode n : root)
      {
        Object o = doAction(n);
        if (o != null)
          list.add(o);
      }
    }

    return list;
  }

  protected List defaultListAction(TreeNode root)
  {
    List list = new LinkedList();

    if (root.hasChildren())
    {
      for (TreeNode n : root)
      {
        Object o = doAction(n);
        if (o != null)
        {
          try {
            list = (List)o;
          } catch (ClassCastException e) {
            list.add(o);
          }
        }
      }
    }

    return list; 
  }

  protected Object defaultNotImplemented(TreeNode root)
  {
    List list = new LinkedList();
    for (TreeNode n : root)
    {
      Object o = doAction(n);
      if (o != null)
        list.add(o);
    }

    throw new RuntimeException("not yet implemented");
  }

  protected abstract Object doAction(TreeNode root);

  public abstract TranslationUnit run(TreeNode root);

  public Declaration symbolLookup(IDExpression id_expr)
  {
    return tunit.findSymbol(id_expr);
  }
}
