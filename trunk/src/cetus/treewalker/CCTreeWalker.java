package cetus.treewalker;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

import cetus.hir.*;

/**
 * TreeWalker to convert a C++ parse tree to Cetus IR.
 */
public class CCTreeWalker extends TreeWalker
{
  private HashMap<String,Integer> action_map;

  public CCTreeWalker(String filename)
  {
    super(filename);
    action_map = new HashMap(200);

    String nonterminals[] =
    {
      "abstract_declarator",
      "access_specifier",
      "additive_expression",
      "ambiguity",
      "and_expression",
      "any_word",
      "asm_clobbers",
      "asm_definition",
      "asm_operand",
      "asm_operands",
      "assignment_expression",
      "assignment_operator",
      "attrib",
      "attribute",
      "attribute_list",
      "attributes",
      "base_clause",
      "base_specifier",
      "base_specifier_list",
      "block_declaration",
      "boolean_literal",
      "cast_expression",
      "class_name",
      "compound_statement",
      "condition",
      "conditional_expression",
      "constant_expression",
      "constant_initializer",
      "conversion_declarator",
      "conversion_function_id",
      "conversion_type_id",
      "ctor_initializer",
      "cv_qualifier",
      "cv_qualifier_seq",
      "decl_specifier",
      "decl_specifier_seq",
      "declaration",
      "declaration_seq",
      "declarator",
      "declarator_head",
      "declarator_id",
      "declarator_tail",
      "declarator_tail_seq",
      "delete_expression",
      "direct_abstract_declarator",
      "direct_declarator",
      "direct_new_declarator",
      "enumerator_definition",
      "enumerator_list",
      "equality_expression",
      "exception_declaration",
      "exception_specification",
      "exclusive_or_expression",
      "explicit_instantiation",
      "explicit_specialization",
      "expression",
      "expression_list",
      "expression_statement",
      "for_init_statement",
      "function_definition",
      "function_prefix",
      "function_specifier",
      "function_try_block",
      "handler",
      "handler_seq",
      "id_expression",
      "inclusive_or_expression",
      "init_declarator",
      "init_declarator_list",
      "initializer",
      "initializer_clause",
      "initializer_list",
      "iteration_statement",
      "jump_statement",
      "labeled_statement",
      "linkage_specification",
      "literal",
      "logical_and_expression",
      "logical_or_expression",
      "mem_initializer",
      "mem_initializer_id",
      "mem_initializer_list",
      "member_declaration",
      "member_declaration_alt",
      "member_declarator",
      "member_declarator_list",
      "member_declarator_list_tail",
      "member_specification",
      "multiplicative_expression",
      "namespace_alias_definition",
      "namespace_definition",
      "nested_name_specifier",
      "new_declarator",
      "new_expression",
      "new_initializer",
      "new_placement",
      "new_type_id",
      "operator_",
      "operator_function_id",
      "parameter_declaration",
      "parameter_declaration_clause",
      "parameter_declaration_list",
      "pm_expression",
      "postfix_expression",
      "primary_expression",
      "ptr_operator",
      "qualified_id",
      "qualified_namespace_specifier",
      "relational_expression",
      "scoped_class_name",
      "scoped_id",
      "scoped_unqualified_id",
      "selection_statement",
      "shift_expression",
      "simple_declaration",
      "simple_type_specifier",
      "statement",
      "statement_seq",
      "storage_class_specifier",
      "template_argument",
      "template_argument_list",
      "template_declaration",
      "template_parameter",
      "template_parameter_list",
      "throw_expression",
      "translation_unit",
      "try_block",
      "type_id",
      "type_id_list",
      "type_parameter",
      "type_specifier",
      "type_specifier_seq",
      "unary_expression",
      "unary_operator",
      "unqualified_id",
      "using_declaration",
      "using_directive"
    };

    for (int i = 0; i < nonterminals.length; ++i)
      action_map.put("action_" + nonterminals[i], i + 1);
  }

  protected List action_abstract_declarator(TreeNode root)
  {
    List list = new ArrayList();
    for (TreeNode n : root)
    {
      Object o = doAction(n);

      if (o instanceof List)
        list.addAll((List)o);
      else
        list.add(o);
    }

    return list;
  }

  protected AccessLevel action_access_specifier(TreeNode root)
  {
    return new AccessLevel(Specifier.fromString(root.getChild(0).getText()));
  }

  protected Object action_ambiguity(TreeNode root)
  {
    ArrayList list = new ArrayList();

    for (TreeNode n : root)
      list.add(doAction(n));

    /* Must return only one item.
       TODO figure out which one to pick. */

    if (list.get(0) instanceof VariableDeclaration)
    {
      List<Specifier> specs = ((VariableDeclaration)list.get(0)).getSpecifiers();

      for (Specifier s : specs)
      {
        if (s instanceof UserSpecifier)
        {
          /* check if the name of the type in the user specifier
             is actually a type name in the symbol table */

          Declaration decl = symbolLookup(((UserSpecifier)s).getIDExpression());

          if (decl instanceof ClassDeclaration
              || decl instanceof cetus.hir.Enumeration
              || (decl instanceof VariableDeclaration
                  && ((VariableDeclaration)decl).isTypedef()))
          {
            /* it's a type. okay */
          }
          else
          {
            /* it's not a type, so this is an illegal specifier
               and we should use the other tree branch */
            //System.err.println("ambig used branch 1");
            return list.get(1);
          }
        }
      }
    }

    //System.err.println("ambig used branch 0");
    return list.get(0);
  }

  protected Object action_block_declaration(TreeNode root)
  {
    return doAction(root.getChild(0));
  }

  protected Expression action_cast_expression(TreeNode root)
  {
    switch (root.getChildCount())
    {
      case 1:
        return (Expression)doAction(root.getChild(0));
      case 3:
        return new StatementExpression((CompoundStatement)doAction(root.getChild(1)));
      case 4:
        List list = (List)doAction(root.getChild(1));
        Expression expr = (Expression)doAction(root.getChild(3));
        return new Typecast(Typecast.NORMAL, list, expr);
      default:
        throw new RuntimeException("bad cast expression");
    }
  }

  protected IDExpression action_class_name(TreeNode root)
  {
    String name = root.getChild(0).getText();

    switch (root.getChildCount())
    {
      case 1:
        return SymbolTools.getOrphanID(name);
      case 3: 
        return new TemplateID(name);
      case 4:
        return new TemplateID(name, (List)doAction(root.getChild(2)));
      default:
        throw new RuntimeException("bad class name");
    }
  }

  protected Object action_condition(TreeNode root)
  {
    if (root.getChildCount() == 1)
      return (Expression)doAction(root.getChild(0));
    else
      throw new RuntimeException("not yet implemented");
  }

  protected Initializer action_constant_initializer(TreeNode root)
  {
    return new Initializer((Expression)doAction(root.getChild(1)));
  }

  protected Specifier action_decl_specifier(TreeNode root)
  {
    Specifier spec = Specifier.fromString(root.getChild(0).getText());

    if (spec == null)
      spec = (Specifier)doAction(root.getChild(0));

    return spec;
  }

  protected IDExpression action_conversion_function_id(TreeNode root)
  {
    return SymbolTools.getOrphanID("FIXME");
  }

  protected Declarator action_declarator(TreeNode root)
  {
    if (root.getChildCount() == 1)
      return (Declarator)doAction(root.getChild(0));
    else
    {
      Declarator d = (Declarator)doAction(root.getChild(1));
      d.getSpecifiers().add(0, (Specifier)doAction(root.getChild(0)));
      return d;
    }
  }

  protected Declarator action_declarator_head(TreeNode root)
  {
    int child_count = root.getChildCount();

    switch (child_count)
    {
      case 1:
        return new VariableDeclarator((IDExpression)doAction(root.getChild(0)));
      case 2:
        Specifier spec = (Specifier)doAction(root.getChild(0));
        Declarator decl = (Declarator)doAction(root.getChild(1));
        decl.getSpecifiers().add(0, spec);
        return decl;
      case 3:
        return new NestedDeclarator((Declarator)doAction(root.getChild(1)));
      default:
        throw new RuntimeException("bad declarator_head");
    }
  }

  protected IDExpression action_declarator_id(TreeNode root)
  {
    if (root.getChildCount() == 1)
      return (IDExpression)doAction(root.getChild(0));
    else
    {
      IDExpression id_expr = (IDExpression)doAction(root.getChild(1));
      id_expr.setGlobal(true);
      return id_expr;
    }
  }

  protected Object action_declarator_tail(TreeNode root)
  {
    int child_count = root.getChildCount();

    if (root.getChild(0).getText().equals("["))
    {
      if (child_count == 2)
        return new ArraySpecifier();
      else
        return new ArraySpecifier((Expression)doAction(root.getChild(1)));
    }

    List params = (List)doAction(root.getChild(1));

    return new ProcedureDeclarator(SymbolTools.getOrphanID("dummy"), params);
  }

  protected Object action_direct_abstract_declarator(TreeNode root)
  {
    if (root.getChild(0).getText().equals("["))
    {
      if (root.getChildCount() == 2)
        return new ArraySpecifier();
      else
        return new ArraySpecifier((Expression)doAction(root.getChild(1)));
    }

    throw new RuntimeException("unknown direct_abstract_declarator"); 
  }

  protected Object action_direct_declarator(TreeNode root)
  {
    /* direct_declarator -> declarator_id */
try {
    if (root.getChildCount() == 1)
      return new VariableDeclarator((IDExpression)doAction(root.getChild(0)));
} catch (NotAnOrphanException e) {
  root.printTree(0);
  throw e;
}

    /* direct_declarator -> '(' declarator ')' */
    if (root.getChild(0).getText().equals("("))
      return new NestedDeclarator((Declarator)doAction(root.getChild(1)));

    Object direct_decl = doAction(root.getChild(0));

    /* direct_declarator -> direct_declarator '[' constant_expression_opt ']' */
    if (root.getChild(1).getText().equals("["))
    {
      ArraySpecifier array_spec = null;
      if (root.getChildCount() == 4)
        array_spec = new ArraySpecifier((Expression)doAction(root.getChild(2)));
      else
        array_spec = new ArraySpecifier();

      if (direct_decl instanceof IDExpression)
        return new VariableDeclarator((IDExpression)direct_decl, array_spec);
      else // it's a VariableDeclarator
      {
        ((VariableDeclarator)direct_decl).getTrailingSpecifiers().add(array_spec);
        return direct_decl;
      } 
    }

    /* direct_declarator '(' parameter_declaration_clause ')' cv_qualifier_seq_opt exception_specification_opt */
    if (root.getChild(1).getText().equals("("))
    {
      List params = (List)doAction(root.getChild(2));

      if (direct_decl instanceof NestedDeclarator)
      {
        Iterator iter = params.iterator();
        while (iter.hasNext())
          ((NestedDeclarator)direct_decl).addParameter((Declaration)iter.next());
        return direct_decl;
      }
      else
        return new ProcedureDeclarator(((VariableDeclarator)direct_decl).getID().clone(),
          (List)doAction(root.getChild(2)));
    }

    throw new RuntimeException("not yet implemented");
  }

  protected Declaration action_explicit_specialization(TreeNode root)
  {
    return new TemplateDeclaration(new LinkedList(), (Declaration)doAction(root.getChild(3)));
  }

  protected Statement action_expression_statement(TreeNode root)
  {
    return new ExpressionStatement(action_expression(root.getChild(0)));
  }

  protected Statement action_for_init_statement(TreeNode root)
  {
    Object o = doAction(root.getChild(0));
    if (o instanceof Declaration)
      return new DeclarationStatement((Declaration)o);
    else
      return (ExpressionStatement)o;
  }

  protected Procedure action_function_definition(TreeNode root)
  {
    int child_count = root.getChildCount();

    Procedure proc = (Procedure)doAction(root.getChild(0));
    CompoundStatement stmt = (CompoundStatement)doAction(root.getChild(child_count - 1));

    proc.setBody(stmt);

    if (child_count == 3)
      proc.setConstructorInitializers((List)doAction(root.getChild(1))); 

    return proc;
  }

  protected Procedure action_function_prefix(TreeNode root)
  {
    int child_count = root.getChildCount();

    if (child_count == 1)
      return new Procedure((Declarator)doAction(root.getChild(0)), new CompoundStatement());

    List specs = (List)doAction(root.getChild(0));
    
    Declarator decl = (Declarator)doAction(root.getChild(1));
    
    ProcedureDeclarator pdecl = null;

    if (child_count == 3)
    {
      List list = (List)doAction(root.getChild(2));
      decl = mergeDeclaratorHeadAndTail(decl, list);
    }

    return new Procedure(specs, decl, new CompoundStatement());
  }

  protected IDExpression action_id_expression(TreeNode root)
  {
    return (IDExpression)doAction(root.getChild(0));
  }

  protected Declarator action_init_declarator(TreeNode root)
  {
    Declarator decl = (Declarator)doAction(root.getChild(0));

    if (root.getChildCount() > 1)
      decl.setInitializer((Initializer)doAction(root.getChild(1)));

    return decl;
  }  

  protected Initializer action_initializer(TreeNode root)
  {
    if (root.getChildCount() == 2)
      return (Initializer)doAction(root.getChild(1));
    else
      return new ConstructorInitializer((List)doAction(root.getChild(1)));
  }

  protected Initializer action_initializer_clause(TreeNode root)
  {
    switch (root.getChildCount())
    {
      case 1:
        return new Initializer((Expression)doAction(root.getChild(0)));
      case 2:
        return new Initializer(new LinkedList());
      case 3:
      case 4:
        return new Initializer((List)doAction(root.getChild(1)));
      default:
        throw new RuntimeException("bad initializer_clause");
    }
  }

  protected Statement action_iteration_statement(TreeNode root)
  {
    String s = root.getChild(0).getText();

    if (s.equals("for"))
    {
      Statement init = (Statement)doAction(root.getChild(2));
      Expression condition = null;
      Expression step = null;

      Object o = doAction(root.getChild(3));
      if (o != null)
        condition = (Expression)o;

      o = doAction(root.getChild(4));
      if (o != null && o instanceof Expression)
        step = (Expression)o;
      else
      {
        o = doAction(root.getChild(5));
        if (o != null && o instanceof Expression)
          step = (Expression)o;
      }

      Statement body = (Statement)doAction(root.getChild(root.getChildCount() - 1));

      return new ForLoop(init, condition, step, body);      
    }
    else if (s.equals("while"))
    {
      Expression expr = (Expression)doAction(root.getChild(2));
      Statement stmt = (Statement)doAction(root.getChild(4));

      return new WhileLoop(expr, stmt);
    }
    else // do loop
    {
      Statement stmt = (Statement)doAction(root.getChild(1));
      Expression expr = (Expression)doAction(root.getChild(4));

      return new DoLoop(stmt, expr);
    }
  }

  protected LinkageSpecification action_linkage_specification(TreeNode root)
  {
    String s = root.getChild(1).getText();

    switch (root.getChildCount())
    {
      case 3:
        return new LinkageSpecification(s, (Declaration)doAction(root.getChild(2)));

      case 4:
        return new LinkageSpecification(s, new LinkedList());

      case 5:
        return new LinkageSpecification(s, (List)doAction(root.getChild(3)));

      default:
        throw new RuntimeException("bad linkage specification");
    }
  }

  protected Expression action_literal(TreeNode root)
  {
    String s = root.getChild(0).getText();

    if (s.equals("string_literal_list"))
    {
      List<String> list = action_string_literal_list(root.getChild(0));

      String actual_string = "";

      for (String lit : list)
      {
        /* The extra backslash-quotes here are because the graphviz
           file format requires escape sequences for quotes.  The
           StringLiteral class automatically prints quotes around
           itself, so both backslashes and outer quotes need
           to be removed. */
        actual_string += lit.substring(2, lit.length() - 2);      
      }

      return new StringLiteral(actual_string);
    }
    else if (Character.isDigit(s.charAt(0)))
    {
      if (s.contains("."))
        return new FloatLiteral(Double.parseDouble(s));
      else
      {
        String suffix = "";

        int i = s.length() - 1;
        for ( ; i >= 0; --i)
        {
          char ch = s.charAt(i);

          /* keep skipping suffix letters but watch
             out for hexadecimal numbers  */

          if (Character.isDigit(ch))
            break;

          if ((ch >= 'a' && ch <= 'f')
              || (ch >= 'A' && ch <= 'F'))
            break;
        }

        suffix = s.substring(i + 1); 
        s = s.substring(0, i + 1);

        if (suffix.equals(""))
          return new IntegerLiteral(Long.decode(s));
        else
          return new IntegerLiteral(Long.decode(s), suffix);
      }
    }
    else if (s.equals("boolean_literal"))
    {
      if (root.getChild(0).getChild(0).getText().equals("true"))
        return new BooleanLiteral(true);
      else
        return new BooleanLiteral(false);
    }

    throw new RuntimeException("bad literal");
  }

  protected Declarator action_mem_initializer(TreeNode root)
  {
    VariableDeclarator decl = new VariableDeclarator((IDExpression)doAction(root.getChild(0)));
    if (root.getChildCount() == 4)
      decl.setInitializer(new ConstructorInitializer((List)doAction(root.getChild(2))));
    else
      decl.setInitializer(new ConstructorInitializer(new LinkedList()));

    return decl;
  }

  protected IDExpression action_mem_initializer_id(TreeNode root)
  {
    int child_count = root.getChildCount();
    IDExpression expr = (IDExpression)doAction(root.getChild(child_count - 1));
    if (child_count == 2)
      expr.setGlobal(true);

    return expr;
  }

  protected Declaration action_member_declaration(TreeNode root)
  {
    Object o = null;

    switch (root.getChildCount())
    {
      case 1:
        o = doAction(root.getChild(0));
        if (o != null)
          return (Declaration)o;
        else
          return new VariableDeclaration(new LinkedList());

      case 2:
        o = doAction(root.getChild(0));
        if (o instanceof List)
        {
          if (root.getChild(0).getText().startsWith("decl"))
            return new VariableDeclaration((List)o, new LinkedList());
          else
            return new VariableDeclaration(new LinkedList(), (List)o);
        }
        else
          return new VariableDeclaration(new VariableDeclarator((IDExpression)o));

      case 3:
        if (root.getChild(0).getText().equals("::"))
        {
          IDExpression expr = action_scoped_unqualified_id(root.getChild(1));
          return new VariableDeclaration(new VariableDeclarator(expr));
        }
        else
        {
          List spec_list = (List)doAction(root.getChild(0));
          List decl_list = (List)doAction(root.getChild(1));
          return new VariableDeclaration(spec_list, decl_list);
        }

      default:
        throw new RuntimeException("not yet implemented");
    }
  }

  protected Declarator action_member_declarator(TreeNode root)
  {
    if (root.getChild(0).getText().equals("declarator"))
    {
      Declarator decl = action_declarator(root.getChild(0));
      if (root.getChildCount() > 1)
        decl.setInitializer(action_constant_initializer(root.getChild(1)));
      return decl;
    }
    else
    {
      String s = root.getChild(0).getText();

      if (root.getChild(0).getText().equals(":"))
        s = "anonymous" + Integer.toString(anonymous_count++);

      return new VariableDeclarator(SymbolTools.getOrphanID(s));
    }
  }

  protected List action_member_specification(TreeNode root)
  {
    int child_count = root.getChildCount();
    List list = null;

    Object member = doAction(root.getChild(0));

    if (child_count == 3 ||
        (child_count == 2 && !root.getChild(1).getText().equals(":")))
      list = (List)doAction(root.getChild(child_count - 1));
    else
      list = new LinkedList();

    list.add(0, member);
 
    return list;
  }

  protected Namespace action_namespace_definition(TreeNode root)
  {
    int child_count = root.getChildCount();

    Namespace space = null;

    if (root.getChild(1).getText().equals("{"))
      space = new Namespace();
    else
      space = new Namespace(SymbolTools.getOrphanID(root.getChild(1).getText()));

    List<Declaration> decl_list = null;
    if (root.getChild(child_count - 2).getText().equals("{"))
      decl_list = new LinkedList();
    else
      decl_list = action_declaration_list(root.getChild(child_count - 2));

    for (Declaration d : decl_list)
      space.addDeclaration(d);

    return space;
  }

  protected OperatorID action_operator_(TreeNode root)
  {
    String s = "";
    for (TreeNode n : root)
      s += n.getText();

    return new OperatorID(s);
  }

  protected OperatorID action_operator_function_id(TreeNode root)
  {
    return (OperatorID)doAction(root.getChild(1));
  }  

  protected VariableDeclaration action_parameter_declaration(TreeNode root)
  {
    int child_count = root.getChildCount();

    List specs = (List)doAction(root.getChild(0));

    if (child_count == 1)
    {
      /* decl_specifier_seq */
      return new VariableDeclaration(specs);
    }

    Declarator decl = null;
    Object o = doAction(root.getChild(1));

    if (o == null)
    {
      /* decl_specifier_seq '=' assignment_expression */
      decl = new VariableDeclarator(SymbolTools.getOrphanID(""));
      decl.setInitializer(new Initializer((Expression)doAction(root.getChild(2))));
    }
    else if (o instanceof Declarator)
    {
      /* decl_specifier_seq declarator */
      decl = (Declarator)o;

      /* decl_specifier_seq declarator '=' assignment_expression */
      if (child_count == 4)
        decl.setInitializer(new Initializer((Expression)doAction(root.getChild(3))));
    }
    else if (o instanceof List)
    {
      /* decl_specifier_seq abstract_declarator */
      /* decl_specifier_seq abstract_declarator '=' assignment_expression */
      List list = (List)o;
      for (Object o2 : list)
      {
        if (o2 instanceof List)
          specs.addAll((List)o2);
        else
          specs.add((Specifier)o2);
      }

      if (child_count == 4)
      {
        decl = new VariableDeclarator(SymbolTools.getOrphanID(""));
        decl.setInitializer((Initializer)doAction(root.getChild(3)));
      } 
    }
    else
    {
      root.printTree(0);
      throw new RuntimeException("bad parameter_declaration");
    }

    if (decl == null)
      return new VariableDeclaration(specs);
    else
      return new VariableDeclaration(specs, decl); 
  }

  protected List action_parameter_declaration_clause(TreeNode root)
  {
    int child_count = root.getChildCount();

    List list = new LinkedList();

    switch (child_count)
    {
      case 0:
        return list;

      case 1:
        if (root.getChild(0).getText().equals("..."))
          list.add(new VariableDeclaration(new VariableDeclarator(SymbolTools.getOrphanID("..."))));
        else
          list = (List)doAction(root.getChild(0));
        return list;

      default:
        list = (List)doAction(root.getChild(0));
        list.add(new VariableDeclaration(new VariableDeclarator(SymbolTools.getOrphanID("..."))));
        return list;
    }
  }

  protected Expression action_pm_expression(TreeNode root)
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

      AccessOperator op = null;
      if (root.getChild(1).getText().equals(".*"))
        op = AccessOperator.MEMBER_DEREF_ACCESS;
      else
        op = AccessOperator.POINTER_MEMBER_ACCESS;

      return new AccessExpression(lhs, op, rhs);
    }
  }

  protected Expression action_postfix_expression(TreeNode root)
  {
    Object o = doAction(root.getChild(0));

    if (o == null)
    {
      String s = root.getChild(0).getText();

      if (s.endsWith("_cast"))
      {
        List specs = (List)doAction(root.getChild(2));
        Expression expr = (Expression)doAction(root.getChild(5));

        if (s.startsWith("const"))
          return new Typecast(Typecast.CONST, specs, expr);
        else if (s.startsWith("dynamic"))
          return new Typecast(Typecast.DYNAMIC, specs, expr);
        else if (s.startsWith("reinterpret"))
          return new Typecast(Typecast.REINTERPRET, specs, expr);
        else
          return new Typecast(Typecast.STATIC, specs, expr);
      }
    }
    else if (o instanceof Expression)
    {
      Expression expr = (Expression)o;
    
      if (root.getChildCount() == 1)
        return expr;
      else
      {
        String s = root.getChild(1).getText();

        if (s.equals("."))
          return new AccessExpression(expr, AccessOperator.MEMBER_ACCESS,
            (Expression)doAction(root.getChild(2)));
        else if (s.equals("->"))
          return new AccessExpression(expr, AccessOperator.POINTER_ACCESS,
            (Expression)doAction(root.getChild(2)));
        else if (s.equals("["))
        {
          Expression index = (Expression)doAction(root.getChild(2));

          if (expr instanceof ArrayAccess)
          {
            ((ArrayAccess)expr).addIndex(index);
            return expr;
          }
          else
            return new ArrayAccess(expr, index);
        } 
        else if (s.equals("++"))
          return new UnaryExpression(UnaryOperator.POST_INCREMENT, expr);
        else if (s.equals("--"))
          return new UnaryExpression(UnaryOperator.POST_DECREMENT, expr);
        else if (s.equals("(") && root.getChild(0).getText().equals("postfix_expression"))
        {
          if (root.getChildCount() == 4)
            return new FunctionCall(expr, (List)doAction(root.getChild(2)));
          else
            return new FunctionCall(expr);
        }
      }
    }
    else if (o instanceof Specifier)
    {
      if (root.getChildCount() == 3)
        return new Typecast(Typecast.NORMAL, (Specifier)o, new LinkedList());
      else
        return new Typecast(Typecast.NORMAL, (Specifier)o, (List)doAction(root.getChild(2)));
    }

    root.printTree(0);
    throw new RuntimeException("unknown postfix expression");
  }

  protected Expression action_primary_expression(TreeNode root)
  {
    String s = root.getChild(0).getText();

    if (s.equals("this"))
      return SymbolTools.getOrphanID("this");

    if (root.getChildCount() == 1)
      return (Expression)doAction(root.getChild(0));

    if (s.equals("("))
      return (Expression)doAction(root.getChild(1));
    else if (s.equals("::"))
    {
      IDExpression expr = (IDExpression)doAction(root.getChild(1));
      expr.setGlobal(true);
      return expr;
    }

    throw new RuntimeException("bad primary expression");
  }

  protected Specifier action_ptr_operator(TreeNode root)
  {
    String s = root.getChild(0).getText();

    if (s.equals("*"))
    {
      PointerSpecifier ps = PointerSpecifier.UNQUALIFIED;

      if (root.getChildCount() == 1)
        return ps;

      List tq_list = (List)doAction(root.getChild(1));
      int n = 0;

      if (tq_list.contains(Specifier.CONST))
        n = n + 1;
      if (tq_list.contains(Specifier.VOLATILE))
        n = n + 2;

      switch (n)
      {
        case 1:
          ps = PointerSpecifier.CONST;
          break;

        case 2:
          ps = PointerSpecifier.VOLATILE;
          break;

        case 3:
          ps = PointerSpecifier.CONST_VOLATILE;
          break;

        default:
          ps = PointerSpecifier.UNQUALIFIED;
          break;
      }

      return ps;
    }
    else if (s.equals("&"))
    {
      return Specifier.REFERENCE;
    }
    else
      throw new RuntimeException("not yet implemented");
  }

  protected QualifiedID action_scoped_class_name(TreeNode root)
  {
    switch (root.getChildCount())
    {
      case 1:
        LinkedList list = new LinkedList();
        list.add(doAction(root.getChild(0)));
        return new QualifiedID(list);

      case 3:
        QualifiedID qid = (QualifiedID)doAction(root.getChild(2));
        qid.getIDExpressionList().add(0, doAction(root.getChild(0)));
        return qid;

      default:
        throw new RuntimeException("not yet implemented");
    }
  }

  protected IDExpression action_scoped_unqualified_id(TreeNode root)
  {
    int child_count = root.getChildCount();

    IDExpression class_name = (IDExpression)doAction(root.getChild(0));
    IDExpression second_id = (IDExpression)doAction(root.getChild(child_count - 1));

    if (second_id instanceof QualifiedID)
    {
      ((QualifiedID)second_id).getIDExpressionList().add(0, class_name);
      return second_id;
    }
    else
    {
      List list = new LinkedList();
      list.add(class_name);
      list.add(second_id);
      return new QualifiedID(list);
    }
  }

  protected Declaration action_simple_declaration(TreeNode root)
  {
    int child_count = root.getChildCount();

    switch (child_count)
    {
      case 1:
        return new VariableDeclaration(new LinkedList(), new LinkedList());
      case 2:
        List list = (List)doAction(root.getChild(0));
        if (list.get(0) instanceof Specifier)
          return new VariableDeclaration(list, new LinkedList());
        else
          return new VariableDeclaration(new LinkedList(), list);
      case 3:
        return new VariableDeclaration((List)doAction(root.getChild(0)),
          (List)doAction(root.getChild(1)));
      default:
        throw new RuntimeException("bad simple declaration");
    }
  }

  protected Specifier action_simple_type_specifier(TreeNode root)
  {
    Object o = Specifier.fromString(root.getChild(0).getText());

    if (o != null)
      return (Specifier)o;

    o = doAction(root.getChild(0));

    if (o != null)
      return new UserSpecifier((IDExpression)o);

    if (root.getChild(1).getText().equals("("))
    {
      o = doAction(root.getChild(2));

      if (o instanceof Expression)
        return new TypeofSpecifier((Expression)o);
      else
        return new TypeofSpecifier((List)o);
    }
    else
    {
      IDExpression expr = (IDExpression)doAction(root.getChild(1));

      if (expr == null)
        root.printTree(0);

      expr.setGlobal(true);
      return new UserSpecifier(expr);
    }
  }

  protected Object action_template_argument(TreeNode root)
  {
    Object o = doAction(root.getChild(0));

    if (o instanceof List)
      o = new VariableDeclaration((List)o, new LinkedList());

    return o;
  }

  protected Object action_template_declaration(TreeNode root)
  {
    int base = 0;

    if (root.getChildCount() == 6)
      base = 1;

    List params = (List)doAction(root.getChild(base + 2));
    Declaration decl = (Declaration)doAction(root.getChild(base + 4));

    return new TemplateDeclaration(params, decl);
  }

  protected Object action_template_parameter(TreeNode root)
  {
    return doAction(root.getChild(0));
  }

  protected List action_type_id(TreeNode root)
  {
    List list = (List)doAction(root.getChild(0));

    if (root.getChildCount() == 2)
      list.addAll((List)doAction(root.getChild(1)));

    return list;
  }

  protected Object action_type_parameter(TreeNode root)
  {
    int child_count = root.getChildCount();
    String s = root.getChild(0).getText();

    /* FIXME */
    return new VariableDeclaration(new VariableDeclarator(SymbolTools.getOrphanID("FIXME")));
/*
    if (s.equals("class") || s.equals("typename"))
    {
      if (child_count == 2 || child_count == 4)
        s += " " + root.getChild(1).getText();

      VariableDeclarator decl = new VariableDeclarator(SymbolTools.getOrphanID(s));

      if (child_count == 3 || child_count == 4)
        decl.setInitializer(new TypeIdInitializer((List)doAction(root.getChild(child_count - 1))));

      return decl;
    }
    else // template
    {
    } */
  }

  protected Specifier action_type_specifier(TreeNode root)
  {
    int child_count = root.getChildCount();

    Specifier spec = (Specifier)doAction(root.getChild(0));

    if (spec != null)
      return spec;

    String s = root.getChild(0).getText();
    
    if (s.equals("enum"))
    {
      String name = null;
      if (root.getChild(1).getText().equals("{"))
        name = "anonymous" + Integer.toString(anonymous_count++);
      else
        name = root.getChild(1).getText();

      if (child_count > 2)
      {
        List decl_list = (List)doAction(root.getChild(child_count - 2));

        if (decl_list != null)
        {
          cetus.hir.Enumeration enum_decl = new cetus.hir.Enumeration(SymbolTools.getOrphanID(name), decl_list);
          saved_decl = enum_decl;
        }
      }

      return new UserSpecifier(SymbolTools.getOrphanID("enum " + name));
    }

    boolean forward_decl = !root.getChild(child_count - 1).getText().equals("}");

    IDExpression class_name = null;

    String name = root.getChild(1).getText();
    if (name.equals("{") || name.equals("base_clause"))
    {
      class_name = SymbolTools.getOrphanID("anonymous" + Integer.toString(anonymous_count++));
    }
    else
      class_name = (IDExpression)doAction(root.getChild(1));

    ClassDeclaration cd = null;
    if (s.equals("class"))
      cd = new ClassDeclaration(ClassDeclaration.CLASS, class_name, forward_decl);
    else if (s.equals("struct"))
      cd = new ClassDeclaration(ClassDeclaration.STRUCT, class_name, forward_decl); 
    else if (s.equals("union"))
      cd = new ClassDeclaration(ClassDeclaration.UNION, class_name, forward_decl);
    else if (s.equals("typename"))
    {
      IDExpression expr = (IDExpression)doAction(root.getChild(child_count - 1));
      expr.setTypename(true);
      if (child_count == 3)
        expr.setGlobal(true);
      return new UserSpecifier(expr);
    }
    else
    {
      root.printTree(0);
      throw new RuntimeException("unknown type specifier");
    }

    Object o = doAction(root.getChild(child_count - 2));
    if (o instanceof List)
    {
      for (Object d : (List)o)
        cd.addDeclaration((Declaration)d);
    }

    saved_decl = cd;
    return new UserSpecifier(SymbolTools.getOrphanID(cd.getKey() + " " + cd.getName()));
  }

  protected Expression action_unary_expression(TreeNode root)
  {
    if (root.getChildCount() == 1)
      return (Expression)doAction(root.getChild(0));
    else if (root.getChild(0).getText().equals("sizeof"))
    {
      if (root.getChildCount() == 2)
        return new SizeofExpression((Expression)doAction(root.getChild(1)));
      else
        return new SizeofExpression((List)doAction(root.getChild(2)));
    }
    else if (root.getChild(0).getText().equals("unary_operator"))
    {
      UnaryOperator op = UnaryOperator.fromString(root.getChild(0).getChild(0).getText());
      return new UnaryExpression(op, (Expression)doAction(root.getChild(1)));
    }

    throw new RuntimeException("unknown unary expression");
  }

  protected IDExpression action_unqualified_id(TreeNode root)
  {
    if (root.getChildCount() == 1)
      return (IDExpression)doAction(root.getChild(0));
    else
      return new DestructorID((IDExpression)doAction(root.getChild(1)));
  }

  protected Declaration action_using_declaration(TreeNode root)
  {
    int child_count = root.getChildCount();

    IDExpression expr = (IDExpression)doAction(root.getChild(child_count - 2));
    
    if (root.getChild(1).getText().equals("typename"))
    {
      expr.setTypename(true);

      if (root.getChild(2).getText().equals("::"))
        expr.setGlobal(true);
    }
    else if (root.getChild(1).getText().equals("::"))
      expr.setGlobal(true);

    return new UsingDeclaration(expr);
  }

  public Object doAction(TreeNode root)
  {
    int action_code = -1;

    try {
      action_code = action_map.get("action_" + root.getText());
    } catch (NullPointerException e) {
      return null;
    }

    switch (action_code)
    {
      case 4:
        return action_ambiguity(root);

      case 3:   // additive_expression
      case 5:   // and_expression
      case 50:  // equality_expression
      case 53:  // exclusive_or_expression
      case 67:  // inclusive_or_expression
      case 78:  // logical_and_expression
      case 79:  // logical_or_expression
      case 89:  // multiplicative_expression
      case 109: // relational_expression
      case 114: // shift_expression
        return defaultBinaryExpressionAction(root);

      case 34:  // cv_qualifier_seq
      case 43:  // declarator_tail_seq
      case 49:  // enumerator_list
      case 57:  // expression_list
      case 69:  // init_declarator_list
      case 72:  // initializer_list
      case 82:  // mem_initializer_list
      case 86:  // member_declarator_list
      case 102: // parameter_declaration_list
      case 118: // statement_seq
      case 121: // template_argument_list
      case 124: // template_parameter_list
      case 129: // type_id_list
      case 132: // type_specifier_seq
        return defaultListAction(root);

      case 27:  // constant_expression
      case 107: // qualified_id
        return doAction(root.getChild(0));

      case 1:
        return action_abstract_declarator(root);
      case 2:
        return action_access_specifier(root);
      case 11:
        return action_assignment_expression(root);
      case 20:
        return action_block_declaration(root);
      case 22:
        return action_cast_expression(root);
      case 23:
        return action_class_name(root);
      case 24:
        return action_compound_statement(root);
      case 25:
        return action_condition(root);
      case 26:
        return action_conditional_expression(root);
      case 28:
        return action_constant_initializer(root);
      case 30:
        return action_conversion_function_id(root);
      case 32: // ctor_initializer
        return doAction(root.getChild(1));
      case 33: // cv_qualifier
        return action_specifier(root);
      case 35:
        return action_decl_specifier(root);
      case 36:
        return action_declaration_specifiers(root);
      case 37:
        return action_declaration(root);
      case 38:
        return action_declaration_list(root);
      case 39:
        return action_declarator(root);
      case 40:
        return action_declarator_head(root);
      case 41:
        return action_declarator_id(root);
      case 42:
        return action_declarator_tail(root);
      case 45:
        return action_direct_abstract_declarator(root);
      case 46:
        return action_direct_declarator(root);
      case 48:
        return action_enumerator(root); // enumerator_definition
      case 55:
        return action_explicit_specialization(root);
      case 56:
        return action_expression(root);
      case 58:
        return action_expression_statement(root);
      case 59:
        return action_for_init_statement(root);
      case 60:
        return action_function_definition(root);
      case 61:
        return action_function_prefix(root);
      case 62:
        return action_specifier(root);
      case 66:
        return action_id_expression(root);
      case 68:
        return action_init_declarator(root);
      case 70:
        return action_initializer(root);
      case 71:
        return action_initializer_clause(root);
      case 73:
        return action_iteration_statement(root);
      case 74:
        return action_jump_statement(root);
      case 75:
        return action_labeled_statement(root);
      case 76:
        return action_linkage_specification(root);
      case 77:
        return action_literal(root);
      case 80:
        return action_mem_initializer(root);
      case 81:
        return action_mem_initializer_id(root);
      case 83:
        return action_member_declaration(root);
      case 85:
        return action_member_declarator(root);
      case 88:
        return action_member_specification(root);
      case 91:
        return action_namespace_definition(root);
      case 98:
        return action_operator_(root);
      case 99:
        return action_operator_function_id(root);
      case 100:
        return action_parameter_declaration(root);
      case 101:
        return action_parameter_declaration_clause(root);
      case 103:
        return action_pm_expression(root);
      case 104:
        return action_postfix_expression(root);
      case 105:
        return action_primary_expression(root);
      case 106:
        return action_ptr_operator(root);
      case 110:
        return action_scoped_class_name(root);
      case 112:
        return action_scoped_unqualified_id(root);
      case 113:
        return action_selection_statement(root);
      case 115:
        return action_simple_declaration(root);
      case 116:
        return action_simple_type_specifier(root);
      case 117:
        return action_statement(root);
      case 119:
        return action_specifier(root);
      case 120:
        return action_template_argument(root);
      case 122:
        return action_template_declaration(root);
      case 123:
        return action_template_parameter(root);
      case 126:
        return action_translation_unit(root);
      case 128:
        return action_type_id(root);
      case 130:
        return action_type_parameter(root);
      case 131:
        return action_type_specifier(root);
      case 133:
        return action_unary_expression(root);
      case 135:
        return action_unqualified_id(root);
      case 136:
        return action_using_declaration(root);

      default:
        root.printTree(0);
        throw new RuntimeException("action " + action_code + " not implemented");
    }
  }

  protected Declarator mergeDeclaratorHeadAndTail(Declarator head, List tail)
  {
    /* The head may be a VariableDeclarator or a NestedDeclarator.
       The list may contain ArraySpecifiers or a ProcedureDeclarator with a
       dummy name. */

    if (tail.get(0) instanceof ArraySpecifier)
    {
      for (Object o : tail)
        head.getArraySpecifiers().add((Specifier)o);

      return head;
    }

    if (tail.get(0) instanceof ProcedureDeclarator)
    {
      ProcedureDeclarator pdecl = (ProcedureDeclarator)tail.get(0);

      List params = new LinkedList();
      for (Declaration d : pdecl.getParameters())
        params.add(d.clone());

      return new ProcedureDeclarator(pdecl.getSpecifiers(),
        ((VariableDeclarator)head).getID().clone(),
        params);
    }

    throw new RuntimeException("cannot merge head and tail");
  }

  public TranslationUnit run(TreeNode root)
  {
//    root.printTree(0);
    return action_translation_unit(root);
  }
}
