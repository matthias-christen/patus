package cetus.hir;

import java.io.*;
import java.util.*;
import cetus.exec.*;
import cetus.analysis.*;

/**
* <b>SymbolTools</b> provides tools for interfacing symbol tables and related
* searches.
*/
public final class SymbolTools
{
  private SymbolTools() {}

  /** Counter for symbol updates */
  private static int num_updates;

  /**
   * Makes links from all {@link IDExpression} objects in the program to
   * their corresponding declarators while generating warnings if there is
   * any undeclared variables or functions. This method is called with the
   * whole program, before any Cetus passes by default and provides a short cut
   * to the declaration point, which enables faster access to the declaration
   * when necessary. Pass writers can call this routine after changing a certain
   * part of the program, e.g., a specific program section within a scope, to
   * reflect the change due to new insertion of declaration.
   *
   * @param tr the input cetus IR.
   */
  public static void linkSymbol(Traversable tr)
  {
    num_updates = 0;
    double timer = Tools.getTime();

    DepthFirstIterator iter = new DepthFirstIterator(tr);

    while ( iter.hasNext() )
    {
      Object o = iter.next();

      // Add enumerations in the table.
      if ( o instanceof Enumeration )
        putEnumeration((Enumeration)o);

      if ( o instanceof Identifier )
        searchAndLink((Identifier)o);
    }

    timer = Tools.getTime(timer);

    PrintTools.printStatus("[LinkSymbol] Updated " + num_updates + " links in "
        + String.format("%.2f seconds\n",timer), 1);
  }

  private static void setSymbol(Identifier id, Symbol symbol)
  {
    id.setSymbol(symbol);
    num_updates++;
  }

  private static void searchAndLink(Identifier id)
  {
    String id_name = id.getName();

    // These cases are skipped intentionally.
    if ( id.getParent() instanceof Declarator || // it is a Symbol object.
    id.getParent() instanceof GotoStatement ||   // not a variable.
    id.getParent() instanceof Label ||           // not a variable.
    id_name.equals("") ||                        // void Symbol.
    id_name.equals("__PRETTY_FUNCTION__") ||     // gcc keyword.
    id_name.equals("__FUNCTION__") ||            // gcc keyword.
    id_name.startsWith("__builtin") )            // gcc keyword.
      return;

    Declaration decls = searchDeclaration(id);

    if ( decls == null )
      return;

    // Lookup the symbol object in the declaration and add links.
    if ( decls instanceof Procedure )
      setSymbol(id, (Symbol)decls);
    else if ( decls instanceof VariableDeclaration ||
        decls instanceof Enumeration ) {
      Declarator decl = null;
      for ( Object child : decls.getChildren() )
        if ( id.equals( ((Declarator)child).getID() ) )
        {
          decl = (Declarator)child;
          setSymbol(id, (Symbol)decl);
          break;
        }
    }
  }

  // Put each enumerated members in the table.
  private static void putEnumeration(Enumeration en)
  {
    Traversable t = en.getParent();
    SymbolTable st = IRTools.getAncestorOfType(en, SymbolTable.class);
    addSymbols(st, en);
  }

  // Returns the type of an expression
  private static List getType(Traversable e)
  {
    if ( !(e instanceof Expression) )
      return null;

    if ( e instanceof Identifier )
    {
      Symbol var = ((Identifier)e).getSymbol();
      return (var==null)? null: var.getTypeSpecifiers();
    }

    else if ( e instanceof AccessExpression )
      return getType(((AccessExpression)e).getRHS());

    else if ( e instanceof ArrayAccess )
      return getType(((ArrayAccess)e).getArrayName());

    else if ( e instanceof ConditionalExpression )
    {
      ConditionalExpression ce = (ConditionalExpression)e;
      List specs = getType(ce.getTrueExpression());
      if ( specs == null || specs.get(0) == Specifier.VOID )
        return getType(ce.getFalseExpression());
      else
        return specs;
    }

    else if ( e instanceof FunctionCall )
      return getType(((FunctionCall)e).getName());

    else if ( e instanceof Typecast )
      return new LinkedList(((Typecast)e).getSpecifiers());

    else if ( e instanceof CommaExpression )
    {
      List children = e.getChildren();
      return getType( (Expression)children.get(children.size()-1) );
    }

    else
    {
      for ( Object child : e.getChildren() )
      {
        List child_type = getType((Expression)child);
        if ( child_type != null )
          return child_type;
      }
      return null;
    }
  }

  // findDeclaration with an arbitrary starting point and a target id.
  private static Declaration findDeclaration(Traversable tr, IDExpression id)
  {
    Traversable t = tr;
    while ( t != null && !(t instanceof SymbolTable) )
      t = t.getParent();
    return ((t==null)? null: findSymbol((SymbolTable)t, id));
  }

  // Serach for declaration of the identifier
  private static Declaration searchDeclaration(Identifier id)
  {
    Declaration ret = null;
    Traversable parent = id.getParent();

    // Broken IR
    if ( parent == null )
      return null;

    // AccessExpression handling.
    if ( parent instanceof AccessExpression &&
    ((AccessExpression)parent).getRHS() == id )
    {
      List specs = getType(((AccessExpression)parent).getLHS());
      Declaration cdecl = findUserDeclaration(id, specs);
      if ( cdecl instanceof ClassDeclaration )
        ret = ((ClassDeclaration)cdecl).findSymbol(id);
    }
    // __builtin__offsetof handling.
    else if ( parent instanceof OffsetofExpression &&
    ((OffsetofExpression)parent).getExpression() == id )
    {
      List specs = ((OffsetofExpression)parent).getSpecifiers();
      Declaration cdecl = findUserDeclaration(id, specs);
      if ( cdecl instanceof ClassDeclaration )
        ret = ((ClassDeclaration)cdecl).findSymbol(id);
    }
    else
    {
      ret = id.findDeclaration();
      // This code section only deals with a situation that name conflicts
      // in a scope; e.g.) { int a = b; float b; ... }
      if ( ret instanceof VariableDeclaration )
      {
        Traversable t1 = IRTools.getAncestorOfType(id, Statement.class);
        Traversable t2 = IRTools.getAncestorOfType(ret, Statement.class);
        if ( t1 != null && t2 != null && t1.getParent() == t2.getParent() )
        {
          List children = t1.getParent().getChildren();
          if ( children.indexOf(t1) < children.indexOf(t2) )
            ret = findDeclaration(t1.getParent().getParent(), id);
        }
      }
    }

    // Prints out warning for undeclared functions/symbols.
    if ( ret == null )
    {
      if ( parent instanceof FunctionCall &&
        ((FunctionCall)parent).getName() == id )
        System.err.print("[WARNING] Function without declaration ");
      else
        System.err.print("[WARNING] Undeclared symbol ");

      System.err.println(id+" from "+parent);
    }
    return ret;
  }

  // Find the body of user-defined class declaration
  private static Declaration findUserDeclaration(Traversable tr, List specs)
  {
    if ( specs == null )
      return null;

    // Find the leading user specifier
    UserSpecifier uspec = null;
    for ( Object o: specs )
    {
      if ( o instanceof UserSpecifier )
      {
        uspec = (UserSpecifier)o;
        break;
      }
    }

    if ( uspec == null )
      return null;

    // Find declaration for the user specifier
    Declaration ret = findDeclaration(tr, uspec.getIDExpression());

    // Handles identifier that shares the same name with its type;
    // e.g. typedef struct {} foo; foo foo;
    if ( ret instanceof VariableDeclaration &&
    ((VariableDeclaration)ret).getSpecifiers() == specs )
    {
      Traversable t =
          (Traversable)IRTools.getAncestorOfType(ret, SymbolTable.class);
      ret = findDeclaration(t.getParent(), uspec.getIDExpression());
    }
    
    // Keep searching through the chain ( i.e. typedef, etc )
    if ( ret instanceof VariableDeclaration )
      return findUserDeclaration(tr,((VariableDeclaration)ret).getSpecifiers());

    // Differentiate prototype and actual declaration (forward declaration)
    if ( ret instanceof ClassDeclaration &&
      ((SymbolTable)ret).getDeclarations().isEmpty() )
    {
      IDExpression class_name = ((ClassDeclaration)ret).getName();
      Traversable t = ret.getParent();
      while ( t != null )
      {
        if ( t instanceof SymbolTable )
          for ( Object child : t.getChildren() )
            if ( child instanceof ClassDeclaration &&
            ((ClassDeclaration)child).getName().equals(class_name) &&
            !((SymbolTable)child).getDeclarations().isEmpty() )
            {
              ret = (Declaration)child;
              break;
            }
        t = t.getParent();
      }
    }
    return ret;
  }

/**
 * Checks if the given symbol is declared as a global variable.
 * @param symbol the symbol of the variable.
 * @return true if the variable is global.
 */
  public static boolean isGlobal(Symbol symbol)
  {
    if (symbol instanceof PseudoSymbol)
      symbol = ((PseudoSymbol)symbol).getIRSymbol();
    Traversable tr = (Traversable)symbol;
    while ( tr != null && !(tr instanceof SymbolTable) )
      tr = tr.getParent();
    return (tr instanceof TranslationUnit);
  }

/**
 * Checks if the given symbol is declared as a local variable.
 * This checking should be made along with other types of checking since
 * the formal parameters are not part of the IR tree.
 * @param symbol the symbol of the variable.
 * @return true if the variable is local.
 */
  public static boolean isLocal(Symbol symbol)
  {
    if (symbol instanceof PseudoSymbol)
      symbol = ((PseudoSymbol)symbol).getIRSymbol();
    Traversable tr = (Traversable)symbol;
    while ( tr != null && !(tr instanceof SymbolTable) )
      tr = tr.getParent();
    return (tr != null && !(tr instanceof TranslationUnit));
  }

/**
 * Checks if the given symbol does not belong to the IR tree.
 * There can be two types of orphan symbol -- 1) formal parameters,
 * 2) other temporary variables not in the tree.
 * @param symbol the symbol of the variable.
 * @return true if the variable is orphan.
 */
  public static boolean isOrphan(Symbol symbol)
  {
    if (symbol instanceof PseudoSymbol)
      symbol = ((PseudoSymbol)symbol).getIRSymbol();
    Traversable tr = (Traversable)symbol;
    while ( tr != null && !(tr instanceof SymbolTable) )
      tr = tr.getParent();
    return (tr == null);
  }

/**
 * Checks if the given symbol is declared as a formal variable.
 * This utility function is using a weak point of the current IR hierarchy;
 * formal parameters are not part of the IR tree but just a satellite data of
 * a procedure object.
 * @param symbol the symbol of the variable.
 * @return true if the variable is formal.
 */
  public static boolean isFormal(Symbol symbol, Procedure proc)
  {
    if (symbol instanceof PseudoSymbol)
      symbol = ((PseudoSymbol)symbol).getIRSymbol();
    Traversable t = (Traversable)symbol;
    while ( !(t instanceof ProcedureDeclarator || t instanceof SymbolTable) )
      t = t.getParent();
    return (t == proc.getDeclarator() && (t != symbol));
  }

/**
 * Checks if the given symbol is declared as a formal variable.
 * @param symbol the symbol to be checked.
 * @return true if the symbol represents a formal parameter, false otherwise.
 */
  public static boolean isFormal(Symbol symbol)
  {
    if (symbol instanceof PseudoSymbol)
      symbol = ((PseudoSymbol)symbol).getIRSymbol();
    Traversable t = (Traversable)symbol;
    while ( !(t instanceof ProcedureDeclarator || t instanceof SymbolTable) )
      t = t.getParent();
    return ( (t instanceof ProcedureDeclarator) && (t != symbol) );
  }

/**
 * Returns a list of symbols by taking symbol object out of the expressions
 * in the given list.
 */
  public static List<Symbol> exprsToSymbols(List<Expression> exprs)
  {
    List<Symbol> ret = new LinkedList<Symbol>();
    for ( Expression e : exprs )
      ret.add(getSymbolOf(e));
    return ret;
  }

  /**
  * Checks if the given symbol is a parameter and a pointer-compatible symbol.
  * Any array declarator is a pointer-compatible symbol.
  */
  public static boolean isPointerParameter(Symbol param)
  {
    return (isFormal(param) && (isPointer(param)||isArray(param)));
  }

/**
 * Returns C-native specifiers for the given symbol.
 * TODO: under development - remove from distribution
 */
  public static List<Specifier>
    getNativeSpecifiers(Traversable t, Symbol symbol)
  {
    List<Specifier> ret = new LinkedList<Specifier>();

    List types = symbol.getTypeSpecifiers();

    System.out.println(types);
    for ( Object o : types )
    {
      if ( o instanceof UserSpecifier )
      {
        SymbolTable symtab = IRTools.getAncestorOfType(t, SymbolTable.class);
        Declaration decln =
          findSymbol(symtab, ((UserSpecifier)o).getIDExpression());
        while ( decln instanceof VariableDeclaration &&
          ((VariableDeclaration)decln).isTypedef() )
        {
        }
        System.out.println(o+" => "+decln);
      }
      else if ( o instanceof Specifier )
        ret.add((Specifier)o);
      else
        throw new InternalError();
      //System.out.println(o+"=>"+o.getClass().getName());
    }
    return null;
  }

/**
 * Checks if the given symbol is a user-defined struct type. Notice that the
 * first parameter symbol should exist as a traversable object.
 * @param symbol the symbol object to be checked.
 * @param tr the traversable object to be searched.
 */
  public static boolean isStruct(Symbol symbol, Traversable tr)
  {
    return (getClassDeclaration(symbol, tr) != null);
  }

/**
 * Returns the user-defined class declaration for the given symbol. Notice that
 * it does not deferentiate "a" and "*a". 
 * @param symbol the symbol object to be checked.
 * @param tr the traversable object to be searched.
 */
  public static ClassDeclaration
      getClassDeclaration(Symbol symbol, Traversable tr)
  {
    Symbol sym = symbol;
    if (sym instanceof PseudoSymbol)
      sym = ((PseudoSymbol)sym).getIRSymbol();
    Declaration ret = findUserDeclaration(tr, sym.getTypeSpecifiers());
    if ( ret instanceof ClassDeclaration )
      return (ClassDeclaration)ret;
    else
      return null;
  }

  /**
  * Returns an incomplete Identifier whose relevant symbol is not defined.
  * This method is equivalent to the constructor of Identifier with a raw
  * string name, which is now hidden to external world.
  */
  public static Identifier getOrphanID(String name)
  {
    return new Identifier(name);
  }

  /**
   * Adds symbols to a symbol table and checks for duplicates.
   *
   * @param table The symbol table to add the symbols to.
   * @param decl The declaration of the symbols.
   * @throws DuplicateSymbolException if there are conflicts with
   *   any existing symbols in the table.
   */
  public static void addSymbols(SymbolTable table, Declaration decl)
  {
    Map<IDExpression, Declaration> symbol_table = getTable(table);

    for (IDExpression id : (List<IDExpression>)decl.getDeclaredIDs()) {
      // Skips dummy identifiers.
      if (id.toString().length()==0)
        continue;
      // Overwriting is not allowed except for the case where Procedure
      // declaration overwrites Procedure declarator. This is so because we
      // assign Procedure as the symbol of the relevant Identifier not the
      // procedure declarator. ProcedureDeclarator is used only if there is no
      // procedure body (library call).
      if (symbol_table.containsKey(id) && !(decl instanceof Procedure))
        continue;
      // Assigns a new identifier if the id represents a variable's name that
      // has a matching declaration/declarator. This table-populating process
      // results in a map from IDExpression objects (NameID/Identifier) to
      // matching declarations. NameID is used for non-variable type IDs such
      // as user-defined type names which cannot have a reference to the Symbol
      // object.
      if (decl instanceof Procedure) {
        symbol_table.put(new Identifier((Symbol)decl), decl);
      } else {
        symbol_table.put(id, decl);
        DepthFirstIterator symbol_iter = new DepthFirstIterator(decl);
        symbol_iter.pruneOn(NestedDeclarator.class);
        while (symbol_iter.hasNext()) {
          Object o = symbol_iter.next();
          if (o instanceof Symbol) {
            Symbol symbol = (Symbol)o;
            if (symbol.getSymbolName().equals(id.toString())) {
              symbol_table.remove(id); // key is not overwritten by put().
              symbol_table.put(new Identifier(symbol), decl);
              break;
            }
          }
        }
      }
    }
  }

  /**
   * Searches for a symbol by name in the table.  If the symbol is
   * not in the table, then parent tables will be searched.
   *
   * @param table The initial table to search.
   * @param name The name of the symbol to locate.
   *
   * @return a Declaration if the symbol is found, or null if it is not found.
   *    The Declaration may contain multiple declarators, of which name will
   *    be one, unless the SingleDeclarator pass has been run on the program.
   */
  public static Declaration findSymbol(SymbolTable table, IDExpression name)
  {
    List<SymbolTable> symtabs = new LinkedList<SymbolTable>();
    symtabs.add(table);
    symtabs.addAll(getParentTables(table));
    for (SymbolTable symtab : symtabs) {
      Declaration ret = getTable(symtab).get(name);
      if (ret != null)
        return ret;
    }
    return null;
  }

  /**
  * Returns a list of parent symbol tables.
  */
  protected static List<SymbolTable> getParentTables(Traversable obj)
  {
    LinkedList<SymbolTable> list = new LinkedList<SymbolTable>();
    Traversable p = obj.getParent();
    while (p != null) {
      if (p instanceof SymbolTable)
        list.add((SymbolTable)p);
      p = p.getParent();
    }
    return list;
  }

  /**
   * Returns a randomly-generated name that is not found in the table.
   *
   * @param table The table to search.
   *
   * @return a unique name.
   */
  public static IDExpression getUnusedID(SymbolTable table)
  {
    String name = null;
    IDExpression ident = null;
    Random rand = new Random();

    do {
      name = "";
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);
      ident = new NameID(name);
    } while (findSymbol(table, ident) == null);

    return ident;
  }

  /**
   * Removes the symbols declared by the declaration from the symbol
   * table.
   *
   * @param table The table from which to remove the symbols.
   * @param decl The declaration of the symbols.
   */
  protected static void removeSymbols(SymbolTable table, Declaration decl)
  {
    List names = decl.getDeclaredIDs();
    Iterator iter = names.iterator();

    Map<IDExpression, Declaration> symbol_table = getTable(table);

    while (iter.hasNext())
    {
      IDExpression symbol = (IDExpression)iter.next();

      if (symbol_table.remove(symbol) == null)
      {
        System.err.println("SymbolTools.removeSymbols could not remove entry for " + symbol.toString());
        System.err.println("table contains only " + symbol_table.toString());
      }
    }
  }

  /**
  * Removes the specified symbol from the symbol table object. This methods
  * consistently modifies the look-up table, and the IR. No action occurs if
  * the specified symbol is not found in the symbol table object.
  *
  * @param table the relevant symbol table object.
  * @param symbol the symbol object to be removed.
  * @throws IllegalArgumentException if <b>symbol</b> is referenced within
  * <b>table</b>.
  */
  protected static void removeSymbol(SymbolTable table, Symbol symbol)
  {
/* TODO: will finish later - remove from release branch
    Set<Symbol> symbols = table.getSymbols();
    if (!symbols.contains(symbol))
      return;
    Set<Symbol> accessed_symbols = getAccessedSymbols(table);
    if (accessed_symbols.contains(symbol))
      throw new IllegalArgumentException("Cannot remove a referenced symbol.");
    if (symbol instanceof Declaration) {
      table.removeDeclaration((Declaration)symbol);
    } else if
*/
  }

  /**
   * Returns a new identifier derived from the given identifier.
   * This method internally calls
   * {@link #getTemp(Identifier, String)}.
   *
   * @param id the identifier from which type and scope are derived.
   * @return the new identifier.
   */
  public static Identifier getTemp(Identifier id)
  {
    return getTemp(id, id.getName());
  }

  /**
   * Returns a new identifier derived from the given IR object and identifier.
   * This method internally calls
   * {@link #getTemp(Traversable, List, String)}.
   *
   * @param where the IR object from which scope is derived.
   * @param id the identifier from which type is derived.
   * @return the new identifier.
   */
  public static Identifier getTemp(Traversable where, Identifier id)
  {
    return getTemp(where, id.getSymbol().getTypeSpecifiers(), id.getName());
  }

  /**
   * Returns a new identifier derived from the given identifier and name.
   * This method internally calls
   * {@link #getTemp(Traversable, List, String)}.
   *
   * @param id the identifier from which scope is derived.
   * @param name the string from which name is derived.
   * @return the new identifier.
   */
  public static Identifier getTemp(Identifier id, String name)
  {
    return getTemp(id, id.getSymbol().getTypeSpecifiers(), name);
  }

  /**
   * Returns a new identifier derived from the given IR object, type, and name.
   * This method internally calls
   * {@link #getTemp(Traversable, List, String)}.
   *
   * @param where the IR object from which scope is derived.
   * @param spec the type specifier.
   * @param name the string from which name is derived.
   * @return the new identifier.
   */
  public static Identifier getTemp(Traversable where, Specifier spec, String name)
  {
    List specs = new LinkedList();
    specs.add(spec);
    return getTemp(where, specs, name);
  }

  /**
   * Returns a new identifier derived from the given IR object, type list, and
   * name.
   * This method internally calls
   * {@link #getArrayTemp(Traversable, List, List, String)}.
   *
   * @param where the IR object from which scope is derived.
   * @param specs the type specifiers.
   * @param name the string from which name is derived.
   * @return the new identifier.
   */
  public static Identifier getTemp(Traversable where, List specs, String name)
  {
    return getArrayTemp(where, specs, (List)null, name);
  }

  /**
   * Returns a new identifier derived from the given IR object, type list,
   * array specifier and name.
   * This method internally calls
   * {@link #getArrayTemp(Traversable, List, List, String)}.
   *
   * @param where the IR object from which scope is derived.
   * @param specs the type specifiers.
   * @param aspec the array specifier.
   * @param name the string from which name is derived.
   * @return the new identifier.
   */
  public static Identifier getArrayTemp
  (Traversable where, List specs, ArraySpecifier aspec, String name)
  {
    List aspecs = new LinkedList();
    aspecs.add(aspec);
    return getArrayTemp(where, specs, aspecs, name);
  }

  /**
   * Returns a new identifier derived from the given IR object, type list,
   * array specifiers and name. If {@code specs} contains any pointer
   * specifiers, they are automatically separated and inserted into the list
   * of specifiers that belong to the new {@code VariableDeclarator} object.
   *
   * @param where the IR object from which scope is derived.
   * @param specs the type specifiers.
   * @param aspecs the array specifier.
   * @param name the string from which name is derived.
   * @return the new identifier.
   */
  public static Identifier getArrayTemp
  (Traversable where, List specs, List aspecs, String name)
  {
    Traversable t = where;
    while ( !(t instanceof SymbolTable) )
      t = t.getParent();
    // Traverse to the parent of a loop statement
    if (t instanceof ForLoop || t instanceof DoLoop || t instanceof WhileLoop) {
      t = t.getParent();
      while ( !(t instanceof SymbolTable) )
        t = t.getParent();
    }
    SymbolTable st = (SymbolTable)t;
    
    String header = (name==null)? "_temp_": name+"_";
    Identifier ret = null;
    for ( int trailer=0; ret==null; ++trailer ) {
      Identifier newid = new Identifier(header+trailer);
      if ( findSymbol(st,newid) == null )
        ret = newid;
    }

    // Separate declarator/declaration specifiers.
    List declaration_specs = new LinkedList();
    List declarator_specs = new LinkedList();
    for ( Object spec : specs )
      if ( spec.toString().contains("*") )
        declarator_specs.add(spec);
      else
        declaration_specs.add(spec);

    Declarator decl = null;
    if ( declarator_specs.isEmpty() )
    {
      if ( aspecs == null || aspecs.isEmpty() )
        decl = new VariableDeclarator(ret);
      else
        decl = new VariableDeclarator(ret, aspecs);
    }
    else
    {
      if ( aspecs == null || aspecs.isEmpty() )
        decl = new VariableDeclarator(declarator_specs, ret);
      else
        decl = new VariableDeclarator(declarator_specs, ret, aspecs);
    }
    Declaration decls = new VariableDeclaration(declaration_specs, decl);
    st.addDeclaration(decls);
    ret.setSymbol((Symbol)decl);

    return ret;
  }
  
  /**
   * Returns a new, pointer-type identifier derived from the given IR object.
   *
   * @param where the IR object from which scope is derived.
   * @param refID the identifier from which type and name are derived.
   * @return the new pointer-type identifier.
   */
  public static Identifier getPointerTemp
  (Traversable where, Identifier refID)
  {
    List pspecs = new LinkedList();
    pspecs.add(PointerSpecifier.UNQUALIFIED);
    return getPointerTemp(where, refID.getSymbol().getTypeSpecifiers(), 
        pspecs, refID.getName());
  }
  
  /**
   * Returns a new, pointer-type identifier derived from the given IR object.
   *
   * @param where the IR object from which scope is derived.
   * @param specs the type specifiers.
   * @param name the string from which name is derived.
   * @return the new pointer-type identifier.
   */
  public static Identifier getPointerTemp
  (Traversable where, List specs, String name)
  {
    List pspecs = new LinkedList();
    pspecs.add(PointerSpecifier.UNQUALIFIED);
    return getPointerTemp(where, specs, pspecs, name);
  }

  /**
   * Returns a new, pointer-type identifier derived from the given IR object.
   *
   * @param where the IR object from which scope is derived.
   * @param specs the type specifiers.
   * @param pspecs the pointer-type specifiers.
   * @param name the string from which name is derived.
   * @return the new pointer-type identifier.
   */
  public static Identifier getPointerTemp
  (Traversable where, List specs, List pspecs, String name)
  {
    Traversable t = where;
    while ( !(t instanceof SymbolTable) )
      t = t.getParent();
    // Traverse to the parent of a loop statement
    if (t instanceof ForLoop || t instanceof DoLoop || t instanceof WhileLoop) {
      t = t.getParent();
      while ( !(t instanceof SymbolTable) )
        t = t.getParent();
    }
    SymbolTable st = (SymbolTable)t;
    
    String header = (name==null)? "_temp_": name+"_";
    Identifier ret = null;
    for ( int trailer=0; ret==null; ++trailer ) {
      Identifier newid = new Identifier(header+trailer);
      if ( findSymbol(st,newid) == null )
        ret = newid;
    }

    Declarator decl =  new VariableDeclarator(pspecs, ret);
    Declaration decls = new VariableDeclaration(specs, decl);
    st.addDeclaration(decls);
  ret.setSymbol((Symbol)decl);

    return ret;
  }

  /**
   * Returns the set of Symbol objects contained in the given SymbolTable
   * object.
   *
   * @param st the symbol table being searched.
   * @return the set of symbols.
   */
  public static Set<Symbol> getSymbols(SymbolTable st)
  {
    Set ret = new LinkedHashSet<Symbol>();
    if (st == null)
      return ret;
    for (IDExpression key : getTable(st).keySet())
      if (key instanceof Identifier && ((Identifier)key).getSymbol() != null)
        ret.add(((Identifier)key).getSymbol());
    return ret;
  }

  /**
   * Returns the set of Symbol objects contained in the given SymbolTable
   * object excluding Procedures.
   *
   * @param st the symbol table being searched.
   * @return the set of symbols.
   */
  public static Set<Symbol> getVariableSymbols(SymbolTable st)
  {
    Set<Symbol> ret = getSymbols(st);
    Iterator<Symbol> iter = ret.iterator();
    while (iter.hasNext()) {
      Symbol symbol = iter.next();
      if (symbol instanceof Procedure ||
          symbol instanceof ProcedureDeclarator)
        iter.remove();
    }
    return ret;
  }

  /**
    * Returns the set of Symbol objects that are global variables 
    * of the File scope 
    */
  public static Set<Symbol> getGlobalSymbols(Traversable t)
  {
    while (true)
    {
      if (t instanceof TranslationUnit) break;
      t = t.getParent(); 
    }
    TranslationUnit t_unit = (TranslationUnit)t;
    return getVariableSymbols(t_unit);
  }

  /**
    * Returns the set of Symbol objects that are formal parameters of 
    * the given Procedure
    */
  public static Set<Symbol> getParameterSymbols(Procedure proc)
  {
    Set<Symbol> ret = new LinkedHashSet<Symbol>();
    DepthFirstIterator iter = new DepthFirstIterator(proc.getDeclarator());
    iter.pruneOn(NestedDeclarator.class);
    iter.next(); // skip procedure declarator itself.
    while (iter.hasNext()) {
      Object child = iter.next();
      if (child instanceof Symbol)
        ret.add((Symbol)child);
    }
    return ret;
  }

  public static Set<Symbol> getSideEffectSymbols(FunctionCall fc)
  {
    Set<Symbol> side_effect_set = new HashSet<Symbol>();
  
    // set of GlobalVariable Symbols that are accessed within a Procedure
    Procedure proc = fc.getProcedure();

    // we assume that there is no global variable access within a procedure
    // if a procedure body is not available for a compiler
    // example: system calls
    if (proc != null)
    {
      Set<Symbol> global_variables = new HashSet<Symbol>();
      Set<Symbol> accessed_symbols = getAccessedSymbols(proc.getBody());
      for (Symbol var : accessed_symbols)
      {
        if ( isGlobal(var, proc) )
        {
          global_variables.add(var);
        }
      }

      if ( !global_variables.isEmpty() )
      {
        side_effect_set.addAll(global_variables);
      }
    }
      
    // find the set of actual parameter Symbols of each function call
    List<Expression> arguments = fc.getArguments();
    HashSet<Symbol> parameters = new HashSet<Symbol>();
    for(Expression e : arguments)
      parameters.addAll(getAccessedSymbols(e));
    
    if ( !parameters.isEmpty() )
    {
      side_effect_set.addAll(parameters);  
    }

    return side_effect_set;
  }

  /**
   * Returns the set of symbols accessed in the traversable object.
   *
   * @param t the traversable object.
   * @return the set of symbols.
   */
  public static Set<Symbol> getAccessedSymbols(Traversable t)
  {
    Set<Symbol> ret = new HashSet<Symbol>();

    if ( t == null )
      return ret;

    DepthFirstIterator iter = new DepthFirstIterator(t);

    while ( iter.hasNext() )
    {
      Object o = iter.next();
      if ( !(o instanceof Identifier) )
        continue;
      Symbol symbol = ((Identifier)o).getSymbol();
      if ( symbol != null )
        ret.add(symbol);
    }

    return ret;
  }

  /**
   * Returns the set of symbols declared within the specified traversable
   * object.
   * @param t the traversable object.
   * @return the set of such symbols.
   */
  public static Set<Symbol> getLocalSymbols(Traversable t)
  {
    Set<Symbol> ret = new LinkedHashSet<Symbol>();

    if ( t == null )
      return ret;

    DepthFirstIterator iter = new DepthFirstIterator(t);
    while ( iter.hasNext() )
    {
      Object o = iter.next();
      if ( o instanceof SymbolTable )
        ret.addAll(getSymbols((SymbolTable)o));
    }

    return ret;
  }

  /**
   * Returns the symbol object having the specified string name.
   * @param name the name to be searched for.
   * @param tr the IR location where searching starts.
   * @return the symbol object.
   */
  public static Symbol getSymbolOfName(String name, Traversable tr)
  {
    Symbol ret = null;
    Traversable t = tr;

    while ( ret == null && t != null )
    {
      if ( t instanceof SymbolTable )
      {
        Set<Symbol> symbols = getSymbols((SymbolTable)t);
        for ( Symbol symbol : symbols )
        {
          if ( name.equals(symbol.getSymbolName()) )
          {
            ret = symbol;
            break;
          }
        }
      }
      t = t.getParent();
    }

    return ret;
  }

  /**
   * Returns the symbol of the expression if it represents an lvalue.
   *
   * @param e the input expression.
   * @return the corresponding symbol object.
   */
  /*
   * The following symbol is returned for each expression types.
   * Identifier         : its symbol.
   * ArrayAccess        : base name's symbol.
   * AccessExpression   : access symbol (list of symbols).
   * Pointer Dereference: the first symbol found in the expression tree.
   */
  public static Symbol getSymbolOf(Expression e)
  {
    if ( e instanceof Identifier )
      return ((Identifier)e).getSymbol();
    else if ( e instanceof ArrayAccess )
      return getSymbolOf( ((ArrayAccess)e).getArrayName() );
    else if ( e instanceof AccessExpression )
      //return ((AccessExpression)e).getSymbol();
      return new AccessSymbol((AccessExpression)e);
    else if ( e instanceof UnaryExpression )
    {
      return getSymbolOf(((UnaryExpression)e).getExpression());
/*
      UnaryExpression ue = (UnaryExpression)e;
      if ( ue.getOperator() == UnaryOperator.DEREFERENCE )
      {
        DepthFirstIterator iter = new DepthFirstIterator(ue.getExpression());
        while ( iter.hasNext() )
        {
          Object o = iter.next();
          if ( o instanceof Identifier )
            return ((Identifier)o).getSymbol();
        }
      }
*/
    }
    return null;
  }

  /**
   * Checks if the symbol is a global variable to the procedure containing the
   * given traversable object.
   *
   * @param symbol The symbol object
   * @param t The traversable object
   * @return true if it is global, false otherwise
   */
  public static boolean isGlobal(Symbol symbol, Traversable t)
  {
    t = IRTools.getAncestorOfType(t, Procedure.class);
    if ( t == null )
      return true; // conservative decision if a bad thing happens.
    for ( SymbolTable symtab : getParentTables(t) )
      if (symtab.containsSymbol(symbol))
        return true;
    return false;
  }

  /**
   * Checks if the symbol is a scalar variable.
   *
   * @param symbol The symbol
   * @return true if it is a scalar variable, false otherwise
   */
  public static boolean isScalar(Symbol symbol)
  {
    if ( symbol == null )
      return false;

    List specs = symbol.getArraySpecifiers();

    return ( specs == null || specs.isEmpty() );
  }

  /**
   * Checks if the symbol is an array variable.
   *
   * @param symbol The symbol
   * @return true if it is an array variable, false otherwise
   */
  public static boolean isArray(Symbol symbol)
  {
    if ( symbol == null )
      return false;

    List specs = symbol.getArraySpecifiers();

    return ( specs != null && !specs.isEmpty() );
  }

  /**
   * Checks if the symbol is a pointer type variable.
   *
   * @param symbol The symbol
   * @return true if it is a pointer type variable, false otherwise
   */
  public static boolean isPointer(Symbol symbol)
  {
    if ( symbol == null )
      return false;

    List specs = symbol.getTypeSpecifiers();

    if ( specs == null )
      return false;

    for ( Object o : specs )
      if ( o instanceof PointerSpecifier )
        return true;

    return false;
  }


  /**
   * Checks if the symbol is a pointer type variable. The input expression
   * should represent a variable. Otherwise it will return true.
   *
   * @param e the expression to be tested.
   */
  public static boolean isPointer(Expression e)
  {
    List spec = getExpressionType(e);
    if ( spec == null || spec.isEmpty() ||
      spec.get(spec.size()-1) instanceof PointerSpecifier )
      return true;
    else
      return false;
  }

  // For use with isInteger()
  private static final Set<Specifier> int_types;
  static {
    int_types = new HashSet<Specifier>();
    int_types.add(Specifier.INT);
    int_types.add(Specifier.LONG);
    int_types.add(Specifier.SIGNED);
    int_types.add(Specifier.UNSIGNED);
  }
  /**
   * Checks if the symbol is an interger type variable.
   *
   * @param symbol the symbol.
   * @return true if it is an integer type variable, false otherwise.
   */
  public static boolean isInteger(Symbol symbol)
  {
    return (symbol != null && isInteger(symbol.getTypeSpecifiers()));
  }
  public static boolean isInteger(List specifiers)
  {
    if (specifiers == null)
      return false;
    boolean ret = false;
    for (Object o : specifiers) {
      if (o == Specifier.CHAR || o instanceof PointerSpecifier)
        return false;
      ret |= int_types.contains(o);
    }
    return ret;
  }

  /**
   * Returns a list of specifiers of the given expression.
   *
   * @param e the given expression.
   * @return the list of specifiers.
   */
  public static List getExpressionType(Expression e)
  {
    if ( e instanceof Identifier )
    {
      Symbol var = ((Identifier)e).getSymbol();
      if ( var != null )
        return var.getTypeSpecifiers();
    }
    else if ( e instanceof ArrayAccess )
    {
      ArrayAccess aa = (ArrayAccess)e;
      List ret = getExpressionType(aa.getArrayName());
      if ( ret != null )
      {
        LinkedList ret0 = new LinkedList(ret);
        for ( int i=0; i < aa.getNumIndices(); ++i )
          if ( ret0.getLast() instanceof PointerSpecifier )
            ret0.removeLast();
        return ret0;
      }
      return ret;
    }
    else if ( e instanceof AccessExpression )
    {
      //Symbol var = ((AccessExpression)e).getSymbol();
      //if ( var != null )
      Symbol var = new AccessSymbol((AccessExpression)e);
      return var.getTypeSpecifiers();
    }
    else if ( e instanceof AssignmentExpression )
    {
      return getExpressionType(((AssignmentExpression)e).getLHS());
    }
    else if ( e instanceof CommaExpression )
    {
      LinkedList children = (LinkedList)e.getChildren();
      return getExpressionType((Expression)children.get(children.size()-1));
    }
    else if ( e instanceof ConditionalExpression )
    {
      return getExpressionType(((ConditionalExpression)e).getTrueExpression());
    }
    else if ( e instanceof FunctionCall )
    {
      Expression fc_name = ((FunctionCall)e).getName();
      if ( fc_name instanceof Identifier )
      {
        Symbol fc_var = ((Identifier)fc_name).getSymbol();
        if ( fc_var != null )
          return fc_var.getTypeSpecifiers();
      }
    }
    else if ( e instanceof IntegerLiteral )
    {
      return new LinkedList(Arrays.asList(Specifier.LONG));
    }
    else if ( e instanceof BooleanLiteral )
    {
      return new LinkedList(Arrays.asList(Specifier.BOOL));
    }
    else if ( e instanceof CharLiteral )
    {
      return new LinkedList(Arrays.asList(Specifier.CHAR));
    }
    else if ( e instanceof StringLiteral )
    {
      return new LinkedList(Arrays.asList(
        Specifier.CHAR,
        PointerSpecifier.UNQUALIFIED
      ));
    }
    else if ( e instanceof FloatLiteral )
    {
      return new LinkedList(Arrays.asList(Specifier.DOUBLE));
    }
    else if ( e instanceof Typecast )
    {
      List ret = new LinkedList();
      for ( Object spec : ((Typecast)e).getSpecifiers() )
      {
        if ( spec instanceof Specifier )
          ret.add(spec);
        else if ( spec instanceof Declarator )
          ret.addAll(((Declarator)spec).getSpecifiers());
      }
      return ret;
    }
    else if ( e instanceof UnaryExpression )
    {
      UnaryExpression ue = (UnaryExpression)e;
      UnaryOperator op = ue.getOperator();
      List ret = getExpressionType(ue.getExpression());
      if ( ret != null )
      {
        LinkedList ret0 = new LinkedList(ret);
        if ( op == UnaryOperator.ADDRESS_OF )
          ret0.addLast(PointerSpecifier.UNQUALIFIED);
        else if ( op == UnaryOperator.DEREFERENCE )
          ret0.removeLast();
        return ret0;
      }
      return ret;
    }
    else if ( e instanceof BinaryExpression )
    {
      Set logical_op =
        new HashSet(Arrays.asList("==",">=",">","<=","<","!=","&&","||"));
      BinaryExpression be = (BinaryExpression)e;
      BinaryOperator op = be.getOperator();
      if ( logical_op.contains(op.toString()) )
        return new LinkedList(Arrays.asList(Specifier.LONG));
      else
        return getExpressionType(be.getLHS());
    }
    else if ( e instanceof VaArgExpression )
    {
      return ((VaArgExpression)e).getSpecifiers();
    }
    PrintTools.printlnStatus("[WARNING] Unknown expression type: "+e, 0);
    return null;
  }

/**
   * Searches for a symbol by String sname in the table. If the symbol is
   * not in the table, then parent tables will be searched breadth-first.
   * If multiple symbols have the same String name, the first one found 
   * during the search will be returned.
   *
   * @param table The initial table to search.
   * @param sname The String name of the symbol (Symbol.getSymbolName()) to locate.
   *
   * @return a Declaration if the symbol is found, or null if it is not found.
   *    The Declaration may contain multiple declarators, of which sname will
   *    be one, unless the SingleDeclarator pass has been run on the program.
   */
  public static Declaration findSymbol(SymbolTable table, String sname)
  {
    return findSymbol(table, new NameID(sname));
  }

  /**
   * Returns a list of specifiers of the expression.
   */
  public static LinkedList getVariableType(Expression e)
  {
    LinkedList ret = new LinkedList();
    if ( e instanceof Identifier )
    {
      Symbol var = ((Identifier)e).getSymbol();
      if ( var != null )
        ret.addAll(var.getTypeSpecifiers());
    }
    else if ( e instanceof ArrayAccess )
    {
      ArrayAccess aa = (ArrayAccess)e;
      ret = getVariableType(aa.getArrayName());
      for ( int i=0; i < aa.getNumIndices(); ++i )
        if ( ret.getLast() instanceof PointerSpecifier )
          ret.removeLast();
    }
    else if ( e instanceof AccessExpression )
    {
      //Symbol var = ((AccessExpression)e).getSymbol();
      Symbol var = new AccessSymbol((AccessExpression)e);
      //if ( var != null )
      ret.addAll(var.getTypeSpecifiers());
    }
    else if ( e instanceof UnaryExpression )
    {
      UnaryExpression ue = (UnaryExpression)e;
      if ( ue.getOperator() == UnaryOperator.DEREFERENCE )
      {
        ret = getVariableType(ue.getExpression());
        if ( ret.getLast() instanceof PointerSpecifier )
          ret.removeLast();
        else
          ret.clear();
      }
    }
    return ret;
  }

  /**
  * Returns the look-up table that is internal to the given symbol table object.
  * Exposing the table access to public may incur inconsistent snapshot of any
  * symbol table object and its look-up table. Only read-only access will be
  * provided through <b>SymbolTable</b> interface.
  *
  * @param symtab the given symbol table object.
  * @return the internal look-up table, null if any exceptions occur.
  */
  protected static Map<IDExpression, Declaration> getTable(SymbolTable symtab)
  {
    try {
      return (Map<IDExpression, Declaration>)
          symtab.getClass().getDeclaredMethod("getTable").invoke(symtab);
    } catch (Exception e) {
      throw new InternalError(e.getMessage());
    }
  }
}
