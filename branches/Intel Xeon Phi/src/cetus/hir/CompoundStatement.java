package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/**
 * <b>CompoundStatement</b> represents a group of statements that are treated
 * as a single statement.
 */
public class CompoundStatement extends Statement implements SymbolTable
{
  /** The default print method for CompoundStatement */
  private static Method class_print_method;

  static
  {
    Class<?>[] params = new Class<?>[2];

    try {
      params[0] = CompoundStatement.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  /** Internal look-up table for symbols */
  private Map<IDExpression, Declaration> symbol_table;

  /** Creates an empty compound statement. */
  public CompoundStatement()
  {
    object_print_method = class_print_method;
    symbol_table = new LinkedHashMap<IDExpression, Declaration>(4);
  }

  /**
   * Adds a declaration after the last declaration; this models
   * the C language's requirement that all declarations appear
   * at the beginning of a block.
   *
   * @param decl The declaration to add.
   * @throws NotAnOrphanException if <b>decl</b> has a parent.
   */
  public void addDeclaration(Declaration decl)
  {
    if (decl.getParent() != null)
      throw new NotAnOrphanException();

    DeclarationStatement stmt = new DeclarationStatement(decl);

    int i = 0;
    int size = children.size();

    while (i < size && (children.get(i) instanceof DeclarationStatement ||
        children.get(i) instanceof AnnotationStatement) )
      i++;

    children.add(i, stmt);
    stmt.setParent(this);

    SymbolTools.addSymbols(this, decl);
  }

  /* SymbolTable interface */
  /**
  * @throws IllegalArgumentException if <b>ref</b> is not found.
  * @throws NotAnOrphanException if <b>decl</b> has a parent.
  */
  public void addDeclarationBefore(Declaration ref, Declaration decl)
  {
    if (decl.getParent() != null)
      throw new NotAnOrphanException();

    int index = Tools.indexByReference(children, ref.getParent());

    if (index == -1)
      throw new IllegalArgumentException();

    DeclarationStatement stmt = new DeclarationStatement(decl);
    children.add(index, stmt);
    stmt.setParent(this);

    SymbolTools.addSymbols(this, decl);
  }

  /* SymbolTable interface */
  /**
  * @throws IllegalArgumentException if <b>ref</b> is not found.
  * @throws NotAnOrphanException if <b>decl</b> has a parent.
  */
  public void addDeclarationAfter(Declaration ref, Declaration decl)
  {
    if (decl.getParent() != null)
      throw new NotAnOrphanException();

    int index = Tools.indexByReference(children, ref.getParent());

    if (index == -1)
      throw new IllegalArgumentException();

    DeclarationStatement stmt = new DeclarationStatement(decl); 
    children.add(index + 1, stmt);
    stmt.setParent(this);

    SymbolTools.addSymbols(this, decl);
  }

  /**
   * Adds a statement to the end of this compound statement. For consistent
   * management of symbol table interface, the input statement <b>stmt</b>
   * must not be a declaration statement; the declaration should be added
   * using the method {@link #addDeclaration}.
   *
   * @param stmt The statement to add.
   * @throws IllegalArgumentException If <b>stmt</b> is null.
   * @throws NotAnOrphanException if <b>stmt</b> has a parent.
   * @throws UnsupportedOperationException if <b>stmt</b> is a declaration
   * statement.
   */
  public void addStatement(Statement stmt)
  {
    if (stmt instanceof DeclarationStatement &&
        !(stmt.getChildren().get(0) instanceof PreAnnotation)) {
      throw new UnsupportedOperationException(
          "Declarations should be inserted by addDeclaration().");
    } else {
      addChild(stmt);
    }
  }  

  /**
   * Add a new statement before the reference statement.
   *
   * @param new_stmt the statement to be added.
   * @param ref_stmt the reference statement.
   * @throws IllegalArgumentException If <b>ref_stmt</b> is not found or
   * <b>new_stmt</b> is null.
   * @throws NotAnOrphanException if <b>new_stmt</b> has a parent.
   * @throws UnsupportedOperationException If <b>new_stmt</b> is a declaration
   * statement.
   */
  public void addStatementBefore(Statement ref_stmt, Statement new_stmt)
  {
    if (new_stmt instanceof DeclarationStatement &&
        !(new_stmt.getChildren().get(0) instanceof PreAnnotation)) {
      throw new UnsupportedOperationException(
          "Declarations should be inserted by addDeclarationBefore().");
    }

    int index = Tools.indexByReference(children, ref_stmt);
    if (index == -1)
      throw new IllegalArgumentException();

    addChild(index, new_stmt);
  }

  /**
   * Add a new statement after the reference statement.
   *
   * @param new_stmt the statement to be added.
   * @param ref_stmt the reference statement.
   * @throws IllegalArgumentException If <b>ref_stmt</b> is not found or
   * <b>new_stmt</b> is null.
   * @throws NotAnOrphanException if <b>new_stmt</b> has a parent.
   * @throws UnsupportedOperationException If <b>stmt</b> is a declaration
   * statement.
   */
  public void addStatementAfter(Statement ref_stmt, Statement new_stmt)
  {
    if (new_stmt instanceof DeclarationStatement &&
        !(new_stmt.getChildren().get(0) instanceof PreAnnotation)) {
      throw new UnsupportedOperationException(
          "Declarations should be inserted by addDeclarationAfter().");
    }

    int index = Tools.indexByReference(children, ref_stmt);
    if (index == -1)
      throw new IllegalArgumentException();

    addChild(index+1, new_stmt);
  }
  
  /**
   * Remove the given statement if it exists.
   *
   * @param stmt the statement to be removed.
   */
  public void removeStatement(Statement stmt)
  {
    removeChild(stmt);
  }

  /**
  * Returns a clone of this compound statement.
  *
  * @return the cloned compound statement.
  */
  @Override
  public CompoundStatement clone()
  {
    CompoundStatement o = (CompoundStatement)super.clone();

    // Builds the look-up table
    o.symbol_table = new LinkedHashMap<IDExpression, Declaration>(4);
    for (Traversable child : o.children)
      if (child instanceof DeclarationStatement)
        SymbolTools.addSymbols(o,
            ((DeclarationStatement)child).getDeclaration());

    return o;
  }

  /**
   * Returns the total number of statements contained within
   * this CompoundStatement.
   *
   * @return the number of statements within this compound statement
   */
  public int countStatements()
  {
    int n = 0;

    FlatIterator iter = new FlatIterator(this);

    for (;;)
    {
      Statement stmt = null;

      try {
        stmt = (Statement)iter.next(Statement.class);
      } catch (NoSuchElementException e) {
        break;
      } 

      if (stmt instanceof IfStatement)
      {
        int then_count = ((CompoundStatement)((IfStatement)stmt).getThenStatement()).countStatements();
        int else_count = ((CompoundStatement)((IfStatement)stmt).getElseStatement()).countStatements();

        if (then_count > else_count)
          n += then_count;
        else
          n += else_count;
      }
      else if (stmt instanceof CompoundStatement)
        n += ((CompoundStatement)stmt).countStatements();
      else
        ++n;
    }

    return n;
  }

  /**
   * Prints a statement to the given print writer.
   *
   * @param s The statement to print.
   * @param o The writer on which to print the statement.
   */
  public static void defaultPrint(CompoundStatement s, PrintWriter o)
  {
    o.println("{");
    PrintTools.printlnList(s.children, o);
    o.print("}");
  }

  /* SymbolTable interface */
  public Declaration findSymbol(IDExpression name)
  {
    return SymbolTools.findSymbol(this, name);
  }

  /* SymbolTable interface */
  public List<SymbolTable> getParentTables()
  {
    return SymbolTools.getParentTables(this);
  }

  /**
  * Returns the internal look-up table for symbols. This method is protected
  * for consistent symbol table management.
  *
  * @return the internal look-up table.
  */
  protected Map<IDExpression, Declaration> getTable()
  {
    return symbol_table;
  }

  /**
  * Removes the specified child object if it exists.
  *
  * @param child the child object to be removed.
  * @throws IllegalArgumentException if <b>child</b> is not found.
  */
  public void removeChild(Traversable child)
  {
    int index = Tools.indexByReference(children, child);
    if (index == -1)
      throw new IllegalArgumentException();
    
    children.remove(index);
    child.setParent(null);

    if (child instanceof DeclarationStatement)
      SymbolTools.removeSymbols(
          this, ((DeclarationStatement)child).getDeclaration());
  }

  /* SymbolTable interface */
  public Set<Symbol> getSymbols()
  {
    return SymbolTools.getSymbols(this);
  }

  /* SymbolTable interface */
  public Set<Declaration> getDeclarations()
  {
    return new LinkedHashSet<Declaration>(symbol_table.values());
  }

  /* SymbolTable interface */
  public boolean containsSymbol(Symbol symbol)
  {
    for (IDExpression id : symbol_table.keySet())
      if (id instanceof Identifier &&
          symbol.equals(((Identifier)id).getSymbol()))
        return true;
    return false;
  }

  /* SymbolTable interface */
  public boolean containsDeclaration(Declaration decl)
  {
    return symbol_table.containsValue(decl);
  }
}
