package cetus.transforms;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

import cetus.hir.*;

/**
 * Transforms a program such that every declaration contains
 * at most one declarator.  The declarations are kept in order,
 * so for example "int x, y, z;" becomes "int x; int y; int z;"
 */
public class SingleDeclarator
{
  private static String pass_name = "[SingleDeclarator]";

  private Program program;

  private SingleDeclarator(Program program)
  {
    this.program = program;
  }

  private void eliminateMultipleDeclarators(VariableDeclaration decl)
  {
    PrintTools.printStatus(pass_name + " eliminating multiples in ", 3);
    PrintTools.printlnStatus(decl, 3);

		SymbolTable outer = null;
		Traversable child = decl, parent = decl.getParent();
		if ( parent instanceof SymbolTable )
			outer = (SymbolTable)parent;
		else if ( parent instanceof DeclarationStatement )
		{
			child = parent;
			parent = child.getParent();
			outer = (SymbolTable)parent;
		}
		else
			return;

		/* now parent is a symbol table and child is either decl or declstmt. */

    VariableDeclaration placeholder = new VariableDeclaration(new LinkedList());
    outer.addDeclarationAfter(decl, placeholder);

		parent.removeChild(child);

    for (int i = decl.getNumDeclarators() - 1; i >= 0; --i)
    {
      Declarator d = decl.getDeclarator(i);

      outer.addDeclarationAfter(placeholder,
        new VariableDeclaration(decl.getSpecifiers(), d.clone()));
    }

		if ( placeholder.getParent() instanceof DeclarationStatement )
			parent.removeChild(placeholder.getParent());
		else
			parent.removeChild(placeholder);
  }

  public static void run(Program program)
  {
    PrintTools.printlnStatus(pass_name + " begin", 1);

    SingleDeclarator pass = new SingleDeclarator(program);
    pass.start();

    PrintTools.printlnStatus(pass_name + " end", 1);
  }

  private void start()
  {
    DepthFirstIterator i = new DepthFirstIterator(program);

    Set<Class> set = new HashSet<Class>();
    set.add(Procedure.class);
    set.add(VariableDeclaration.class);

    for (;;)
    {
      Procedure proc = null;
      VariableDeclaration decl = null;

      try {
        Object o = i.next(set);
        if (o instanceof Procedure)
          proc = (Procedure)o;
        else
          decl = (VariableDeclaration)o;
      } catch (NoSuchElementException e) {
        break;
      }

      if (proc != null)
      {
        PrintTools.printlnStatus(pass_name + " examining procedure " + proc.getName(), 2);
      }
      else  
      {
        if (decl.getNumDeclarators() > 1)
          eliminateMultipleDeclarators(decl);
      }
    }
  }
}
