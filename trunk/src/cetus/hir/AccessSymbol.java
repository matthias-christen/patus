package cetus.hir;

import java.util.*;

/**
 * AccessSymbol is intended to express an object of AccessExpression as a
 * Symbol object. This will replace AccessSymbol if everything works fine.
 */
/*
  Failed case:
  base: ConditionalExpression, BinaryExpression
*/
public class AccessSymbol extends PseudoSymbol implements Symbol
{
  // The base symbol.
  private Symbol base;

  // The member symbol.
  private Symbol member;

  /**
   * Constructs an object with the given access expression.
   * @param ae the access expression which a symbol is created from.
   */
  public AccessSymbol(AccessExpression ae)
  {
    base = parse(ae.getLHS());
    member = ((Identifier)ae.getRHS()).getSymbol();
    // This is converted to invisible symbol.
    if (ae.getOperator() == AccessOperator.POINTER_ACCESS) {
      if ( ae.getLHS() instanceof UnaryExpression &&
          ((UnaryExpression)ae.getLHS()).getOperator()
          == UnaryOperator.ADDRESS_OF )
        ;
      else
        base = new DerefSymbol(base);
    }
  }

  /**
   * Constructs a struct symbol with the given base and member symbol.
   * @param base the base symbol.
   * @param member the member symbol.
   */
  public AccessSymbol(Symbol base, Symbol member)
  {
    this.base = base;
    this.member = member;
  }

  /**
   * Analyzes the given expression and returns the symbol representation.
   * @param e the expression being analyzed.
   * @return the symbol representation of the given expression.
   */
  private Symbol parse(Expression e)
  {
    Symbol ret = null;
    if ( e instanceof AccessExpression )
      ret = new AccessSymbol((AccessExpression)e);
    else if ( e instanceof Identifier )
      ret = ((Identifier)e).getSymbol();
    else if ( e instanceof ArrayAccess )
      ret = parse(((ArrayAccess)e).getArrayName());
    else if ( e instanceof Typecast )
      ret = parse((Expression)e.getChildren().get(0));
    // ConditionalExpression, FunctionCall, CommaExpression, and
    // BinaryExpression is not handled in general. Temporarily, the most
    // representative expression is parsed, which may be incorrect, so the
    // high-level passes that uses AccessSymbol must handle these cases
    // properly.
    ////////////////////////////////////////////////////////////////////////////
    else if ( e instanceof ConditionalExpression )
    {
      PrintTools.printlnStatus("[WARNING] Symbol is obscure for "+e, 0);
      ret = parse(((ConditionalExpression)e).getTrueExpression());
    }
    else if ( e instanceof FunctionCall )
    {
      PrintTools.printlnStatus("[WARNING] Symbol is obscure for "+e, 0);
      ret = parse(((FunctionCall)e).getName());
    }
    else if ( e instanceof CommaExpression )
    {
      PrintTools.printlnStatus("[WARNING] Symbol is obscure for "+e, 0);
      ret = parse((Expression)e.getChildren().get(e.getChildren().size()-1));
    }
    else if ( e instanceof BinaryExpression )
    {
      PrintTools.printlnStatus("[WARNING] Symbol is obscure for "+e, 0);
      ret = parse(((BinaryExpression)e).getLHS());
    }
    ////////////////////////////////////////////////////////////////////////////
    else if ( e instanceof UnaryExpression )
    {
      UnaryOperator op = ((UnaryExpression)e).getOperator();
      Expression ue = ((UnaryExpression)e).getExpression();
      ret = parse(ue);
      if ( op == UnaryOperator.DEREFERENCE )
        ret = new DerefSymbol(ret);
    }
    else
      throw new InternalError
          ("failed to parse "+e+"("+e.getClass().getName()+")");
    return ret;
  }

  /**
   * Returns the list of type specifiers.
   */
  public List getTypeSpecifiers()
  {
    return member.getTypeSpecifiers();
  }

  /**
   * Returns the list of array specifiers.
   */
  public List getArraySpecifiers()
  {
    return member.getArraySpecifiers();
  }

  /**
   * Returns the name representation of the symbol.
   */
  public String getSymbolName()
  {
    return "<"+base.getSymbolName()+"."+member.getSymbolName()+">";
  }

  /**
   * Returns debug information.
   */
  private static String getInfo(Symbol symbol)
  {
    String ret = "<SYMBOL> " + symbol.getSymbolName() + " {\n";
    if ( symbol instanceof AccessSymbol )
    {
      AccessSymbol ss = (AccessSymbol)symbol;
      ret += "base = \n" + getInfo(ss.base) + "\n";
      ret += "member = " + getInfo(ss.member) + "\n";
    }
    else if ( symbol instanceof DerefSymbol )
    {
      ret += "ref = \n" + getInfo(((DerefSymbol)symbol).getRefSymbol())
          + "\n";
    }
    else
      ret += "symbol = " + symbol.getSymbolName() + "\n";
    ret += "types = " + symbol.getTypeSpecifiers() + "\n}";

    return ret;
  }

  /**
   * Recovers an expression from this symbol. This may produce an illegal
   * expression because not every aspect of the original expression is stored
   * while its corresponding struct symbol is created. This functionality is
   * useful only in limited cases in that sense.
   * @return the expression representation.
   */
  private Expression toExpression()
  {
    Expression lhs = null, rhs = null;
    if ( base instanceof DerefSymbol )
      lhs = ((DerefSymbol)base).toExpression();
    else if ( base instanceof AccessSymbol )
      lhs = ((AccessSymbol)base).toExpression();
    else if ( base instanceof Identifier )
      lhs = new Identifier(base);
    else
    {
      PrintTools.printlnStatus("[WARNING] Unexpected access expression type", 0);
      return null;
    }
    rhs = new Identifier(member);
    return new AccessExpression(lhs, AccessOperator.MEMBER_ACCESS, rhs);
  }

  /**
   * Checks if the this symbol is equal to the the given object. The comparison
   * comes down to symbol comparison which is hashcode comparison assuming the
   * composed symbol such as AccessSymbol or DerefSymbol provides a
   * correct equals method.
   * @param o the object to be compared with.
   * @return true if they are equal, false otherwise.
   */
  @Override
  public boolean equals(Object o)
  {
    if ( o instanceof AccessSymbol )
    {
      AccessSymbol other = (AccessSymbol)o;
      return (base.equals(other.base) && member.equals(other.member));
    }
    else
      return false;
  }

  /**
   * Returns the hash code of the current struct symbol.
   * @return the integer hash code for this struct symbol.
   */
  @Override
  public int hashCode()
  {
    return ( base.hashCode() ^ member.hashCode() );
  }

  /**
   * Returns the base symbol.
   * @return the base symbol.
   */
  public Symbol getBaseSymbol()
  {
    return base;
  }

  /**
   * Returns the member symbol.
   * @return the member symbol.
   */
  public Symbol getMemberSymbol()
  {
    return member;
  }

  /**
   * Returns the IR symbol from which the current symbol is derived. For this
   * struct symbol the member symbol is returned since it is the one that
   * describes the type information of this struct symbol.
   * @return the symbol object that exist in the IR.
   */
  public Symbol getIRSymbol()
  {
    if (base instanceof PseudoSymbol)
      return ((PseudoSymbol)base).getIRSymbol();
    else
      return base;
  }
}
