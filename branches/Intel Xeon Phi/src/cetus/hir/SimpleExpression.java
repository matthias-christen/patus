package cetus.hir;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * SimpleExpression is another form of representing an expression which is
 * effective in symbolic manipulation. Expressions with commutative and
 * associative operators are flattend for easier manipulation, and only a
 * subset of the Cetus expressions is represented specifically with an
 * abbreviated operator name. Other types are marked as general "TREE" whose
 * children are recursively represented using SimpleExpression.
 */
public class SimpleExpression implements
    Comparator<SimpleExpression>, Comparable<SimpleExpression>, Cloneable
{
  // Short operator name for SimpleExpression. All non-tractable expressions are
  // named as TREE expression.
  protected static final int
    ADD  = 0,
    MUL  = 1,
    DIV  = 2,
    MOD  = 3,
    SFTL = 4,
    SFTR = 5,
    BAND = 6,
    BOR  = 7,
    BXOR = 8,
    BCMP = 9,
    AND  = 10,
    OR   = 11,
    EQ   = 12,
    NE   = 13,
    LE   = 14,
    LT   = 15,
    GE   = 16,
    GT   = 17,
    NEG  = 18,
    TREE = 19,
    ID   = 20,
    LIT  = 21,
    LEAF = 22,
    MIN  = 23,
    MAX  = 24;

  // Map from simple operators to Cetus operators.
  private static final List cop = Arrays.asList(
    BinaryOperator.ADD,
    BinaryOperator.MULTIPLY,
    BinaryOperator.DIVIDE,
    BinaryOperator.MODULUS,
    BinaryOperator.SHIFT_LEFT,
    BinaryOperator.SHIFT_RIGHT,
    BinaryOperator.BITWISE_AND,
    BinaryOperator.BITWISE_INCLUSIVE_OR,
    BinaryOperator.BITWISE_EXCLUSIVE_OR,
    UnaryOperator.BITWISE_COMPLEMENT,
    BinaryOperator.LOGICAL_AND,
    BinaryOperator.LOGICAL_OR,
    BinaryOperator.COMPARE_EQ,
    BinaryOperator.COMPARE_NE,
    BinaryOperator.COMPARE_LE,
    BinaryOperator.COMPARE_LT,
    BinaryOperator.COMPARE_GE,
    BinaryOperator.COMPARE_GT,
    UnaryOperator.LOGICAL_NEGATION,
    "TREE",
    "ID",
    "LIT",
    "LEAF",
    "MIN",
    "MAX"
  );

  // Frequently-used static simple expressions
  protected static final SimpleExpression sone = getInt(1);
  protected static final SimpleExpression szero = getInt(0);

  // Debug flag
  protected static final int verbosity = 0;//Integer.valueOf(cetus.exec.Driver.getOptionValue("verbosity")).intValue();

  // Masks for each options
  protected static final int
    FACTORIZE  = 1<<4,
    FOLD       = 1<<3,
    DISTRIBUTE = 1<<2,
    DIVIDE     = 1<<1,
    LOGIC      = 1<<0;

  // Maximum order for a simple expression
  private static final int TREE_ORDER = 100;

  // Normalization option (switch on everything by default).
  private static int option = 0xff;

  // Reads normalization options from cetus' command-line option
  static
  {
    String s = cetus.exec.Driver.getOptionValue("simplify");
    if ( s != null )
    {
      int opt;
      try {
        opt = Integer.parseInt(s, 2);
      } catch ( NumberFormatException ex ) {
        opt = 0xff;
      }
      setOption(opt);
    }
  }

  // List containing the children
  private LinkedList<SimpleExpression> children;

  // Expression's complexity based on the number of symbolic variables.
  private int order;

  // Abbreviated operator name.
  private int sop;

  // Reference to the IR for TREE-, ID-, and LIT-type expression.
  private Expression expr;

  // Flag for side-effect expression.
  private boolean contains_side_effect;

  // Constructs an empty simple expression
  protected SimpleExpression()
  {
    children = new LinkedList<SimpleExpression>();
    order = 0;
    expr = null;
    contains_side_effect = false;
  }

  // Constructs an empty simple expression with the given operator
  protected SimpleExpression(int sop)
  {
    this();
    this.sop = sop;
  }

  // Constructs an empty simple expression with the given expression copying
  // the expr reference.
  protected SimpleExpression(SimpleExpression se)
  {
    this(se.sop);
    this.expr = se.expr;
  }

  // Constructs a simple expression with the given operator and operands.
  protected SimpleExpression(SimpleExpression se1, int op, SimpleExpression se2)
  {
    this(op);
    add(se1);
    add(se2);
  }

  // Constructs a simple expression from the given Cetus expression
  protected SimpleExpression(Expression e)
  {
    this();
    if ( e instanceof UnaryExpression )
      parse( (UnaryExpression)e );
    else if ( e instanceof BinaryExpression )
      parse( (BinaryExpression)e );
    else if ( e.getChildren()!=null && !e.getChildren().isEmpty() )
      parseTree( e );
    else
      parseLeaf( e );
  }

  // Parses a unary expression
  private void parse(UnaryExpression ue)
  {
    UnaryOperator uop = ue.getOperator();
    SimpleExpression child = new SimpleExpression(ue.getExpression());
    contains_side_effect |= child.contains_side_effect;

    if ( uop == UnaryOperator.MINUS )
    {
      sop = MUL;
      add(getInt(-1));
      add(child);
    }
    else if ( uop == UnaryOperator.PLUS )
    {
      sop = child.sop;
      expr = child.expr;
      addAll(child);
    }
    else if ( uop == UnaryOperator.LOGICAL_NEGATION ||
        uop == UnaryOperator.BITWISE_COMPLEMENT )
    {
      sop = cop.indexOf(uop);
      add(child);
    }
    else
    {
      sop = TREE;
      expr = ue;
      add(child);
    }
  }

  // Parses a binary expression
  private void parse(BinaryExpression be)
  {
    BinaryOperator bop = be.getOperator();
    SimpleExpression lhs = new SimpleExpression(be.getLHS());
    SimpleExpression rhs = new SimpleExpression(be.getRHS());
    contains_side_effect |=
      (lhs.contains_side_effect || rhs.contains_side_effect);

    if ( bop == BinaryOperator.SUBTRACT )
    {
      sop = ADD;
      SimpleExpression new_rhs = new SimpleExpression(MUL);
      new_rhs.add(getInt(-1));
      new_rhs.add(rhs);
      add(lhs);
      add(new_rhs);
    }
    else
    {
      sop = cop.indexOf(bop);
      if ( sop == -1 )
      {
        sop = TREE;
        expr = be;
      }
      add(lhs);
      add(rhs);
    }
  }

  // Parses a generic expression and returns true if it contains side effect.
  private void parseTree(Expression e)
  {
    if ( e instanceof MinMaxExpression )
    {
      MinMaxExpression mme = (MinMaxExpression)e;
      sop = (mme.isMin())? MIN: MAX;
    }
    else
    {
      sop = TREE;
      expr = e;
      order = TREE_ORDER;
      if ( e instanceof FunctionCall || e instanceof VaArgExpression )
        contains_side_effect = true;
    }

    if ( e instanceof StatementExpression ) // parsing stops here.
      return;

    for ( Object o : e.getChildren() )
    {
      if ( o instanceof Expression )
      {
        SimpleExpression child = new SimpleExpression((Expression)o);
        contains_side_effect |= child.contains_side_effect;
        add(child);
      }
      else
        Tools.exit("[SimpleExpression] unable to parse a generic expression");
    }
  }

  // Parses a leaf expression
  private void parseLeaf(Expression e)
  {
    expr = e;
    if ( e instanceof Identifier )
    {
      sop = ID;
      order = 1;
    }
    else if ( e instanceof IntegerLiteral || e instanceof FloatLiteral )
    {
      sop = LIT;
      order = 0;
    }
    else
    {
      sop = LEAF;
      order = 0;
    }
  }

  // Adds the given simple expression to the list of children while flattening
  // commutative/associative operations.
  protected void add(SimpleExpression se)
  {
    if ( isCommAssoc() && sop == se.sop )
      addAll(se);
    else
    {
      children.add(se);
      order += se.order;
    }
  }

  // Adds the given collection of simple expressions to the list of children.
  protected void addAll(Collection<SimpleExpression> ses)
  {
    for ( SimpleExpression se : ses )
    {
      children.add(se);
      order += se.order;
    }
  }

  // Adds the children of the given simple expression to the list of children.
  protected void addAll(SimpleExpression se)
  {
    addAll(se.children);
  }

  // Returns a simple expression from the given integer number
  protected static SimpleExpression getInt(int num)
  {
    return new SimpleExpression(new IntegerLiteral(num));
  }

  // Returns a simple expression from the givne floating-point number
  protected static SimpleExpression getDouble(double num)
  {
    return new SimpleExpression(new FloatLiteral(num));
  }

  /**
   * Returns a string representation of the simple expression.
   * @return the string representation.
   */
  @Override
public String toString()
  {
    if ( children.isEmpty() )
      return expr.toString();
    StringBuilder str = new StringBuilder(80);
    str.append("(");
    str.append(cop.get(sop).toString());
    str.append(", ");
    str.append(PrintTools.listToString(children, ", "));
    str.append(")");
    return str.toString();
  }

  /**
   * Returns a reconstructed Cetus expression from the simple expression.
   * @return the Cetus expression.
   */
  protected Expression getExpression()
  {
    Expression ret = null;

    if ( children.isEmpty() )
      ret = expr.clone();
    else if ( sop == TREE )
    {
      ret = expr.clone();
      for ( int i=0; i<children.size(); ++i )
        ret.setChild(i, getChild(i).getExpression());
    }
    else if ( sop == MIN || sop == MAX )
    {
      ret = new MinMaxExpression(true);
      for ( SimpleExpression child : children )
        ((MinMaxExpression)ret).add(child.getExpression());
      if ( sop == MAX )
        ((MinMaxExpression)ret).setMin(false);
    }
    else if ( cop.get(sop) instanceof UnaryOperator )
    {
      UnaryOperator uop = (UnaryOperator)cop.get(sop);
      ret = new UnaryExpression(uop, getChild(0).getExpression());
    }
    else if ( cop.get(sop) instanceof BinaryOperator )
    {
      BinaryOperator bop = (BinaryOperator)cop.get(sop);
      Iterator<SimpleExpression> iter = children.iterator();
      if ( iter.hasNext() )
        ret = iter.next().getExpression();
      while ( iter.hasNext() )
        ret = new BinaryExpression(ret, bop, iter.next().getExpression());
    }
    else
      Tools.exit("[SimpleExpression] unknown simple expression");

    return ret;
  }

  // Returns the child simple expression at the given position
  protected SimpleExpression getChild(int id)
  {
    return children.get(id);
  }

  // Sets the id-th child with the given simple expression
  protected void setChild(int id, SimpleExpression child)
  {
    children.set(id, child);
    order += child.order - getChild(id).order;
  }

  // Sets the simplification option with the given integer
  protected static void setOption(int opt)
  {
    option = opt;
  }

  // Checks if the given option is allowed now.
  private static boolean allow(int opt)
  {
    return ((option & opt) != 0);
  }

  // Checks if the operator is commutative and associative
  protected boolean isCommAssoc()
  {
    return (sop==ADD|| sop==MUL|| sop==AND|| sop==OR|| sop==MIN|| sop==MAX);
  }

  // Checks if the operator is a comparison operator.
  protected boolean isCompare()
  {
    return (sop >= EQ && sop <= GT);
  }

  // Checks if the simple expression contains a child with the given operator
  protected boolean containsChildOfType(int op)
  {
    for ( SimpleExpression child : children )
      if ( child.sop == op )
        return true;
    return false;
  }

  // Returns the number of descendants with the specified type.
  private int countsOperations(int op)
  {
    int ret = 0;
    if ( children == null )
      return ret;
    for ( SimpleExpression child : children )
      ret += child.countsOperations(op);
    if ( sop == op )
      ret += children.size()-1;
    return ret;
  }

/*
  public static void runtest(Program p)
  {
    DepthFirstIterator iter = new DepthFirstIterator(p);

    while ( iter.hasNext() )
    {
      Object o = iter.next();
      if ( o instanceof ExpressionStatement )
      {
        Expression e = ((ExpressionStatement)o).getExpression();
        SimpleExpression se = new SimpleExpression(e);
        System.out.println(e.toString()+":");
        int[] iopt = {8, 4, 2, 16, 1, 31};
        String[] sopt = {" FOLD"," DIST","  DIV"," FACT","LOGIC"," SIMP"};
        for ( int i=0; i<6; i++ )
        {
          SimpleExpression.setOption(iopt[i]);
          Expression expr = se.normalize().getExpression();
          System.out.println(sopt[i]+" -> "+expr);
        }
        System.out.println("");
      }
    }
  }
*/

  /**
   * Compares this simple expression with the given simple expression.
   * @param se the given simple expression.
   * @return the result of Expression.compareTo(Expression).
   */
  @Override
public int compareTo(SimpleExpression se)
  {
    return getExpression().compareTo(se.getExpression());
  }

  /**
   * Returns a clone of this simple expression.
   * @return a clone object.
   */
  @Override
  public SimpleExpression clone()
  {
    SimpleExpression ret = new SimpleExpression(sop);
    ret.order = order;
    ret.expr = expr;
    ret.contains_side_effect = contains_side_effect;
    ret.children.addAll(children);
    return ret;
  }

  /**
   * Checks if this simple expression is equal to the given object.
   * @param o the given object.
   * @return true if it is, false otherwise.
   */
  @Override
public boolean equals(Object o)
  {
    if ( o instanceof SimpleExpression )
      return (compareTo((SimpleExpression)o) == 0);
    else
      return false;
  }

  /**
   * Returns if a simple expression is greater than, less than, or equal to
   * the other.
   * @param se1 the first simple expression.
   * @param se2 the second simple expression.
   * @return the comparison result (-1, 0, 1).
   */
  @Override
public int compare(SimpleExpression se1, SimpleExpression se2)
  {
    if ( se1.order < se2.order )
      return -1;
    else if ( se1.order > se2.order )
      return 1;

    se1 = se1.getTerm();
    se2 = se2.getTerm();
    if ( se1.children.size() < se2.children.size() )
      return -1;
    else if ( se1.children.size() > se2.children.size() )
      return 1;
    else
      return se1.compareTo(se2);
  }

  // Returns the non-literal term in the simple expression.
  protected SimpleExpression getTerm()
  {
    SimpleExpression ret = null;
    if ( sop == LIT )
      ret = sone;
    else if ( sop != MUL )
      ret = this;
    else
    {
      ret = new SimpleExpression(MUL);
      for ( SimpleExpression child : children )
        if ( child.sop != LIT )
          ret.add(child);

      if ( ret.children.size() == 0 )
        ret = sone;
      else if ( ret.children.size() == 1 )
        ret = ret.getChild(0);
    }
    return ret;
  }

  // Returns a Double value if this simple expression is literal
  protected Double getValue()
  {
    Double ret = null;
    if ( expr instanceof IntegerLiteral )
      return new Double( ((IntegerLiteral)expr).getValue() );
    else if ( expr instanceof FloatLiteral )
      return new Double( ((FloatLiteral)expr).getValue() );
    return ret;
  }

  // Returns se1+se2 with constant evaluation.
  protected static SimpleExpression
      add(SimpleExpression se1, SimpleExpression se2)
  {
    return compute(se1, ADD, se2);
  }

  // Returns se1-se2 with constant evaluation.
  protected static SimpleExpression
      subtract(SimpleExpression se1, SimpleExpression se2)
  {
    return compute(se1, ADD, compute(getInt(-1), MUL, se2));
  }

  // Returns se1*se2 with constant evaluation.
  protected static SimpleExpression
      multiply(SimpleExpression se1, SimpleExpression se2)
  {
    return compute(se1, MUL, se2);
  }

  // Returns se1/se2 with constant evaluation.
  protected static SimpleExpression
      divide(SimpleExpression se1, SimpleExpression se2)
  {
    return compute(se1, DIV, se2);
  }

  // Returns se1%se2 with constant evaluation.
  protected static SimpleExpression
      mod(SimpleExpression se1, SimpleExpression se2)
  {
    return compute(se1, MOD, se2);
  }

  // Returns se1<op>se2 with constant evaluation.
  private static SimpleExpression compute
      (SimpleExpression se1, int op, SimpleExpression se2)
  {
    SimpleExpression ret = null;
    if ( se1.sop == LIT && se2.sop == LIT )
    {
      Double result = null, v1 = se1.getValue(), v2 = se2.getValue();
      switch ( op )
      {
        case ADD: result = v1+v2; break;
        case MUL: result = v1*v2; break;
        case DIV: result = v1/v2; break;
        case MOD: result = v1%v2; break;
        default :
          Tools.exit("[SimpleExpression] unknown operation in normalization");
      }
      if ( se1.expr instanceof IntegerLiteral &&
          se2.expr instanceof IntegerLiteral )
        ret = getInt(result.intValue());
      else
        ret = getDouble(result.doubleValue());
    }
    else
    {
      ret = new SimpleExpression(se1, op, se2);
      ret = ret.normalize();
    }
    return ret;
  }

  // Sorts the child terms if they are commutative and associative
  protected void sort()
  {
    for ( SimpleExpression child : children )
      child.sort();
    if ( isCommAssoc() )
      Collections.sort(children, this);
  }

  // Returns the literal terms of this simple expression -- assumes it is
  // already simplified.
  protected SimpleExpression getCoef()
  {
    SimpleExpression ret = null;
    if ( sop == LIT )
      ret = this;
    else if ( sop != MUL || getChild(0).sop != LIT )
      ret = sone;
    else
      ret = getChild(0);
    return ret;
  }

  // Normalizes this simple expression recursively.
  protected SimpleExpression normalize()
  {
    SimpleExpression ret = new SimpleExpression(this);

    for ( SimpleExpression child : children )
      ret.add(child.normalize());

    if ( contains_side_effect )
      return ret;

    switch ( ret.sop )
    {
      case ID: case LIT: case LEAF:
        ret = this; break;
      case ADD:
        ret = ret.normalizeADD(); break;
      case MUL:
        ret = ret.normalizeMUL(); break;
      case DIV:
        ret = ret.normalizeDIV(); break;
      case MOD:
        ret = ret.normalizeMOD(); break;
      case SFTL: case SFTR: case BAND: case BOR: case BXOR:
        ret = ret.normalizeBitOperation(); break;
      case BCMP:
        ret = ret.normalizeBCMP(); break;
      case AND: case OR:
        ret = ret.normalizeLogic(); break;
      case EQ: case NE: case LE: case LT: case GE: case GT:
        ret = ret.normalizeCompare(); break;
      case NEG:
        ret = ret.normalizeNEG(); break;
      case MIN: case MAX:
        ret = ret.normalizeMINMAX(); break;
      default:
    }

    if ( ret.isCommAssoc() )
      ret.sort();

    return ret;
  }

  // Normalizes an ADD expression
  private SimpleExpression normalizeADD()
  {
    if ( !allow(FOLD) )
      return this;
    TreeMap<SimpleExpression, SimpleExpression> terms =
      new TreeMap<SimpleExpression, SimpleExpression>();
    for ( SimpleExpression child : children )
    {
      SimpleExpression term = child.getTerm(), coef = child.getCoef();
      if ( terms.containsKey(term) )
        terms.put(term, add(terms.get(term), coef));
      else
        terms.put(term, coef);
    }

    SimpleExpression ret = new SimpleExpression(ADD);
    for ( SimpleExpression term : terms.keySet() )
    {
      SimpleExpression coef = terms.get(term);
      if ( !coef.equals(szero) )
        ret.add( (coef.equals(sone))? term: multiply(coef, term));
    }

    if ( ret.children.size() == 0 )
      ret = szero;
    else if ( ret.children.size() == 1 )
      ret = ret.getChild(0);

    return ret;
  }

  // Normalizes a MUL expression
  private SimpleExpression normalizeMUL()
  {
    SimpleExpression ret = this;
    if ( allow(FOLD) )
    {
      SimpleExpression coef = sone;
      LinkedList<SimpleExpression> terms = new LinkedList<SimpleExpression>();
      for ( SimpleExpression child : children )
      {
        if ( child.sop == LIT )
          coef = multiply(coef, child);
        else
          terms.add(child);
      }
      ret = new SimpleExpression(MUL);
      if ( coef.equals(szero) )
        ret = szero;
      else
      {
        if ( !coef.equals(sone) || terms.size()==0 )
          ret.add(coef);
        ret.addAll(terms);
        if ( ret.children.size() == 1 )
          ret = ret.getChild(0);
      }
    }
    ret = ret.distribute();
    return ret;
  }

  // Normalizes a DIV expression
  private SimpleExpression normalizeDIV()
  {
    if ( !allow(DIVIDE) )
      return this;
    SimpleExpression lhs = getChild(0), rhs = getChild(1), ret = null;
    if ( rhs.equals(szero) )
      ret = this;                // Don't do anything with division by zero
    else if ( lhs.equals(szero) )
      ret = szero;               // 0/<expr>: expr=0 is an exception anyhow
    else if ( lhs.sop == LIT && rhs.sop == LIT )
      ret = divide(lhs, rhs);    // Call compute method
    else if ( rhs.equals(sone) || rhs.equals(getInt(-1)) )
      ret = multiply(rhs, lhs);  // Division by one -> multiplication by one
    if ( ret == null )
    {
      lhs = lhs.factorize();
      rhs = rhs.factorize();
      List<SimpleExpression> gcd = computeGCD(lhs, rhs);
      if ( gcd.get(0).equals(sone) )
        ret = this;
      else
        ret = divide(gcd.get(1), gcd.get(2));
    }
    return ret;
  }

  // Normalizes a MOD expression
  private SimpleExpression normalizeMOD()
  {
    if ( !allow(DIVIDE) )
      return this;
    SimpleExpression lhs = getChild(0), rhs = getChild(1), ret = null;
    if ( rhs.equals(szero) )
      ret = this;
    else if ( lhs.sop == LIT && rhs.sop == LIT )
      ret = mod(lhs, rhs);
    else if ( rhs.equals(sone) || rhs.equals(getInt(-1)) )
      ret = szero;
    if ( ret == null )
    {
      lhs = lhs.factorize();
      rhs = rhs.factorize();
      List<SimpleExpression> gcd = computeGCD(lhs, rhs);
      if ( gcd.get(0).equals(sone) )
        ret = this;
      else // (a*x)%(b*x) = (a%b)*x
        ret = multiply(gcd.get(0),
          new SimpleExpression(gcd.get(1), MOD, gcd.get(2)));
    }
    return ret;
  }

  // Normalizes a BIT operation
  private SimpleExpression normalizeBitOperation()
  {
    if ( !(getChild(0).expr instanceof IntegerLiteral &&
        getChild(1).expr instanceof IntegerLiteral) )
      return this;

    int lhs = getChild(0).getValue().intValue();
    int rhs = getChild(1).getValue().intValue();

    switch ( sop )
    {
      case SFTL: return getInt(lhs << rhs);
      case SFTR: return getInt(lhs >> rhs);
      case BAND: return getInt(lhs & rhs);
      case BOR:  return getInt(lhs | rhs);
      case BXOR: return getInt(lhs ^ rhs);
      default:
        Tools.exit("[SimpleExpression] unknown bit operation");
        return null;
    }
  }

  // Normalizes a BCMP operation
  private SimpleExpression normalizeBCMP()
  {
    if ( !(getChild(0).expr instanceof IntegerLiteral) )
      return this;

    int val = getChild(0).getValue().intValue();

    return getInt(~val);
  }

  // Normalizes an AND|OR operation
  private SimpleExpression normalizeLogic()
  {
    if ( !allow(LOGIC) )
      return this;
    TreeSet<SimpleExpression> set = new TreeSet<SimpleExpression>(children);
    TreeSet<SimpleExpression> neg = new TreeSet<SimpleExpression>();
    SimpleExpression ret = new SimpleExpression(sop);
    for ( SimpleExpression child : set )
    {
      if ( sop == AND )
      {
        if ( child.equals(szero) || neg.contains(child) )
          return szero;
        else if ( child.sop != LIT ) // ==LIT means non-zero literal.
          ret.add(child);
      }
      else // sop == OR
      {
        if ( (child.sop==LIT && !child.equals(szero)) || neg.contains(child) )
          return sone;
        else if ( child.sop != LIT ) // ==LIT means zero literal.
          ret.add(child);
      }
      neg.add(child.negate());
    }
    if ( ret.children.size() == 0 ) // skipped literals.
      ret = (ret.sop==AND)? sone: szero;
    else if ( ret.children.size() == 1 )
      ret = ret.getChild(0);
    return ret;
  }

  // Normalizes a comparison operation
  private SimpleExpression normalizeCompare()
  {
    SimpleExpression lhs = getChild(0), rhs = getChild(1);

    // Before normalization ( lhs <op> rhs )
    if ( lhs.sop == LIT && rhs.sop == LIT )
    {
      double diff = lhs.getValue().doubleValue()-rhs.getValue().doubleValue();
      switch ( sop )
      {
        case EQ: return (diff==0)? sone: szero;
        case NE: return (diff!=0)? sone: szero;
        case LE: return (diff<=0)? sone: szero;
        case LT: return (diff<0)? sone: szero;
        case GE: return (diff>=0)? sone: szero;
        case GT: return (diff>0)? sone: szero;
        default: Tools.exit("[SimpleExpression] unknown comparison expression");
      }
    }
    else if ( lhs.equals(rhs) )
      return (sop==EQ||sop==LE||sop==GE)? sone: szero;

    // Normalization ( lhs-rhs <op> 0 )
    SimpleExpression ret = new SimpleExpression(sop);
    if ( compare(lhs, rhs) < 0 )
    {
      ret.add(subtract(rhs, lhs));
      ret.sop = exchangeOp(sop);
    }
    else
      ret.add(subtract(lhs, rhs));
    ret.add(szero);

    return ret;
  }

  // Normalizes a NEG expression
  private SimpleExpression normalizeNEG()
  {
    return getChild(0).negate();
  }

  // Normalizes a MIN/MAX expression
  private SimpleExpression normalizeMINMAX()
  {
    // Literal/non-literal separation
    TreeSet<Double> literals = new TreeSet<Double>();
    TreeSet<SimpleExpression> exprs = new TreeSet<SimpleExpression>();
    SimpleExpression ret = new SimpleExpression(sop);
    for ( SimpleExpression child : children )
    {
      if ( child.sop == LIT )
        literals.add(child.getValue());
      else
        exprs.add(child);
    }
    if (!literals.isEmpty()) {
      if ( sop == MIN )
        ret.add(getInt(literals.first().intValue()));
      else
        ret.add(getInt(literals.last().intValue()));
    }
    ret.addAll(exprs);

    // Quick return for single-entry min/max.
    if ( ret.children.size() == 1 )
      return ret.getChild(0);

    // Match min(a,max(a,b))=a or max(a,min(a,b))=a
    if ( ret.sop == MIN && ret.children.size() == 2 &&
        ret.getChild(1).sop == MAX && ret.getChild(1).children.size() == 2 &&
        ret.getChild(1).children.contains(ret.getChild(0)) ||
        ret.sop == MAX && ret.children.size() == 2 &&
        ret.getChild(1).sop == MIN && ret.getChild(1).children.size() == 2 &&
        ret.getChild(1).children.contains(ret.getChild(0)) )
      return ret.getChild(0);

    return ret;
  }

  // Distributes terms; a*(b+c) --> a*b+a*c
  private SimpleExpression distribute()
  {
    if ( !allow(DISTRIBUTE) || sop != MUL || !containsChildOfType(ADD) )
      return this;
    SimpleExpression ret = sone;
    for ( SimpleExpression child : children )
    {
      SimpleExpression lhs = new SimpleExpression(szero, ADD, ret);
      SimpleExpression rhs = new SimpleExpression(szero, ADD, child);
      ret = new SimpleExpression(ADD);
      for ( SimpleExpression lhs_child : lhs.children )
        for ( SimpleExpression rhs_child : rhs.children )
          ret.add(multiply(lhs_child, rhs_child));
    }
    ret = ret.normalizeADD();
    return ret;
  }

  // Factorizes terms; a*b+a*c --> a*(b+c)
  private SimpleExpression factorize()
  {
    if ( !allow(FACTORIZE) || sop != ADD )
      return this;
    SimpleExpression ret = null, gcd = getChild(0); // for normalized form.
    for ( SimpleExpression child : children )
    {
      gcd = computeGCD(gcd, child).get(0);
      if ( gcd.equals(sone) )
      {
        ret = this;
        break;
      }
    }
    if ( ret == null )
    {
      SimpleExpression rhs = szero;
      for ( SimpleExpression child : children )
        rhs = add(rhs, computeGCD(gcd, child).get(2));
      ret = new SimpleExpression(gcd, MUL, rhs);
      ret.sort();
    }
    return ret;
  }

  // Negates this simple expression
  private SimpleExpression negate()
  {
    SimpleExpression ret = null;

    if ( sop == NEG )
      ret = getChild(0);
    else if ( sop == LIT )
      ret = (equals(szero))? sone: szero;
    else if ( sop >= EQ && sop <= GT )
    {
      ret = clone();
      ret.sop = negateOp(sop);
    }
    else if ( sop == AND || sop == OR )
    {
      ret = new SimpleExpression((sop==AND)? OR: AND);
      for ( SimpleExpression child : children )
        ret.add(child.negate());
    }
    else
    {
      ret = new SimpleExpression(NEG);
      ret.add(this);
    }

    return ret;
  }

  // Computes the least common multiple of the two simple expression.
  // Assumes se1 and se2 have been normalized.
  private static SimpleExpression
      computeLCM(SimpleExpression se1, SimpleExpression se2)
  {
    se1 = se1.factorize();
    se2 = se2.factorize();
    List<SimpleExpression> gcd = computeGCD(se1, se2);
    SimpleExpression ret = multiply(gcd.get(0), gcd.get(1));
    ret = multiply(ret, gcd.get(2));
    return ret;
  }

  // Computes the LCM of the given simple expressions.
  protected static SimpleExpression getLCM(List<SimpleExpression> ses)
  {
    SimpleExpression ret = sone;
    for ( SimpleExpression se : ses )
      ret = computeLCM(ret, se);
    return ret;
  }

  // Computes the least common denominator of the two simple expression.
  private static SimpleExpression
      computeLCD(SimpleExpression se1, SimpleExpression se2)
  {
    SimpleExpression divider1 = sone, divider2 = sone;
    if ( se1.sop == DIV )
      divider1 = se1.getChild(1);
    if ( se2.sop == DIV )
      divider2 = se2.getChild(1);
    return computeLCM(divider1, divider2);
  }

  // Computes the LCM of the given simple expressions.
  protected static SimpleExpression getLCD(List<SimpleExpression> ses)
  {
    SimpleExpression ret = sone;
    for ( SimpleExpression se : ses )
      ret = computeLCD(ret, se);
    return ret;
  }

  // Assumes se1 and se2 have been normalized.
  // Returns a triplet (gcd, dividend1, dividend2).
  private static List<SimpleExpression>
      computeGCD(SimpleExpression se1, SimpleExpression se2)
  {
    // Compute symbolic parts.
    List<SimpleExpression> terms1 = new LinkedList<SimpleExpression>();
    List<SimpleExpression> terms2 = new LinkedList<SimpleExpression>();
    SimpleExpression term1 = se1.getTerm(), term2 = se2.getTerm();
    if ( term1.sop == MUL )
      terms1.addAll(term1.children);
    else
      terms1.add(term1);
    if ( term2.sop == MUL )
      terms2.addAll(term2.children);
    else
      terms2.add(term2);
    TreeSet<SimpleExpression> gcd_terms = new TreeSet<SimpleExpression>(terms1);
    gcd_terms.retainAll(terms2);
    for ( SimpleExpression gcd_term : gcd_terms )
    {
      terms1.remove(gcd_term); // removes the first occurrence
      terms2.remove(gcd_term); //   of the gcd term
    }
    // Compute numeric parts.
    SimpleExpression s1 = se1.getCoef(), s2 = se2.getCoef(), sgcd=null;
    if (s1.expr instanceof IntegerLiteral && s2.expr instanceof IntegerLiteral)
    {
      int n1 = s1.getValue().intValue(), n2 = s2.getValue().intValue();
      int gcd = cetus.analysis.GCD.compute(n1, n2);
      s1 = getInt(n1/gcd);
      s2 = getInt(n2/gcd);
      sgcd = getInt(gcd);
    }
    else
      sgcd = sone;
    // Combine two parts.
    for ( SimpleExpression child : gcd_terms )
      sgcd = multiply(sgcd, child);
    for ( SimpleExpression child : terms1 )
      s1 = multiply(s1, child);
    for ( SimpleExpression child : terms2 )
      s2 = multiply(s2, child);

    List<SimpleExpression> ret = new LinkedList<SimpleExpression>();
    ret.add(sgcd);
    ret.add(s1);
    ret.add(s2);
    return ret;
  }

  // Removes division by replacing with modulus operations.
  // The returned list contains the modified simple expression (get(0)) and
  // the additionally multiplied value (get(1)).
  // e.g., a*(b/c)*(d/e) returns {a*(b-b%c)*(d-d%e), c*e}.
  // It is important to notice that the legality check of this transformation
  // is up to the callers.
  // It is assumed that the original simple expressions has been normalized.
  protected List<SimpleExpression> multiplyByLCM()
  {
    List<SimpleExpression> ret = new ArrayList<SimpleExpression>(2);
    if ( sop == DIV )
    {
      ret.add(subtract(getChild(0), mod(getChild(0), getChild(1))));
      ret.add(getChild(1));
    }
    else if ( sop == ADD )
    {
      List<SimpleExpression> terms = new LinkedList<SimpleExpression>();
      List<SimpleExpression> factors = new LinkedList<SimpleExpression>();
      SimpleExpression lcm = sone;
      for ( SimpleExpression child : children )
      {
        List<SimpleExpression> ret0 = child.multiplyByLCM();
        terms.add(ret0.get(0));
        factors.add(ret0.get(1));
        lcm = computeLCM(lcm, ret0.get(1));
      }
      SimpleExpression ret1 = new SimpleExpression(ADD);
      for ( int i=0; i<terms.size(); i++ )
        ret1.add(multiply(divide(lcm, factors.get(i)), terms.get(i)));
      ret.add(ret1.normalize());
      ret.add(lcm);
    }
    else if ( sop == MUL )
    {
      SimpleExpression terms = new SimpleExpression(MUL);
      SimpleExpression factors = sone;
      for ( SimpleExpression child : children )
      {
        List<SimpleExpression> ret0 = child.multiplyByLCM();
        terms.add(ret0.get(0));
        factors = multiply(factors, ret0.get(1));
      }
      ret.add(terms.normalize());
      ret.add(factors);
    }
    else
    {
      ret.add(this);
      ret.add(sone);
    }
    return ret;
  }

  // Aggressively normalize divisible expressions to minimize ADD operations.
  // This method is called only by induction variable substitution where
  // the divisibility of an expression is defined well.
  protected SimpleExpression normalizeDivisible()
  {
    SimpleExpression ret = this;
    if ( sop == DIV )
    {
      if ( getChild(0).sop == DIV )
        ret = divide(getChild(0).getChild(0),
            multiply(getChild(0).getChild(1), getChild(1)));
    }
    else if ( sop == MUL )
    {
      ret = toDivision().normalize();
    }
    else if ( sop == ADD )
    {
      ret = toDivision();
      if ( ret.sop == DIV && ret.getChild(0).sop == ADD )
      {
        SimpleExpression non_div = szero, dividend = szero;
        SimpleExpression divider = ret.getChild(1);
        for ( SimpleExpression child : ret.getChild(0).children )
        {
          SimpleExpression divided = divide(child, divider);
          if ( divided.sop == DIV )
            dividend = add(dividend, child);
          else
            non_div = add(non_div, divided);
        }
        if ( non_div.equals(szero) &&
            ret.countsOperations(ADD) >= countsOperations(ADD) )
          ret = this; // heuristics: no benefit from the simplification.
        else
          ret = add(non_div, divide(dividend, divider));
      }
    }

    return ret;
  }

  // Converts ADD and MUL to division. e.g., a/2+1 --> (a+2)/2
  // This is called only when the expression is divisible.
  protected SimpleExpression toDivision()
  {
    SimpleExpression ret;
    if ( sop == ADD )
    {
      List<SimpleExpression> converted = new LinkedList<SimpleExpression>();
      for ( SimpleExpression child : children )
        converted.add(child.toDivision());
      SimpleExpression lcd = getLCD(converted);
      if ( lcd.equals(sone) )
        ret = this;
      else
      {
        SimpleExpression dividend = szero;
        for ( SimpleExpression child : converted )
        {
          if ( child.sop == DIV )
            dividend = add(dividend, multiply(child.getChild(0),
              divide(lcd, child.getChild(1))));
          else
            dividend = add(dividend, multiply(child, lcd));
        }
        ret = divide(dividend, lcd);
      }
    }
    else if ( sop == MUL )
    {
      SimpleExpression dividend = sone, divider = sone;
      for ( SimpleExpression child : children )
      {
        if ( child.sop == DIV )
        {
          dividend = multiply(dividend, child.getChild(0));
          divider = multiply(divider, child.getChild(1));
        }
        else
          dividend = multiply(dividend, child);
      }
      if ( divider.equals(sone) )
        ret = this;
      else
        ret = divide(dividend, divider);
    }
    else
      ret = this;

    return ret;
  }

  // Returns a negated comparison operator
  protected static int negateOp(int op)
  {
    switch ( op )
    {
      case EQ: return NE;
      case NE: return EQ;
      case LE: return GT;
      case LT: return GE;
      case GE: return LT;
      case GT: return LE;
      default: return op;
    }
  }

  // Returns an exchanged comparison operator
  protected static int exchangeOp(int op)
  {
    switch ( op )
    {
      case LE: return GE;
      case LT: return GT;
      case GE: return LE;
      case GT: return LT;
      default: return op;
    }
  }

  // Returns the current normalization option
  protected static int getOption()
  {
    return option;
  }

  // Returns the order of the simple expression
  protected int getOrder()
  {
    return order;
  }

  // Returns the Cetus expression reference in the simple expression
  protected Expression getExprRef()
  {
    return expr;
  }

  // Returns the operator
  protected int getOP()
  {
    return sop;
  }

  // Returns the list of children
  protected List<SimpleExpression> getChildren()
  {
    return children;
  }

  // Returns the Cetus operator for this simple expression
  protected static Object getCetusOP(int op)
  {
    return cop.get(op);
  }

}
