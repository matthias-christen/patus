package cetus.analysis;

import java.util.*;

import cetus.hir.*;

import cetus.exec.*;

/**
 * Class Section represents a list of array subscripts that expresses a subset
 * of the whole array elements. Each element in the list should be an object
 * of {@link Section.ELEMENT}.
 *
 * @see Section.ELEMENT
 * @see Section.MAP
 */
public class Section extends ArrayList<Section.ELEMENT> implements Cloneable
{
  private static final long serialVersionUID = 12L;
  // Dimension
  private int dimension;

  /**
   * Constructs a section with the specified dimension.
   *
   * @param dimension the dimension of the array. -1 for scalar variables.
   */
  public Section(int dimension)
  {
    super();
    this.dimension = dimension;
  }

  /**
   * Constructs a section with the specified array access.
   *
   * @param acc the array access expression.
   */
  public Section(ArrayAccess acc)
  {
    this(acc.getNumIndices());

    add(new ELEMENT(acc));
  }


  /**
   * Clones a section object.
   *
   * @return the cloned section.
   */
  public Object clone()
  {
    Section o = new Section(dimension);

    // Make a deep copy since ArrayList makes only a shallow copy.
    for ( ELEMENT elem : this )
      o.add((ELEMENT)elem.clone());

    return o;
  }


  /**
   * Adds a new element in the section.
   *
   * @param elem the new element to be added.
   * @return true (as per the general contract of Collection.add).
   */
  public boolean add(ELEMENT elem)
  {
    if ( !contains(elem) )
      super.add(elem);

    return true;
  }

  /**
   * returns a dimension
   */
  public int getDimension()
  {
    return dimension;
  }

  /**
   * Checks if the section is for a scalar variable.
   */
  public boolean isScalar()
  {
    return ( isEmpty() && dimension == -1 );
  }


  /**
   * Checks if the section is for an array variable.
   */
  public boolean isArray()
  {
    return ( dimension > 0 );
  }


  /**
   * Checks if the section contains the specified variables.
   */
  public boolean containsSymbols(Set<Symbol> vars)
  {
    for ( ELEMENT elem : this )
      for ( Expression e : elem )
        if ( IRTools.containsSymbols(e, vars) )
          return true;
    return false;
  }


  /**
   * Expand every section under the constraints given by the range domain.
   *
   * @param rd the given range domain.
   * @param vars the set of symbols to be expanded.
   */
  public void expandMay(RangeDomain rd, Set<Symbol> vars)
  {
    PrintTools.println("  [Section.expandMay] vars = " + PrintTools.collectionToString(vars, ","), 5);
    PrintTools.println("                      rd = " + rd.toString(), 5);
    for ( ELEMENT elem : this )
      for ( int i=0; i<dimension; ++i )
      {
        PrintTools.println("  [Section.expandMay] section = " + elem.get(i).toString(), 5);

        Expression expanded = rd.expandSymbols(elem.get(i), vars);
        PrintTools.println("  [Section.expandMay] expanded = " + expanded.toString(), 5);
        elem.set(i, expanded);
      }
    simplifyMe();
  }

  /**
   * Expand every section under the constraints given by the range domain.
   *
   * @param rd the given range domain.
   * @param ivs the set of symbols to be expanded.
   * @param vars the set of symbols that should not be part of the expansion.
   */
  public void expandMust(RangeDomain rd, Set<Symbol> ivs, Set<Symbol> vars)
  {
    Iterator<ELEMENT> iter = iterator();
    while ( iter.hasNext() )
    {
      ELEMENT elem = iter.next();

      for ( int i=0; i<dimension; ++i )
      {
        Expression expanded = rd.expandSymbols(elem.get(i), ivs);

        if ( expanded == null ||
        expanded instanceof RangeExpression &&
        !((RangeExpression)expanded).isBounded() ||
        IRTools.containsSymbols(expanded, vars) )
        {
          iter.remove();
          break;
        }

        elem.set(i, expanded);
      }
    }
    simplifyMe();
  }

  /**
   * Substitutes any variables having symbolic constant values in the section
   * avoiding cases that produces expression having a variable in the
   * "avoid" set.
   */
  public void substituteForward(RangeDomain rd, Set<Symbol> avoid)
  {
    for ( ELEMENT elem : this )
    {
      for ( int i=0; i<dimension; ++i )
      {
        Expression substituted = rd.substituteForward(elem.get(i));
        if ( !IRTools.containsSymbols(substituted, avoid) )
          elem.set(i, substituted);
      }
    }
  }

  /**
   * Performs intersection operation between two sections with the specified
   * range domain.
   *
   * @param other the section being intersected with.
   * @param rd the supporting range domain.
   * @return the resulting section.
   */
  public Section intersectWith(Section other, RangeDomain rd)
  {
    // No intersection is possible; returns null (check at the higher level)
    if ( dimension != other.dimension )
      return null;

    Section ret = new Section(dimension);

    for ( ELEMENT elem1 : this )
    {
      for ( ELEMENT elem2 : other )
      {
        ELEMENT intersected = elem1.intersectWith(elem2, rd);

        if ( intersected != null )
          ret.add(intersected);
      }

      if ( ret.isEmpty() )
        break;
    }

    ret = ret.simplify();

    PrintTools.printlnStatus(this+" (^) "+other+" = "+ret+" under "+rd, 5);

    return ret;
  }


  /**
   * Performs union operation between two sections with the specified range
   * domain.
   *
   * @param other the section being unioned with.
   * @param rd the supporting range domain.
   * @return the resulting section.
   */
  public Section unionWith(Section other, RangeDomain rd)
  {
    if ( dimension != other.dimension )
      return null;

    Section ret = new Section(dimension);

    Iterator<ELEMENT> iter1 = iterator();
    Iterator<ELEMENT> iter2 = other.iterator();

    while ( iter1.hasNext() || iter2.hasNext() )
    {
      if ( !iter1.hasNext() )
        ret.add((ELEMENT)iter2.next().clone());

      else if ( !iter2.hasNext() )
        ret.add((ELEMENT)iter1.next().clone());

      else
      {
        ELEMENT elem1 = iter1.next(), elem2 = iter2.next();

        ELEMENT unioned = elem1.unionWith(elem2, rd);

        if ( unioned == null ) // union was not merged
        {
          ret.add((ELEMENT)elem1.clone());
          ret.add((ELEMENT)elem2.clone());
        }
        else                   // union was merged
          ret.add(unioned);
      }
    }

    ret = ret.simplify();

    PrintTools.printlnStatus(this+" (v) "+other+ " = "+ret+" under "+rd, 5);

    return ret;
  }


  /**
   * Performs difference operation between two sections with the specified
   * range domain.
   *
   * @param other the other section from which this section is differenced.
   * @param rd the supporting range domain.
   * @return the resulting section.
   */
  public Section differenceFrom(Section other, RangeDomain rd)
  {
    String mesg = this+" (-) "+other+" = ";

    Section ret = (Section)clone();

    // Just return a clone upon dimension mismatch
    if ( dimension != other.dimension )
    {
      PrintTools.printlnStatus(mesg+ret, 5);
      return ret;
    }

    for ( ELEMENT elem2 : other )
    {
      Section curr = new Section(dimension);

      for ( ELEMENT elem1 : ret )
      {
        Section diffed = elem1.differenceFrom(elem2, rd);

        for ( ELEMENT elem : diffed )
          //if ( !curr.contains(elem) )
          curr.add(elem);
      }

      ret = curr;
    }

    ret = ret.simplify();

    PrintTools.printlnStatus(mesg+ret+" under "+rd, 5);

    return ret;
  }


  /**
   * Returns union of two symbolic bounds
   */
  private static Expression unionBound
  (Expression e1, Expression e2, RangeDomain rd)
  {
    Expression intersected = intersectBound(e1, e2, rd);

    //System.out.println("intersected = "+intersected);

    if ( intersected == null ) // Either it has no intersection or unknown.
      return null;             // Merging i,i+1 => i:i+1 disregarded for now.

    RangeExpression re1 = RangeExpression.toRange(e1);
    RangeExpression re2 = RangeExpression.toRange(e2);

    Expression lb = null, ub = null;

    Relation rel = rd.compare(re1.getLB(), re2.getLB());

    if ( rel.isLE() )
      lb = re1.getLB();
    else if ( rel.isGE() )
      lb = re2.getLB();
    else
      return null;

    rel = rd.compare(re1.getUB(), re2.getUB());

    if ( rel.isGE() )
      ub = re1.getUB();
    else if ( rel.isLE() )
      ub = re2.getUB();
    else
      return null;

    return (new RangeExpression(lb.clone(), ub.clone())).toExpression();
  }


  /**
   * Returns intersection of two symbolic intervals
   */
  private static Expression intersectBound
  (Expression e1, Expression e2, RangeDomain rd)
  {
    RangeExpression re1 = RangeExpression.toRange(e1);
    RangeExpression re2 = RangeExpression.toRange(e2);

    Expression lb = null, ub = null;

    Relation rel = rd.compare(re1.getLB(), re2.getLB());

    if ( rel.isGE() )
      lb = re1.getLB();
    else if ( rel.isLE() )
      lb = re2.getLB();
    else
      return null;

    rel = rd.compare(re1.getUB(), re2.getUB());

    if ( rel.isLE() )
      ub = re1.getUB();
    else if ( rel.isGE() )
      ub = re2.getUB();
    else
      return null;

    // Final check if lb>ub.
    rel = rd.compare(lb, ub);

    if ( !rel.isLE() )
      return null;
    else
      return (new RangeExpression(lb.clone(), ub.clone())).toExpression();

    /* temporary fix for the replacement of the above commented section */
    /* use this fix with caution - it may break many innocent things.
    return (new RangeExpression(lb, ub)).toExpression();
    */
  }


  /**
   * Removes section elements that contain the specified variable.
   */
  public void removeAffected(Symbol var)
  {
    Iterator<ELEMENT> iter = iterator();
    while ( iter.hasNext() )
    {
      boolean kill = false;

      for ( Expression e : iter.next() )
      {
        if ( IRTools.containsSymbol(e, var) )
        {
          kill = true;
          break;
        }
      }

      if ( kill )
        iter.remove();
    }
  }


  /**
   * Removes section elements that is affected by the specified function call.
   */
  public void removeSideAffected(FunctionCall fc)
  {
    Set<Symbol> params = SymbolTools.getAccessedSymbols(fc);
    Iterator<ELEMENT> iter = iterator();
    while ( iter.hasNext() )
    {
      boolean kill = false;

      for ( Expression e : iter.next() )
      {
        Set<Symbol> vars = SymbolTools.getAccessedSymbols(e);
        vars.retainAll(params);
        // Case 1: variables in section representation are used as parameters.
        if ( !vars.isEmpty() )
        {
          kill = true;
          break;
        }
        // Case 2: variables in section representation are global.
        for ( Symbol var : SymbolTools.getAccessedSymbols(e) )
        {
          if ( SymbolTools.isGlobal(var, fc) )
          {
            kill = true;
            break;
          }
        }
        if ( kill )
          break;
      }
      if ( kill )
        iter.remove();
    }
  }

  /**
   * Converts this section to a string.
   *
   * @return the string representation of the section.
   */
  public String toString()
  {
    return ( "{" + PrintTools.listToString(this, ", ") + "}" );
  }

  // Simplify the adjacent section elements.
  private Section simplify()
  {
    if ( dimension < 1 )
      return this;

    Section ret = null;

    // Adjacenct elements. [1][0],[1][1] => [1][0:1]
    for ( int i=0; i<dimension; ++i )
    {
      Section temp = (ret==null)? (Section)this.clone(): ret;

      ret = new Section(dimension);

      while ( !temp.isEmpty() )
      {
        ELEMENT elem1 = temp.remove(0);
        Iterator<ELEMENT> iter = temp.iterator();
        while ( iter.hasNext() )
        {
          ELEMENT elem2 = iter.next();
          if ( elem2.isAdjacentWest(elem1, i) )
          {
            RangeExpression re1 = RangeExpression.toRange(elem1.get(i));
            RangeExpression re2 = RangeExpression.toRange(elem2.get(i));
            elem1.set(i, new RangeExpression(
                re2.getLB().clone(), re1.getUB().clone()));
            iter.remove();
          }
          else if ( elem2.isAdjacentEast(elem1, i) )
          {
            RangeExpression re1 = RangeExpression.toRange(elem1.get(i));
            RangeExpression re2 = RangeExpression.toRange(elem2.get(i));
            elem1.set(i, new RangeExpression(
                re1.getLB().clone(), re2.getUB().clone()));
            iter.remove();
          }
        }
        ret.add(elem1);
      }
    }

    // Enclosed elements. [1][0:1],[1][0] => [1][0:1]
    RangeDomain rd = new RangeDomain();
    Section temp = (Section)ret.clone();
    ret.clear();
    while ( !temp.isEmpty() )
    {
      ELEMENT elem1 = temp.remove(0);
      Iterator<ELEMENT> iter = temp.iterator();
      while ( iter.hasNext() )
      {
        ELEMENT elem2 = iter.next();
        if ( elem2.enclose(elem1, rd) )
        {
          elem1 = elem2;
          iter.remove();
        }
        else if ( elem1.enclose(elem2, rd) )
          iter.remove();
      }
      ret.add(elem1);
    }
    return ret;
  }

  // In-place simplification.
  private void simplifyMe()
  {
    Section simplified = this.simplify();
    this.clear();
    this.addAll(simplified);
  }

  // Returns predicates from the unresolved set of bounds from the section.
  private RangeDomain getPredicates(RangeDomain rd)
  {
    RangeDomain ret = new RangeDomain();
    if ( !isArray() )
      return ret;

    for ( ELEMENT elem : this )
      for ( Expression e : elem )
        if ( e instanceof RangeExpression )
        {
          RangeExpression re = (RangeExpression)e;
          Relation rel = rd.compare(re.getLB(), re.getUB());
          if ( !rel.isLE() )
          {
            ret.intersectRanges(RangeAnalysis.extractRanges(
              Symbolic.le(re.getLB(), re.getUB())));
          }
        }
    return ret;
  }


  /**
   * Represents the elements contained in a section.
   */
  public static class ELEMENT extends ArrayList<Expression>
  implements Cloneable
  {
    private static final long serialVersionUID = 13L;
    /**
     * Constructs an empty element.
     */
    public ELEMENT()
    {
      super();
    }

    /**
     * Constructs a new element from the given array access.
     *
     * @param acc the array access from which the new element is constructed.
     */
    public ELEMENT(ArrayAccess acc)
    {
      for ( int i=0; i < acc.getNumIndices(); ++i )
        add(Symbolic.simplify(acc.getIndex(i)));
    }

    /**
     * Returns a clone of this element.
     *
     * @return the cloned object.
     */
    public Object clone()
    {
      ELEMENT o = new ELEMENT();

      for ( Expression e : this )
        o.add(e.clone());

      return o;
    }

    /**
     * Checks if this element is equal to the specified object.
     *
     * @param o the object to be compared with.
     */
    public boolean equals(Object o)
    {
      if ( o == null || o.getClass() != this.getClass() )
        return false;

      ELEMENT other = (ELEMENT)o;

      if ( size() != other.size() )
        return false;

      for ( int i=0; i < size(); ++i )
      {
        if ( !get(i).equals(other.get(i)) )
          return false;
      }

      return true;
    }

    // Checks if this element is adjacent west to the other element in the
    // specified dimension while other subscripts are identical in the other
    // dimensions. [0][2:5] isAdjacentWest to [1][2:5].
    private boolean isAdjacentWest(ELEMENT other, int loc)
    {
      return ( isAdjacentTo(other, loc ) == -1 );
    }

    // Checks if this element is adjacent east to the other element.
    private boolean isAdjacentEast(ELEMENT other, int loc)
    {
      return ( isAdjacentTo(other, loc ) == 1 );
    }

    // Base check method for isAdjacent method.
    private int isAdjacentTo(ELEMENT other, int loc)
    {
      if ( other == null || size() != other.size() )
        return 0;

      // All subscripts in the other dimensions should be equal to the other
      // elements' subscripts.
      for ( int i=0; i<size(); ++i )
        if ( i != loc && !get(i).equals(other.get(i)) )
          return 0;

      RangeExpression re1 = RangeExpression.toRange(get(loc));
      RangeExpression re2 = RangeExpression.toRange(other.get(loc));

      Expression e12 = Symbolic.subtract(re2.getLB(), re1.getUB());
      Expression e21 = Symbolic.subtract(re1.getLB(), re2.getUB());
      Expression one = new IntegerLiteral(1);

      if ( e12.equals(one) )
        return -1; // UB(e1)+1 = LB(e2)
      else if ( e21.equals(one) )
        return 1;  // UB(e2)+1 = LB(e1);
      else
        return 0;
    }

    // Check if this element encloses the other.
    private boolean enclose(ELEMENT other, RangeDomain rd)
    {
      for ( int i=0; i<size(); i++ )
      {
        RangeExpression re0 = RangeExpression.toRange(get(i));
        RangeExpression re1 = RangeExpression.toRange(other.get(i));
        Expression lb0 = re0.getLB(), ub0 = re0.getUB();
        Expression lb1 = re1.getLB(), ub1 = re1.getUB();
        if ( rd.isLE(lb0, lb1) && rd.isGE(ub0, ub1) )
          ; // it encloses !!!
        else
          return false;
      }
      return true;
    }

    /**
     * Converts this element to a string.
     *
     * @return the string representation of this element.
     */
    public String toString()
    {
      StringBuilder str = new StringBuilder(80);

      str.append("[");

      for ( int i=0; i<size(); i++ )
      {
        if ( i > 0 ) str.append("][");
        Expression e = get(i);
        if ( e instanceof RangeExpression )
        {
          RangeExpression re = (RangeExpression)e;
          str.append(re.getLB()+":"+re.getUB());
        }
        else
          str.append(e.toString());
      }

      str.append("]");

      return str.toString();
    }

    /**
     * Performs intersection operation between two section elements with the
     * specified range domain.
     *
     * @param other the other element.
     * @param rd the specified range domain.
     * @return the result of the intersection.
     */
    public ELEMENT intersectWith(ELEMENT other, RangeDomain rd)
    {
      ELEMENT ret = new ELEMENT();

      for ( int i=0; i < size(); ++i )
      {
        Expression intersected = intersectBound(get(i), other.get(i), rd);

        if ( intersected == null ) // Either it is empty or unknown
          return null;

        ret.add(intersected);
      }

      return ret;
    }

    /**
     * Performs union operation between two section elements with the
     * specified range domain.
     *
     * @param other the other element.
     * @param rd the specified range domain.
     * @return the result of the union.
     */
    public ELEMENT unionWith(ELEMENT other, RangeDomain rd)
    {
      ELEMENT ret = new ELEMENT();

      for ( int i=0; i < size(); ++i )
      {
        Expression unioned = unionBound(get(i), other.get(i), rd);

        if ( unioned == null ) // Either it has holes or unknown
          return null;

        ret.add(unioned);
      }

      return ret;
    }

    /**
     * Performs difference operation between two section elements with the
     * specified range domain.
     *
     * @param other the other element.
     * @param rd the specified range domain.
     * @return the resulting section of the difference.
     */
    public Section differenceFrom(ELEMENT other, RangeDomain rd)
    {
      // Temporary list containing the result of differences for each dimension
      Section ret = new Section(size());

      // Process easy case: other encloses this.
      if ( other.enclose(this, rd) )
        return ret;

      for ( int i=0; i < size(); ++i )
      {
        List<Expression> temp_i = new ArrayList<Expression>();

        Expression intersected = intersectBound(get(i), other.get(i), rd);

        //System.out.println("intersected="+intersected);

        if ( intersected == null )
          temp_i.add(get(i).clone());

        else
        {
          RangeExpression re_inct = RangeExpression.toRange(intersected);
          RangeExpression re_from = RangeExpression.toRange(get(i));
          Expression one = new IntegerLiteral(1);

          Expression left_ub = Symbolic.subtract(re_inct.getLB(),one);
          Expression right_lb = Symbolic.add(re_inct.getUB(), one);

          Relation rel = rd.compare(re_from.getLB(), left_ub);

          if ( !rel.isGT() )
            temp_i.add(
            (new RangeExpression(re_from.getLB().clone(), left_ub))
            .toExpression());

          rel = rd.compare(right_lb, re_from.getUB());

          if ( !rel.isGT() )
            temp_i.add(
            (new RangeExpression(right_lb, re_from.getUB().clone()))
            .toExpression());
        }

        for ( Expression e : temp_i )
        {
          ELEMENT new_section = (ELEMENT)clone();
          new_section.set(i, e);
          ret.add(new_section);
        }
      }

      return ret;
    }
  }


  /**
   * Class MAP represents map from variables to their sections. For the
   * convenience of implementation, we assign empty section for scalar
   * variables.
   */
  public static class MAP extends HashMap<Symbol,Section> implements Cloneable
  {
    private static final long serialVersionUID = 14L;
    /**
     * Constructs an empty map.
     */
    public MAP()
    {
      super();
    }

    /**
     * Constructs a map with a pair of variable and section.
     *
     * @param var the key variable.
     * @param section the section associated with the variable.
     */
    public MAP(Symbol var, Section section)
    {
      super();
      put(var, section);
    }

    /**
     * Returns a clone object.
     */
    public Object clone()
    {
      MAP o = new MAP();

      for ( Symbol var : keySet() )
        o.put(var, (Section)get(var).clone());

      return o;
    }

    /**
     * Cleans up empty sections.
     */
    public void clean()
    {
      Set<Symbol> vars = new HashSet<Symbol>(keySet());

      for ( Symbol var : vars )
        if ( var == null || get(var).dimension > 0 && get(var).isEmpty() )
          remove(var);
    }

    /**
     * Performs intersection operation between the two section maps with the
     * specified range domain.
     *
     * @param other the other section map to be intersected with.
     * @param rd the specified range domain.
     * @return the resulting section map after intersection.
     */
    public MAP intersectWith(MAP other, RangeDomain rd)
    {
      MAP ret = new MAP();

      if ( other == null )
        return ret;

      for ( Symbol var : keySet() )
      {
        Section s1 = get(var);
        Section s2 = other.get(var);

        if ( s1 == null || s2 == null )
          continue;

        if ( s1.isScalar() && s2.isScalar() )
          ret.put(var, (Section)s1.clone());

        //else if ( !s1.isScalar() && !s2.isScalar() )
        else
        {
          Section intersected = s1.intersectWith(s2, rd);

          if ( intersected == null )
            PrintTools.printlnStatus("[WARNING] Dimension mismatch", 0);

          else
            ret.put(var, intersected);
        }
      }

      ret.clean();
      return ret;
    }

    /**
     * Performs union operation between the two section maps with the
     * specified range domain.
     *
     * @param other the other section map to be united with.
     * @param rd the specified range domain.
     * @return the resulting section map after union.
     */
    public MAP unionWith(MAP other, RangeDomain rd)
    {
      if ( other == null )
        return (MAP)clone();

      MAP ret = new MAP();

      Set<Symbol> vars = new HashSet<Symbol>(keySet());
      vars.addAll(other.keySet());

      for ( Symbol var : vars )
      {
        Section s1 = get(var);
        Section s2 = other.get(var);

        if ( s1 == null && s2 == null )
          continue;

        if ( s1 == null )
        {
          ret.put(var, (Section)s2.clone());
        }
        else if ( s2 == null )
          ret.put(var, (Section)s1.clone());
        else if ( s1.isScalar() && s2.isScalar() )
          ret.put(var, (Section)s1.clone());
        else
        {
          Section unioned = s1.unionWith(s2, rd);

          if ( unioned == null )
            ret.put(var, (Section)s2.clone()); // heuristics -- second operand
          else
            ret.put(var, unioned);
        }
      }

      ret.clean();
      return ret;
    }

    /**
     * Performs difference operation between the two section maps with the
     * specified range domain.
     *
     * @param other the other section map to be differenced from.
     * @param rd the specified range domain.
     * @return the resulting section map after difference.
     */
    public MAP differenceFrom(MAP other, RangeDomain rd)
    {
      if ( other == null )
        return (MAP)clone();

      MAP ret = new MAP();

      Set<Symbol> vars = new HashSet<Symbol>(keySet());

      for ( Symbol var : vars )
      {
        Section s1 = get(var);
        Section s2 = other.get(var);

        if ( s2 == null )
          ret.put(var, (Section)s1.clone());

        //else if ( !s1.isScalar() )
        else if ( s1.isArray() || s2.isArray() )
          ret.put(var, s1.differenceFrom(s2, rd));
      }

      ret.clean();
      return ret;
    }

    /**
     * Performs conditional difference operation after adding unresolved
     * bound relation from the current section map. This operation enhances
     * the coverage of difference operation if there exist some bound
     * expressions that are not guaranteed to be true (lb<=ub). This should
     * be used only for upward-exposed set computation.
     */
    public MAP differenceFrom2(MAP other, RangeDomain rd)
    {
      if ( other == null )
        return (MAP)clone();

      MAP ret = new MAP();
      Set<Symbol> vars = new HashSet<Symbol>(keySet());

      for ( Symbol var : vars )
      {
        Section s1 = get(var), s2 = other.get(var);

        if ( s2 == null )
          ret.put(var, (Section)s1.clone());

        else if ( s1.isArray() || s2.isArray() )
        {
          RangeDomain modified = rd.clone();
          RangeDomain predicates = s1.getPredicates(rd);
          for ( Symbol symbol : predicates.getSymbols() )
            modified.setRange(symbol, predicates.getRange(symbol));
          ret.put(var, s1.differenceFrom(s2, modified));
        }
      }
      ret.clean();
      return ret;
    }

    /**
     * Removes sections that contains the specified symbol.
     */
    public void removeAffected(Symbol var)
    {
      Set<Symbol> keys = new HashSet<Symbol>(keySet());

      for ( Symbol key : keys )
        get(key).removeAffected(var);

      clean();
    }

    /**
     * Removes sections that contains the specified set of variables.
     */
    public void removeAffected(Collection<Symbol> vars)
    {
      for ( Symbol var : vars )
        removeAffected(var);
    }

    /**
     * Removes sections that are unsafe in the given traversable object due to
     * function calls.
     */
    public void removeSideAffected(Traversable tr)
    {
      DepthFirstIterator iter = new DepthFirstIterator(tr);

      iter.pruneOn(FunctionCall.class);

      while ( iter.hasNext() )
      {
        Object o = iter.next();
        if ( o instanceof FunctionCall )
        {
          Set<Symbol> vars = new HashSet<Symbol>(keySet());

          for ( Symbol var : vars )
            get(var).removeSideAffected((FunctionCall)o);

          clean();
        }
      }
    }

    public void print(String str) { print(str, 7); }

    public void print(String str, int verbosity)
    {
      if (isEmpty())
        PrintTools.println(str + " is empty", verbosity);
      else
      {
        if ( keySet() == null )
          PrintTools.println(str + " keySet() is null", verbosity);
        else if ( keySet().size() == 0 )
          PrintTools.println(str + " keySet() is empty", verbosity);
        else
        {
          PrintTools.print(str + " = [", verbosity);
          int count=0;
          for ( Symbol symbol : keySet() )
          {
            if (++count > 1) PrintTools.print(", ", verbosity);
            Section section = get(symbol);
            if (section.getDimension() < 1)    // scalar variable
              PrintTools.print(symbol.getSymbolName(), verbosity);  
            else
              PrintTools.print(symbol.getSymbolName() + "=" + section.toString(), verbosity);  
          }
          PrintTools.println("]", verbosity);
        }
      }
    }

    public String toString()
    {
      StringBuilder str = new StringBuilder(80);

      if ( isEmpty() )
        str.append("{}");
      else
      {
        str.append(str + " = [");
        int count=0;
        for ( Symbol symbol : keySet() )
        {
          if (++count > 1) str.append(", ");

          Section section = get(symbol);
          if (section.getDimension() < 1)    // scalar variable
            str.append(symbol.getSymbolName());  
          else
            str.append(symbol.getSymbolName() + "=" + section.toString());  
        }
        str.append("]");
      }
      return str.toString();
    }
  }

}
