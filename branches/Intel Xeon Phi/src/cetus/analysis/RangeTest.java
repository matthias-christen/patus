package cetus.analysis;

import java.util.*;
import java.io.*;
import cetus.hir.*;

/**
 * Symbolic Range Test for disproving loop-carried dependences between two
 * subscript expressions. Unlike the Banerjee-Wolfe test, the Range test does
 * not solve the problem under the all possible direction vectors; it disproves
 * dependences between a subscript pairs. Because of the DDT driver calling a
 * DDT solver with a specific direction vector, the Range test should also
 * return a correct answer when given such a direction vector. To address this
 * issue, the Range test records a list of direction vectors that do not incur
 * any dependences once it disproves a dependence and returns an appropriate
 * answer based on the "independent direction vectors".
 * The Range test utilizes the power of the symbolic range analysis as much as
 * possible when performing the test. For example, when computing minimum and
 * maximum symbolic values of a subscript expression at any given iteration, it
 * just expands the index variables of the enclosing loops using the utility
 * methods given by RangeDomain.
 */
public class RangeTest implements DDTest
{
/*-----------------------------------------------------------------------------/
  Static fields.
/-----------------------------------------------------------------------------*/
  // Static reference to the current procedure
  private static Procedure procedure = null;

  // Static range map for the current procedure
  private static Map<Statement, RangeDomain> range_map = null;

  // Expressions' monotonisties
  private static final int
    MONO_NONINC  =-1, // monotonically non-increasing
    MONO_CONST   = 0, // constant
    MONO_NONDEC  = 1, // monotonically non-decreasing
    MONO_FLIP    = 2, // is not monotonic
    MONO_UNKNOWN = 3; // unknown

  // Which test has passed
  private static final String
    TEST1_PASS = "T1", // test1 has passed
    TEST2_PASS = "T2"; // test2 has passed

  // Mask for direction vectors
  private static final int
    DV_ANY = 1,
    DV_LT  = 2,
    DV_GT  = 4,
    DV_EQ  = 8;

  // Loop variants cache
  private static Cache<Loop, Set<Symbol>>
    loop_variants = new Cache<Loop, Set<Symbol>>();

  // Expression range cache.
  private static Cache<List<Object>, Expression>
    expr_range_cache = new Cache<List<Object>, Expression>();

  // Tag
  private static final String tag = "[RangeTest] ";

  // Debug level
  private static final int verbosity = 
    Integer.valueOf(cetus.exec.Driver.getOptionValue("verbosity")).intValue();

  // Test map for measuring redundant computations.
  private static Set<String> get_range_history = new LinkedHashSet<String>();
  private static int get_range_called = 0;
  private static int get_range_exist = 0;

/*-----------------------------------------------------------------------------/
  Object fields.
/-----------------------------------------------------------------------------*/
  // Subscript expressions f and g
  private Expression f, g;

  // Statements that contain f and g in the IR.
  private Statement f_stmt, g_stmt;

  // Common range domain for f and g
  private RangeDomain f_range, g_range, common_range;

  // Common/local loop nest
  private LinkedList<Loop>
    common_loops,        // enclose both f and g
    f_loops,             // enclose only f
    g_loops,             // enclose only g
    pseudo_common_loops; // separately enclose f and g with the same loopinfo.

  // Relevant loops that enclose both f and g, and index vars appear in f or g
  private LinkedHashSet<Loop> relevant_loops;

  // Parallel loops found during the test; map value is whether test1 passed
  // If test1 passed no dependence on * vector.
  private LinkedHashMap<Loop, String> parallel_loops;

  // Set of dependence vectors on which a data dependence does not exist.
  private int[] independent_vectors;

  // Problem already solved
  private boolean was_solved;

/*-----------------------------------------------------------------------------/
  Methods.
/-----------------------------------------------------------------------------*/
  /**
   * Constructs a range test problem from the given subscript pair.
   */
  // Subscript pair doesn't have enough information, so range test needs to
  // look at the IR again.
  public RangeTest(SubscriptPair pair)
  {
    f = pair.getSubscript1();
    g = pair.getSubscript2();
    f_stmt = pair.getStatement1(); // Get an access to the IR.
    g_stmt = pair.getStatement2();
    setRanges(pair);
    setLoops(pair);
    was_solved = false;
  }

  // Builds a new range map if necessary and sets the current procedure.
  private void setRanges(SubscriptPair pair)
  {
    Procedure proc = f_stmt.getProcedure();
    if ( proc != procedure )
    {
      if ( verbosity >= 1 )
        PrintTools.printlnStatus(tag+"for procedure "+proc.getName(), 1);
      range_map = RangeAnalysis.getRanges(proc);
    }
    procedure = proc;
    f_range = range_map.get(f_stmt);
    g_range = range_map.get(g_stmt);
    common_range = f_range.clone();
    common_range.unionRanges(g_range);
  }

  // Collects enclosing loops differentiating common loops, relevant loops,
  // and local loops.
  private void setLoops(SubscriptPair pair)
  {
    // Collects common loops and local loops.
    Loop outer_most = pair.getEnclosingLoopsList().get(0);
    f_loops = getLocalLoops(f_stmt, outer_most);
    g_loops = getLocalLoops(g_stmt, outer_most);
    common_loops = new LinkedList<Loop>();
    common_loops.addAll(f_loops);
    common_loops.retainAll(g_loops);
    f_loops.removeAll(common_loops);
    g_loops.removeAll(common_loops);
    // Relevant loops belong to a subset of the common loops.
    relevant_loops = new LinkedHashSet<Loop>();
    for ( Loop loop : common_loops )
    {
      Identifier index = (Identifier)LoopTools.getIndexVariable(loop);
      Symbol index_sym = index.getSymbol();
      if ( IRTools.containsSymbol(f, index_sym) ||
          IRTools.containsSymbol(g, index_sym) )
        relevant_loops.add(loop);
    }
    parallel_loops = new LinkedHashMap<Loop, String>();
    independent_vectors = new int[common_loops.size()];
    // Pseudo-common loops; will be considered in the future if profitable.
  }

  // Set local_loops (f_loops/g_loops) as any loops contained in the outer loop.
  private LinkedList<Loop> getLocalLoops(Statement stmt, Loop outer)
  {
    LinkedList<Loop> ret = new LinkedList<Loop>();
    Traversable tr = stmt;
    for ( ; tr != outer; tr = tr.getParent() )
      if ( tr instanceof ForLoop )
        ret.addFirst((Loop)tr);
    ret.addFirst(outer);
    return ret;
  }

  // Returns the mono state of the expression w.r.t. the loop.
  // Issues to be handled
  // 1. Loop variants
  // 2. Negative loop steps
  private int getMonoState(Expression e, Expression ref, Loop loop)
  {
    RangeDomain rd = (ref==f)? f_range: g_range;
    Identifier index = (Identifier)LoopTools.getIndexVariable(loop);
    Expression step = LoopTools.getIncrementExpression(loop);
    Expression next = IRTools.replaceSymbol(e, index.getSymbol(),
      Symbolic.add(index, step));

    Relation rel = rd.compare(e, next);
    int ret = MONO_UNKNOWN;
    if ( rel.isEQ() )
      ret = MONO_CONST;
    else if ( rel.isLE() )
      ret = MONO_NONDEC;
    else if ( rel.isGE() )
      ret = MONO_NONINC;

    if ( verbosity >= 5 )
      PrintTools.printlnStatus
        (tag+"mono("+e+") w.r.t. "+loopToString(loop)+" = "+ret, 2);
    
    return ret;
  }

  // Returns the range of the given expression with respect to the given set
  // of loops.
  private Expression getRange(Expression e, Set<Loop> loops)
  {
    if ( verbosity >= 5 )
      PrintTools.printlnStatus(tag+"range("+e+") w.r.t. "+loopsToString(loops), 2);

    // Access the cache first; Use this if profitable.
    /*
    List<Object> signature = new LinkedList<Object>();
    signature.add(e);
    signature.addAll(loops);
    Expression ret = expr_range_cache.get(signature);
    if ( ret != null )
    {
      if ( verbosity >= 2 )
        PrintTools.printlnStatus(tag+"  = "+ret+" from the cache", 2);
      return ret;
    }

    ret = e;
    */

    Expression ret = e;

    if ( loops.isEmpty() )
    {
      if ( verbosity >= 5 )
        PrintTools.printlnStatus(ret.toString(), 2);
      return ret;
    }

    for ( int i=common_loops.size()-1; i>=0; i-- )
    {
      Loop curr_loop = common_loops.get(i);

      if ( !loops.contains(curr_loop) )
        continue;

      // Just pick the first range domain and expand the index variable
      // from there.
      RangeDomain rd = range_map.get(getFirstStatement(curr_loop));
      if ( rd == null )
        rd = new RangeDomain();

      Symbol index =
        ((Identifier)LoopTools.getIndexVariable(curr_loop)).getSymbol();
        
      ret = rd.expandSymbol(ret, index);

      if ( verbosity >= 5 )
        PrintTools.printlnStatus(
          tag+"   = "+ret+" after expanding "+index+" under "+rd, 2);
    }

/* Use this cache if profitable.
    expr_range_cache.put(signature, ret);
*/

    return ret;
  }

  private static Statement getFirstStatement(Loop loop)
  {
    for ( Traversable child : loop.getBody().getChildren() )
    {
      if ( !(child instanceof DeclarationStatement) &&
          !(child instanceof AnnotationStatement) &&
          child instanceof Statement )
        return (Statement)child;
    }
    return null;
  }

  /**
   * Test dependence on the specified direction vector.
   * @param dvec direction vector to be tested on.
   * @return whether dependence exists.
   */
  public boolean testDependence(DependenceVector dvec)
  {
    boolean ret = false;
    solve();
    for ( int i=0; i<common_loops.size() && !ret; i++ )
      switch ( dvec.getDirection(common_loops.get(i)) )
      {
        case DependenceVector.any:
          ret = (independent_vectors[i] & DV_ANY) == 0;
          break;
        case DependenceVector.equal:
          ret = (independent_vectors[i] & DV_EQ) == 0;
          break;
        case DependenceVector.less:
          ret = (independent_vectors[i] & DV_LT) == 0;
          break;
        case DependenceVector.greater:
          ret = (independent_vectors[i] & DV_GT) == 0;
          break;
        default:
          ret = true;
      }
    if ( verbosity >= 5 )
      PrintTools.printlnStatus(tag+"under "+dvec+" --> "+ret, 1);
    return ret;
  }

  /**
   * Checks if this test can proceed.
   * @return true if it is, false otherwise.
   */
  public boolean isTestEligible()
  {
    return true;
  }

  /**
   * Returns the list of loops that commonly enclose the subscript pair.
   * @return the list of common loops.
   */
  public LinkedList<Loop> getCommonEnclosingLoops()
  {
    return common_loops;
  }

  // Driver for a single range test problem.
  private void solve()
  {
    // Return quickly if the problem was already solved.
    if ( was_solved )
      return;

    List<Loop> permuted = new LinkedList<Loop>();

    // Iterate from innermost to outermost
    for ( int i=common_loops.size()-1; i>=0; i-- )
    {
      Loop loop = common_loops.get(i);
      Set<Loop> inner_permuted = new LinkedHashSet<Loop>();
      inner_permuted.addAll(permuted);
      boolean placed = false;

      if ( verbosity >= 5 )
        PrintTools.printlnStatus(tag+"for "+loopToString(loop), 2);

      // Exclude non-relevant loops in the permuted loops.
      if ( !relevant_loops.contains(loop) )
      {
        if ( test1(loop, inner_permuted) )
          parallel_loops.put(loop, TEST1_PASS);
        placed = true;
      }

      Iterator perm_iter = permuted.iterator();
      while ( perm_iter.hasNext() && !placed )
      {
        Loop perm_loop = (Loop)perm_iter.next();

        if ( test1(loop, inner_permuted) )
        {
          parallel_loops.put(loop, TEST1_PASS);
          placed = true;
        }
        else if ( test2(loop, inner_permuted) )
        {
          parallel_loops.put(loop, TEST2_PASS);
          placed = true;
        }
        else if ( !parallel_loops.containsKey(perm_loop) ||
            inner_permuted.isEmpty() )
        {
          placed = true;
        }
        else
        {
          inner_permuted.remove(perm_loop);
          inner_permuted.add(loop);
          if ( !test2(perm_loop, inner_permuted) )
            placed = true;
          inner_permuted.remove(loop);
        }

        if ( placed )
          permuted.add(permuted.indexOf(perm_loop), loop);
      }

      if ( !placed )
      {
        // Assert inner_permuted.size() == 0
        if ( test1(loop, inner_permuted) )
          parallel_loops.put(loop, TEST1_PASS);
        else if ( test2(loop, inner_permuted) )
          parallel_loops.put(loop, TEST2_PASS);
        permuted.add(loop);
      }
    }
    was_solved = true;
    setIndependentVectors();
    if ( verbosity >= 3 )
      PrintTools.printlnStatus(tag+this, 1);
  }

  // Converts the test result to set of dependence vectors with which no
  // loop-carried dependence exists.
  //                                            +--> current loop
  //                                            |
  // test 1 disproves dependences with {=,...,=,*,*,...,*}
  // test 2 disproves dependences with {=,...,=,<,*,...,*}
  //                                   {=,...,=,>,*,...,*}
  // the outer independent vectors do not need to be '=' if both f and g
  // do not have any loop variants of the loop for that vector.
  private void setIndependentVectors()
  {
    for ( Loop loop : parallel_loops.keySet() )
    {
      String result = parallel_loops.get(loop);
      int loop_id = common_loops.indexOf(loop);

      // Outer loops
      for ( int i=0; i<loop_id; i++ )
      {
        independent_vectors[i] |= DV_EQ;  // This is guaranteed.
        Identifier index = (Identifier)LoopTools.getIndexVariable(loop);
        Symbol index_symbol = index.getSymbol();
        Set<Symbol> symbols_in_pair = SymbolTools.getAccessedSymbols(f);
        symbols_in_pair.addAll(SymbolTools.getAccessedSymbols(g));
        symbols_in_pair.remove(index_symbol);
        if ( !symbols_in_pair.isEmpty() )
          symbols_in_pair.retainAll(getLoopVariants(common_loops.get(i)));
        if ( symbols_in_pair.isEmpty() )  // Don't care non-relevant loops.
          independent_vectors[i] |= DV_ANY+DV_LT+DV_GT;
      }

      // Current loop
      independent_vectors[loop_id] |= (DV_LT+DV_GT);    // for test2
      if ( result == TEST1_PASS )
        independent_vectors[loop_id] |= (DV_ANY+DV_EQ); // for test1

      // Inner loops
      for ( int i=loop_id+1; i<common_loops.size(); i++ )
        independent_vectors[i] |= DV_ANY+DV_LT+DV_GT+DV_EQ;
    }
  }

  // Returns loop variants of the specified loop -- cached.
  private Set<Symbol> getLoopVariants(Loop loop)
  {
    Set<Symbol> ret = loop_variants.get(loop);
    if ( ret == null )
    {
      ret = DataFlowTools.getDefSymbol((Traversable)loop);
      loop_variants.put(loop, ret);
    }
    return ret;
  }

  // Range test rule 1: min/max check w.r.t the given set of loops.
  private boolean test1(Loop loop, Set<Loop> loops)
  {
    boolean ret = ( f!=g &&
      (rtest1(f, g, f_loops, g_loops, loop, loops) ||
      rtest1(g, f, g_loops, f_loops, loop, loops)));
    if ( verbosity >= 5 )
      PrintTools.printlnStatus(tag+"test1 = "+ret, 2);
    return ret;
  }

  // Core test for the range test 1.
  private boolean rtest1(
    Expression e1,           // first expression
    Expression e2,           // second expression
    List<Loop> local_loops1, // first local loops
    List<Loop> local_loops2, // second local loops
    Loop loop,               // current loop
    Set<Loop> inner_loops    // inner loops (possibly permuted)
  )
  {
    Set<Loop> inner_loops1 = new LinkedHashSet<Loop>(inner_loops);
    Set<Loop> inner_loops2 = new LinkedHashSet<Loop>(inner_loops);
    inner_loops1.addAll(local_loops1);
    inner_loops2.addAll(local_loops2);

    // Use getRange to compute min/max of the expressions w.r.t. the loops
    Expression max1 = 
      RangeExpression.toRange(getRange(e1, inner_loops1)).getUB();
    Expression min2 = 
      RangeExpression.toRange(getRange(e2, inner_loops2)).getLB();

    if ( max1 instanceof InfExpression || min2 instanceof InfExpression )
    {  
      if ( verbosity >= 5 )
        PrintTools.printlnStatus(tag+"max1 = "+max1+", min2 = "+min2, 2);
      return false;
    }

    Identifier index = (Identifier)LoopTools.getIndexVariable(loop);
    Symbol index_sym = index.getSymbol();

    // Special case handling -- will visit later.
    /*
    int mono1 = getMonoState(max1, e1, loop);
    int mono2 = getMonoState(min2, e2, loop);
    if ( mono1 == MONO_NONDEC && mono2 == MONO_NONINC )
    {
      // will visit later.
    }
    else if ( mono1 == MONO_NONINC && mono2 == MONO_NONDEC )
    {
      // will visit later.
    }
    */

    inner_loops1.add(loop);
    inner_loops2.add(loop);
    max1 = RangeExpression.toRange(getRange(max1, inner_loops1)).getUB();
    min2 = RangeExpression.toRange(getRange(min2, inner_loops2)).getLB();

    if ( max1 instanceof InfExpression || min2 instanceof InfExpression )
    {
      if ( verbosity >= 5 )
        PrintTools.printlnStatus(tag+"max1 = "+max1+", min2 = "+min2, 2);
      return false;
    }

    max1 = removeLoopVariants(max1, e1, inner_loops1);
    min2 = removeLoopVariants(min2, e2, inner_loops2);

    Relation rel = common_range.compare(max1, min2);
    if ( verbosity >= 5 )
      PrintTools.printlnStatus(tag+"compare "+max1+" "+rel+" "+min2, 2);
    return rel.isLT();
  }

  // Removes all loop variants of loops excluding the index variables of the
  // loops that enclose the given reference expression.
  private Expression removeLoopVariants(
    Expression e,   // loop variants are removed from
    Expression ref, // enclosed by loops whose indices are skipped
    Set<Loop> loops // of which variants are removed
  )
  {
    boolean is_f = (ref == f);
    RangeDomain range = (is_f)? f_range: g_range;
    List<Loop> local_loops = (is_f)? f_loops: g_loops;

    Set<Symbol> variants = new LinkedHashSet<Symbol>();
    for ( Loop loop : loops )
      variants.addAll(getLoopVariants(loop));

    List<Loop> enclosers = new LinkedList<Loop>(common_loops);
    enclosers.addAll(local_loops);
    for ( Loop loop : enclosers )
    {
      Identifier index = (Identifier)LoopTools.getIndexVariable(loop);
      variants.remove(index.getSymbol());
    }
    Expression ret = range.expandSymbols(e, variants);

    if ( verbosity >= 5 )
      PrintTools.printlnStatus(
        tag+"removeLV("+e+") w.r.t. "+loopsToString(loops)+" = "+ret, 2);

    return ret;
  }

  // Range test rule 2: min/max check w.r.t the given set of loops with
  // monotonicity hints.
  private boolean test2(Loop loop, Set<Loop> loops)
  {
    int f_mono = getMonoState(f, f, loop);
    boolean ret = (
      (f_mono == MONO_NONINC || f_mono == MONO_NONDEC) &&
      (f_mono == getMonoState(g, g, loop)) &&
      rtest2(f, g, f_loops, g_loops, loop, loops) &&
      (f==g || rtest2(g, f, g_loops, f_loops, loop, loops)) );
    if ( verbosity >= 5 )
      PrintTools.printlnStatus(tag+"test2 = "+ret, 2);
    return ret;
  }

  private boolean rtest2(
    Expression e1,           // first expression
    Expression e2,           // second expression
    List<Loop> local_loops1, // first local loops
    List<Loop> local_loops2, // second local loops
    Loop loop,               // current loop
    Set<Loop> inner_loops    // inner loops (possibly permuted)
  )
  {
    Set<Loop> inner_loops1 = new LinkedHashSet<Loop>(inner_loops);
    Set<Loop> inner_loops2 = new LinkedHashSet<Loop>(inner_loops);
    inner_loops1.addAll(local_loops1);
    inner_loops2.addAll(local_loops2);

    Expression max1 =
      RangeExpression.toRange(getRange(e1, inner_loops1)).getUB();
    Expression min2 =
      RangeExpression.toRange(getRange(e2, inner_loops2)).getLB();

    if ( max1 instanceof InfExpression || min2 instanceof InfExpression )
    {
      if ( verbosity >= 5 )
        PrintTools.printlnStatus(tag+"max1 = "+max1+", min2 = "+min2, 2);
      return false;
    }

    max1 = removeLoopVariants(max1, e1, inner_loops1);
    min2 = removeLoopVariants(min2, e2, inner_loops2);

    if ( RangeExpression.toRange(max1).isOmega() ||
        RangeExpression.toRange(min2).isOmega() )
    {
      if ( verbosity >= 5 )
        PrintTools.printlnStatus(tag+"max1 = "+max1+", min2 = "+min2, 2);
      return false;
    }

    RangeDomain rd = common_range.clone();
    // Tighten the constraints on the current loop's index.
    // Check if this is a correct step (positive/negative?)
    Expression index = LoopTools.getIndexVariable(loop);
    Symbol id = ((Identifier)index).getSymbol();
    Expression step = LoopTools.getIncrementExpression(loop);
    Expression new_index = null;
    RangeExpression new_index_range = null;
    Expression index_range = rd.getRange(id);
    boolean has_index_range = (index_range != null &&
      index_range instanceof RangeExpression &&
      RangeExpression.toRange(index_range).isBounded() );
    
    if ( getMonoState(min2, e2, loop) == MONO_NONDEC )
    {
      new_index = Symbolic.add(index, step);
      if ( has_index_range )
      {
        new_index_range = (RangeExpression)index_range.clone();
        new_index_range.setUB(Symbolic.subtract(new_index_range.getUB(), step));
      }
    }
    else
    {
      new_index = Symbolic.subtract(index, step);
      if ( has_index_range )
      {
        new_index_range = (RangeExpression)index_range.clone();
        new_index_range.setLB(Symbolic.add(new_index_range.getLB(), step));
      }
    }

    min2 = IRTools.replaceSymbol(min2, id, new_index);

    if ( has_index_range )
      rd.setRange(id, new_index_range);

    Relation rel = rd.compare(max1, min2);
    if ( verbosity >= 5 )
      PrintTools.printlnStatus(tag+"compare "+max1+" "+rel+" "+min2, 2);
    return rel.isLT();
  }

  /**
   * Returns a string that shows a snapshot of the current range test problem.
   * @return test information in string.
   */
  public String toString()
  {
    StringBuilder str = new StringBuilder(200);
    str.append("f = "+f+", g = "+g);
    str.append(", f_range = "+f_range+", g_range = "+g_range);
    str.append(", f_loops = "+loopsToString(f_loops));
    str.append(", g_loops = "+loopsToString(g_loops));
    str.append(", common_range = "+common_range);
    str.append(", common_loops = "+loopsToString(common_loops));
    str.append(", relevant_loops = "+loopsToString(relevant_loops));
    str.append(", parallel_loops = "+loopsToString2(parallel_loops));
    str.append(", independent_vectors = {");
    str.append(Integer.toBinaryString(independent_vectors[0]));
    for ( int i=1; i<common_loops.size(); i++ )
      str.append(", "+Integer.toBinaryString(independent_vectors[i]));
    str.append("}");
    return str.toString();
  }

  // Pretty print method for loop information.
  private String loopsToString(Collection<Loop> loops)
  {
    StringBuilder str = new StringBuilder(80);
    str.append("{");
    Iterator<Loop> iter = loops.iterator();
    if ( iter.hasNext() )
      str.append(loopToString(iter.next()));
    while ( iter.hasNext() )
      str.append(", "+loopToString(iter.next()));
    str.append("}");
    return str.toString();
  }

  // Pretty print method for loop information.
  private String loopsToString2(Map<Loop, String> loops)
  {
    StringBuilder str = new StringBuilder(80);
    str.append("{");
    Iterator<Loop> iter = loops.keySet().iterator();
    if ( iter.hasNext() )
    {
      Loop curr = iter.next();
      str.append(loopToString(curr)+"["+loops.get(curr)+"]");
    }
    while ( iter.hasNext() )
    {
      Loop curr = iter.next();
      str.append(", "+loopToString(curr)+"["+loops.get(curr)+"]");
    }
    str.append("}");
    return str.toString();
  }

  // Pretty print method for loop information.
  private String loopToString(Object loop)
  {
    if ( loop.getClass() != ForLoop.class )
      return "non-for loop";
    ForLoop floop = (ForLoop)loop;
    return "["+floop.getInitialStatement()+" "+
      floop.getCondition()+"; "+
      floop.getStep()+"]";
  }
}
