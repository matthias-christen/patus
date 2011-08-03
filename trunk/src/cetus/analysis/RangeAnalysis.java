package cetus.analysis;

import java.io.*;
import java.util.*;
import java.lang.reflect.*;
import cetus.hir.*;
import cetus.exec.*;

/**
 * RangeAnalysis performs symbolic range propagation for the programs by
 * symbolically executing the program and abstracting the values of integer
 * variables with a symbolic bounds. The implementation is based on Symbolic
 * Range Propagation by Blume and Eigenmann, which was implemented for the
 * Fortran77 language.
 * <p>
 * The goal of Range Analysis is to collect, at each program statement, a map
 * from integer-typed scalar variables to their symbolic value ranges,
 * represented by a symbolic lower bound and an upper bound.
 * In other words, a symbolic value range expresses the relationship between the
 * variables that appear in the range. We use a similar approach as in Symbolic
 * Range Propagation in the Polaris parallelizing compiler, with necessary
 * adjustment for the C language, to compute the set of value ranges before
 * each statement. The set of value ranges at each statement can be used in
 * several ways. Pass writers can directly query the symbolic bound of a
 * variable or can compare two symbolic expressions using the constraints given
 * by the set of value ranges.
 * <p>
 * The high-level algorithm performs fix-point iteration in two phases when
 * propagating the value ranges throughout the program.
 * The first phase applies widening operations at nodes that have incoming back
 * edges to guarantee the termination of the algorithm. The second phase
 * compensates the loss of information due to the widening operations by
 * applying narrowing operation to the node at which widening has occurred.
 * During the fix-point iteration, the value ranges are merged at nodes that
 * have multiple predecessors, and outgoing value ranges are computed by
 * symbolically executing the statement. Two typical types of program semantics
 * that cause such changes of value ranges are constraints from conditional
 * expressions and assignments to variables.
 * <p>
 * Range analysis returns a map from each statement to its corresponding
 * {@link RangeDomain} that contains the set of valid value ranges before
 * each statement. The result of this analysis was verified with the
 * C benchmarks in the SPEC CPU2006 suite. Range analysis does not handle
 * integer overflow as it does not model the overflow but such cases are rare
 * in normal correct programs. Following example shows how to invoke a range
 * analyzer in a compiler pass:
 * <pre>
 * Map&lt;Statement, RangeDomain&gt; range_map = RangeAnalysis.getRanges(procedure);
 * RangeDomain range_domain = range_map.get(statement);
 * // range_domain now contains the set of value ranges for the statement.
 * </pre>
 * Following example shows a function and its range map created after the
 * range analysis.
 * <pre>
 *          int foo(int k) {                             
 *                  []                    
 *            int i, j;                   
 *                  []                    
 *            double a;                   
 *                  []                    
 *            for ( i=0; i&lt;10; ++i ) {                           
 *                  [0&lt;=i&lt;=9]             
 *              a=(0.5*i);                
 *            }                           
 *                  [i=10]                
 *            j=(i+k);                    
 *                  [i=10, j=(i+k)]       
 *            return j;                   
 *          }                             
 * </pre>
 */
public class RangeAnalysis extends AnalysisPass
{
  // Pass tag
  private static final String tag = "[RangeAnalysis]";

  // Set of tractable expression types
  private static final Set<Class> tractable_class;

  // Set of tractable operation types
  private static final Set<Printable> tractable_op;

  // Debug level
  private static int debug;

  // Range analysis option
  // 0: don't do
  // 1: intra analysis (default)
  // 2: inter analysis
  private static int option;
  /** Option for disabling range computation */
  public static final int RANGE_EMPTY = 0;
  /** Option for enforcing intra-procedural analysis */
  public static final int RANGE_INTRA = 1;
  /** Option for enforcing inter-procedural analysis */
  public static final int RANGE_INTER = 2;

  static
  {
    tractable_class = new HashSet<Class>();
    tractable_class.add(ArrayAccess.class);
    tractable_class.add(BinaryExpression.class);
    tractable_class.add(Identifier.class);
    tractable_class.add(InfExpression.class);
    tractable_class.add(IntegerLiteral.class);
    tractable_class.add(MinMaxExpression.class);
    tractable_class.add(RangeExpression.class);
    tractable_class.add(UnaryExpression.class);
    tractable_op = new HashSet<Printable>();
    tractable_op.add(BinaryOperator.ADD);
    tractable_op.add(BinaryOperator.DIVIDE);
    tractable_op.add(BinaryOperator.MODULUS);
    tractable_op.add(BinaryOperator.MULTIPLY);
    tractable_op.add(UnaryOperator.MINUS);
    tractable_op.add(UnaryOperator.PLUS);
    debug = 0;
    if (Driver.getOptionValue("verbosity") != null)
      debug = Integer.valueOf(Driver.getOptionValue("verbosity")).intValue();
    option = RANGE_INTRA;
    if (Driver.getOptionValue("range") != null)
      option = Integer.valueOf(Driver.getOptionValue("range")).intValue();
  }

  // Flag for checking if it is generating ranges for orphan IR.
  // isValidScope test is skipped if is_orphan == true.
  private static boolean is_orphan = false;

  // Validation flag
  private int validation;

  // Interprocedural input
  // ip_node the procedure node to be processed
  // ip_cfg the procedure control flow graph to be processed.
  private static IPANode ip_node = null;
  private static CFGraph ip_cfg = null;

  // Result of interprocedural range analysis.
  private static Map<Procedure, Map<Statement, RangeDomain>> ip_ranges = null;

  /**
   * Constructs a range analyzer for the program. The instantiated constructor
   * is used only internally.
   *
   * @param program  the input program.
   */
  private RangeAnalysis(Program program)
  {
    super(program);

    // Range Validation
    try {
      validation = new Integer(System.getenv("CETUS_RANGE_VALIDATION"));
    } catch ( Exception ex ) {
      validation = 0;
    }
  }

  /**
   * Returns the pass name.
   *
   * @return the pass name in string.
   */
  @Override
  public String getPassName()
  {
    return tag;
  }

  /**
   * Starts range analysis for the whole program. this is useful only for
   * testing the range analysis.
   */
  @Override
  public void start()
  {
    if ( validation == 0 )
      return;

    double timer = Tools.getTime();

    DepthFirstIterator iter = new DepthFirstIterator(program);
    iter.pruneOn(Procedure.class);
    iter.pruneOn(Statement.class);
    iter.pruneOn(Declaration.class);

    while ( iter.hasNext() )
    {
      Object o = iter.next();
      if ( o.getClass() != Procedure.class )
        continue;
      Map range_map = getRangeMap((Procedure)o);
      // Test codes for tools with range information.
      //testSubstituteForward((Procedure)o, range_map);
    }
    
    timer = Tools.getTime(timer);

    PrintTools.printStatus(tag+" "+String.format("%.2f\n",timer),1);

  }

  private void testSubstituteForward(Procedure proc, Map range_map)
  {
    DepthFirstIterator iter = new DepthFirstIterator(proc);
    while ( iter.hasNext() )
    {
      Traversable tr = (Traversable)iter.next();
      if ( !(tr instanceof Statement) )
        continue;
      RangeDomain rd = (RangeDomain)range_map.get(tr);
      if ( rd == null )
        continue;
      PrintTools.printlnStatus("[RD] = " + rd, 0);
      Set<Expression> use = DataFlowTools.getUseSet(tr);
      for ( Expression e : use )
      {
        PrintTools.printStatus("  [SubstituteForward] " + e + " => ", 0);
        PrintTools.printlnStatus(rd.substituteForward(e).toString(), 0);
      }
    }
  }

  /**
   * Performs range analysis for the procedure and returns the result.
   *
   * @param proc    the input procedure.
   * @return the range map for the procedure.
   */
  public Map<Statement, RangeDomain> getRangeMap(Procedure proc)
  {
    return getRanges(proc);
/*
    double timer = Tools.getTime();
    PrintTools.printStatus(tag+" Analyzing " + proc.getName() + " ... ", 1);

    // Builds a directed graph for the procedure
    CFGraph g = new CFGraph(proc);

    // Expands the graph whose node has at most one assignment
    g.normalize();

    if ( debug >= 5 )
    {
      PrintTools.printlnStatus(tag+" CFG of "+proc.getName(), 5);
      PrintTools.printlnStatus(g.toDot(), 5);
    }

    // Computes the topological order of the node disregarding back edges
    g.topologicalSort(g.getNodeWith("stmt","ENTRY"));

    // Widening phase of fixpoint iteration
    iterateToFixpoint(g, true);

    if ( debug >= 5 )
    {
      PrintTools.printlnStatus(tag+" CFG after widening", 5);
      PrintTools.printlnStatus(g.toDot("top-order,ranges,ir,tag", 3), 5);
      PrintTools.printlnStatus(tag+" Range Map after widening", 5);
      PrintTools.printlnStatus(
        toPrettyRanges(proc, getRangeMap(g), new Integer(0)), 5);
    }

    // Narrowing phase of fixpoint iteration
    iterateToFixpoint(g, false);

    Map<Statement, RangeDomain> range_map = getRangeMap(g);

    if ( debug >= 5 )
    {
      PrintTools.printlnStatus(tag+" CFG after narrowing", 5);
      PrintTools.printlnStatus(g.toDot("top-order,ranges,ir,tag", 3), 5);
      PrintTools.printlnStatus(tag+" Range Map after narrowing", 5);
      PrintTools.printlnStatus(
        toPrettyRanges(proc, range_map, new Integer(0)), 5);
    }

    PrintTools.printlnStatus(tag+" took "+
        String.format("%.2f", Tools.getTime(timer)) + " seconds.", 1);

    return range_map;
*/
  }

  /**
   * Generates a range map from the work space graph.
   *
   * @param g     the control flow graph used in the analysis.
   * @return      the map from statements to sets of value ranges.
   */
  private static Map<Statement, RangeDomain> getRangeMap(CFGraph g)
  {
    Map<Statement, RangeDomain> ret = new HashMap<Statement, RangeDomain>();
    for ( int i=0; i < g.size(); ++i )
    {
      DFANode node = g.getNode(i);
      Object ir = node.getData("stmt");
      RangeDomain rd = node.getData("ranges");
      if ( ir == null || rd == null || !(ir instanceof Statement) )
        continue;
      Statement stmt = (Statement)ir;

      // Eliminate variables with pointer type.
      Set<Symbol> vars = new LinkedHashSet<Symbol>(rd.getSymbols());
      for ( Symbol var : vars )
        if (SymbolTools.isPointer(var) ||containsPointerType(rd.getRange(var)))
          rd.removeRange(var);

      // Skip scope checking if ranges are for an orphan IR.
      if ( is_orphan )
      {
        ret.put(stmt, rd);
        continue;
      }

      // Does not include out-of-scope ranges with same name.
      vars = new LinkedHashSet<Symbol>(rd.getSymbols());
      for ( Symbol var : vars )
      {
        // Test with key
        IDExpression id = new Identifier(var);

        if ( !isValidScope(stmt, id, var) )
        {
          rd.removeRange(var);
          continue;
        }

        // Test with data
        DepthFirstIterator iter = new DepthFirstIterator(rd.getRange(var));

        while ( iter.hasNext() )
        {
          Object o = iter.next();
          if ( o instanceof Identifier )
          {
            Identifier ident = (Identifier)o;
            if ( !isValidScope(stmt, ident, ident.getSymbol()) )
            {
              rd.removeRange(var);
              break;
            }
          }
        }
      }

      // Eliminate variables with pointer type.
      /*
      vars = new HashSet(rd.getSymbols());
      for ( Symbol var : vars )
        if ( SymbolTools.isPointer(var) || containsPointerType(rd.getRange(var)) )
          rd.removeRange(var);
      */
        
      ret.put(stmt, rd);
    }

    return ret;
  }

  /**
   * Returns the range map in a pretty format.
   *
   * @param t        the traversable object.
   * @param ranges   the range map.
   * @param indent   the indent for pretty printing.
   * @return         the string in a pretty format.
   */
  public static String toPrettyRanges
  (Traversable t, Map ranges, Integer indent)
  {
    String ret = "", tab = "";

    for ( int i=0; i<indent; ++i )
      tab += "  ";

    if ( t instanceof Procedure )
    {
      Procedure p = (Procedure)t;
      ret += tag+" Range Domain for Procedure "+p.getName()+"\n";
      ret += toPrettyRanges(p.getBody(), ranges, indent);
    }

    else if ( t instanceof CompoundStatement )
    {
      ret += tab+"{\n";
      indent++;

      for ( Object o: t.getChildren() )
        ret += toPrettyRanges((Traversable)o, ranges, indent);

      indent--;
      ret += tab+"}\n";
    }

    else if ( t instanceof DoLoop )
    {
      DoLoop d = (DoLoop)t;
      ret += tab+"do\n";
      ret += toPrettyRanges(d.getBody(), ranges, indent);
      ret += tab+"while ( "+d.getCondition()+" );\n";
    }

    else if ( t instanceof ForLoop )
    {
      ForLoop f = (ForLoop)t;
      ret += tab+"for ( ";
      Statement init = f.getInitialStatement();
      ret += ((init==null)? ";": init)+" ";
      Expression condition = f.getCondition();
      ret += ((condition==null)? " ": condition)+"; ";
      Expression step = f.getStep();
      ret += ((step==null)? "": step)+" )\n";
      ret += toPrettyRanges(f.getBody(), ranges, indent);
    }

    else if ( t instanceof IfStatement )
    {
      IfStatement i = (IfStatement)t;
      ret += tab+"if ( "+i.getControlExpression()+" )\n";
      ret += toPrettyRanges(i.getThenStatement(), ranges, indent);
      Statement els = i.getElseStatement();
      if ( els != null )
        ret += tab+"else\n"+toPrettyRanges(els, ranges, indent);
    }

    else if ( t instanceof SwitchStatement )
    {
      SwitchStatement s = (SwitchStatement)t;
      ret += tab+"switch ( "+s.getExpression()+" )\n";
      ret += toPrettyRanges(s.getBody(), ranges, indent);
    }

    else if ( t instanceof WhileLoop )
    {
      WhileLoop w = (WhileLoop)t;
      ret += tab+"while ( "+w.getCondition()+" )\n";
      ret += toPrettyRanges(w.getBody(), ranges, indent);
    }

    else if ( t instanceof Statement )
      ret += tab+t+"\n";


    if ( t instanceof Statement )
    {
      RangeDomain rd = (RangeDomain)ranges.get(t);
      ret = "        "+((rd==null)? "[]": rd)+"\n"+ret;
    }
    return ret;
  }

  // Returns an empty map of ranges - fallback behavior.
  private static Map<Statement, RangeDomain> getEmptyRanges(Traversable t)
  {
    Map<Statement, RangeDomain> ret = new HashMap<Statement, RangeDomain>();
    RangeDomain empty = new RangeDomain();
    DepthFirstIterator iter = new DepthFirstIterator(t);
    while (iter.hasNext()) {
      Object o = iter.next();
      if (o instanceof Statement)
        ret.put((Statement)o, empty);
    }
    return ret;
  }

  /** Enforces use of the specified option for computing the ranges */
  public static Map<Statement, RangeDomain>
      getRanges(SymbolTable symtab, int temporal_option)
  {
    int original_option = option;
    option = temporal_option;
    Map<Statement, RangeDomain> ret = getRanges(symtab);
    option = original_option;
    return ret;
  }

  /**
   * Returns a range map created for the symbol table object. It is more
   * general to support range map for a traversable object but supporting this
   * functionality on an object that contains collection of statement object,
   * i.e. symbol table, is more natural.
   *
   * @param symtab the symbol table object whose range map is collected.
   * @return the range map for the traversable object.
   */
  public static Map<Statement,RangeDomain> getRanges(SymbolTable symtab)
  {
    Map<Statement, RangeDomain> ret = null;
    if ( symtab instanceof TranslationUnit )
    {
      PrintTools.printlnStatus(tag +
        " Range analysis for whole program unavailable", 0);
      return ret;
    }
    else if (symtab instanceof Procedure)
    {
      String proc_name = " "+((Procedure)symtab).getSymbolName();
      switch (option) {
        case RANGE_INTER:
          if (ip_ranges == null) {
            Program prog = IRTools.getAncestorOfType(symtab, Program.class);
            if (prog != null) {
              PrintTools.printlnStatus(tag + " Recomputing IP ranges ...", 1);
              IPRangeAnalysis.compute(prog);
            }
          }
          if (ip_ranges != null && (ret=ip_ranges.get(symtab)) != null)
            PrintTools.printlnStatus(tag + proc_name +
                ": Retrieving IP ranges ...", 1);
          else
            PrintTools.printlnStatus(tag + proc_name +
                ": Falling back to intra analysis ...", 1);
          break;
        case RANGE_EMPTY:
          PrintTools.printlnStatus(tag + proc_name +
              ": Returning empty ranges ...", 1);
          ret = getEmptyRanges(symtab);
          break;
        case RANGE_INTRA:
        default:
          PrintTools.printlnStatus(tag + proc_name +
              ": Computing local ranges ...", 1);
      }
      if (ret != null) {
        if (debug >= 3) {
          //System.err.println("Range Map for the symbol table object:");
          System.err.println(toPrettyRanges(symtab, ret, new Integer(0)));
        }
        return ret;
      }
    }

    Traversable root = (Traversable)symtab;
    while ( root != null && root.getClass() != Program.class )
      root = root.getParent();
    is_orphan = (root == null);

    CFGraph cfg = new CFGraph((Traversable)symtab);
    cfg.normalize();
    cfg.topologicalSort(cfg.getNodeWith("stmt","ENTRY"));
    iterateToFixpoint(cfg, true);
    iterateToFixpoint(cfg, false);
    if ( debug >= 5 )
      PrintTools.printlnStatus(cfg.toDot("top-order,ranges,ir,tag", 3), 5);
    ret = getRangeMap(cfg);

    // Inserts an empty range if there is no associated range domain.
    RangeDomain empty = new RangeDomain();
    DepthFirstIterator iter = new DepthFirstIterator((Traversable)symtab);
    while ( iter.hasNext() )
    {
      Object o = iter.next();
      if ( o instanceof Statement )
      {
        Statement stmt = (Statement)o;
        if ( ret.get(stmt) == null )
          ret.put(stmt, empty);
      }
    }

    if ( debug >= 3 )
    {
      //PrintTools.printlnStatus("Range Map for the symbol table object:", 0);
      PrintTools.printlnStatus(toPrettyRanges((Traversable)symtab, ret, new Integer(0)), 0);
    }
    is_orphan = false;
    return ret;
  }

  /**
   * Returns the fine-grain result of range analysis with the given
   * interprocedural input. "node" represents a procedure whose IPA results are
   * stored inside. "ip_node" and "ip_cfg" are temporarily stored for a single
   * invocation of getRangeCFG, and are accessed during intraprocedural analysis
   * that interprets the flow entry of a procedure and the call site inside a
   * procedure.
   */
  protected static CFGraph getRangeCFG(IPANode node)
  {
    ip_node = node;
    debug = PrintTools.getVerbosity();
    CFGraph ret = new CFGraph(node.getProcedure());
    ret.normalize();
    DFANode entry_node = ret.getEntry();
    ret.topologicalSort(entry_node);
    ip_cfg = ret;
    Domain in0 = node.in();
    if ( in0 instanceof RangeDomain )
      entry_node.putData("ranges", ((RangeDomain)in0).clone());
    iterateToFixpoint(ret, true);
    iterateToFixpoint(ret, false);
    ip_node = null;
    ip_cfg = null;
    return ret;
  }

  /**
   * Returns the coarse-grain result of range analysis with the given
   * interprocedural input. It internally calls getRangeCFG and jsut converts
   * the cfg-based workspace into a map from statements to range domains.
   */
  protected static Map<Statement, RangeDomain> getRanges(IPANode node)
  {
    Procedure proc = node.getProcedure();
    Map<Statement, RangeDomain> ret = getRangeMap(getRangeCFG(node));
    RangeDomain empty = new RangeDomain();
    DepthFirstIterator iter = new DepthFirstIterator(proc);
    while ( iter.hasNext() )
    {
      Object o = iter.next();
      if ( o instanceof Statement )
      {
        Statement stmt = (Statement)o;
        if ( ret.get(stmt) == null )
          ret.put(stmt, empty);
      }
    }

    if ( debug >= 3 )
    {
      PrintTools.printlnStatus(toPrettyRanges(proc, ret, new Integer(0)), 3);
    }

    return ret;
  }

  /**
   * Returns a range domain for the given statement.
   *
   * @param stmt the input statement.
   * @return the out range domain for the statement.
   */
  public static RangeDomain getRangeDomain(Statement stmt)
  {
    if ( stmt == null || !(stmt instanceof ExpressionStatement) )
      return new RangeDomain();

    CFGraph cfg = new CFGraph(stmt);
    cfg.normalize();
    cfg.topologicalSort(cfg.getNodeWith("stmt","ENTRY"));
    DFANode last = cfg.getNodeWith("top-order", new Integer(cfg.size()-1));
    DFANode exit = new DFANode("stmt","EXIT");
    exit.putData("top-order", new Integer(cfg.size()));
    cfg.addEdge(last, exit);
    // Exit node is necessary because Range Analysis computes only in-range.
    // Maybe CFGraph needs flow exit always but it breaks the range analysis,
    // so it needs to be revisited later.

    iterateToFixpoint(cfg, true);

    RangeDomain ret = exit.getData("ranges");
    if ( ret == null )
      ret = new RangeDomain();

    return ret;
  }

  /**
  * Sets the interprocedural range maps with the specified new map.
  */
  protected static void
  setRanges(Map<Procedure, Map<Statement, RangeDomain>> ip_ranges)
  {
    RangeAnalysis.ip_ranges = ip_ranges;
  }

  // Check if the identifier has a valid scope.
  // TODO: this should be supported by SymbolTools.
  private static boolean
      isValidScope(Traversable from, IDExpression id, Symbol symbol)
  {
    while (from != null) {
      from = (Traversable)IRTools.getAncestorOfType(from, SymbolTable.class);
      Set<Symbol> symbols = SymbolTools.getSymbols((SymbolTable)from);
      if (symbols.contains(symbol))
        return true;
    }
    return false;
  }

  // Eliminates variables with pointer type.
  private static boolean containsPointerType(Traversable t)
  {
    if ( t == null )
      return false;

    DepthFirstIterator iter = new DepthFirstIterator(t);
    while ( iter.hasNext() )
    {
      Object o = iter.next();
      if ( o instanceof Identifier &&
      SymbolTools.isPointer(((Identifier)o).getSymbol()) )
        return true;
    }
    return false;
  }

  // Marks the node having a back edge by comparing topological order
  private static void setBackedge(DFANode node)
  {
    int my_order = (Integer)node.getData("top-order");
    for ( DFANode pred : node.getPreds() )
      if ( my_order < (Integer)pred.getData("top-order") )
      {
        node.putData("has-backedge", new Boolean(true));
        break;
      }
  }

  // Fixpoint iteration
  private static void iterateToFixpoint(CFGraph g, boolean widen)
  {
    TreeMap<Integer, DFANode> work_list = new TreeMap<Integer, DFANode>();

    // Add the entry node to the work list for widening phase.
    if ( widen )
    {
      DFANode entry = g.getNodeWith("stmt", "ENTRY");
      if ( entry.getData("ranges") == null )
        entry.putData("ranges", new RangeDomain());
      work_list.put((Integer)entry.getData("top-order"), entry);
    }

    // Add the widened nodes to the work list for narrowing phase.
    else
    {
      for ( int i=0; i<g.size(); ++i )
      {
        DFANode widened = g.getNode(i);
        if ( widened.getData("has-backedge") != null )
          work_list.put((Integer)widened.getData("top-order"), widened);
      }
    }

    while ( work_list.size() > 0 )
    {
      // Get the first node in topological order from the work list.
      Integer node_num = work_list.firstKey();
      DFANode node = work_list.remove(node_num);

      // Record number of iterations for each node.
      Integer visits = node.getData("num-visits");
      if ( visits == null )
      {
        node.putData("num-visits", new Integer(1));
        setBackedge(node);
      }
      else
        node.putData("num-visits", new Integer(visits+1));

      if ( debug >= 5 )
      {
        PrintTools.printlnStatus(tag+" Visited Node#"+node_num, 2);
        PrintTools.printlnStatus(tag+"   IR = "+CFGraph.getIR(node), 5);
      }

      // Merge incoming states from predecessors.
      RangeDomain curr_ranges = null;
      for ( DFANode pred : node.getPreds() )
      {
        RangeDomain pred_range_out = (RangeDomain)node.getPredData(pred);

        // Skip BOT-state predecessors that has not been visited.
        if ( pred_range_out == null )
          continue;

        if ( curr_ranges == null )
          curr_ranges = new RangeDomain(pred_range_out);
        else
          curr_ranges.unionRanges(pred_range_out);
      }

      if ( debug >= 5 )
        PrintTools.printlnStatus(tag+"   IN0 = "+curr_ranges, 5);

      // Add initial values from declarations
      enterScope(node, curr_ranges);

      // Widening/Narrowing operations.
      RangeDomain prev_ranges = node.getData("ranges");
      if ( prev_ranges != null && node.getData("has-backedge") != null )
      {
        if ( widen )
        {
          Set<Symbol> widener = node.getData("loop-variants");

          // Selective widening only with loop-variant symbols.
          if ( widener != null && widener.size() > 0 )
            curr_ranges.widenAffectedRanges(prev_ranges, widener);

          else
            curr_ranges.widenRanges(prev_ranges);
        }
        else
          curr_ranges.narrowRanges(prev_ranges);
      }

      if ( debug >= 5 )
        PrintTools.printlnStatus(tag+"   IN1 = "+curr_ranges, 5);

      if ( prev_ranges == null || !prev_ranges.equals(curr_ranges) )
      {
        if ( node != g.getEntry() ) // Keep the IPA result for the entry node.
          node.putData("ranges", curr_ranges);

        // Apply state changes due to the execution of the node.
        updateRanges(node);

        // Clean up after exiting a scope.
        exitScope(node);

        for ( DFANode succ : node.getSuccs() )
        {
          // Do not add successors for infeasible paths
          if ( succ.getPredData(node) != null )
          {
            work_list.put((Integer)succ.getData("top-order"), succ);
            if ( debug >= 5 )
            {
              PrintTools.printStatus(tag+"   OUT#"+succ.getData("top-order"), 5);
              PrintTools.printlnStatus(" = "+succ.getPredData(node), 5);
            }
          }
        }
      }
    }
  }

  // Add intialized values from the declarations.
  private static void enterScope(DFANode node, RangeDomain ranges)
  {
    SymbolTable st = node.getData("symbol-entry");
    if ( st == null )
      return;

    for ( Symbol var : SymbolTools.getSymbols(st) )
    {
      // Not interested in other declarators.
      if ( !(var instanceof VariableDeclarator) )
        continue;

      Initializer init = ((VariableDeclarator)var).getInitializer();

      // Flow of initializer is not guaranteed.
      if ( IRTools.containsSideEffect(init) )
      {
        ranges.clear();
        return;
      }

      // Extract only scalar integers' initial values.
      List specs = var.getTypeSpecifiers();
      if (
        init == null ||
        init.getChildren().size() != 1 || // #initializer must be one.
        !SymbolTools.isScalar(var) ||
        !isTractableType(var) ||
        specs.contains(Specifier.VOLATILE) ||
        specs.contains(Specifier.STATIC)
      ) continue;

      Expression new_range = 
        Symbolic.simplify((Expression)init.getChildren().get(0));

      if ( isTractableRange(new_range) )
        ranges.setRange(var, new_range);
    }
  }

  // Clean up variables that are not present in the scope.
  private static void exitScope(DFANode node)
  {
    List<SymbolTable> symbol_exits = node.getData("symbol-exit");

    if ( symbol_exits != null )
    {
      Set<Symbol> symbols = new LinkedHashSet<Symbol>();

      // symbol-exit assumed to be list of symbol tables
      for ( SymbolTable st : symbol_exits )
        symbols.addAll(SymbolTools.getSymbols(st));

      for ( DFANode succ : node.getSuccs() )
      {
        RangeDomain ranges_out = succ.getPredData(node);

        if ( ranges_out != null )
          ranges_out.removeSymbols(symbols);
      }
    }
  }

  // Check if the expression is tractable by range analysis.
  private static boolean isTractableRange(Expression e)
  {
    DepthFirstIterator iter = new DepthFirstIterator(e);
    while ( iter.hasNext() )
    {
      Object o = iter.next();

      if ( o instanceof IntegerLiteral )
      {
        long value = ((IntegerLiteral)o).getValue();
        if ( value >= Integer.MAX_VALUE/4 || value <= Integer.MIN_VALUE/4 )
          return false; // Avoid overflow from potentially large numbers.
      }
      if ( !tractable_class.contains(o.getClass()) )
        return false;
      if ( o instanceof Identifier )
      {
        Identifier id = (Identifier)o;
        if ( !isTractableType(id.getSymbol()) )
          return false;
      }
      else if ( o instanceof BinaryExpression &&
        !tractable_op.contains(((BinaryExpression)o).getOperator()) )
        return false;
      else if ( o instanceof UnaryExpression &&
        !tractable_op.contains(((UnaryExpression)o).getOperator()) )
        return false;
      // Other cases should be all tractable.
    }
    return true;
  }


  // Check if the symbol is tractable by range analysis.
  protected static boolean isTractableType(Symbol symbol)
  {
    return ( SymbolTools.isInteger(symbol) &&
      !symbol.getTypeSpecifiers().contains(Specifier.UNSIGNED) );
  }

  // Detect invertible assignment and return the inverted expression.
  private static Expression invertExpression(Expression to, Expression from)
  {
    Symbol var = SymbolTools.getSymbolOf(to);

    Expression diff = Symbolic.subtract(to, from);

    if ( IRTools.containsSymbol(diff, var) )
      return null;

    return Symbolic.add(to, diff);
  }

  // Methods for updating edges to successors.
  private static void updateRanges(DFANode node)
  {
    Object o = CFGraph.getIR(node);

    if ( o instanceof ExpressionStatement )
      o = ((ExpressionStatement)o).getExpression();

    // Side-effect node
    if ( o instanceof Traversable )
    {
      Traversable t = (Traversable)o;
      if ( IRTools.containsClass(t, VaArgExpression.class) )
      {
        updateUnsafeNode(node);
        return;
      }
      if ( IRTools.containsClass(t, FunctionCall.class) )
      {
        if ( ip_node == null )
          updateFunctionCall(node);
        else
          updateFunctionCallWithIPA(node);
        return;
      }
    }

    // Assignments
    if ( o instanceof AssignmentExpression )
      updateAssignment(node, (AssignmentExpression)o);

    // Binary expressions (logical expression)
    else if ( o instanceof BinaryExpression )
      updateBinary(node, (BinaryExpression)o);

    // Switch statements
    else if ( o instanceof SwitchStatement )
      updateSwitch(node, (SwitchStatement)o);

    // Side-effect-free node
    else
      updateSafeNode(node);
  }

  // Node containing function calls.
  private static void updateFunctionCall(DFANode node)
  {
    if ( Driver.getOptionValue("no-side-effect") == null )
    {
      updateUnsafeNode(node);
      return;
    }

    // no-side-effect is assumed by user: do aggressive update.
    RangeDomain range = new RangeDomain((RangeDomain)node.getData("ranges"));
    Set<Symbol> defs = DataFlowTools.getDefSymbol((Traversable)CFGraph.getIR(node));
    for ( Symbol def : defs )
      range.removeRangeWith(def);
    for ( DFANode succ : node.getSuccs() )
      succ.putPredData(node, range);
  }

  // Returns a set of symbols whose addresses were taken, hence possibly
  // unsafe to have a value range.
  private static Set<Symbol> getAddressTakenID(Traversable tr)
  {
    Set<Symbol> ret = new HashSet<Symbol>();
    DepthFirstIterator iter = new DepthFirstIterator(tr);
    while ( iter.hasNext() )
    {
      Object o = iter.next();
      if ( o instanceof UnaryExpression )
      {
        UnaryExpression ue = (UnaryExpression)o;
        if ( ue.getExpression() instanceof Identifier &&
            ue.getOperator() == UnaryOperator.ADDRESS_OF )
          ret.add(SymbolTools.getSymbolOf(ue.getExpression()));
      }
    }
    return ret;
  }

  // Update function call's ranges using the interprocedural input.
  @SuppressWarnings("unchecked")
  private static void updateFunctionCallWithIPA(DFANode node)
  {
    Map<CallSite, Domain> maymods = ip_node.getData(MayMod.tag+"CallOUT");
    Domain range = new RangeDomain((RangeDomain)node.getData("ranges"));
    Domain fc_gen = null;
    Traversable tr = (Traversable)CFGraph.getIR(node);

    // Kill ranges due to visible modifications in the node.
    range.kill(DataFlowTools.getDefSymbol(tr));

    // Kill ranges due to side effects.
    // Collect new ranges returning from the callee.
    DepthFirstIterator iter = new DepthFirstIterator(tr);
    while ( iter.hasNext() )
    {
      Object o = iter.next();
      if ( o instanceof FunctionCall )
      {
        FunctionCall fc = (FunctionCall)o;
        CallSite call_site = ip_node.getCallSite(fc);
        Domain maymod = maymods.get(call_site);
        if ( maymod instanceof SetDomain )
          range.kill((Set<Symbol>)maymod);
        else if ( StandardLibrary.isSideEffectFreeExceptIO(fc) )
          ;
          //range.kill(SymbolTools.getAccessedSymbols(fc));
          // TODO: verify this is safe.
        else
          ((RangeDomain)range).killGlobal();

        // Additionally kill symbols whose address is taken.
        range.kill(getAddressTakenID(fc));

        if ( fc_gen == null )
          fc_gen = call_site.out();
        else
          fc_gen = fc_gen.union(call_site.out());
      }
    }
    range = range.intersect(fc_gen);

    for ( DFANode succ : node.getSuccs() )
      succ.putPredData(node, range);
  }

  // Side-effect-present node.
  private static void updateUnsafeNode(DFANode node)
  {
    for ( DFANode succ : node.getSuccs() )
      succ.putPredData(node, new RangeDomain());
  }

  // Side-effect-free node.
  private static void updateSafeNode(DFANode node)
  {
    RangeDomain ranges_in = node.getData("ranges");

    for ( DFANode succ : node.getSuccs() )
      succ.putPredData(node, new RangeDomain(ranges_in));
  }

  // Update assignments.
  private static void updateAssignment(DFANode node, AssignmentExpression e)
  {
    // Compute invariant information and cache it.
    if ( (Integer)node.getData("num-visits") < 2 )
    {
      Expression to = e.getLHS();
      Expression from = Symbolic.simplify(e.getRHS());
      String direction = "nochange";

      // Case 1. Dereference is treated conservatively.
      if ( IRTools.containsUnary(to, UnaryOperator.DEREFERENCE) )
        direction = "discard";

      else if ( to instanceof Identifier )
      {
        Symbol var = ((Identifier)to).getSymbol();

        // Case 1. Unknown types are treated conservatively.
        if ( var == null )
          direction = "discard";

        // Case 2. Lvalue is not tractable by range analysis.
        else if ( !isTractableType(var) )
          direction = "nochange";

        // Case 3. Lvalue is tractable but range is not.
        else if ( !isTractableRange(from) )
          direction = "kill";

        // Case 4. Typical assignment with no self dependence.
        else if ( !IRTools.containsSymbol(from, var) )
          direction = "normal";

        else 
        {
          Expression inverted = invertExpression(to, from);

          // Case 5. There is a self dependence, and it is not invertible.
          if ( inverted == null )
            direction = "recurrence";

          // Case 5.5. Inverted expression contains array access - kill it.
          else if ( IRTools.containsClass(inverted, ArrayAccess.class) )
            direction = "kill";

          // Case 6. There is a self dependence, and it is invertible.
          else
          {
            direction = "invertible";
            node.putData(direction, inverted);
          }
        }
        node.putData("assign-to", to);
        node.putData("assign-from", from);
      }

      else
      {
        node.putData("assign-to", to);
        direction = "kill";
      }

      // Case 7. Lvalue is not simple; no source of information.
      node.putData("assign-direction", direction);
    }

    RangeDomain ranges_in = node.getData("ranges");

    RangeDomain ranges_out = new RangeDomain(ranges_in);

    String direction = node.getData("assign-direction");

    Object node_data = null;

    // Dicards everything after this node.
    if ( direction.equals("discard") )
      ranges_out = new RangeDomain();

    // Kills ranges containing non-identifiers that modified at this node.
    else if ( !((node_data=node.getData("assign-to")) instanceof Identifier) )
      ranges_out.removeRangeWith(
        SymbolTools.getSymbolOf((Expression)node_data));

    // Cases where identifiers are modified.
    else if ( !direction.equals("nochange") )
    {
      Symbol var = ((Identifier)node.getData("assign-to")).getSymbol();

      // Preprocess the range to avoid replacement in array subscripts.
      ranges_out.killArraysWith(var);

      Expression replace_with = ( direction.equals("invertible") )?
        (Expression)node.getData("invertible"): ranges_out.getRange(var);

      Expression from = node.getData("assign-from");

      // Expand expressions that contain the killed symbol.
      if ( direction.equals("invertible") || direction.equals("recurrence") )
      {
        from = ranges_out.expandSymbol(from, var);
        ranges_out.expandSymbol(var);
      }

      // Eliminate the assigned symbol in the range.
      ranges_out.replaceSymbol(var, replace_with);

      // Postprocess the range by discarding cyclic ranges.
      ranges_out.removeRecurrence();

      // Remove or keep the range for the assigned symbol.
      if ( direction.equals("kill") )
        ranges_out.removeRange(var);
      else
        ranges_out.setRange(var, from);

      // Add additional ranges from the equality condition -- disable if
      // there is any problem.
      /*
      Expression eq_expr = Symbolic.eq(e.getLHS(), e.getRHS());
      RangeDomain eq_range = extractRanges(eq_expr);
      eq_range.removeRange(var);
      for ( Symbol eq_var : eq_range.getSymbols() )
        if ( ranges_out.getRange(eq_var) == null )
          ranges_out.setRange(eq_var, eq_range.getRange(eq_var));
      */
    }

    // Update successors.
    for ( DFANode succ : node.getSuccs() )
      succ.putPredData(node, ranges_out);
  }

  // Apply conditional expressions.
  private static void updateBinary(DFANode node, BinaryExpression e)
  {
    // Adjustment for unexpanded expression due to may-be evaluated expression.
    // Only safe evaluation is taken out of any expressions used in conditional
    // branches, and there should be a checking mechanism that detects unsafe
    // evaluations.
    if ( IRTools.containsSideEffect(e) )
    {
      updateUnsafeNode(node);
      return;
    }
    // This is not a branch; residue of normalization.
    else if ( node.getSuccs().size() < 2 )
    {
      updateSafeNode(node);
      return;
    }

    // Compute or retrieve the ranges from the conditional expression.
    if ( (Integer)node.getData("num-visits") < 2 )
    {
      RangeDomain true_range = extractRanges(e);
      RangeDomain false_range = extractRanges(Symbolic.negate(e));

      node.putSuccData((DFANode)node.getData("true"), true_range);
      node.putSuccData((DFANode)node.getData("false"), false_range);
    }

    RangeDomain ranges_in = node.getData("ranges");

    for ( DFANode succ : node.getSuccs() )
    {
      RangeDomain ranges_out = new RangeDomain(ranges_in);

      ranges_out.intersectRanges((RangeDomain)node.getSuccData(succ));

      // Infeasible path detected, so remove the data
      if ( ranges_in.size() > ranges_out.size() )
        ranges_out = null;

      succ.putPredData(node, ranges_out);
    }
  }

  // Apply conditional expressions from switch statement.
  private static void updateSwitch(DFANode node, SwitchStatement s)
  {
    Expression lhs = s.getExpression();
    if ( IRTools.containsSideEffect(lhs) )
    {
      updateUnsafeNode(node);
      return;
    }

    // Compute and cache the extracted ranges. 
    if ( (Integer)node.getData("num-visits") < 2 )
    {
      for ( DFANode succ : node.getSuccs() )
      {
        Object o = CFGraph.getIR(node);

        if ( o instanceof Case )
          node.putSuccData(succ, 
            extractRanges(Symbolic.eq(lhs,((Case)o).getExpression())));
      }
    }

    RangeDomain ranges_in = node.getData("ranges");

    for ( DFANode succ : node.getSuccs() )
    {
      RangeDomain ranges_out = new RangeDomain(ranges_in);

      RangeDomain edge_range = (RangeDomain)node.getSuccData(succ);

      if ( edge_range != null )
        ranges_out.intersectRanges(edge_range);

      // Infeasible path detected, so remove the data
      if ( ranges_in.size() > ranges_out.size() )
        ranges_out = null;

      succ.putPredData(node, ranges_out);
    }
  }

  // Extract a set of ranges from the preprocessed list
  private static RangeDomain extractRanges(List solved)
  {
    RangeDomain ret = new RangeDomain();
    for ( Object o : solved )
    {
      if ( !(o instanceof BinaryExpression) )
        continue;

      BinaryExpression be = (BinaryExpression)o;
      BinaryOperator op = be.getOperator();
      Expression lhs = be.getLHS();
      Expression rhs = Symbolic.simplify(be.getRHS());

      Symbol var = ((Identifier)lhs).getSymbol();

      if ( !(lhs instanceof Identifier) ||
        !isTractableType(var) ||
        !isTractableRange(rhs) )
        continue;

      if ( op == BinaryOperator.COMPARE_EQ )
        ret.setRange(var, rhs.clone());

      else if ( op == BinaryOperator.COMPARE_GE )
        ret.setRange(var, new RangeExpression(
          Symbolic.simplify(rhs),
          new InfExpression(1)));

      else if ( op == BinaryOperator.COMPARE_GT )
        ret.setRange(var, new RangeExpression(
          Symbolic.add(rhs, new IntegerLiteral(1)),
          new InfExpression(1)));

      else if ( op == BinaryOperator.COMPARE_LE )
        ret.setRange(var, new RangeExpression(
          new InfExpression(-1),
          Symbolic.simplify(rhs)));

      else if ( op == BinaryOperator.COMPARE_LT )
        ret.setRange(var, new RangeExpression(
          new InfExpression(-1),
          Symbolic.subtract(rhs, new IntegerLiteral(1))));

      else if ( op == BinaryOperator.COMPARE_NE )
        ; // too wide
      else
        ; // nothing to be done
    }
    return ret;
  }

  // Extract a set of ranges from the given binary expression
  public static RangeDomain extractRanges(Expression e)
  {
    RangeDomain ret = new RangeDomain();

    Expression simp = Symbolic.simplify(e);

    if ( !(simp instanceof BinaryExpression) )
      return ret;

    BinaryExpression be = (BinaryExpression)simp;

    List solved = Symbolic.getVariablesOnLHS(be);

    BinaryOperator op = be.getOperator();

    if ( op == BinaryOperator.LOGICAL_AND )
      for ( Object o : solved )
        ret.intersectRanges(extractRanges((List)o));

    else if ( op == BinaryOperator.LOGICAL_OR )
      for ( Object o : solved )
        ret.unionRanges(extractRanges((List)o));

    else
      ret = extractRanges(solved);

    return ret;
  }
}
