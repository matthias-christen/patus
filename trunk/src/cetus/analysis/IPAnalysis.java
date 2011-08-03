package cetus.analysis;

import java.util.*;
import cetus.hir.*;
import cetus.transforms.*;

/**
 * Class IPAnalysis provides common framework for interprocedural analysis
 * passes. The following features are supported now.
 *   - Generation of call graphs.
 *   - Three solvers -- top-down, bottom-up, and worklist.
 *
 * These abstract methods need to be implemented in the child class.
 *   - analyzeProcedure(IPANode node) -- procedure analysis
 *   - updateCall(IPANode node)       -- forward data update
 *   - updateCalls(IPANode node)      -- forward data update (worklist only)
 *   - updateReturn(IPANode node)     -- backward data update
 * 
 * For example, a side-effect problem needs to provide the analyzer that
 * collects the side effects of a procedure in the "analyzerProcedure", and the
 * summarization routine in the "updateReturn" method.
 */
public abstract class IPAnalysis extends AnalysisPass
{
  // Verbosity
  protected static final int verbosity = PrintTools.getVerbosity();

  /**
   * Work space for interprocedural analysis.
   */
  protected IPAGraph callgraph;

  /**
   * The name of the analysis.
   */
  protected String name;

  /**
   * Options for the analysis.
   */
  protected Set<Option> option;

  /** Possible options to interprocedural problems */
  protected static enum Option
  {
    FORWARD,  // Data is propagated from caller to callee.
    BACKWARD, // Data is propagated from callee to caller.
    TOPDOWN,  // Traversal is from top to bottom.
    BOTTOMUP, // Traversal is from bottom to top.
    WORKLIST, // Traversal depends on the state change.
    CONTEXT1, // Minimal context-sensitivity (call site).
    NORMARG,  // Performs argument normalization.
    NORMRET,  // Normalize return statements.
    FAILFAST; // Whether the analysis fails fast upon exceptional call graph.
  }

  /**
   * Analyzes the individual procedure.
   * @param node the node associated with the procedure.
   */
  protected abstract void analyzeProcedure(IPANode node);

  /**
   * Updates the effect due to procedure call in the callee.
   * @param node the node that was just analyzed.
   * @return true if the IN data of any call site has been updated.
   */
  protected abstract boolean updateCall(IPANode node);

  /**
   * Updates the effect due to procedure return in the caller.
   * @param node the node that was just analyzed.
   * @return true if the OUT data of the node has been updated.
   */
  protected abstract boolean updateReturn(IPANode node);

  /**
   * Updates the effect due to procedure calls in the callees.
   * @param node the node that was just analyzed.
   * @return the set of call sites that were updated.
   */
  protected abstract Set<CallSite> updateCalls(IPANode node);

  /**
   * Constructs an interprocedural analysis for the given program.
   */
  protected IPAnalysis(Program program, Option... opts)
  {
    super(program);
    option = EnumSet.noneOf(Option.class);
    for ( Option opt : opts )
      option.add(opt);
    // Normalize return statements.
    if ( option.contains(Option.NORMRET) )
      TransformPass.run(new NormalizeReturn(program));
    callgraph = new IPAGraph(program);
    // Fail fast if the call graph is difficult to handle.
    if ( !isAnalyzable() )
      return;
    // Initializes IN domain and normalize arguments if requested.
    Iterator<IPANode> iter = callgraph.topiterator();
    while ( iter.hasNext() )
    {
      IPANode node = iter.next();
      for ( CallSite calling_site : node.getCallingSites() )
      {
        if ( option.contains(Option.NORMARG) )
          calling_site.normalizeArguments();
        if ( callgraph.getTopOrder(calling_site.getCaller()) >= 0 )
          node.in(calling_site, NullDomain.getNull());
      }
    }
    if ( verbosity >= 1 )
      PrintTools.printlnStatus("[IPA] Created a call graph", 0);
    if ( verbosity >= 5 )
    {
      PrintTools.printlnStatus("[IPA] Graph dump =\n" + callgraph, 0);
      PrintTools.printlnStatus("[IPA] Graph was\n" + callgraph.toDot("name", 1), 0);
    }
  }

  /**
   * Constructs an interprocedural analysis on the given call graph.
   */
  protected IPAnalysis(IPAGraph callgraph, Option... opts)
  {
    super(callgraph.getProgram());
    this.callgraph = callgraph;
    option = EnumSet.noneOf(Option.class);
    for ( Option opt : opts )
      option.add(opt);
  }

  /** Checks if it is possible to proceed with the analysis. */
  protected boolean isAnalyzable()
  {
    if ( callgraph.getRoot() == null )
    {
      PrintTools.printlnStatus(name+" Stops due to no flow entry\n",0);
      return false;
    }

    if ( !option.contains(Option.FAILFAST) )
      return true;

    // From here, the decision depends on the analysis since there might be
    // a get-around for these exceptions.
    if ( callgraph.containsFunctionPointer() )
    {
      PrintTools.printlnStatus(name+" Stops due to function pointer\n",0);
      return false;
    }

    return true;
  }

  /**
   * Starts interprocedural analysis.
   */
  public void start()
  {
    if ( !isAnalyzable() )
      return;

    compute();

    // Final intra phase.
    Iterator<IPANode> iter = callgraph.topiterator();
    while ( iter.hasNext() )
    {
      IPANode node = iter.next();
      if ( option.contains(Option.CONTEXT1) )
        removeContexts(node);
      analyzeProcedure(node.countVisits());
    }
  }

  /**
   * Returns the list of calling contexts for the given callee node.
   * Creates an empty list if there is no previously used contexts.
   * @param node the callee node.
   * @return the list.
   */
  protected static Set<CallSite> getContexts(IPANode node)
  {
    Set<CallSite> ret = node.getData("contexts");
    if ( ret == null )
    {
      ret = new LinkedHashSet<CallSite>();
      node.putData("contexts", ret);
    }
    return ret;
  }

  /**
   * Returns the outstanding context specifically stored in the given node.
   * @param node the callee node.
   * @return the outstanding calling context. 
   */
  protected static CallSite getContext(IPANode node)
  {
    CallSite ret = node.getData("context");
    return ret;
  }

  /**
   * Removes context information from the given node.
   */
  protected static void removeContexts(IPANode node)
  {
    node.removeData("context");
    node.removeData("contexts");
  }

  /**
   * Computes the data using the given type of approach.
   */
  protected void compute()
  {
    if ( option.contains(Option.CONTEXT1) )
    {
      if ( option.contains(Option.WORKLIST) )
        computeWorkListContext();
      else if ( option.contains(Option.TOPDOWN) )
        computeTopDownContext();
      else if ( option.contains(Option.BOTTOMUP) )
        computeBottomUpContext();
      else
        PrintTools.printlnStatus("[WARNING] Unknown IPAnalysis type", 0);
    }
    else
    {
      if ( option.contains(Option.WORKLIST) )
        computeWorkList();
      else if ( option.contains(Option.TOPDOWN) )
        computeTopDown();
      else if ( option.contains(Option.BOTTOMUP) )
        computeBottomUp();
      else
        PrintTools.printlnStatus("[WARNING] Unknown IPAnalysis type", 0);
    }

    PrintTools.printlnStatus(name+" finished", 1);

    if ( verbosity >= 3 )
      PrintTools.printlnStatus(callgraph.toString(""), 3);
  }

  /**
   * Drives forward or top-down iteration of the call graph with an option of
   * tailing backward phase.
   */
  private void computeTopDown()
  {
    boolean changed = true;

    while ( changed )
    {
      changed = false;
      for ( int i=0; i<=callgraph.getLastOrder(); i++ )
      {
        IPANode node = callgraph.getNode(i);
        analyzeProcedure(node.countVisits());
        if ( option.contains(Option.FORWARD) )
          changed |= updateCall(node);
        if ( option.contains(Option.BACKWARD) )
          changed |= updateReturn(node);
      }
    }
  }

  // not tested.
  private void computeTopDownContext()
  {
    boolean changed = true;
    
    while ( changed )
    {
      changed = false;
      for ( int i=0; i<=callgraph.getLastOrder(); i++ )
      {
        IPANode node = callgraph.getNode(i);
        resetContexts(node);
        while ( i==0 || hasNextContext(node) )
        {
          if ( hasNextContext(node) )
            setNextContext(node);
          analyzeProcedure(node.countVisits());
          if ( option.contains(Option.FORWARD) )
            changed |= updateCall(node);
          if ( option.contains(Option.BACKWARD) )
            changed |= updateReturn(node);
          if ( i==0 )
            break;
        }
      }
    }
  }

  /**
   * Drives backward or bottom-up iteration of the call graph with an option of
   * tailing forward phase.
   */
  private void computeBottomUp()
  {
    boolean changed = true;

    int total_visits = 0;
    while ( changed )
    {
      changed = false;
      for ( int i=callgraph.getLastOrder(); i>=0; i--,total_visits++ )
      {
        IPANode node = callgraph.getNode(i);
        analyzeProcedure(node.countVisits());
        if ( option.contains(Option.FORWARD) )
          changed |= updateCall(node);
        if ( option.contains(Option.BACKWARD) )
          changed |= updateReturn(node);
      }
    }
    if ( verbosity >=3 )
      PrintTools.printlnStatus(name+" Total visits = "+total_visits, 3);
  }

  // not tested.
  private void computeBottomUpContext()
  {
    boolean changed = true;

    int total_visits = 0;
    while ( changed )
    {
      changed = false;
      for ( int i=callgraph.getLastOrder(); i>=0; i--,total_visits++ )
      {
        IPANode node = callgraph.getNode(i);
        resetContexts(node);
        // Enters this loop only if it is the main procedure or there are
        // jobs to be done.
        while ( i==0 || hasNextContext(node) )
        {
          if ( hasNextContext(node) )
            setNextContext(node);
          analyzeProcedure(node.countVisits());
          if ( option.contains(Option.FORWARD) )
            changed |= updateCall(node);
          if ( option.contains(Option.BACKWARD) )
            changed |= updateReturn(node);
          if ( i==0 )
            break;
        }
      }
    }
    if ( verbosity >=3 )
      PrintTools.printlnStatus(name+" Total visits = "+total_visits, 3);
  }


  /**
   * Drives worklist iteration.
   */
  private void computeWorkList()
  {
    TreeSet<Integer> work_list = new TreeSet<Integer>();

    // Worklist solver chooses the preferred the node first.
    if ( option.contains(Option.TOPDOWN) )
      work_list.add(0);
    else
      work_list.add(callgraph.getLastOrder());

    int total_visits = 0;
    while ( !work_list.isEmpty() )
    {
      int node_id;

      // Worklist solver chooses the preferred the node first.
      if ( option.contains(Option.TOPDOWN) )
        node_id = work_list.first();
      else
        node_id = work_list.last();

      work_list.remove(node_id);

      if ( node_id < 0 )
        continue;

      total_visits++;

      IPANode node = callgraph.getNode(node_id);

      analyzeProcedure(node.countVisits());

      // Updates forward data.
      if ( option.contains(Option.FORWARD) )
      {
        Set<CallSite> changed = updateCalls(node);
        for ( CallSite call_site : changed )
          work_list.add(callgraph.getTopOrder(call_site.getCallee()));
      }

      // Updates backward data.
      if ( option.contains(Option.BACKWARD) )
      {
        boolean changed = updateReturn(node);
        if ( changed )
          for ( IPANode callers : node.getCallers() )
            work_list.add(callgraph.getTopOrder(callers));
      }
    }
    if ( verbosity >=3 )
      PrintTools.printlnStatus(name+" Total visits = "+total_visits, 3);
  }

  // Context-sensitive work-list solver.
  private void computeWorkListContext()
  {
    TreeSet<Integer> work_list = new TreeSet<Integer>();

    // Worklist solver chooses the preferred the node first.
    if ( option.contains(Option.TOPDOWN) )
      work_list.add(0);
    else
      work_list.add(callgraph.getLastOrder());

    int total_visits = 0;
    while ( !work_list.isEmpty() )
    {
      int node_id;

      // Worklist solver chooses the preferred the node first.
      if ( option.contains(Option.TOPDOWN) )
        node_id = work_list.first();
      else
        node_id = work_list.last();

      work_list.remove(node_id);

      if ( node_id < 0 )
        continue;

      total_visits++;

      IPANode node = callgraph.getNode(node_id);
      resetContexts(node);

      PrintTools.printStatus(name+" visiting procedure ["+node.getName()+"]", 3);
      PrintTools.printlnStatus(", #contexts = "+getContexts(node).size(), 3);

      // Enters this loop only if it is the main procedure or there are
      // jobs to be done.
      while ( node_id == 0 || hasNextContext(node) )
      {
        // Stages the next context to be analyzed. The result of this operation
        // is used while invoking node.in().
        if ( hasNextContext(node) )
          setNextContext(node);

        PrintTools.printStatus(name+" context = "+getContext(node), 3);
        PrintTools.printlnStatus(", in = "+node.in(), 3);

        analyzeProcedure(node.countVisits());

        // Updates forward data.
        if ( option.contains(Option.FORWARD) )
        {
          Set<CallSite> changed = updateCalls(node);
          PrintTools.printlnStatus(name + " changed callee " + changed, 3);
          for ( CallSite call_site : changed )
          {
            // Add affected call site (context) to the callee's work list.
            if ( call_site.getCallee() != null )
              getContexts(call_site.getCallee()).add(call_site);
            work_list.add(callgraph.getTopOrder(call_site.getCallee()));
          }
        }

        // Updates backward data.
        if ( option.contains(Option.BACKWARD) )
        {
          boolean changed = updateReturn(node);
          PrintTools.printlnStatus(name + " changed caller " + changed, 3);
          if ( changed )
            for ( IPANode callers : node.getCallers() )
              work_list.add(callgraph.getTopOrder(callers));
        }

        if ( node_id == 0 )
          break;
      }
    }
    if ( verbosity >=3 )
      PrintTools.printlnStatus(name+" Total visits = "+total_visits, 3);
  }

  /**
   * Returns the name of this analysis.
   */
  public String getPassName()
  {
    return name;
  }

  /**
   * Stores the result of the analysis to the node database, to be used with
   * other analysis passes.
   */
  protected void saveData()
  {
    Iterator<IPANode> iter = callgraph.topiterator();
    while ( iter.hasNext() )
    {
      IPANode node = iter.next();
      Map<CallSite, Domain> call_data; 
      if ( option.contains(Option.FORWARD) )
      {
        node.putData(name+"IN", node.in());
        call_data = new LinkedHashMap<CallSite, Domain>();
        for ( CallSite call_site : node.getCallSites() )
          call_data.put(call_site, call_site.in());
        node.putData(name+"CallIN", call_data);
      }
      if ( option.contains(Option.BACKWARD) )
      {
        node.putData(name+"OUT", node.out());
        call_data = new LinkedHashMap<CallSite, Domain>();
        for ( CallSite call_site : node.getCallSites() )
          call_data.put(call_site, call_site.out());
        node.putData(name+"CallOUT", call_data);
      }
    }
    if ( verbosity >= 5 )
    {
      if ( option.contains(Option.FORWARD) && option.contains(Option.BACKWARD) )
        PrintTools.printlnStatus(callgraph.toDot(
        name+"IN,"+name+"CallIN,"+name+"OUT,"+name+"CallOUT",4),5);
      else if ( option.contains(Option.FORWARD) )
        PrintTools.printlnStatus(callgraph.toDot(
        name+"IN,"+name+"CallIN",2),5);
      else if ( option.contains(Option.BACKWARD) )
        PrintTools.printlnStatus(callgraph.toDot(
        name+"OUT,"+name+"CallOUT",2),5);
    }
  }

  // Returns the next context of the given callee node.
  private static CallSite getNextContext(IPANode node)
  {
    assert hasNextContext(node) : "inconsistent calling contexts.";
    Iterator<CallSite> iter = node.getData("context-iter");
    return iter.next();
  }

  // Checks if the given node contains the next calling context to analyze.
  private static boolean hasNextContext(IPANode node)
  {
    Iterator<CallSite> iter = node.getData("context-iter");
    assert iter != null : "inconsistent calling contexts";
    return iter.hasNext();
  }

  // Resets the the context iterator from the stored list of contexts.
  private static void resetContexts(IPANode node)
  {
    Set<CallSite> contexts = getContexts(node);
    node.putData("context-iter", contexts.iterator());
  }

  // Stages the next context to be analyzed.
  private static void setNextContext(IPANode node)
  {
    node.putData("context", getNextContext(node));
  }

  /**
   * Returns the pretty-printed domain information for the given input.
   * @param t the traversable object to be visited.
   * @param domain the map to be printed.
   * @param indent the size of indentation for the visited object.
   * @return the pretty-printed domain information.
   */
  protected static <D extends Domain> String toPrettyDomain
  (Traversable t, Map<Statement, D> domain, Integer indent)
  {
    String ret = "", tab = "";

    for ( int i=0; i<indent; ++i )
      tab += "  ";

    if ( t instanceof Procedure )
    {
      Procedure p = (Procedure)t;
      ret += "Domain for Procedure "+p.getName()+"\n";
      ret += toPrettyDomain(p.getBody(), domain, indent);
    }

    else if ( t instanceof CompoundStatement )
    {
      ret += tab+"{\n";
      indent++;

      for ( Object o: t.getChildren() )
        ret += toPrettyDomain((Traversable)o, domain, indent);

      indent--;
      ret += tab+"}\n";
    }

    else if ( t instanceof DoLoop )
    {
      DoLoop d = (DoLoop)t;
      ret += tab+"do\n";
      ret += toPrettyDomain(d.getBody(), domain, indent);
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
      ret += toPrettyDomain(f.getBody(), domain, indent);
    }

    else if ( t instanceof IfStatement )
    {
      IfStatement i = (IfStatement)t;
      ret += tab+"if ( "+i.getControlExpression()+" )\n";
      ret += toPrettyDomain(i.getThenStatement(), domain, indent);
      Statement els = i.getElseStatement();
      if ( els != null )
        ret += tab+"else\n"+toPrettyDomain(els, domain, indent);
    }

    else if ( t instanceof SwitchStatement )
    {
      SwitchStatement s = (SwitchStatement)t;
      ret += tab+"switch ( "+s.getExpression()+" )\n";
      ret += toPrettyDomain(s.getBody(), domain, indent);
    }

    else if ( t instanceof WhileLoop )
    {
      WhileLoop w = (WhileLoop)t;
      ret += tab+"while ( "+w.getCondition()+" )\n";
      ret += toPrettyDomain(w.getBody(), domain, indent);
    }

    else if ( t instanceof Statement )
      ret += tab+t+"\n";


    if ( t instanceof Statement )
    {
      Domain d = domain.get(t);
      ret = "        "+((d==null)? "[]": d)+"\n"+ret;
    }
    return ret;
  }

}
