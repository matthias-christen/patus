package cetus.transforms;

import cetus.hir.*;
import cetus.analysis.IPPointsToAnalysis;
import cetus.analysis.IPRangeAnalysis;

/**
 * Base class of all transformation passes. For consistent compilation, there
 * are a series of checking processes at the end of every transformation pass.
 */
public abstract class TransformPass
{
  /** The associated program */
  protected Program program;

  /** Constructs a transform pass with the given program */
  protected TransformPass(Program program)
  {
    this.program = program;
  }

  /** Returns the name of the transform pass */
  public abstract String getPassName();

  /** 
   * Invokes the specified transform pass.
   * @param pass the transform pass that is to be run.
   */
  public static void run(TransformPass pass)
  {
    double timer = Tools.getTime();
    PrintTools.println(pass.getPassName() + " begin", 0);
    //pass.startAndCheck();
    pass.start();
    PrintTools.println(pass.getPassName() + " end in " +
      String.format("%.2f seconds", Tools.getTime(timer)), 0);
    if ( !IRTools.checkConsistency(pass.program) )
      throw new InternalError("Inconsistent IR after " + pass.getPassName());
    // Updates symbol link from each identifier.
    SymbolTools.linkSymbol(pass.program);
    // Invalidates points-to relations.
    IPPointsToAnalysis.clearPointsToRelations();
    IPRangeAnalysis.clear();
    // TODO: what about ddgraph ?
  }

  /** Starts a transform pass */
  public abstract void start();
}
