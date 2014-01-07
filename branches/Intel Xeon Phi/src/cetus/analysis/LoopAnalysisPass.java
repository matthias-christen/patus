package cetus.analysis;

import java.util.*;

import cetus.hir.*;

public abstract class LoopAnalysisPass extends AnalysisPass
{
  protected LoopAnalysisPass(Program program)
  {
    super(program);
  }

  public abstract void analyzeLoop(Loop loop);

  public void start()
  {
    DepthFirstIterator iter = new DepthFirstIterator(program);

    for (;;)
    {
      Loop loop = null;

      try {
        loop = (Loop)iter.next(Loop.class);
      } catch (NoSuchElementException e) {
        break;
      }

      analyzeLoop(loop);
    }
  }
}
