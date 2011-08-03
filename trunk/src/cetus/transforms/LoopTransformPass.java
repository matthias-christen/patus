package cetus.transforms;

import java.util.*;

import cetus.hir.*;

public abstract class LoopTransformPass extends TransformPass
{
  protected LoopTransformPass(Program program)
  {
    super(program);
  }

  public abstract void transformLoop(Loop loop);

  public void start()
  {
    PostOrderIterator iter = new PostOrderIterator(program);

    for (;;)
    {
      Loop loop = null;

      try {
        loop = (Loop)iter.next(Loop.class);
      } catch (NoSuchElementException e) {
        break;
      }

      System.out.println("calling transformLoop");
      transformLoop(loop);
    }
  }
}
