package cetus.codegen;

import cetus.hir.*;
import cetus.exec.*;
import cetus.analysis.*;

import java.util.*;

/**
 * This pass looks for Annotations that provide
 * enough information to add OpenMP pragmas and
 * then inserts those pragmas.
 */
public class ompGen extends CodeGenPass
{
  /**
  * Possible options for handling the existing OpenMP and Cetus-internal
  * pragmas.
  */
  private static enum Option
  {
    REMOVE_OMP_PRAGMA,   // Avoid printing of the existing OpenMP pragma.
    REMOVE_CETUS_PRAGMA, // Avoid printing of the existing Cetus pragma.
    COMMENT_OMP_PRAGMA,  // Comment out the existing OpenMP pragam.
    COMMENT_CETUS_PRAGMA // Comment out the existing Cetus pragma.
  }

  /** Set of options for OpenMP generation */
  private static Set<Option> option;

  // Reads in the command line option.
  static
  {
    option = EnumSet.noneOf(Option.class);
    Integer option_value;
    try {
      option_value = Integer.valueOf(Driver.getOptionValue("ompGen"));
    } catch (NumberFormatException e) {
      option_value = 1;
    }
    switch (option_value) {
      case 2: option.add(Option.REMOVE_OMP_PRAGMA); break;
      case 3: option.add(Option.REMOVE_CETUS_PRAGMA); break; 
      case 4: option.add(Option.REMOVE_OMP_PRAGMA);
              option.add(Option.REMOVE_CETUS_PRAGMA); break;
      default:
    }
  }

  public ompGen(Program program)
  {
    super(program);
  }

  public String getPassName()
  {
    return new String("[ompGen]");
  }

	public void start()
	{
		DepthFirstIterator iter = new DepthFirstIterator(program);
		LinkedList<ForLoop> loops = iter.getList(ForLoop.class);

		for (ForLoop loop : loops)
		{
			genOmpParallelLoops(loop);
		}
	}

	private void genOmpParallelLoops(ForLoop loop)
	{
    // handling of existing annotations.
    if ( option.contains(Option.REMOVE_OMP_PRAGMA) )
      Annotation.hideAnnotations(loop, OmpAnnotation.class);
    if ( option.contains(Option.REMOVE_CETUS_PRAGMA) )
      Annotation.hideAnnotations(loop, CetusAnnotation.class);

		// currently, we check only omp parallel for construct
		if ( !loop.containsAnnotation(CetusAnnotation.class, "parallel") )
			return;

		// if the loop already contains an OpenMP parallel construct,
		// return
		if ( loop.containsAnnotation(OmpAnnotation.class, "for") )
			return;

		OmpAnnotation omp_annot = new OmpAnnotation();
		List<CetusAnnotation> cetus_annots =
				loop.getAnnotations(CetusAnnotation.class);

		for ( CetusAnnotation cetus_annot : cetus_annots )
			omp_annot.putAll(cetus_annot);

		omp_annot.put("for", "true");
		loop.annotateBefore(omp_annot);
	}
}
