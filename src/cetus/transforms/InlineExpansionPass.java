package cetus.transforms;

import cetus.analysis.InlineExpansion;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Procedure;
import cetus.hir.Program;

/**
 * Transforms a program by performing simple subroutine in-line expansion in its main function.
 */

public class InlineExpansionPass extends TransformPass {
	
	/** Name of the inline expansion pass */
	private static final String NAME = "[InlineExpansionPass]";
	
	/**
	 * Constructs an inline expansion pass 
	 * @param program - the program to perform inline expansion on
	 */
	public InlineExpansionPass(Program program) {
		super(program);
	}
	
	@Override
	public void start() {
	    DepthFirstIterator i = new DepthFirstIterator(program);
	    i.pruneOn(Procedure.class);
	    // find the main function and perform inlining in its code
	    while(i.hasNext()){
			Object o = i.next();
			if (o instanceof Procedure){
				Procedure proc = (Procedure)o;
				String proc_name = proc.getName().toString();
				if(proc_name.equals("main") || proc_name.equals("MAIN__")){ //f2c code uses MAIN__
					new InlineExpansion().inline(proc);
				}	
			}
	    	
	    }
	}
	
	@Override
	public String getPassName() {
		return NAME;
	}
}
