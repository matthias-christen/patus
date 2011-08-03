package ch.unibas.cs.hpwc.patus.codegen.backend;

import cetus.hir.Expression;
import ch.unibas.cs.hpwc.patus.ast.StatementList;

/**
 *
 * @author Matthias-M. Christen
 */
public interface INonKernelFunctions
{
	/**
	 * Initialize code generator data structures
	 */
	public abstract void initializeNonKernelFunctionCG ();
	
	
	///////////////////////////////////////////////////////////////////
	// Patus Pragmas

	public abstract StatementList forwardDecls ();

	public abstract StatementList declareGrids ();

	public abstract StatementList allocateGrids ();

	public abstract StatementList initializeGrids ();

	public abstract StatementList sendData ();

	public abstract StatementList receiveData ();

	public abstract StatementList computeStencil ();

	public abstract StatementList validateComputation ();

	public abstract StatementList deallocateGrids ();


	///////////////////////////////////////////////////////////////////
	// Patus Variables

	public abstract Expression getFlopsPerStencil ();

	public abstract Expression getGridPointsCount ();

	public abstract Expression getBytesTransferred ();

	public abstract Expression getDoValidation ();

	public abstract Expression getValidates ();
}
