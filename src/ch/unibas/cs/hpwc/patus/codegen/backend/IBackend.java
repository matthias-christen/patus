package ch.unibas.cs.hpwc.patus.codegen.backend;

/**
 * The backend code generator.
 * @author Matthias-M. Christen
 */
public interface IBackend extends IParallel, IDataTransfer, IIndexing, IArithmetic, IAdditionalKernelSpecific, INonKernelFunctions
{
}
