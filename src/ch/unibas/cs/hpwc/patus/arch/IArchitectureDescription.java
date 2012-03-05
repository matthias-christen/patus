/*******************************************************************************
 * Copyright (c) 2011 Matthias-M. Christen, University of Basel, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 *
 * Contributors:
 *     Matthias-M. Christen, University of Basel, Switzerland - initial API and implementation
 ******************************************************************************/
package ch.unibas.cs.hpwc.patus.arch;

import java.io.File;
import java.util.List;

import cetus.hir.BinaryOperator;
import cetus.hir.FunctionCall;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Assembly;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Build;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;

/**
 * Describes data types, SIMD vector lengths, etc. for a certain
 * type of hardware in a certain programming model for which the
 * code is being generated.
 *
 * @author Matthias-M. Christen
 */
public interface IArchitectureDescription
{
	/**
	 * Returns the name of the backend for which to generate code.
	 * @return The backend name
	 */
	public abstract String getBackend ();
	
	/**
	 * Returns the name of the backend code generator used to generate the innermost loops
	 * containing the stencil computation.
	 * @return The name of the innermost loop code generator
	 */
	public abstract String getInnermostLoopCodeGenerator ();

	/**
	 * Returns the file suffix of the file to which the code generator writes the output.
	 * @return The suffix of the output file
	 */
	public abstract String getGeneratedFileSuffix ();

	/**
	 * Specifies whether function pointers can be used to select code variants.
	 * @return
	 */
	public abstract boolean useFunctionPointers ();

	/**
	 * Returns the number of parallelism level this hardware has
	 * (the finest level being threads).
	 * CPUs have one level, for instance, GPUs can have parallel
	 * blocks and parallel threads within a block, hence there are
	 * two levels.
	 * @return The number of parallelism levels the hardware supports
	 */
	public abstract int getNumberOfParallelLevels ();

	/**
	 * Specifies whether explicit local copies of the data are allocated
	 * (c.f. Cell BE).
	 * @param nParallelismLevel The parallelism level. 0 is the outermost level with no parallelism,
	 * 	1 is the first level with parallelism, etc.
	 * @return
	 */
	public abstract boolean hasExplicitLocalDataCopies (int nParallelismLevel);

	/**
	 * Specifies whether the parallelism supports asynchronous IO, i.e.,
	 * overlapping computation and communication.
	 * @param nParallelismLevel The parallelism level. 0 is the outermost level with no parallelism,
	 * 	1 is the first level with parallelism, etc.
	 * @return
	 */
	public abstract boolean supportsAsynchronousIO (int nParallelismLevel);

	/**
	 * Returns a barrier intrinsic statement, which will synchronize all execution
	 * units in parallelism level <code>nParallelismLevel</code>
	 * @param nParallelismLevel The parallelism level on which to generate a barrier
	 * @return A barrier statement for parallelism level <code>nParallelismLevel</code>
	 */
	public abstract Statement getBarrier (int nParallelismLevel);

	/**
	 * Returns a variable type specifier (acknowledging SIMD) for
	 * a given basic data type, <code>specType</code>.
	 * @param specifier
	 * @return
	 */
	public abstract List<Specifier> getType (Specifier specType);

	/**
	 * Returns <code>true</code> iff the code is to be vectorized
	 * (i.e., SIMD datatypes were specified in the architecture description).
	 * @return <code>true</code> iff the code is to be vectorized
	 */
	public abstract boolean useSIMD ();

	/**
	 * Returns the length of a SIMD vector for the given basic data type
	 * <code>specType</code>.
	 * @return
	 */
	public abstract int getSIMDVectorLength (Specifier specType);

	/**
	 * Returns the factor at which memory addresses must be aligned for the
	 * promoted type corresponding to the primitive type <code>specType</code>.
	 * Returns 1 if no restrictions apply.
	 * @return
	 */
	public abstract int getAlignmentRestriction (Specifier specType);
	
	/**
	 * Determines whether the architecture supports unaligned SIMD vector loads and stores.
	 * @return <code>true</code> iff unaligned vector moves are supported
	 */
	public abstract boolean supportsUnalignedSIMD ();

	/**
	 * Gets additional declspecs for a type (e.g., the kernel function)
	 * @param strType The type of object for which to get additional declspecs
	 * @return A list of specifiers
	 */
	public abstract List<Specifier> getDeclspecs (TypeDeclspec type);

	/**
	 * Returns the {@link NameID} for the intrinsic replacing the operation given by its name, <code>strOperation</code>
	 * for a given type <code>specType</code>.
	 * @param strOperation The operation name:
	 * @param specType The type for which to get the intrinsic
	 * @return The {@link NameID} for the intrinsic replacing the operation <code>strOperation</code>
	 */
	public abstract Intrinsic getIntrinsic (String strOperation, Specifier specType);

	/**
	 * Returns the {@link NameID} for the intrinsic replacing the unary arithmetic operator <code>op</code>
	 * for a given type <code>specType</code>.
	 * @param op The arithmetic operator
	 * @param specType The type for which to get the intrinsic
	 * @return The {@link NameID} for the intrinsic replacing the arithmetic operator <code>op</code>
	 */
	public abstract Intrinsic getIntrinsic (UnaryOperator op, Specifier specType);

	/**
	 * Returns the {@link NameID} for the intrinsic replacing the arithmetic operator <code>op</code>
	 * for a given type <code>specType</code>.
	 * @param op The arithmetic operator
	 * @param specType The type for which to get the intrinsic
	 * @return The {@link NameID} for the intrinsic replacing the arithmetic operator <code>op</code>
	 */
	public abstract Intrinsic getIntrinsic (BinaryOperator op, Specifier specType);

	/**
	 * Returns the name of the intrinsic for the function <code>fnx</code>.
	 * @param fnx
	 * @param specType
	 * @return
	 */
	public abstract Intrinsic getIntrinsic (FunctionCall fnx, Specifier specType);
	
	/**
	 * Returns the assembly specification (registers, ...)
	 * @return
	 */
	public abstract Assembly getAssemblySpec ();

	/**
	 * Returns the number of available registers of type <code>type</code>
	 * @param type The register type
	 * @return The number of available registers
	 */
	public abstract int getRegistersCount (TypeRegisterType type);

	/**
	 * Returns a list of architecture-specific header files that need to be included.
	 * @return
	 */
	public abstract List<String> getIncludeFiles ();

	/**
	 * Returns information on how to build the benchmark harness.
	 * @return The build information
	 */
	public abstract Build getBuild ();

	/**
	 * Clones the architecture description.
	 * @return A copy of this architecture description
	 */
	public abstract IArchitectureDescription clone ();

	/**
	 * Returns the file from which the hardware description was loaded or <code>null</code> if it wasn't loaded from file.
	 * @return The file from which the hardware description was loaded or <code>null</code> if it wasn't loaded from file
	 */
	public abstract File getFile ();
}
