/*******************************************************************************
 * Copyright (c) 2011 Matthias-M. Christen, University of Basel, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 * Matthias-M. Christen, University of Basel, Switzerland - initial API and implementation
 ******************************************************************************/
package ch.unibas.cs.hpwc.patus.arch;

import java.io.File;
import java.util.Collection;
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
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;

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
	 * 
	 * @return The backend name
	 */
	public abstract String getBackend ();

	/**
	 * Returns the name of the backend code generator used to generate the innermost loops
	 * containing the stencil computation.
	 * 
	 * @return The name of the innermost loop code generator
	 */
	public abstract String getInnermostLoopCodeGenerator ();

	/**
	 * Returns the file suffix of the file to which the code generator writes the output.
	 * 
	 * @return The suffix of the output file
	 */
	public abstract String getGeneratedFileSuffix ();

	/**
	 * Specifies whether function pointers can be used to select code variants.
	 * 
	 * @return
	 */
	public abstract boolean useFunctionPointers ();

	/**
	 * Returns the number of parallelism level this hardware has
	 * (the finest level being threads).
	 * CPUs have one level, for instance, GPUs can have parallel
	 * blocks and parallel threads within a block, hence there are
	 * two levels.
	 * 
	 * @return The number of parallelism levels the hardware supports
	 */
	public abstract int getNumberOfParallelLevels ();

	/**
	 * Specifies whether explicit local copies of the data are allocated
	 * (c.f. Cell BE).
	 * 
	 * @param nParallelismLevel
	 *            The parallelism level. 0 is the outermost level with no parallelism,
	 *            1 is the first level with parallelism, etc.
	 * @return
	 */
	public abstract boolean hasExplicitLocalDataCopies (int nParallelismLevel);

	/**
	 * Specifies whether the parallelism supports asynchronous IO, i.e.,
	 * overlapping computation and communication.
	 * 
	 * @param nParallelismLevel
	 *            The parallelism level. 0 is the outermost level with no parallelism,
	 *            1 is the first level with parallelism, etc.
	 * @return
	 */
	public abstract boolean supportsAsynchronousIO (int nParallelismLevel);

	/**
	 * Returns a barrier intrinsic statement, which will synchronize all execution
	 * units in parallelism level <code>nParallelismLevel</code>
	 * 
	 * @param nParallelismLevel
	 *            The parallelism level on which to generate a barrier
	 * @return A barrier statement for parallelism level <code>nParallelismLevel</code>
	 */
	public abstract Statement getBarrier (int nParallelismLevel);

	/**
	 * Returns a variable type specifier (acknowledging SIMD) for
	 * a given basic data type, <code>specType</code>.
	 * 
	 * @param specifier
	 * @return
	 */
	public abstract List<Specifier> getType (Specifier specType);

	/**
	 * Returns <code>true</code> iff the code is to be vectorized
	 * (i.e., SIMD datatypes were specified in the architecture description).
	 * 
	 * @return <code>true</code> iff the code is to be vectorized
	 */
	public abstract boolean useSIMD ();

	/**
	 * Returns the length of a SIMD vector for the given basic data type <code>specType</code>.
	 * 
	 * @return
	 */
	public abstract int getSIMDVectorLength (Specifier specType);

	public abstract int getSIMDVectorLengthInBytes ();

	/**
	 * Returns the factor at which memory addresses must be aligned for the
	 * promoted type corresponding to the primitive type <code>specType</code>.
	 * Returns 1 if no restrictions apply.
	 * 
	 * @return
	 */
	public abstract int getAlignmentRestriction (Specifier specType);

	/**
	 * Determines whether the architecture supports unaligned SIMD vector loads and stores.
	 * 
	 * @return <code>true</code> iff unaligned vector moves are supported
	 */
	public abstract boolean supportsUnalignedSIMD ();

	/**
	 * Gets additional declspecs for a type (e.g., the kernel function)
	 * 
	 * @param strType
	 *            The type of object for which to get additional declspecs
	 * @return A list of specifiers
	 */
	public abstract List<Specifier> getDeclspecs (TypeDeclspec type);

	/**
	 * Returns the {@link NameID} for the intrinsic replacing the operation given by its name, <code>strOperation</code> for a given type <code>specType</code>.
	 * 
	 * @param strOperation
	 *            The operation name:
	 * @param specType
	 *            The type for which to get the intrinsic
	 * @return The {@link NameID} for the intrinsic replacing the operation <code>strOperation</code>
	 */
	public abstract Intrinsic getIntrinsic (String strOperation, Specifier specType);

	/**
	 * Returns the {@link NameID} for the intrinsic replacing the unary arithmetic operator <code>op</code> for a given type <code>specType</code>.
	 * 
	 * @param op
	 *            The arithmetic operator
	 * @param specType
	 *            The type for which to get the intrinsic
	 * @return The {@link NameID} for the intrinsic replacing the arithmetic operator <code>op</code>
	 */
	public abstract Intrinsic getIntrinsic (UnaryOperator op, Specifier specType);

	/**
	 * Returns the {@link NameID} for the intrinsic replacing the arithmetic operator <code>op</code> for a given type <code>specType</code>.
	 * 
	 * @param op
	 *            The arithmetic operator
	 * @param specType
	 *            The type for which to get the intrinsic
	 * @return The {@link NameID} for the intrinsic replacing the arithmetic operator <code>op</code>
	 */
	public abstract Intrinsic getIntrinsic (BinaryOperator op, Specifier specType);

	/**
	 * Returns the name of the intrinsic for the function <code>fnx</code>.
	 * 
	 * @param fnx
	 * @param specType
	 * @return
	 */
	public abstract Intrinsic getIntrinsic (FunctionCall fnx, Specifier specType);
	
	/**
	 * Returns the merged intrinsic for the base intrinsic <code>type</code> and the datatype <code>specType</code>.
	 * @param type
	 * @param specType
	 * @return
	 */
	public abstract Intrinsic getIntrinsic (TypeBaseIntrinsicEnum type, Specifier specType);
	
	/**
	 * Returns a particular intrinsic for the base intrinsic <code>type</code>,
	 * the datatype <code>specType</code>, and a set of operands, which have to
	 * correspond to the types of <code>rgOperands</code>.
	 * 
	 * @param type
	 * @param specType
	 * @param rgOperands
	 * @return
	 */
	public abstract Intrinsic getIntrinsic (TypeBaseIntrinsicEnum type, Specifier specType, IOperand[] rgOperands);
	
	/**
	 * 
	 * @param strIntrinsicName
	 * @return
	 */
	public abstract Collection<Intrinsic> getIntrinsicsByIntrinsicName (String strIntrinsicName);

	/**
	 * Returns the assembly specification (registers, ...)
	 * 
	 * @return
	 */
	public abstract Assembly getAssemblySpec ();

	/**
	 * Returns the number of available registers of type <code>type</code>
	 * 
	 * @param type
	 *            The register type
	 * @return The number of available registers
	 */
	public abstract int getRegistersCount (TypeRegisterType type);
	
	/**
	 * Returns an iterable over all the register classes for the register type
	 * <code>type</code>.
	 * The iterable will iterate over the register classes in descending order
	 * of the register widths.
	 * 
	 * @param type
	 *            The register type for which to find the register classes
	 * @return An iterable iterating in descending order of register width over
	 *         all the register classes
	 *         of type <code>type</code>
	 */
	public abstract Iterable<TypeRegisterClass> getRegisterClasses (TypeRegisterType type);
	
	/**
	 * Returns the default register class (the one declared in the outermost
	 * register) corresponding to the register type <code>type</code>.
	 * 
	 * @param type
	 *            The register type for which to retrieve the default register
	 *            class
	 * @return The default register class for the register type
	 *         <code>type</code>
	 */
	public abstract TypeRegisterClass getDefaultRegisterClass (TypeRegisterType type);

	/**
	 * Returns a list of architecture-specific header files that need to be included.
	 * 
	 * @return
	 */
	public abstract List<String> getIncludeFiles ();

	/**
	 * Returns information on how to build the benchmark harness.
	 * 
	 * @return The build information
	 */
	public abstract Build getBuild ();

	/**
	 * Clones the architecture description.
	 * 
	 * @return A copy of this architecture description
	 */
	public abstract IArchitectureDescription clone ();

	/**
	 * Returns the file from which the hardware description was loaded or
	 * <code>null</code> if it wasn't loaded from file.
	 * 
	 * @return The file from which the hardware description was loaded or
	 *         <code>null</code> if it wasn't loaded from file
	 */
	public abstract File getFile ();

	/**
	 * Returns <code>true</code> iff the intrinsics described in the
	 * architecture description are non-destructive operations, i.e., if the
	 * input operands are not overwritten with the output.
	 * 
	 * @return <code>true</code> iff the operations have non-destructive
	 *         operands
	 */
	public abstract boolean hasNonDestructiveOperations ();
	
	/**
	 * Returns the number of instructions the processor can issue simultaneously
	 * in one clock cycle.
	 * 
	 * @return The number of instructions the processor can issue per clock
	 *         cycle
	 */
	public abstract int getIssueRate ();
	
	/**
	 * Returns the minimum number of execution units of the execution unit types
	 * to which the intrinsics in <code>itIntrinsics</code> are mapped.
	 * E.g., if <code>itIntrinsics</code> only contains &quot;load&quot;
	 * instructions and there are two execution units capable of
	 * &quot;load&quot;s, the function will return two.
	 * If, in addition, <code>itIntrinsics</code> also contains a
	 * &quot;store&quot; and there is only one unit that can handle
	 * &quot;store&quot;s, one is returned instead.
	 * 
	 * @param itIntrinsics
	 *            An iterable over intrinsics for which to get the minimum
	 *            number of execution units
	 * @return The minimum number of execution units of the execution unit types
	 *         to which the intrinsics in <code>itIntrinsics</code> are mapped
	 */
	public abstract int getMinimumNumberOfExecutionUnitsPerType (Iterable<Intrinsic> itIntrinsics);
	
	/**
	 * Returns the number of execution unit types defined in the architecture
	 * specification.
	 * 
	 * @return The number of execution unit types
	 */
	public abstract int getExecutionUnitTypesCount ();
	
	/**
	 * Finds an execution unit type by its ID. The IDs are used in the
	 * <code>exec-unit-type-ids</code> attribute of intrinsics.
	 * 
	 * @param nID
	 *            The logical ID of the execution unit
	 * @return The execution unit type corresponding to the ID <code>nID</code>
	 *         or <code>null</code> if there is no execution unit type for the
	 *         ID specified
	 */
	public abstract TypeExecUnitType getExecutionUnitTypeByID (int nID);
	
	public abstract List<TypeExecUnitType> getExecutionUnitTypesByIDs (List<?> listIDs);
}
