package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.test;

import java.io.File;
import java.util.Collection;
import java.util.List;

import cetus.hir.BinaryOperator;
import cetus.hir.FunctionCall;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Assembly;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Build;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeDeclspec;
import ch.unibas.cs.hpwc.patus.arch.TypeExecUnitType;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterClass;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;

public class DummyArchitectureDescription implements IArchitectureDescription
{
	@Override
	public String getBackend ()
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getInnermostLoopCodeGenerator ()
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getGeneratedFileSuffix ()
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean useFunctionPointers ()
	{
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public int getNumberOfParallelLevels ()
	{
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public boolean hasExplicitLocalDataCopies (int nParallelismLevel)
	{
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean supportsAsynchronousIO (int nParallelismLevel)
	{
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public Statement getBarrier (int nParallelismLevel)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<Specifier> getType (Specifier specType)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean useSIMD ()
	{
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public int getSIMDVectorLength (Specifier specType)
	{
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int getSIMDVectorLengthInBytes ()
	{
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int getAlignmentRestriction (Specifier specType)
	{
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public boolean supportsUnalignedSIMD ()
	{
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public List<Specifier> getDeclspecs (TypeDeclspec type)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Intrinsic getIntrinsic (String strOperation, Specifier specType)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Intrinsic getIntrinsic (UnaryOperator op, Specifier specType)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Intrinsic getIntrinsic (BinaryOperator op, Specifier specType)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Intrinsic getIntrinsic (FunctionCall fnx, Specifier specType)
	{
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public Intrinsic getIntrinsic (TypeBaseIntrinsicEnum type, Specifier specType)
	{
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public Intrinsic getIntrinsic (TypeBaseIntrinsicEnum type, Specifier specType, IOperand[] rgOperands)
	{
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public Collection<Intrinsic> getIntrinsicsByIntrinsicName (String strIntrinsicName)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Assembly getAssemblySpec ()
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getRegistersCount (TypeRegisterType type)
	{
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public Iterable<TypeRegisterClass> getRegisterClasses (TypeRegisterType type)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public TypeRegisterClass getDefaultRegisterClass (TypeRegisterType type)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<String> getIncludeFiles ()
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Build getBuild ()
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public File getFile ()
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean hasNonDestructiveOperations ()
	{
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public int getIssueRate ()
	{
		// TODO Auto-generated method stub
		return 0;
	}
	
	@Override
	public int getMinimumNumberOfExecutionUnitsPerType (Iterable<Intrinsic> itIntrinsics)
	{
		// TODO Auto-generated method stub
		return 0;
	}
	
	@Override
	public int getExecutionUnitTypesCount ()
	{
		// TODO Auto-generated method stub
		return 0;
	}
	
	@Override
	public TypeExecUnitType getExecutionUnitTypeByID (int nID)
	{
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public List<TypeExecUnitType> getExecutionUnitTypesByIDs (List<?> listIDs)
	{
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public IArchitectureDescription clone ()
	{
		// TODO Auto-generated method stub
		return null;
	}
}
