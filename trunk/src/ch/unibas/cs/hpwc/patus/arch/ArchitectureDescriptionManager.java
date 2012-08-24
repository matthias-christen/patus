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
import java.lang.reflect.Field;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.annotation.XmlType;

import cetus.hir.AnnotationStatement;
import cetus.hir.BinaryOperator;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.NameID;
import cetus.hir.PragmaAnnotation;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.UnaryOperator;
import cetus.hir.UserSpecifier;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Assembly;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Build;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Datatypes.Datatype;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Declspecs.Declspec;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Includes.Include;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Parallelism.Level;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Parallelism.Level.Barrier;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Argument;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Arguments;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class ArchitectureDescriptionManager
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	protected static class HardwareDescription implements IArchitectureDescription
	{
		private File m_file;

		private TypeArchitectureType m_type;

		private Map<String, Datatype> m_mapDataTypes;
		private Map<TypeDeclspec, Declspec> m_mapDeclspecs;
		private Map<String, Datatype> m_mapDataTypesFromBase;
		private Map<String, List<Intrinsic>> m_mapOperationsToIntrinsics;
		private Map<String, List<Intrinsic>> m_mapOperationsToMergedIntrinsics;
		private Map<String, List<Intrinsic>> m_mapIntrinsicNamesToIntrinsics;
		private Map<Integer, TypeExecUnitType> m_mapExecUnitTypes;
		
		private boolean m_bHasNonDestructiveOperations;


		public HardwareDescription (File file, TypeArchitectureType type)
		{
			m_file = file;
			m_type = type;

			m_mapDataTypes = new HashMap<> ();

			m_mapDeclspecs = new HashMap<> ();
			if (m_type.getDeclspecs () != null)
				for (Declspec d : m_type.getDeclspecs ().getDeclspec ())
					m_mapDeclspecs.put (d.getType (), d);

			m_mapDataTypesFromBase = new HashMap<> ();
			if (m_type.getDatatypes () != null)
				for (Datatype datatype : m_type.getDatatypes ().getDatatype ())
				{
					m_mapDataTypes.put (datatype.getName (), datatype);

					if (m_mapDataTypesFromBase.containsKey (datatype.getBasetype ().toString ()))
						throw new RuntimeException ("A hardware description must not define multiple datatypes for one base type.");
					m_mapDataTypesFromBase.put (datatype.getBasetype ().value (), datatype);
				}

			m_bHasNonDestructiveOperations = true;
			
			m_mapOperationsToIntrinsics = new HashMap<> ();
			m_mapOperationsToMergedIntrinsics = new HashMap<> ();
			m_mapIntrinsicNamesToIntrinsics = new HashMap<> ();
			
			if (m_type.getIntrinsics () != null)
			{
				for (Intrinsic intrinsic : m_type.getIntrinsics ().getIntrinsic ())
				{
					List<Intrinsic> listIntrinsics = m_mapOperationsToIntrinsics.get (intrinsic.getBaseName ());
					if (listIntrinsics == null)
						m_mapOperationsToIntrinsics.put (intrinsic.getBaseName (), listIntrinsics = new ArrayList<> ());
					listIntrinsics.add (intrinsic);
					
					List<Intrinsic> listMergedIntrinsics = m_mapOperationsToMergedIntrinsics.get (intrinsic.getBaseName ());
					if (listMergedIntrinsics == null)
						m_mapOperationsToMergedIntrinsics.put (intrinsic.getBaseName (), listMergedIntrinsics = new ArrayList<> ());
					mergeIntrinsics (listMergedIntrinsics, intrinsic);
					
					List<Intrinsic> listSameNameIntrinsics = m_mapIntrinsicNamesToIntrinsics.get (intrinsic.getName ());
					if (listSameNameIntrinsics == null)
						m_mapIntrinsicNamesToIntrinsics.put (intrinsic.getName (), listSameNameIntrinsics = new ArrayList<> ());
					listSameNameIntrinsics.add (intrinsic);
					
					if (m_bHasNonDestructiveOperations && (
						intrinsic.getBaseName ().equals (TypeBaseIntrinsicEnum.PLUS.value ()) ||
						intrinsic.getBaseName ().equals (TypeBaseIntrinsicEnum.MINUS.value ()) ||
						intrinsic.getBaseName ().equals (TypeBaseIntrinsicEnum.MULTIPLY.value ()) ||
						intrinsic.getBaseName ().equals (TypeBaseIntrinsicEnum.DIVIDE.value ())))
					{
						if (intrinsic.getArguments () != null && Arguments.parseArguments (intrinsic.getArguments ()).length != 3)
							m_bHasNonDestructiveOperations = false;
					}
				}
			}
			
			m_mapExecUnitTypes = new HashMap<> ();
			if (m_type.getAssembly () != null && m_type.getAssembly ().getExecUnitTypes () != null)
			{
				for (TypeExecUnitType t : m_type.getAssembly ().getExecUnitTypes ().getExecUnitType ())
					m_mapExecUnitTypes.put (t.getId ().intValue (), t);
				checkExecUnitTypeIDs ();
			}
		}
		
		private static void mergeIntrinsics (List<Intrinsic> listIntrinsics, Intrinsic intrinsic)
		{
			String strName = intrinsic.getName ();
			
			Intrinsic intrinsicToRemove = null;
			Intrinsic intrinsicNew = null;
			
			for (Intrinsic i : listIntrinsics)
			{
				if (i.getName ().equals (strName))
				{
					// create a new intrinsic
					intrinsicNew = new Intrinsic ();
					intrinsicNew.setBaseName (i.getBaseName ());
					intrinsicNew.setDatatype (i.getDatatype ());
					intrinsicNew.setLatency (Math.max (i.getLatency (), intrinsic.getLatency ()));
					intrinsicNew.setName (i.getName ());
					
					// merge the arguments of intrinsics i and intrinsic
					Argument[] rgArgs0 = Arguments.parseArguments (i.getArguments ());
					Argument[] rgArgs1 = Arguments.parseArguments (intrinsic.getArguments ());
					
					Argument[] rgArgsNew = null;
					if (rgArgs0.length == 0)
						rgArgsNew = rgArgs1;
					else if (rgArgs1.length == 0)
						rgArgsNew = rgArgs0;
					else if (rgArgs0.length == rgArgs1.length)
					{
						rgArgsNew = new Argument[rgArgs0.length];
						for (int j = 0; j < rgArgs0.length; j++)
						{
							rgArgsNew[j] = new Argument (
								rgArgs0[j].isRegister () || rgArgs1[j].isRegister (),
								rgArgs0[j].isMemory () || rgArgs1[j].isMemory (),
								rgArgs0[j].isOutput () || rgArgs1[j].isOutput (),
								rgArgs0[j].getName () == null ? rgArgs1[j].getName () : rgArgs0[j].getName (),
								j
							);
						}
					}
					else
						throw new RuntimeException (StringUtil.concat ("The intrinsics to merge (", intrinsic.toString (), " and ", i.toString (), ") have different number of arguments."));
					
					intrinsicNew.setArguments (Arguments.encode (rgArgsNew));
					
					// merge exec unit IDs
					intrinsicNew.getExecUnitTypeIds ().addAll (intrinsic.getExecUnitTypeIds ());
					intrinsicNew.getExecUnitTypeIds ().addAll (i.getExecUnitTypeIds ());
					
					intrinsicToRemove = i;
					break;
				}
			}
			
			if (intrinsicToRemove != null)
			{
				listIntrinsics.remove (intrinsicToRemove);
				listIntrinsics.add (intrinsicNew);
			}
			else
				listIntrinsics.add (intrinsic);
		}

		@Override
		public String getBackend ()
		{
			return m_type.getCodegenerator ().getBackend ();
		}
		
		@Override
		public String getInnermostLoopCodeGenerator ()
		{
			return m_type.getCodegenerator ().getInnermostLoopCg ();
		}

		@Override
		public String getGeneratedFileSuffix ()
		{
			return m_type.getCodegenerator ().getSrcSuffix ();
		}

		@Override
		public boolean useFunctionPointers ()
		{
			return m_type.getCodegenerator ().isUseFunctionPointers ();
		}

		@Override
		public int getNumberOfParallelLevels ()
		{
			return m_type.getParallelism ().getLevel ().size ();
		}

		private Level getParallelismLevel (int nParallelismLevel)
		{
			// get the parallelism level from the list; if nIdx exceeds the size of
			// the list, this hardware doesn't have that many parallelism levels, i.e.,
			// there are also no explicit data copies -- return null

			for (Level level : m_type.getParallelism ().getLevel ())
				if (level.getNumber () == nParallelismLevel)
					return level;
			return null;
		}

		@Override
		public boolean supportsAsynchronousIO (int nParallelismLevel)
		{
			Level level = getParallelismLevel (nParallelismLevel);
			return level == null ? false : level.isAsyncIO ();
		}

		@Override
		public boolean hasExplicitLocalDataCopies (int nParallelismLevel)
		{
			Level level = getParallelismLevel (nParallelismLevel);
			return level == null ? false : level.isHasExplicitLocalDatacopy ();
		}

		@Override
		public Statement getBarrier (int nParallelismLevel)
		{
			Level level = getParallelismLevel (nParallelismLevel);
			if (level == null)
				return null;

			Barrier barrier = level.getBarrier ();
			if (barrier == null)
				return null;

			switch (barrier.getType ())
			{
			case PRAGMA:
				return new AnnotationStatement (new PragmaAnnotation (barrier.getImplementation ()));
			case FUNCTIONCALL:
				return new ExpressionStatement (new FunctionCall (new NameID (barrier.getImplementation ())));
			case STATEMENT:
				return new ExpressionStatement (new NameID (barrier.getImplementation ()));
			}

			return null;
		}

		@Override
		public List<Specifier> getType (Specifier specType)
		{
			Datatype type = m_mapDataTypesFromBase.get (specType.toString ());
			List<Specifier> listSpecifiers = new ArrayList<> ();

			// add all the types separated by whitespaces to the list as user specifiers
			if (type != null)
				for (String strType : type.getName ().split ("\\s+"))
					listSpecifiers.add (new UserSpecifier (new NameID (strType)));
			else
				listSpecifiers.add (specType);

			return listSpecifiers;
		}

		@Override
		public boolean useSIMD ()
		{
			int nMaxSIMDVecLen = 0;
			for (Specifier specType : Globals.BASE_DATATYPES)
				nMaxSIMDVecLen = Math.max (getSIMDVectorLength (specType), nMaxSIMDVecLen);
			return nMaxSIMDVecLen > 1;
		}

		@Override
		public int getSIMDVectorLength (Specifier specType)
		{
			Datatype type = m_mapDataTypesFromBase.get (specType.toString ());

			// if the type couldn't be found, return the default value (1)
			if (type == null)
				return 1;

			return type.getSimdVectorLength ();
		}
		
		@Override
		public int getSIMDVectorLengthInBytes ()
		{
			Specifier spec = Globals.BASE_DATATYPES[0];
			return getSIMDVectorLength (spec) * getTypeSize (spec);
		}

		@Override
		public int getAlignmentRestriction (Specifier specType)
		{
			Datatype type = m_mapDataTypesFromBase.get (specType.toString ());

			// if the type couldn't be found, return the default value (1)
			if (type == null)
				return 1;

			if (type.getAlignment () == null)
				return 1;
			
			return type.getAlignment ();
		}
		
		@Override
		public boolean supportsUnalignedSIMD ()
		{
			return m_mapOperationsToIntrinsics.containsKey (TypeBaseIntrinsicEnum.LOAD_FPR_UNALIGNED.value ()) &&
				m_mapOperationsToIntrinsics.containsKey (TypeBaseIntrinsicEnum.STORE_FPR_UNALIGNED.value ());
		}

		@Override
		public List<Specifier> getDeclspecs (TypeDeclspec type)
		{
			List<Specifier> list = new ArrayList<> ();
			Declspec declspec = m_mapDeclspecs.get (type);
			if (declspec != null)
			{
				String[] rgSpecs = declspec.getSpecifiers ().trim ().split ("\\s+");
				for (String strSpec : rgSpecs)
				{
					Specifier spec = Specifier.fromString (strSpec);
					list.add (spec == null ? new UserSpecifier (new NameID (strSpec)) : spec);
				}
			}

			return list;
		}

		private Intrinsic getIntrinsicInternal (String strIntrinsicName, Specifier specType, IOperand[] rgOperands)
		{
			Datatype datatype = m_mapDataTypesFromBase.get (specType.toString ());

			// get the list of intrinsics for the function base name
			List<Intrinsic> listIntrinsics = rgOperands == null ?
				m_mapOperationsToMergedIntrinsics.get (strIntrinsicName) : m_mapOperationsToIntrinsics.get (strIntrinsicName);
			if (listIntrinsics == null || listIntrinsics.size () == 0)
				return null;

			// find the intrinsic for the correct data type
			for (Intrinsic intrinsic : listIntrinsics)
			{
				String strDatatype = intrinsic.getDatatype ();
				if (strDatatype == null)
				{
					if (datatype == null || datatype.getName () == null || "".equals (datatype.getName ()))
					{
						if (rgOperands == null || intrinsic.getArguments () == null || "".equals (intrinsic.getArguments ()))
							return intrinsic;
						
						// check operands
						if (matchArgs (Arguments.parseArguments (intrinsic.getArguments ()), rgOperands))
							return intrinsic;
					}
				}
				else
				{
					if (datatype != null && strDatatype.equals (datatype.getName ()))
					{
						if (rgOperands == null || intrinsic.getArguments () == null || "".equals (intrinsic.getArguments ()))
							return intrinsic;

						// check operands
						if (matchArgs (Arguments.parseArguments (intrinsic.getArguments ()), rgOperands))
							return intrinsic;
					}
				}
			}

			return null;
		}
		
		private boolean matchArgs (Argument[] rgArgs, IOperand[] rgOperands)
		{
			if (rgArgs.length != rgOperands.length)
				return false;
			
			for (int i = 0; i < rgArgs.length; i++)
			{
				if (rgOperands[i] instanceof IOperand.IRegisterOperand)
				{
					if (!rgArgs[i].isRegister ())
						return false;
				}
				
				if (rgOperands[i] instanceof IOperand.Address)
				{
					if (!rgArgs[i].isMemory ())
						return false;
				}
			}
			
			return true;
		}

		@Override
		public Intrinsic getIntrinsic (String strOperationOrBaseName, Specifier specType)
		{
			TypeBaseIntrinsicEnum type = Globals.getIntrinsicBase (strOperationOrBaseName);
			return getIntrinsicInternal (type == null ? strOperationOrBaseName : type.value (), specType, null);
		}

		@Override
		public Intrinsic getIntrinsic (UnaryOperator op, Specifier specType)
		{
			TypeBaseIntrinsicEnum type = Globals.getIntrinsicBase (op);
			return type == null ? null : getIntrinsicInternal (type.value (), specType, null);
		}

		@Override
		public Intrinsic getIntrinsic (BinaryOperator op, Specifier specType)
		{
			TypeBaseIntrinsicEnum type = Globals.getIntrinsicBase (op);
			return type == null ? null : getIntrinsicInternal (type.value (), specType, null);
		}

		@Override
		public Intrinsic getIntrinsic (FunctionCall fnx, Specifier specType)
		{
			String strFnx = fnx.getName ().toString ();
			/*
			if (Globals.FNX_BARRIER.getName ().equals (strFnx))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.BARRIER.value (), specType);
			if (Globals.FNX_FMA.getName ().equals (strFnx))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.FMA.value (), specType);
			if (Globals.NUMBER_OF_THREADS.getName ().equals (strFnx))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.NUMTHREADS.value (), specType);
			if (Globals.THREAD_NUMBER.getName ().equals (strFnx))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.THREADID.value (), specType);
			*/

			Intrinsic intfnx = getIntrinsicInternal (strFnx, specType, null);
			if (intfnx != null)
				return intfnx;

			intfnx = new Intrinsic ();

			String strName = fnx.getName ().toString ();
			intfnx.setBaseName (strName);
			intfnx.setName (strName);

			return intfnx;
		}
		
		@Override
		public Intrinsic getIntrinsic (TypeBaseIntrinsicEnum type, Specifier specType)
		{
			return getIntrinsic (type, specType, null);
		}
		
		@Override
		public Intrinsic getIntrinsic (TypeBaseIntrinsicEnum type, Specifier specType, IOperand[] rgOperands)
		{
			if (type == null)
				return null;
			return getIntrinsicInternal (type.value (), specType, rgOperands);
		}
		
		@Override
		public Collection<Intrinsic> getIntrinsicsByIntrinsicName (String strIntrinsicName)
		{
			return m_mapIntrinsicNamesToIntrinsics.get (strIntrinsicName);
		}
		
		@Override
		public Assembly getAssemblySpec ()
		{
			return m_type.getAssembly ();
		}
		
		@Override
		public boolean hasNonDestructiveOperations ()
		{
			if (getAssemblySpec () == null)
				return true;
			return m_bHasNonDestructiveOperations;
		}
		
		@Override
		public int getRegistersCount (TypeRegisterType type)
		{
			int nRegsCount = 0;
			for (TypeRegister reg : m_type.getAssembly ().getRegisters ().getRegister ())
				if (((TypeRegisterClass) reg.getClazz ()).getType ().equals (type))
					nRegsCount++;
			return nRegsCount;
		}
		
		@Override
		public Iterable<TypeRegisterClass> getRegisterClasses (TypeRegisterType type)
		{
			List<TypeRegisterClass> list = new ArrayList<> ();
			if (m_type.getAssembly () != null && m_type.getAssembly ().getRegisterClasses () != null && m_type.getAssembly ().getRegisterClasses ().getRegisterClass () != null)
			{
				for (TypeRegisterClass cls : m_type.getAssembly ().getRegisterClasses ().getRegisterClass ())
					if (cls.getType ().equals (type))
						list.add (cls);
			}
			
			Collections.sort (list, new Comparator<TypeRegisterClass> ()
			{
				@Override
				public int compare (TypeRegisterClass c1, TypeRegisterClass c2)
				{
					return c2.getWidth () - c1.getWidth ();
				}
			});

			return list;
		}
		
		@Override
		public TypeRegisterClass getDefaultRegisterClass (TypeRegisterType type)
		{
			if (m_type.getAssembly () == null)
				return null;
			if (m_type.getAssembly ().getRegisters () == null)
				return null;
			
			// search for a register of type "type" and return its class
			for (TypeRegister reg : m_type.getAssembly ().getRegisters ().getRegister ())
				if (((TypeRegisterClass) reg.getClazz ()).getType ().equals (type))
					return (TypeRegisterClass) reg.getClazz ();

			// no outer-most register of type "type" found
			return null;
		}
		
		@Override
		public int getIssueRate ()
		{
			if (m_type.getAssembly () == null)
				return 1;
			
			return Math.max (m_type.getAssembly ().getProcessorIssueRate (), 1);
		}
		
		@Override
		public int getMinimumNumberOfExecutionUnitsPerType (Iterable<Intrinsic> itIntrinsics)
		{
			int nMin = Integer.MAX_VALUE;
			
			for (Intrinsic i : itIntrinsics)
			{
				if (i.getExecUnitTypeIds () == null)
					return 1;
				
				for (int nID : i.getExecUnitTypeIds ())
				{
					TypeExecUnitType t = getExecutionUnitTypeByID (nID);
					if (t == null)
						return 1;
					
					nMin = Math.min (nMin, t.getQuantity ().intValue ());
					
					// no need to search further if already 1
					if (nMin == 1)
						return 1;
				}
			}
			
			return nMin == Integer.MAX_VALUE ? 1 : nMin;
		}
		
		/**
		 * Checks whether all the execution unit type IDs defined in the
		 * intrinsics were also defined as execution unit types in the assembly
		 * specification. Throws an exception if an ID was found for which no
		 * corresponding type was defined.
		 */
		private void checkExecUnitTypeIDs ()
		{
			if (m_type.getIntrinsics () != null)
			{
				for (Intrinsic intrinsic : m_type.getIntrinsics ().getIntrinsic ())
				{
					if (intrinsic.getExecUnitTypeIds () != null)
					{
						for (int nID : intrinsic.getExecUnitTypeIds ())
							if (!m_mapExecUnitTypes.containsKey (nID))
								throw new RuntimeException (StringUtil.concat ("No execution unit type defined for ID ", nID));
					}
				}
			}
		}
		
		@Override
		public int getExecutionUnitTypesCount ()
		{
			return m_mapExecUnitTypes.size ();
		}
		
		@Override
		public TypeExecUnitType getExecutionUnitTypeByID (int nID)
		{
			return m_mapExecUnitTypes.get (nID);
		}
		
		@Override
		public List<TypeExecUnitType> getExecutionUnitTypesByIDs (List<?> listIDs)
		{
			List<TypeExecUnitType> listResult = new ArrayList<> (listIDs.size ());
			for (Object oID : listIDs)
			{
				if (oID instanceof Number)
				{
					TypeExecUnitType t = getExecutionUnitTypeByID (((Number) oID).intValue ());
					if (t != null)
						listResult.add (t);
				}
			}

			return listResult;
		}
		
		@Override
		public List<String> getIncludeFiles ()
		{
			List<String> listIncludes = new ArrayList<> ();
			for (Include include : m_type.getIncludes ().getInclude ())
				listIncludes.add (include.getFile ());
			return listIncludes;
		}

		@Override
		public Build getBuild ()
		{
			return m_type.getBuild ();
		}

		@Override
		public IArchitectureDescription clone ()
		{
			return new HardwareDescription (m_file, m_type);
		}

		@Override
		public File getFile ()
		{
			return m_file;
		}
	}


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private Map<String, HardwareDescription> m_mapDescriptions;


	///////////////////////////////////////////////////////////////////
	// Implementation

	/**
	 * Constructs the hardware description manager. The hardware descriptions are contained in the XML file
	 * <code>fileHardwareDescriptions</code>.
	 * @param fileHardwareDescriptions The hardware descriptions file, an XML file conforming to the hardware-config
	 * 	XSD schema
	 * @throws JAXBException
	 */
	public ArchitectureDescriptionManager (File fileHardwareDescriptions)
	{
		try
		{
			// unmarshal the architecture description
			JAXBContext context = JAXBContext.newInstance (getClass ().getPackage ().getName ());
			ArchitectureTypes types = (ArchitectureTypes) context.createUnmarshaller ().unmarshal (fileHardwareDescriptions);

			resolveInheritsFrom (types);

			// construct hardware description objects
			m_mapDescriptions = new HashMap<> ();
			for (TypeArchitectureType type : types.getArchitectureType ())
				m_mapDescriptions.put (type.getName (), new HardwareDescription (fileHardwareDescriptions, type));
		}
		catch (JAXBException e)
		{
			if (e.getLinkedException () == null)
			{
				System.err.println (StringUtil.concat (
					"An error occurred while trying to read the hardware definition (",
					e.getClass ().getSimpleName (), ")"));
			}
			else
			{
				System.err.println (StringUtil.concat (
					"An error occurred while trying to read the hardware definition: ",
					e.getLinkedException ().getMessage ()));
			}
			System.exit (-1);
		}
	}

	private void resolveInheritsFrom (ArchitectureTypes types)
	{
		// resolve "inherits-from" attributes
		Map<String, TypeArchitectureType> mapResolved = new HashMap<> ();
		Map<String, TypeArchitectureType> mapToResolve = new HashMap<> ();

		// add types without "inherits-from" attributes to the map of resolved types
		for (TypeArchitectureType type : types.getArchitectureType ())
		{
			if (type.getInheritsFrom () == null || "".equals (type.getInheritsFrom ()))
				mapResolved.put (type.getName (), type);
			else
				mapToResolve.put (type.getName (), type);
		}

		// resolve types iteratively
		while (!mapToResolve.isEmpty ())
		{
			// find a type that is resolvable
			TypeArchitectureType type = ArchitectureDescriptionManager.findResolvableType (mapResolved, mapToResolve);
			if (type == null)
				throw new RuntimeException (StringUtil.concat ("The architecture type definition \"", mapToResolve.keySet ().iterator ().next (), "\" is not resolvable."));

			copyFields (type, mapResolved.get (type.getInheritsFrom ()));
			mapResolved.put (type.getName (), type);
			mapToResolve.remove (type.getName ());
		}
	}

	private static TypeArchitectureType findResolvableType (Map<String, TypeArchitectureType> mapResolved, Map<String, TypeArchitectureType> mapToResolve)
	{
		for (TypeArchitectureType type : mapToResolve.values ())
		{
			TypeArchitectureType typeParent = mapResolved.get (type.getInheritsFrom ());
			if (typeParent != null)
				return type;
		}

		return null;
	}

	private void copyFields (Object objDest, Object objSrc)
	{
		for (Field fieldSrc : objSrc.getClass ().getDeclaredFields ())
		{
			try
			{
				Field fieldDest = objDest.getClass ().getDeclaredField (fieldSrc.getName ());
				
				// check whether the XML node can have child nodes
				// we use the JAXB annotation XmlType and look for the propOrder element
				boolean bHasChildElements = false;
				XmlType xmlType = fieldDest.getType ().getAnnotation (XmlType.class);
				if (xmlType != null && xmlType.propOrder () != null && xmlType.propOrder ().length > 0 && !"".equals (xmlType.propOrder ()[0]))
					bHasChildElements = true;
				
				boolean bFieldExists = fieldDest.get (objDest) != null;

				// nothing to do if the dest field is already set
				if (!bHasChildElements && bFieldExists)
					continue;

				// copy the src field contents

				fieldSrc.setAccessible (true);
				fieldDest.setAccessible (true);

				if (fieldSrc.get (objSrc) == null && !bFieldExists)
					fieldDest.set (objDest, null);
				else if (fieldSrc.getType ().equals (String.class))
					fieldDest.set (objDest, fieldSrc.get (objSrc));
				else if (fieldSrc.getType ().equals (Integer.class))
					fieldDest.set (objDest, new Integer ((Integer) fieldSrc.get (objSrc)));
				else if (fieldSrc.getType ().equals (int.class))
					fieldDest.set (objDest, ((Integer) fieldSrc.get (objSrc)).intValue ());
				else if (fieldSrc.getType ().equals (BigInteger.class))
					fieldDest.set (objDest, new BigInteger (((BigInteger) fieldSrc.get (objSrc)).toByteArray ()));
				else if (fieldSrc.getType ().equals (Boolean.class))
					fieldDest.set (objDest, new Boolean ((Boolean) fieldSrc.get (objSrc)));
				else if (fieldSrc.getType ().equals (boolean.class))
					fieldDest.set (objDest, ((Boolean) fieldSrc.get (objSrc)).booleanValue ());
				else if (fieldSrc.getType ().isEnum ())
					fieldDest.set (objDest, fieldSrc.get (objSrc));
				else if (fieldSrc.getType ().equals (List.class))
				{
					List<Object> list = new ArrayList<> ();
					for (Object objSrcEntry : (List<?>) fieldSrc.get (objSrc))
					{
						if (objSrcEntry == null)
							list.add (null);
						else
						{
							if (objSrcEntry.getClass ().equals (Integer.class))
								list.add (new Integer (((Integer) objSrcEntry).intValue ()));
							if (objSrcEntry.getClass ().equals (BigInteger.class))
								list.add (new BigInteger (((BigInteger) objSrcEntry).toByteArray ()));
							else
							{
								Object objDestEntry = objSrcEntry.getClass ().newInstance ();
								copyFields (objDestEntry, objSrcEntry);
								list.add (objDestEntry);
							}
						}
					}
					fieldDest.set (objDest, list);
				}
				else
				{
					Object objSrcVal = fieldSrc.get (objSrc);
					if (objSrcVal == null)
					{
						if (!bFieldExists)
							fieldDest.set (objDest, null);
					}
					else if (objSrcVal instanceof TypeRegisterClass)
					{
						// special treatment because of key
						fieldDest.set (objDest, objSrcVal);
					}
					else
					{
						Object objVal = null;
						if (!bFieldExists)
						{
							objVal = fieldSrc.getType ().newInstance ();
							fieldDest.set (objDest, objVal);
						}
						else
							objVal = fieldDest.get (objDest);
						
						copyFields (objVal, fieldSrc.get (objSrc));
					}
				}
			}
			catch (SecurityException e)
			{
				e.printStackTrace();
			}
			catch (NoSuchFieldException e)
			{
				e.printStackTrace();
			}
			catch (IllegalArgumentException e)
			{
				e.printStackTrace();
			}
			catch (IllegalAccessException e)
			{
				e.printStackTrace();
			}
			catch (InstantiationException e)
			{
				e.printStackTrace();
			}
		}
	}

	/**
	 * Returns all the hardware description names found in the description file.
	 * @return The hardware description names
	 */
	public Iterable<String> getHardwareNames ()
	{
		return m_mapDescriptions.keySet ();
	}

	/**
	 * Returns a hardware description object for a specific hardware name. The name must be one returned by
	 * {@link ArchitectureDescriptionManager#getHardwareNames()}, otherwise <code>null</code> is returned.
	 * @param strHardwareName
	 * @return
	 */
	public HardwareDescription getHardwareDescription (String strHardwareName)
	{
		return m_mapDescriptions.get (strHardwareName);
	}

	/**
	 * Returns the size of a floating point data type.
	 * 
	 * @param specDatatype
	 * @return
	 */
	public static int getTypeSize (Specifier specDatatype)
	{
		if (specDatatype.equals (Specifier.FLOAT))
			return Float.SIZE / 8;
		if (specDatatype.equals (Specifier.DOUBLE))
			return Double.SIZE / 8;
		return 0;
	}
}
