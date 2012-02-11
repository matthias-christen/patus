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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;

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
		private Map<String, List<Intrinsic>> m_mapIntrinsics;


		public HardwareDescription (File file, TypeArchitectureType type)
		{
			m_file = file;
			m_type = type;

			m_mapDataTypes = new HashMap<String, Datatype> ();

			m_mapDeclspecs = new HashMap<TypeDeclspec, Declspec> ();
			if (m_type.getDeclspecs () != null)
				for (Declspec d : m_type.getDeclspecs ().getDeclspec ())
					m_mapDeclspecs.put (d.getType (), d);

			m_mapDataTypesFromBase = new HashMap<String, Datatype> ();
			if (m_type.getDatatypes () != null)
				for (Datatype datatype : m_type.getDatatypes ().getDatatype ())
				{
					m_mapDataTypes.put (datatype.getName (), datatype);

					if (m_mapDataTypesFromBase.containsKey (datatype.getBasetype ().toString ()))
						throw new RuntimeException ("A hardware description must not define multiple datatypes for one base type.");
					m_mapDataTypesFromBase.put (datatype.getBasetype ().value (), datatype);
				}

			m_mapIntrinsics = new HashMap<String, List<Intrinsic>> ();
			if (m_type.getIntrinsics () != null)
				for (Intrinsic intrinsic : m_type.getIntrinsics ().getIntrinsic ())
				{
					List<Intrinsic> listIntrinsics = m_mapIntrinsics.get (intrinsic.getBaseName ());
					if (listIntrinsics == null)
						m_mapIntrinsics.put (intrinsic.getBaseName (), listIntrinsics = new ArrayList<Intrinsic> ());
					listIntrinsics.add (intrinsic);
				}
		}

		@Override
		public String getBackend ()
		{
			return m_type.getCodegenerator ().getBackend ();
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
			// there are also no explicit data copies -- return false

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
			return level == null ? false : level.isHasExplicitLocalDataCopy ();
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
			List<Specifier> listSpecifiers = new ArrayList<Specifier> ();

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
			return Math.max (getSIMDVectorLength (Specifier.FLOAT), getSIMDVectorLength (Specifier.DOUBLE)) > 1;
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
		public int getAlignmentRestriction (Specifier specType)
		{
			Datatype type = m_mapDataTypesFromBase.get (specType.toString ());

			// if the type couldn't be found, return the default value (1)
			if (type == null)
				return 1;

			return type.getAlignment ();
		}

		@Override
		public List<Specifier> getDeclspecs (TypeDeclspec type)
		{
			List<Specifier> list = new ArrayList<Specifier> ();
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

		private Intrinsic getIntrinsicInternal (String strIntrinsicName, Specifier specType)
		{
			Datatype datatype = m_mapDataTypesFromBase.get (specType.toString ());

			// get the list of intrinsics for the function base name
			List<Intrinsic> listIntrinsics = m_mapIntrinsics.get (strIntrinsicName);
			if (listIntrinsics == null || listIntrinsics.size () == 0)
				return null;

			// find the intrinsic for the correct data type
			for (Intrinsic intrinsic : listIntrinsics)
			{
				String strDatatype = intrinsic.getDatatype ();
				if (strDatatype == null)
				{
					if (datatype == null || datatype.getName () == null || "".equals (datatype.getName ()))
						return intrinsic;
				}
				else
				{
					if (datatype != null && strDatatype.equals (datatype.getName ()))
						return intrinsic;
				}
			}

			return null;
		}

		@Override
		public Intrinsic getIntrinsic (String strOperation, Specifier specType)
		{
			if ("+".equals (strOperation))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.PLUS.value (), specType);
			if ("-".equals (strOperation))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.MINUS.value (), specType);
			if ("*".equals (strOperation))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.MULTIPLY.value (), specType);
			if ("/".equals (strOperation))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.DIVIDE.value (), specType);

			return getIntrinsicInternal (strOperation, specType);
		}

		@Override
		public Intrinsic getIntrinsic (UnaryOperator op, Specifier specType)
		{
			if (UnaryOperator.PLUS.equals (op))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.UNARY_PLUS.value (), specType);
			if (UnaryOperator.MINUS.equals (op))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.UNARY_MINUS.value (), specType);

			// intrinsic not found
			return null;
		}

		@Override
		public Intrinsic getIntrinsic (BinaryOperator op, Specifier specType)
		{
			if (BinaryOperator.ADD.equals (op))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.PLUS.value (), specType);
			if (BinaryOperator.SUBTRACT.equals (op))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.MINUS.value (), specType);
			if (BinaryOperator.MULTIPLY.equals (op))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.MULTIPLY.value (), specType);
			if (BinaryOperator.DIVIDE.equals (op))
				return getIntrinsicInternal (TypeBaseIntrinsicEnum.DIVIDE.value (), specType);
			return null;
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

			Intrinsic intfnx = getIntrinsicInternal (strFnx, specType);
			if (intfnx != null)
				return intfnx;

			intfnx = new Intrinsic ();
			String strName = fnx.getName ().toString ();
			intfnx.setBaseName (strName);
			intfnx.setName (strName);
			return intfnx;
		}
		
		@Override
		public Assembly getAssemblySpec ()
		{
			return m_type.getAssembly ();
		}

		@Override
		public List<String> getIncludeFiles ()
		{
			List<String> listIncludes = new ArrayList<String> ();
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
			m_mapDescriptions = new HashMap<String, ArchitectureDescriptionManager.HardwareDescription> ();
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
		Map<String, TypeArchitectureType> mapResolved = new HashMap<String, TypeArchitectureType> ();
		Map<String, TypeArchitectureType> mapToResolve = new HashMap<String, TypeArchitectureType> ();

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
			TypeArchitectureType type = findResolvableType (mapResolved, mapToResolve);
			if (type == null)
				throw new RuntimeException (StringUtil.concat ("The architecture type definition \"", mapToResolve.keySet ().iterator ().next (), "\" is not resolvable."));

			copyFields (type, mapResolved.get (type.getInheritsFrom ()));
			mapResolved.put (type.getName (), type);
			mapToResolve.remove (type.getName ());
		}
	}

	private TypeArchitectureType findResolvableType (Map<String, TypeArchitectureType> mapResolved, Map<String, TypeArchitectureType> mapToResolve)
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

				// nothing to do if the dest field is already set
				if (fieldDest.get (objDest) != null)
					continue;

				// copy the src field contents

				fieldSrc.setAccessible (true);
				fieldDest.setAccessible (true);

				if (fieldSrc.get (objSrc) == null)
					fieldDest.set (objDest, null);
				else if (fieldSrc.getType ().equals (String.class))
					fieldDest.set (objDest, fieldSrc.get (objSrc));
				else if (fieldSrc.getType ().equals (Integer.class))
					fieldDest.set (objDest, new Integer ((Integer) fieldSrc.get (objSrc)));
				else if (fieldSrc.getType ().equals (int.class))
					fieldDest.set (objDest, ((Integer) fieldSrc.get (objSrc)).intValue ());
				else if (fieldSrc.getType ().equals (Boolean.class))
					fieldDest.set (objDest, new Boolean ((Boolean) fieldSrc.get (objSrc)));
				else if (fieldSrc.getType ().equals (boolean.class))
					fieldDest.set (objDest, ((Boolean) fieldSrc.get (objSrc)).booleanValue ());
				else if (fieldSrc.getType ().isEnum ())
					fieldDest.set (objDest, fieldSrc.get (objSrc));
				else if (fieldSrc.getType ().equals (List.class))
				{
					List<Object> list = new ArrayList<Object> ();
					for (Object objSrcEntry : (List<?>) fieldSrc.get (objSrc))
					{
						if (objSrcEntry == null)
							list.add (null);
						else
						{
							Object objDestEntry = objSrcEntry.getClass ().newInstance ();
							copyFields (objDestEntry, objSrcEntry);
							list.add (objDestEntry);
						}
					}
					fieldDest.set (objDest, list);
				}
				else
				{
					if (fieldSrc.get (objSrc) == null)
						fieldDest.set (objDest, null);
					else
					{
						Object objVal = fieldSrc.getType ().newInstance ();
						fieldDest.set (objDest, objVal);
						copyFields (objVal, fieldSrc.get (objSrc));
					}
				}
			}
			catch (SecurityException e)
			{
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			catch (NoSuchFieldException e)
			{
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			catch (IllegalArgumentException e)
			{
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			catch (IllegalAccessException e)
			{
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			catch (InstantiationException e)
			{
				// TODO Auto-generated catch block
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
}
