package ch.unibas.cs.hpwc.patus.arch;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;

import cetus.hir.BinaryOperator;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.UnaryOperator;
import cetus.hir.UserSpecifier;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Build;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Datatypes;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Datatypes.Alignment;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Datatypes.Datatype;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Declspecs.Declspec;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Includes.Include;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Parallelism.Level;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class ArchitectureDescriptionManager
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	protected static class HardwareDescription implements IArchitectureDescription
	{
		private TypeArchitectureType m_type;

		private Map<String, Datatype> m_mapDataTypes;
		private Map<TypeDeclspec, Declspec> m_mapDeclspecs;
		private Map<String, Datatype> m_mapDataTypesFromBase;
		private Map<String, List<Intrinsic>> m_mapIntrinsics;


		public HardwareDescription (TypeArchitectureType type)
		{
			m_type = type;

			m_mapDataTypes = new HashMap<String, Datatype> ();

			m_mapDeclspecs = new HashMap<TypeDeclspec, Declspec> ();
			for (Declspec d : m_type.getDeclspecs ().getDeclspec ())
				m_mapDeclspecs.put (d.getType (), d);

			m_mapDataTypesFromBase = new HashMap<String, Datatype> ();
			for (Datatype datatype : m_type.getDatatypes ().getDatatype ())
			{
				m_mapDataTypes.put (datatype.getName (), datatype);

				if (m_mapDataTypesFromBase.containsKey (datatype.getBasetype ().toString ()))
					throw new RuntimeException ("A hardware description must not define multiple datatypes for one base type.");
				m_mapDataTypesFromBase.put (datatype.getBasetype ().value (), datatype);
			}

			m_mapIntrinsics = new HashMap<String, List<Intrinsic>> ();
			for (Intrinsic intrinsic : m_type.getIntrinsics ().getIntrinsic ())
			{
				List<Intrinsic> listIntrinsics = m_mapIntrinsics.get (intrinsic.getBaseName ().value ());
				if (listIntrinsics == null)
					m_mapIntrinsics.put (intrinsic.getBaseName ().value (), listIntrinsics = new ArrayList<Intrinsic> ());
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
			// the first parallelism level is first entry in the (zero-based) list
			int nIdx = nParallelismLevel - 1;

			// get the parallelism level from the list; if nIdx exceeds the size of
			// the list, this hardware doesn't have that many parallelism levels, i.e.,
			// there are also no explicit data copies -- return false
			List<Level> listLevels = m_type.getParallelism ().getLevel ();
			if (nIdx < 0 || nIdx >= listLevels.size ())
				return null;

			return listLevels.get (nIdx);
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
		public int getAlignmentRestriction ()
		{
			Datatypes datatypes = m_type.getDatatypes ();
			if (datatypes == null)
				return 1;
			Alignment alignment = datatypes.getAlignment ();
			if (alignment == null)
				return 1;
			return alignment.getRestrict () ==  null ? 1 : alignment.getRestrict ();
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

		private NameID getNameID (TypeBaseIntrinsicName name, Specifier specType)
		{
			Datatype datatype = m_mapDataTypesFromBase.get (specType.toString ());

			List<Intrinsic> listIntrinsics = m_mapIntrinsics.get (name.value ());
			if (listIntrinsics == null || listIntrinsics.size () == 0)
				return null;

			for (Intrinsic intrinsic : listIntrinsics)
			{
				String strDatatype = intrinsic.getDatatype ();
				if (strDatatype == null)
				{
					if (datatype == null || datatype.getName () == null || "".equals (datatype.getName ()))
						return new NameID (intrinsic.getName ());
				}
				else
				{
					if (datatype != null && strDatatype.equals (datatype.getName ()))
						return new NameID (intrinsic.getName ());
				}
			}

			return null;
		}

		@Override
		public IDExpression getIntrinsicName (String strOperation, Specifier specType)
		{
			if ("+".equals (strOperation))
				return getNameID (TypeBaseIntrinsicName.PLUS, specType);
			if ("-".equals (strOperation))
				return getNameID (TypeBaseIntrinsicName.MINUS, specType);
			if ("*".equals (strOperation))
				return getNameID (TypeBaseIntrinsicName.MULTIPLY, specType);
			if ("/".equals (strOperation))
				return getNameID (TypeBaseIntrinsicName.DIVIDE, specType);
			if (Globals.FNX_FMA.getName ().equals (strOperation))
				return getNameID (TypeBaseIntrinsicName.FMA, specType);

			return null;
		}

		@Override
		public IDExpression getIntrinsicName (UnaryOperator op, Specifier specType)
		{
			if (UnaryOperator.PLUS.equals (op))
				return getNameID (TypeBaseIntrinsicName.UNARY_PLUS, specType);
			if (UnaryOperator.MINUS.equals (op))
				return getNameID (TypeBaseIntrinsicName.UNARY_MINUS, specType);

			// intrinsic not found
			return null;
		}

		@Override
		public IDExpression getIntrinsicName (BinaryOperator op, Specifier specType)
		{
			if (BinaryOperator.ADD.equals (op))
				return getNameID (TypeBaseIntrinsicName.PLUS, specType);
			if (BinaryOperator.SUBTRACT.equals (op))
				return getNameID (TypeBaseIntrinsicName.MINUS, specType);
			if (BinaryOperator.MULTIPLY.equals (op))
				return getNameID (TypeBaseIntrinsicName.MULTIPLY, specType);
			if (BinaryOperator.DIVIDE.equals (op))
				return getNameID (TypeBaseIntrinsicName.DIVIDE, specType);
			return null;
		}

		@Override
		public IDExpression getIntrinsicName (FunctionCall fnx, Specifier specType)
		{
			String strFnx = fnx.getName ().toString ();
			if (Globals.FNX_BARRIER.getName ().equals (strFnx))
				return getNameID (TypeBaseIntrinsicName.BARRIER, specType);
			if (Globals.FNX_FMA.getName ().equals (strFnx))
				return getNameID (TypeBaseIntrinsicName.FMA, specType);
			if (Globals.NUMBER_OF_THREADS.getName ().equals (strFnx))
				return getNameID (TypeBaseIntrinsicName.NUMTHREADS, specType);
			if (Globals.THREAD_NUMBER.getName ().equals (strFnx))
				return getNameID (TypeBaseIntrinsicName.THREADID, specType);

			return fnx.getName () instanceof IDExpression ? (IDExpression) fnx.getName () : new NameID (fnx.getName ().toString ());
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
			JAXBContext context = JAXBContext.newInstance (getClass ().getPackage ().getName ());
			ArchitectureTypes types = (ArchitectureTypes) context.createUnmarshaller ().unmarshal (fileHardwareDescriptions);

			m_mapDescriptions = new HashMap<String, ArchitectureDescriptionManager.HardwareDescription> ();
			for (TypeArchitectureType type : types.getArchitectureType ())
				m_mapDescriptions.put (type.getName (), new HardwareDescription (type));
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
