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
package ch.unibas.cs.hpwc.patus.codegen.backend.cuda;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import omp2gpu.hir.CUDASpecifier;
import omp2gpu.hir.Dim3Specifier;
import omp2gpu.hir.KernelFunctionCall;
import cetus.hir.AccessExpression;
import cetus.hir.AccessOperator;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.DeclarationStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.PointerSpecifier;
import cetus.hir.SizeofExpression;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Typecast;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.UserSpecifier;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.analysis.StrategyAnalyzer;
import ch.unibas.cs.hpwc.patus.ast.StatementList;
import ch.unibas.cs.hpwc.patus.ast.StatementListBundle;
import ch.unibas.cs.hpwc.patus.ast.SubdomainIdentifier;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers.EVariableType;
import ch.unibas.cs.hpwc.patus.codegen.GlobalGeneratedIdentifiers.Variable;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObject;
import ch.unibas.cs.hpwc.patus.codegen.MemoryObjectManager;
import ch.unibas.cs.hpwc.patus.codegen.ValidationCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.backend.AbstractBackend;
import ch.unibas.cs.hpwc.patus.codegen.backend.AbstractNonKernelFunctionsImpl.EOutputGridType;
import ch.unibas.cs.hpwc.patus.codegen.backend.IIndexing;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.geometry.Size;
import ch.unibas.cs.hpwc.patus.geometry.Vector;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.CodeGeneratorUtil;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 *
 * @author Matthias-M. Christen
 */
public abstract class AbstractCUDACodeGenerator extends AbstractBackend
{
	// /////////////////////////////////////////////////////////////////
	// Constants

	private final static NameID[] ACCESS_NAMES = new NameID[] { new NameID ("x"), new NameID ("y"), new NameID ("z") };

	private final static String SUFFIX_GPU = "_gpu";

	private final static IDExpression GPU_PTR = new NameID ("gpu_ptr_t");
	private final static Specifier SPEC_GPU_PTR = new UserSpecifier (GPU_PTR);


	private final IIndexing.IIndexingLevel[] INDEXING_LEVELS = new IIndexing.IIndexingLevel[] {
		// deepest indexing level
		new IIndexing.IIndexingLevel ()
		{
			@Override
			public int getDimensionality ()
			{
				return getIndexingLevelDimensionality (0);
			}

			@Override
			public boolean isVariable ()
			{
				return true;
			}

			@Override
			public Expression getIndexForDimension (int nDimension)
			{
				return new AccessExpression (new NameID ("threadIdx"), AccessOperator.MEMBER_ACCESS, AbstractCUDACodeGenerator.ACCESS_NAMES[nDimension].clone ());
			}

			@Override
			public Expression getSizeForDimension (int nDimension)
			{
				return new AccessExpression (new NameID ("blockDim"), AccessOperator.MEMBER_ACCESS, AbstractCUDACodeGenerator.ACCESS_NAMES[nDimension].clone ());
			};

			@Override
			public int getDefaultBlockSize (int nDimension)
			{
				// not needed
				return 0;
			}
		},

		// second indexing level
		new IIndexing.IIndexingLevel ()
		{
			@Override
			public int getDimensionality ()
			{
				return getIndexingLevelDimensionality (1);
			}

			@Override
			public boolean isVariable ()
			{
				return true;
			}

			@Override
			public Expression getIndexForDimension (int nDimension)
			{
				return new AccessExpression (new NameID ("blockIdx"), AccessOperator.MEMBER_ACCESS, AbstractCUDACodeGenerator.ACCESS_NAMES[nDimension].clone ());
			}

			@Override
			public Expression getSizeForDimension (int nDimension)
			{
				return new AccessExpression (new NameID ("gridDim"), AccessOperator.MEMBER_ACCESS, AbstractCUDACodeGenerator.ACCESS_NAMES[nDimension].clone ());
			};

			@Override
			public int getDefaultBlockSize (int nDimension)
			{
				// not needed
				return 0;
			}
		}
	};


	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;

	private Expression m_exprTotalSharedMemory;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public AbstractCUDACodeGenerator (CodeGeneratorSharedObjects data)
	{
		super (data);
		m_data = data;
		m_exprTotalSharedMemory = null;
	}

	protected CodeGeneratorSharedObjects getData ()
	{
		return m_data;
	}

	protected abstract int getIndexingLevelDimensionality (int nIndexingLevel);


	///////////////////////////////////////////////////////////////////
	// IArithmetic Implementation

	@Override
	public Expression shuffle (Expression expr1, Expression expr2, Specifier specDatatype, int nOffset)
	{
		return null;
	}

	@Override
	public Expression splat (Expression expr, Specifier specDatatype)
	{
		return null;
	}


	///////////////////////////////////////////////////////////////////
	// IIndexing Implementation

	@Override
	public int getIndexingLevelsCount ()
	{
		return 2;
	}

	@Override
	public IIndexingLevel getIndexingLevel (int nIndexingLevel)
	{
		return INDEXING_LEVELS[nIndexingLevel];
	}

	@Override
	public EThreading getThreading ()
	{
		return IIndexing.EThreading.MANY;
	}


	///////////////////////////////////////////////////////////////////
	// IDataTransfer Implementation

	private static class TransferData implements Comparable<TransferData>
	{
		/**
		 * The subdomain identifier representing the iterator on the level of the
		 * large subdomain, e.g., if the strategy is
		 *
		 * <pre>
		 * 	for subdomain v(size) in u(:; t) parallel
		 * 		...
		 * </pre>
		 *
		 * the iterator is <code>v</code> representing the offset within
		 * <code>u</code>.
		 */
		private SubdomainIdentifier m_sdidSourceIterator;

		/**
		 * The &quot;minimum&quot; stencil node within the memory object
		 * transferred
		 */
		private StencilNode m_nodeMin;

		private MemoryObject m_moDestination;

		private Size m_sizeDestination;

		private MemoryObject m_moSource;

		// private Expression m_exprMemobjSource;
		private Size m_sizeSource;

		private boolean m_bIsDestinationSmall;

		// public TransferData (
		// SubgridIdentifier sdidLargeLevel,
		// StencilNode nodeDestination, MemoryObject moDestination, //Expression
		// exprMemoryObjectDestination,
		// StencilNode nodeSource, MemoryObject moSource, //Expression
		// exprMemoryObjectSource,
		// boolean bIsDestinationSmall)
		// {
		// m_sdidLargeLevel = sdidLargeLevel;
		//
		// m_nodeDestination = nodeDestination;
		// m_moDestination = moDestination;
		// // m_exprMemobjDestination = exprMemoryObjectDestination;
		// m_sizeDestination = m_moDestination.getSize ();
		//
		// m_nodeSource = nodeSource;
		// m_moSource = moSource;
		// // m_exprMemobjSource = exprMemoryObjectSource;
		// m_sizeSource = m_moSource.getSize ();
		//
		// m_bIsDestinationSmall = bIsDestinationSmall;
		// }

		public TransferData (StencilNode node, SubdomainIdentifier sdidSourceIterator, MemoryObject moDestination, MemoryObject moSource, boolean bIsDestinationSmall)
		{
			m_nodeMin = node;
			m_sdidSourceIterator = sdidSourceIterator;

			m_moDestination = moDestination;
			m_sizeDestination = moDestination.getSize ();
			m_moSource = moSource;
			m_sizeSource = moSource.getSize ();

			m_bIsDestinationSmall = bIsDestinationSmall;
		}

		public final StencilNode getMinimumStencilNode ()
		{
			return m_nodeMin;
		}

		public final MemoryObject getDestinationMemoryObject ()
		{
			return m_moDestination;
		}

		public final Size getDestinationSize ()
		{
			return m_sizeDestination;
		}

		public final MemoryObject getSourceMemoryObject ()
		{
			return m_moSource;
		}

		public final Size getSourceSize ()
		{
			return m_sizeSource;
		}

		public final Size getSmallSize ()
		{
			return m_bIsDestinationSmall ? m_sizeDestination : m_sizeSource;
		}

		public final SubdomainIdentifier getSubdomainIdentifier ()
		{
			return m_sdidSourceIterator;
		}

		@Override
		public int compareTo (TransferData td)
		{
			Size sizeThis = getSmallSize ();
			Size sizeOther = td.getSmallSize ();

			for (int i = 0; i < sizeThis.getDimensionality (); i++)
			{
				int nResult = sizeThis.getCoord (i).compareTo (sizeOther.getCoord (i));
				if (nResult != 0)
					return nResult;
			}

			return 0;
		}
	}

	private Set<TransferData> m_setLoadData = new TreeSet<TransferData> ();

	private Set<TransferData> m_setStoreData = new TreeSet<TransferData> ();

	/**
	 * Creates a datatransfer loop nest.
	 *
	 * @param listLoad
	 *            The list of {@link TransferData} objects that are loaded in
	 *            one loop nest
	 * @param sizeSmallMemobj
	 *            The size of the small memory objects
	 * @param nDim
	 *            The number of dimensions of the stencil
	 * @param nHWDim
	 *            The number of dimensions of the indices in the programming
	 *            model/architecture
	 * @param nParallelismLevel
	 *            The parallelism level on which the smaller memory objects
	 *            reside
	 * @param slbCode
	 *            The statement list bundle to which the code is added
	 */
	private void createDataTransfer (List<TransferData> listTransfer, Size sizeSmallMemobj, int nDim, int nHWDim,
		int nParallelismLevel, boolean bIsDestinationSmall, StatementListBundle slbCode, CodeGeneratorRuntimeOptions options)
	{
		if (listTransfer.isEmpty ())
			return;

		// > for i_N = 0 .. b_N
		// >     for i_{N-1} = 0 .. b_{N-1}
		// >         ...
		// >             for i_3 = threadIdx.z .. b_3 by blockDim.z
		// >                 for i_2 = threadIdx.y .. b_2 by blockDim.y
		// >                     for i_1 = threadIdx.x .. b_1 by blockDim.x
		// >                         a[i_1, i_2, ..., i_N] = b[(i_1, ..., i_N) + <offset>]

		IIndexingLevel indexingLevel = getIndexingLevelFromParallelismLevel (nParallelismLevel);
		MemoryObjectManager mgr = m_data.getData ().getMemoryObjectManager ();

		// create the identifiers for the loop indices
		Identifier[] rgLoopIndices = new Identifier[nDim];
		for (int i = 0; i < nDim; i++)
		{
			VariableDeclarator decl = new VariableDeclarator (new NameID (StringUtil.concat ("i_", nParallelismLevel, "_", i)));
			m_data.getData ().addDeclaration (new VariableDeclaration (Globals.SPECIFIER_INDEX, decl));
			rgLoopIndices[i] = new Identifier (decl);
		}

		// create the loop body
		StatementList slBody = new StatementList (new CompoundStatement ());
		for (TransferData td : listTransfer)
		{
//			Size sizeDest = bIsDestinationSmall ? sizeSmallMemobj : td.getDestinationSize ();
//			Size sizeSource = bIsDestinationSmall ? td.getSourceSize () : sizeSmallMemobj;
//
//			// convert the point index to a linear index
//			// convention: unit stride is in the first coordinate
//			// index (x1,x2,...,xn) in box of size (w1,w2,...,wn) has the linear index
//			//     x1 + x2*w1 + x3*w1*w2 + ... + xn*w1*w2*...*w{n-1} =                             (*)
//			//       x1 + w1(x2 + w2(x3 + ... + w{n-2}(x{n-1} + w{n-1}*xn) ... ))
//
//			Expression exprIdxDest = rgLoopIndices[nDim - 1].clone ();
//			Expression exprIdxSource = rgLoopIndices[nDim - 1].clone ();
//			for (int i = nDim - 2; i >= 0; i--)
//			{
//				// multiply / add according to formula (*)
//				exprIdxDest = new BinaryExpression (
//					new BinaryExpression (sizeDest.getCoord (i).clone (), BinaryOperator.MULTIPLY, exprIdxDest.clone ()),
//					BinaryOperator.ADD,
//					rgLoopIndices[i].clone ());
//				exprIdxSource = new BinaryExpression (
//					new BinaryExpression (sizeSource.getCoord (i).clone (), BinaryOperator.MULTIPLY, exprIdxSource.clone ()),
//					BinaryOperator.ADD,
//					rgLoopIndices[i].clone ());
//			}

			// add assignment statement
			if (bIsDestinationSmall)
			{
	 			slBody.addStatement (new ExpressionStatement (new AssignmentExpression (
	 				mgr.getMemoryObjectExpression (
	 					td.getMinimumStencilNode (),
	 					td.getDestinationMemoryObject (),
	 					null, /* TODO: what if no pointer swapping can be used? */
	 					null,
	 					td.getDestinationMemoryObject ().index (null, td.getMinimumStencilNode (), new Vector (rgLoopIndices), slBody, options),
	 					options
	 				),
	 				AssignmentOperator.NORMAL,
	 				mgr.getMemoryObjectExpression (
	 					td.getSubdomainIdentifier (), td.getMinimumStencilNode (), new Vector (rgLoopIndices), true, true, true, slBody, options)
	 				)
	 			));
			}
			else
			{

			}
		}

		// create the loop nest
		Statement stmtInner = slBody.getCompoundStatement ();
		for (int i = 0; i < nDim; i++)
		{
			boolean bIsHWIndex = i < nHWDim;

			stmtInner = new ForLoop (
				// initialization statement: i=0 if non-HW index / i=thdIdx.* if HW index
				new ExpressionStatement (new AssignmentExpression (
					rgLoopIndices[i].clone (),
					AssignmentOperator.NORMAL,
					bIsHWIndex ? indexingLevel.getIndexForDimension (i).clone () : new IntegerLiteral (0))),

				// condition
				new BinaryExpression (rgLoopIndices[i].clone (), BinaryOperator.COMPARE_LT, sizeSmallMemobj.getCoord (i)),

				// step
				bIsHWIndex ?
					new AssignmentExpression (rgLoopIndices[i].clone (), AssignmentOperator.ADD, indexingLevel.getSizeForDimension (i).clone ()) :
					new UnaryExpression (UnaryOperator.PRE_INCREMENT, rgLoopIndices[i].clone ()),

				// loop body
				stmtInner);
		}

		slbCode.addStatement (stmtInner);
	}

	/**
	 *
	 * @param nParallelismLevel
	 * @param slbCode
	 * @param setTransfer
	 * @param bIsDestinationSmall
	 *            Determines the direction of the data transfers:
	 *            <ul>
	 *            <li><code>true</code>: transfer large &rarr; small (&rArr;
	 *            load)</li>
	 *            <li><code>false</code>: transfer small &rarr; large (&rArr;
	 *            store)</li>
	 *            </ul>
	 */
	private void doDataTransfer (int nParallelismLevel, StatementListBundle slbCode, Set<TransferData> setTransfer, boolean bIsDestinationSmall, CodeGeneratorRuntimeOptions options)
	{
		Size sizePrev = null;
		List<TransferData> listTransfer = new ArrayList<TransferData> (setTransfer.size ());

		// get the number of index dimensions in the architecture/programming
		// model
		int nHWDim = getIndexingLevelFromParallelismLevel (nParallelismLevel).getDimensionality ();

		// get the number of dimensions of the stencil
		int nDim = m_data.getStencilCalculation ().getStencilBundle ().getDimensionality ();

		//

		for (TransferData td : setTransfer)
		{
			// size has changed, create a new group of load loops
			Size size = td.getSmallSize ();
			if (sizePrev != null && !size.equals (sizePrev))
			{
				createDataTransfer (listTransfer, size, nDim, nHWDim, nParallelismLevel, bIsDestinationSmall, slbCode, options);
				listTransfer.clear ();
			}

			listTransfer.add (td);
			sizePrev = size;
		}
		createDataTransfer (listTransfer, sizePrev, nDim, nHWDim, nParallelismLevel, bIsDestinationSmall, slbCode, options);

		setTransfer.clear ();
	}

	@Override
	public void loadData (
		// SubdomainIdentifier sdidLargeLevel,
		// StencilNode nodeDestination, MemoryObject moDestination, //Expression
		// exprDestinationMemoryObject,
		// StencilNode nodeSource, MemoryObject moSource, //Expression
		// exprSourceMemoryObject,
		// int nParallelismLevel,
		// StatementListBundle slbCode)
		StencilNode node, SubdomainIdentifier sdidSourceIterator, MemoryObject moDestination, MemoryObject moSource, int nParallelismLevel, StatementListBundle slbCode,
		CodeGeneratorRuntimeOptions options)
	{
		// only for parallelism level 1 (shared memory)
		if (nParallelismLevel != 1)
			return;

		// m_setLoadData.add (new TransferData (
		// sgidLargeLevel,
		// nodeDestination, moDestination, //exprDestinationMemoryObject,
		// nodeSource, moSource, //exprSourceMemoryObject,
		// true));
		m_setLoadData.add (new TransferData (node, sdidSourceIterator, moDestination, moSource, true));
	}

	@Override
	public void doLoadData (int nParallelismLevel, StatementListBundle slbCode, CodeGeneratorRuntimeOptions options)
	{
		// only for parallelism level 1 (shared memory)
		if (nParallelismLevel != 1)
			return;

		// use parallelism level 2 to create the datatransfer code
		doDataTransfer (2, slbCode, m_setLoadData, true, options);
	}

	@Override
	public void storeData (
		// SubgridIdentifier sgidLargeLevel,
		// StencilNode nodeDestination, MemoryObject moDestination, //Expression
		// exprDestinationMemoryObject,
		// StencilNode nodeSource, MemoryObject moSource, //Expression
		// exprSourceMemoryObject,
		// int nParallelismLevel,
		// StatementListBundle slbCode)
		StencilNode node, SubdomainIdentifier sdidSourceIterator, MemoryObject moDestination, MemoryObject moSource, int nParallelismLevel, StatementListBundle slbCode,
		CodeGeneratorRuntimeOptions options)
	{
		// only for parallelism level 1 (shared memory)
		if (nParallelismLevel != 1)
			return;

		// m_setStoreData.add (new TransferData (
		// sgidLargeLevel,
		// nodeSource, moSource, //exprSourceMemoryObject,
		// nodeDestination, moDestination, //exprDestinationMemoryObject,
		// false));
		m_setStoreData.add (new TransferData (node, sdidSourceIterator, moDestination, moSource, false));
	}

	@Override
	public void doStoreData (int nParallelismLevel, StatementListBundle slbCode, CodeGeneratorRuntimeOptions options)
	{
		// only for parallelism level 1 (shared memory)
		if (nParallelismLevel != 1)
			return;

		// use parallelism level 2 to create the datatransfer code
		doDataTransfer (2, slbCode, m_setStoreData, false, options);
	}

	@Override
	public void waitFor (StencilNode node, MemoryObject mo, int nParallelismLevel, StatementListBundle slbCode)
	{
	}

	@Override
	public void doWaitFor (int nParallelismLevel, StatementListBundle slbCode, CodeGeneratorRuntimeOptions options)
	{
		// only for parallelism level 1 (shared memory)
		if (nParallelismLevel != 1)
			return;

		slbCode.addStatement (new ExpressionStatement (new FunctionCall (new NameID ("__syncthreads"), CodeGeneratorUtil.expressions ())));
	}

	@Override
	public void allocateData (StencilNode node, MemoryObject mo, Expression exprMemoryObject, int nParallelismLevel, StatementListBundle slbCode)
	{
		Expression exprSize = mo.getSize ().getVolume ();
		if (m_exprTotalSharedMemory == null)
			m_exprTotalSharedMemory = exprSize;
		else
			m_exprTotalSharedMemory = new BinaryExpression (m_exprTotalSharedMemory, BinaryOperator.ADD, exprSize);
	}

	@Override
	public void doAllocateData (int nParallelismLevel, StatementListBundle slbCode, CodeGeneratorRuntimeOptions options)
	{
	}


	///////////////////////////////////////////////////////////////////
	// IAdditionalKernelSpecific Implementation

	@Override
	public String getAdditionalKernelSpecificCode ()
	{
		// forward declaration for barrier
		//return "#include \"barrier.cu\"";
		return "";
	}


	///////////////////////////////////////////////////////////////////
	// INonKernelFunction Implementation

	private Map<GlobalGeneratedIdentifiers.Variable, Identifier> m_mapGPUGrids = new HashMap<GlobalGeneratedIdentifiers.Variable, Identifier> ();

	private VariableDeclarator m_declThds;
	private VariableDeclarator m_declBlks;
	private Identifier m_idThds;
	private Identifier m_idBlks;


	@Override
	public void initializeNonKernelFunctionCG ()
	{
		initThreadConfiguration ();
	}

	public StatementList declareGPUGrids ()
	{
		StatementList sl = new StatementList (new ArrayList<Statement> ());

		for (GlobalGeneratedIdentifiers.Variable var : m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables ())
		{
			if (var.isGrid ())
			{
				Expression exprId = getExpressionForVariable (var);
				if (exprId instanceof Identifier)
				{
					NameID nid = new NameID (StringUtil.concat (var.getName (), SUFFIX_GPU));
					VariableDeclarator decl = new VariableDeclarator (nid);
					m_mapGPUGrids.put (var, new Identifier (decl));

					if (var.getType () == GlobalGeneratedIdentifiers.EVariableType.INPUT_GRID)
						sl.addDeclaration (new VariableDeclaration (var.getSpecifiers (), decl));
					else if (var.getType () == GlobalGeneratedIdentifiers.EVariableType.OUTPUT_GRID)
						sl.addDeclaration (new VariableDeclaration (CodeGeneratorUtil.specifiers (SPEC_GPU_PTR, PointerSpecifier.UNQUALIFIED), decl));
				}
			}
		}

		addConfigurationDeclaration (sl);
		return sl;
	}

	private void initThreadConfiguration ()
	{
		StrategyAnalyzer analyzer = m_data.getCodeGenerators ().getStrategyAnalyzer ();

		Dim3Specifier dim3Thds = null;
		Dim3Specifier dim3Blks = null;

		if (analyzer.getParallelismLevelsCount () == 0)
		{
			// no parallelism
			dim3Thds = new Dim3Specifier (new IntegerLiteral (1), new IntegerLiteral (1), new IntegerLiteral (1));
			dim3Blks = new Dim3Specifier (new IntegerLiteral (1), new IntegerLiteral (1), new IntegerLiteral (1));
		}
		else
		{
			// TODO: if < 2 parallelism levels, don't resort to default block
			// size, but add to autotuning parameters!!!

			// total size (grid size * block size)
			Size sizeLevel1 = analyzer.getDomainSizeForParallelismLevel (1);

			// block size (# threads in each dimension per block)
			Size sizeLevel2 = analyzer.getDomainSizeForParallelismLevel (2);
			Size sizeIteratorLevel1 = null;
			if (sizeLevel2 == null)
			{
				// there is no second parallel iterator; use default block size
				int nDim = sizeLevel1.getDimensionality ();
				Expression[] rgBlockSize = new Expression[nDim];
				for (int i = 0; i < nDim; i++)
				{
					String strBlockSizeName = StringUtil.concat ("thds_", CodeGeneratorUtil.getDimensionName (i));
					VariableDeclarator decl = new VariableDeclarator (new NameID (strBlockSizeName));

					rgBlockSize[i] = new Identifier (decl);
					m_data.getData ().getGlobalGeneratedIdentifiers ().addStencilFunctionArguments (
						new Variable (
							EVariableType.INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER,
							new VariableDeclaration (Globals.SPECIFIER_SIZE, decl),
							strBlockSizeName,
							strBlockSizeName,
							null,
							m_data)
					);
				}

				sizeLevel2 = new Size (rgBlockSize);
				sizeIteratorLevel1 = analyzer.getIteratorSizeForParallismLevel (1);
			}

			// #blocks is size(level1) ./ size(level2)
			for (int i = 0; i < sizeLevel1.getDimensionality (); i++)
			{
				Expression exprSizeLevel2 = sizeLevel2.getCoord (i).clone ();

				// if there is only one parallelism level, adjust the number of blocks:
				// must be divided by the size of the first parallel iterator
				if (sizeIteratorLevel1 != null)
					exprSizeLevel2 = new BinaryExpression (exprSizeLevel2, BinaryOperator.MULTIPLY, sizeIteratorLevel1.getCoord (i).clone ());

				sizeLevel1.setCoord (i, ExpressionUtil.ceil (sizeLevel1.getCoord (i).clone (), exprSizeLevel2));
			}

			// map target dimension to HW dim [quick and dirty]
			final int nDimGrid = getIndexingLevelDimensionality (1);
			Expression exprProduct = null;
			for (int i = nDimGrid - 1; i < sizeLevel1.getDimensionality (); i++)
			{
				if (exprProduct == null)
					exprProduct = sizeLevel1.getCoord (i).clone ();
				else
					exprProduct = new BinaryExpression (exprProduct, BinaryOperator.MULTIPLY, sizeLevel1.getCoord (i).clone ());
			}
			sizeLevel1.setCoord (nDimGrid - 1, exprProduct);
			for (int i = nDimGrid; i < sizeLevel1.getDimensionality (); i++)
				sizeLevel1.setCoord (i, new IntegerLiteral (1));

			dim3Thds = new Dim3Specifier (Arrays.asList (sizeLevel2.getCoords ()));
			dim3Blks = new Dim3Specifier (Arrays.asList (sizeLevel1.getCoords ()));
		}

		m_declThds = new VariableDeclarator (new NameID ("thds"), dim3Thds);
		m_declBlks = new VariableDeclarator (new NameID ("blks"), dim3Blks);

		m_idThds = new Identifier (m_declThds);
		m_idBlks = new Identifier (m_declBlks);
	}

	private void addConfigurationDeclaration (StatementList sl)
	{
		sl.addStatement (new DeclarationStatement (new VariableDeclaration (CUDASpecifier.CUDA_DIM3, m_declThds)));
		sl.addStatement (new DeclarationStatement (new VariableDeclaration (CUDASpecifier.CUDA_DIM3, m_declBlks)));
	}

	public StatementList allocateGPUGrids ()
	{
		StatementList sl = new StatementList (new ArrayList<Statement> ());

		// cudaMalloc ((void**) &gpugrid, size);
		for (GlobalGeneratedIdentifiers.Variable var : m_mapGPUGrids.keySet ())
		{
			sl.addStatement (new ExpressionStatement (new FunctionCall (
				new NameID ("cudaMalloc"),
				CodeGeneratorUtil.expressions (
					new Typecast (
						CodeGeneratorUtil.specifiers (Specifier.VOID, PointerSpecifier.UNQUALIFIED, PointerSpecifier.UNQUALIFIED),
						new UnaryExpression (UnaryOperator.ADDRESS_OF, m_mapGPUGrids.get (var).clone ())
					),
					var.getType () == EVariableType.INPUT_GRID ? var.getSize ().clone () : new SizeofExpression (CodeGeneratorUtil.specifiers (SPEC_GPU_PTR))
				)
			)));
		}

		return sl;
	}

	public StatementList copyGridsToGPU ()
	{
		StatementList sl = new StatementList (new ArrayList<Statement> ());

		// cudaMemcpy ((void*) gpugrid, (void*) cpugrid, size, cudaMemcpyHostToDevice);
		for (GlobalGeneratedIdentifiers.Variable var : m_mapGPUGrids.keySet ())
		{
			if (var.getType ().equals (EVariableType.INPUT_GRID))
			{
				sl.addStatement (new ExpressionStatement (new FunctionCall (new NameID ("cudaMemcpy"), CodeGeneratorUtil.expressions (
					new Typecast (CodeGeneratorUtil.specifiers (Specifier.VOID, PointerSpecifier.UNQUALIFIED), m_mapGPUGrids.get (var).clone ()),
					new Typecast (CodeGeneratorUtil.specifiers (Specifier.VOID, PointerSpecifier.UNQUALIFIED), getExpressionForVariable (var).clone ()),
					var.getSize ().clone (),
					new NameID ("cudaMemcpyHostToDevice")
				))));
			}
		}

		return sl;
	}

	private StatementList doCopyInputGridsFromGPU (String strSuffix)
	{
		StatementList sl = new StatementList (new ArrayList<Statement> ());

		// cudaMemcpy ((void*) cpugrid, (void*) gpugrid, size, cudaMemcpyDeviceToHost);
		for (GlobalGeneratedIdentifiers.Variable var : m_mapGPUGrids.keySet ())
		{
			if (var.getType ().equals (EVariableType.INPUT_GRID))
			{
				Expression exprCPUGrid = getExpressionForVariable (var);
				if (strSuffix == null || "".equals (strSuffix))
					exprCPUGrid = exprCPUGrid.clone ();
				else if (exprCPUGrid instanceof IDExpression)
					exprCPUGrid = new NameID (StringUtil.concat (((IDExpression) (exprCPUGrid)).getName (), strSuffix));
				else
					exprCPUGrid = exprCPUGrid.clone ();

				sl.addStatement (new ExpressionStatement (new FunctionCall (new NameID ("cudaMemcpy"), CodeGeneratorUtil.expressions (
					new Typecast (CodeGeneratorUtil.specifiers (Specifier.VOID, PointerSpecifier.UNQUALIFIED), exprCPUGrid),
					new Typecast (CodeGeneratorUtil.specifiers (Specifier.VOID, PointerSpecifier.UNQUALIFIED), m_mapGPUGrids.get (var).clone ()),
					var.getSize ().clone (),
					new NameID ("cudaMemcpyDeviceToHost")
				))));
			}
		}

		return sl;
	}

	public StatementList copyInputGridsFromGPU ()
	{
		return doCopyInputGridsFromGPU ("");
	}

	public StatementList copyInputGridsFromGPUToReferenceGrids ()
	{
		return doCopyInputGridsFromGPU (ValidationCodeGenerator.SUFFIX_REFERENCE);
	}

	/**
	 * Determines whether the input grid variable <code>v</code> has a corresponding output grid.
	 */
	private boolean hasOutputGrid (Variable v)
	{
		String strOutputName = StringUtil.concat (v.getName (), MemoryObjectManager.SUFFIX_OUTPUTGRID);
		for (GlobalGeneratedIdentifiers.Variable var : m_mapGPUGrids.keySet ())
		{
			if (var.getType ().equals (EVariableType.OUTPUT_GRID))
				if (var.getName ().equals (strOutputName))
					return true;
		}

		return false;
	}

	public StatementList copyOutputGridsFromGPU ()
	{
		StatementList sl = new StatementList (new ArrayList<Statement> ());

		// gpu_ptr_t ptr;
		// cudaMemcpy ((void*) &ptr, (void*) gpu_outgrid_ptr, sizeof (gpu_ptr_t), cudaMemcpyDeviceToHost);
		// cudaMemcpy ((void*) cpu_grid, (void*) ptr, size, cudaMemcpyDeviceToHost);

		for (GlobalGeneratedIdentifiers.Variable var : m_mapGPUGrids.keySet ())
		{
			if (var.getType ().equals (EVariableType.OUTPUT_GRID))
			{
				String strCorrespondingInputGrid = var.getName ().substring (0, var.getName ().length () - MemoryObjectManager.SUFFIX_OUTPUTGRID.length ());

				Identifier idGPUOutputGrid = m_mapGPUGrids.get (var);

				VariableDeclarator declGPUPointer = new VariableDeclarator (new NameID (StringUtil.concat (idGPUOutputGrid.getName (), "_ptr")));
				Identifier idGPUPointer = new Identifier (declGPUPointer);

				sl.addStatement (new DeclarationStatement (new VariableDeclaration (SPEC_GPU_PTR, declGPUPointer)));
				sl.addStatement (new ExpressionStatement (new AssignmentExpression (
					getExpressionForVariable (var).clone (), AssignmentOperator.NORMAL, new NameID (strCorrespondingInputGrid))));

				sl.addStatement (new ExpressionStatement (new FunctionCall (new NameID ("cudaMemcpy"), CodeGeneratorUtil.expressions (
					new Typecast (CodeGeneratorUtil.specifiers (Specifier.VOID, PointerSpecifier.UNQUALIFIED), new UnaryExpression (UnaryOperator.ADDRESS_OF, idGPUPointer.clone ())),
					new Typecast (CodeGeneratorUtil.specifiers (Specifier.VOID, PointerSpecifier.UNQUALIFIED), idGPUOutputGrid.clone ()),
					new SizeofExpression (CodeGeneratorUtil.specifiers (SPEC_GPU_PTR)),
					new NameID ("cudaMemcpyDeviceToHost")
				))));

				sl.addStatement (new ExpressionStatement (new FunctionCall (new NameID ("cudaMemcpy"), CodeGeneratorUtil.expressions (
					new Typecast (CodeGeneratorUtil.specifiers (Specifier.VOID, PointerSpecifier.UNQUALIFIED), getExpressionForVariable (var).clone ()),
					new Typecast (CodeGeneratorUtil.specifiers (Specifier.VOID, PointerSpecifier.UNQUALIFIED), idGPUPointer.clone ()),
					var.getSize ().clone (),
					new NameID ("cudaMemcpyDeviceToHost")
				))));
			}
		}

		return sl;
	}

	@Override
	public List<Expression> getExpressionsForVariables (List<Variable> listVariables, EOutputGridType typeOutputGrid)
	{
		List<Expression> list = new ArrayList<Expression> (listVariables.size ());
		for (Variable v : listVariables)
		{
			Identifier id = m_mapGPUGrids.get (v);
			if (id != null)
			{
				if (v.getType ().equals (EVariableType.OUTPUT_GRID))
				{
					switch (typeOutputGrid)
					{
					case OUTPUTGRID_DEFAULT:
						list.add (id.clone ());
						break;
					case OUTPUTGRID_POINTER:
						list.add (new UnaryExpression (UnaryOperator.ADDRESS_OF, id.clone ()));
						break;
					case OUTPUTGRID_TYPECAST:
						list.add (new Typecast (v.getSpecifiers (), id.clone ()));
						break;
					}
				}
				else
					list.add (id.clone ());
			}
			else
				list.add (getExpressionForVariable (v, typeOutputGrid));
		}
		return list;
	}

	@Override
	public StatementList initializeGrids ()
	{
		NameID nidInitialize = m_data.getData ().getGlobalGeneratedIdentifiers ().getInitializeFunctionName ();
		if (nidInitialize == null)
			return null;

		return new StatementList (new ExpressionStatement (new KernelFunctionCall (
			nidInitialize.clone (),
			getExpressionsForVariables (
				m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables (
					~EVariableType.OUTPUT_GRID.mask () & ~EVariableType.INTERNAL_AUTOTUNE_PARAMETER.mask () & ~EVariableType.INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER.mask ()
				),
				EOutputGridType.OUTPUTGRID_DEFAULT),
			CodeGeneratorUtil.expressions (m_idBlks.clone (), m_idThds.clone ())
		)));
	}

	@Override
	public StatementList computeStencil ()
	{
		// launch the GPU kernel
		return new StatementList (new ExpressionStatement (new KernelFunctionCall (
			// kernel name
			m_data.getData ().getGlobalGeneratedIdentifiers ().getStencilFunctionName ().clone (),

			// kernel arguments
			getExpressionsForVariables (
				m_data.getData ().getGlobalGeneratedIdentifiers ().getVariables (~EVariableType.INTERNAL_NONKERNEL_AUTOTUNE_PARAMETER.mask ()),
				EOutputGridType.OUTPUTGRID_TYPECAST),

			// configuration
			CodeGeneratorUtil.expressions (m_idBlks.clone (), m_idThds.clone ()))));
	}

	@Override
	public StatementList deallocateGrids ()
	{
		StatementList sl = new StatementList (new ArrayList<Statement> ());

		// deallocate GPU grids
		for (Variable v : m_mapGPUGrids.keySet ())
		{
			if (v.getType () == EVariableType.INPUT_GRID)
			{
				sl.addStatement (new ExpressionStatement (new FunctionCall (
					new NameID ("cudaFree"),
					CodeGeneratorUtil.expressions (new Typecast (
						CodeGeneratorUtil.specifiers (Specifier.VOID, PointerSpecifier.UNQUALIFIED),
						m_mapGPUGrids.get (v).clone ()
					))
				)));
			}
		}

		// deallocate CPU grids
		sl.addStatements (super.deallocateGrids ());
		return sl;
	}
}
