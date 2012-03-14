package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import cetus.hir.BinaryExpression;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Literal;
import cetus.hir.Specifier;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * Creates a list of assembly instructions from an {@link Expression.
 * 
 * @author Matthias-M. Christen
 */
public class AssemblyExpressionCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Constants

	private static final Comparator<int[]> COMPARATOR_REGSCOUNT = new Comparator<int[]> ()
	{
		@Override
		public int compare (int[] rg1, int[] rg2)
		{
			// sort by register count, which is stored in the second array entry
			return rg2[1] - rg1[1];
		}
	};

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;
	
	private StencilAssemblySection m_assemblySection;
	
	private Map<StencilNode, IOperand.IRegisterOperand> m_mapReuseNodesToRegisters;
	
	private Map<Expression, IOperand.IRegisterOperand> m_mapConstantsAndParams;
		
	private RegisterAllocator m_allocator;

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public AssemblyExpressionCodeGenerator (StencilAssemblySection as, CodeGeneratorSharedObjects data,
		Map<StencilNode, IOperand.IRegisterOperand> mapReuseNodesToRegisters,
		Map<Expression, IOperand.IRegisterOperand> mapConstantsAndParams)
	{
		m_data = data;
		m_assemblySection = as;
		m_mapReuseNodesToRegisters = mapReuseNodesToRegisters;
		m_mapConstantsAndParams = mapConstantsAndParams;
		
		Map<StencilNode, Boolean> mapReuse = new HashMap<StencilNode, Boolean> ();
		for (StencilNode node : m_mapReuseNodesToRegisters.keySet ())
			mapReuse.put (node, true);
		
		m_allocator = new RegisterAllocator (m_data, m_assemblySection, mapReuse);
	}
	
	/**
	 * 
	 * @param expr
	 * @param options
	 * @return
	 */
	public void generate (Expression expr, StencilNode nodeOutput, InstructionList il, CodeGeneratorRuntimeOptions options)
	{
		// get code generation options
		int nUnrollFactor = options.getIntValue (InnermostLoopCodeGenerator.OPTION_INLINEASM_UNROLLFACTOR);
		
		// generate the inline assembly code
		m_allocator.countRegistersNeeded (expr);
		IOperand[] rgResult = traverse (expr, m_assemblySection.getDatatype (), nUnrollFactor, il);
		
		// write the result back
		// TODO: handle temporaries (scalar stencil nodes)
		if (nodeOutput.isScalar ())
		{
			
		}
		
		IOperand[] rgDest = processStencilNode (nodeOutput, nUnrollFactor, il);
		for (int i = 0; i < nUnrollFactor; i++)
			il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.MOVE_FPR, rgResult[i], rgDest[i]));
	}
	
	/**
	 * 
	 * @param expr
	 * @param specDatatype
	 * @param nUnrollFactor
	 * @param il
	 * @return
	 */
	private IOperand[] traverse (Expression expr, Specifier specDatatype, int nUnrollFactor, InstructionList il)
	{
		if (expr instanceof StencilNode)
			return processStencilNode ((StencilNode) expr, nUnrollFactor, il);
//		if (expr instanceof IDExpression)
//			return processVariable ((IDExpression) expr, nUnrollFactor, il);
		if (expr instanceof Literal || expr instanceof IDExpression)
			return processConstantOrParam (expr, nUnrollFactor, il);
		
		if (RegisterAllocator.isAddSubSubtree (expr))
			return processAddSubSubtree (expr, specDatatype, nUnrollFactor, il);
		if (expr instanceof BinaryExpression)
			return processBinaryExpression ((BinaryExpression) expr, specDatatype, nUnrollFactor, il);
		if (expr instanceof FunctionCall)
			return processFunctionCall ((FunctionCall) expr, specDatatype, nUnrollFactor, il);
		if (expr instanceof UnaryExpression)
			return processUnaryExpression ((UnaryExpression) expr, specDatatype, nUnrollFactor, il);
		
		return null;
	}
	
	/**
	 * Adds the instruction based on <code>strIntrinsicBaseName</code> <code>nUnrollFactor</code>-times
	 * to the instruction list <code>il</code>.
	 * 
	 * @param il
	 *            The instruction list to which the instructions will be added
	 * @param nUnrollFactor
	 *            The unroll factor determining how many times to add the instruction to
	 *            the instruction list
	 * @param strIntrinsicBaseName
	 *            The base name of the intrinsic for which to generate an instruction
	 * @param rgResult
	 *            An array into which the result operands will be written, or <code>null</code>,
	 *            in which case a new result array will be created and returned by the method
	 * @param rgArguments
	 *            The operand arguments to the instruction
	 * @return An array of operands containing the result for each of the unrollings
	 */
	private IOperand[] addInstruction (InstructionList il, int nUnrollFactor, String strIntrinsicBaseName, IOperand[] rgResult, IOperand[]... rgArguments)
	{
		boolean bHasResult = rgResult != null;
		if (!bHasResult)
			rgResult = new IOperand[nUnrollFactor];
		
		for (int i = 0; i < nUnrollFactor; i++)
		{
			if (!bHasResult)
				rgResult[i] = new IOperand.PseudoRegister ();
			
			IOperand[] rgArgs = new IOperand[rgArguments.length + 1];
			for (int j = 0; j < rgArguments.length; j++)
				rgArgs[j] = rgArguments[j][i];
			rgArgs[rgArguments.length] = rgResult[i];
			
			il.addInstruction (new Instruction (strIntrinsicBaseName, rgArgs));
		}
		
		return rgResult;
	}

	/**
	 * Generates the code for a add-sub subtree.
	 * @param expr
	 * @param specDatatype
	 * @param nUnrollFactor
	 * @param il
	 * @return
	 */
	private IOperand[] processAddSubSubtree (Expression expr, Specifier specDatatype, int nUnrollFactor, InstructionList il)
	{
		List<AddSub> list = new LinkedList<AddSub> ();
		RegisterAllocator.linearizeAddSubSubtree (expr, list);
		
		IOperand[] rgResult = null;
		
		int i = 0;
		for (AddSub addsub : list)
		{
			if (i == 0)
				rgResult = traverse (addsub.getExpression (), specDatatype, nUnrollFactor, il);
			else
			{
				rgResult = addInstruction (il, nUnrollFactor,
					addsub.getBaseIntrinsic (),
					i == 1 ? null : rgResult,
					traverse (addsub.getExpression (), specDatatype, nUnrollFactor, il),
					rgResult
				);
			}
			
			i++;
		}
		
		return rgResult;
	}
	
	/**
	 * Processes the argument expressions <code>rgExpressions</code> in the order such that
	 * the expression that requires most registers is processed first and the expression
	 * with the least register requirements is processed last.
	 * 
	 * @param specDatatype
	 *            The data type of the expression
	 * @param nUnrollFactor
	 *            The unrolling factor determining how many results will be generated
	 * @param il
	 *            The instruction list to which intermediately generated instructions are added
	 * @param rgExpressions
	 *            The array of expressions to process
	 * @return An array of operands containing the results for each of the expressions <code>rgExpressions</code>
	 *         (first array index) and for each of the unrolling factors (second array index)
	 */
	private IOperand[][] processArguments (Specifier specDatatype, int nUnrollFactor, InstructionList il, Expression... rgExpressions)
	{
		// find the register counts for each of the expressions and sort by descending number of registers
		List<int[]> listRegsCount = new ArrayList<int[]> (rgExpressions.length);
		for (int i = 0; i < rgExpressions.length; i++)
			listRegsCount.add (new int[] { i, m_allocator.getNumRegistersUsed (rgExpressions[i]) });
		Collections.sort (listRegsCount, COMPARATOR_REGSCOUNT);

		// traverse the expressions in rgExpressions in the order of descending register counts
		IOperand[][] rgResult = new IOperand[rgExpressions.length][nUnrollFactor];
		for (int[] rgRegsCount : listRegsCount)
			rgResult[rgRegsCount[0]] = traverse (rgExpressions[rgRegsCount[0]], specDatatype, nUnrollFactor, il);
		
		return rgResult;
	}
	
	/**
	 * 
	 * @param expr
	 * @param specDatatype
	 * @param nUnrollFactor
	 * @param il
	 * @return
	 */
	private IOperand[] processBinaryExpression (BinaryExpression expr, Specifier specDatatype, int nUnrollFactor, InstructionList il)
	{
		return addInstruction (il, nUnrollFactor, Globals.getIntrinsicBase (expr.getOperator ()).value (),
			null, processArguments (specDatatype, nUnrollFactor, il, expr.getLHS (), expr.getRHS ()));

		//isReservedRegister (rgOpRHS[i]) ? new IOperand.PseudoRegister () : rgOpRHS[i];
	}
	
	/**
	 * 
	 * @param fnxCall
	 * @param specDatatype
	 * @param nUnrollFactor
	 * @param il
	 * @return
	 */
	private IOperand[] processFunctionCall (FunctionCall fnxCall, Specifier specDatatype, int nUnrollFactor, InstructionList il)
	{
		Expression exprFuncName = fnxCall.getName ();
		Intrinsic i = m_data.getArchitectureDescription ().getIntrinsic (exprFuncName.toString (), specDatatype);

		// only fma and fms supported for now...
		if (exprFuncName.equals (Globals.FNX_FMA) || exprFuncName.equals (Globals.FNX_FMS))
			return processFusedMultiplyAddSub (i, fnxCall.getArgument (0), fnxCall.getArgument (1), fnxCall.getArgument (2), specDatatype, nUnrollFactor, il);
		else
			throw new RuntimeException (StringUtil.concat ("The function '", exprFuncName.toString (), "' is currently not supported."));
	}
	
	/**
	 * 
	 * @param intrinsic
	 * @param exprSummand
	 * @param exprFactor1
	 * @param exprFactor2
	 * @param specDatatype
	 * @param nUnrollFactor
	 * @param il
	 * @return
	 */
	private IOperand[] processFusedMultiplyAddSub (Intrinsic intrinsic, Expression exprSummand, Expression exprFactor1, Expression exprFactor2,
		Specifier specDatatype, int nUnrollFactor, InstructionList il)
	{
		return addInstruction (il, nUnrollFactor, intrinsic.getBaseName (), null,
			processArguments (specDatatype, nUnrollFactor, il, exprSummand, exprFactor1, exprFactor2));
	}

	/**
	 * Processes an unary expression.
	 * 
	 * @param expr
	 *            The unary expression to process
	 * @param specDatatype
	 *            The datatype of the expression
	 * @param nUnrollFactor
	 *            The unrolling factor; this will add <code>nUnrollFactor</code> instructions to the instruction list
	 * @param il
	 *            The instruction list to which the instructions generated by the unary expression are added
	 * @return The operands in which the results are stored. Returns <code>nUnrollFactor</code> results.
	 */
	private IOperand[] processUnaryExpression (UnaryExpression expr, Specifier specDatatype, int nUnrollFactor, InstructionList il)
	{
		IOperand[] rgOps = traverse (expr, specDatatype, nUnrollFactor, il);
		
		if (((UnaryExpression) expr).getOperator ().equals (UnaryOperator.MINUS))
			rgOps = addInstruction (il, nUnrollFactor, TypeBaseIntrinsicEnum.UNARY_MINUS.value (), null, rgOps);

		return rgOps;
	}

	/**
	 * 
	 * @param node
	 * @param nUnrollFactor
	 * @param il
	 * @return
	 */
	private IOperand[] processStencilNode (StencilNode node, int nUnrollFactor, InstructionList il)
	{
		IOperand[] rgOpResults = new IOperand[nUnrollFactor];
		boolean bIsReuse = true;

		// check whether the stencil node is saved in a reuse register
		StencilNode nodeTest = new StencilNode (node);
		for (int i = 0; i < nUnrollFactor; i++)
		{
			nodeTest.getSpaceIndex ()[0] = node.getSpaceIndex ()[0] + i;
			IOperand.IRegisterOperand op = m_mapReuseNodesToRegisters.get (nodeTest);
			if (op == null)
			{
				bIsReuse = false;
				break;
			}
			
			rgOpResults[i] = op;
		}
		
		// load the value of the stencil node into a register
		if (!bIsReuse)
		{
			for (int i = 0; i < nUnrollFactor; i++)
			{
				IOperand op = m_assemblySection.getGrid (node, i);
//				if (op instanceof IOperand.Address)
//				{
//					// if the operand is an address, check whether memory operands are supported for the argument
//					if (!RegisterAllocator.canBeMemoryOperand (node, m_data, m_assemblySection))
//					{
//						// no: we need to load the data into a register first
//						IOperand opAddr = op;
//						op = new IOperand.PseudoRegister ();
//						il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.MOVE_FPR.value (), new IOperand[] { opAddr, op }));
//					}
//				}
				
				rgOpResults[i] = op;
			}
		}
		
		return rgOpResults;
	}
	
	private IOperand[] processVariable (IDExpression id, int nUnrollFactor, InstructionList il)
	{
		if (m_mapConstantsAndParams.containsKey (id))
			return new IOperand[] { m_mapConstantsAndParams.get (id) };
		
/*		
		// if id is constant, we need only one register, otherwise we need nUnrollFactor registers
		boolean bIsConstant =
			m_data.getStencilCalculation ().getStencilBundle ().isConstantOutputStencilNode (id) ||
			m_data.getStencilCalculation ().isArgument (id.getName ());
		
		// allocate pseudo registers
		IOperand.IRegisterOperand[] rgRegs = new IOperand.IRegisterOperand[bIsConstant ? 1 : nUnrollFactor];		
		for (int i = 0; i < rgRegs.length; i++)
		{
			rgRegs[i] = new IOperand.PseudoRegister ();

			// if id is a constant, load the value from memory
			// otherwise assume it's a temporary variable
			if (bIsConstant)
			{
				Intrinsic intrinsic = m_data.getArchitectureDescription ().getIntrinsic (
					TypeBaseIntrinsicEnum.MOVE_FPR.value (), m_assemblySection.getDatatype ());
				if (intrinsic == null)
					throw new RuntimeException ("No FRP move instruction defined");
				
				il.addInstruction (new Instruction (intrinsic.getName (), new IOperand[] { X, rgRegs[i] }));
			}
		}
		
		m_mapVariables.put (id.getName (), rgRegs);
		
		return rgRegs;
*/
		return null;
	}
	
	private IOperand[] processConstantOrParam (Expression exprConstantOrParam, int nUnrollFactor, InstructionList il)
	{
		// find the register the constant is saved in or load the constant into a register (if too many registers)
		IOperand[] rgResult = new IOperand[nUnrollFactor];
		
		IOperand op = m_mapConstantsAndParams.get (exprConstantOrParam);
		if (op == null)
		{
			// the constant or parameter is not contained in the map, i.e., there are no special registers
			// reserved for them => load into a temporary register from memory
			
			int nConstParamIdx = m_assemblySection.getConstantOrParamIndex (exprConstantOrParam);
			if (nConstParamIdx == -1)
			{
				// the constant/parameter was not found in the assembly section constant/param input array
				if (exprConstantOrParam instanceof IDExpression)
					return processVariable ((IDExpression) exprConstantOrParam, nUnrollFactor, il);
				
				throw new RuntimeException (StringUtil.concat ("Don't know how to process ", exprConstantOrParam.toString ()));
			}
			else
			{
				// the constant/parameter was found in the assembly section constant/param input array
				
				Specifier specType = m_assemblySection.getDatatype ();
				int nSIMDVectorLength = m_data.getArchitectureDescription ().getSIMDVectorLength (specType);
				
//				il.addInstruction (new Instruction (
//					TypeBaseIntrinsicEnum.MOVE_FPR.value (),
//					new IOperand[] {
//						new IOperand.Address (
//							(IOperand.IRegisterOperand) m_assemblySection.getInput (StencilAssemblySection.INPUT_CONSTANTS_ARRAYPTR),
//							nConstParamIdx * AssemblySection.getTypeSize (specType) * nSIMDVectorLength),
//						op = new IOperand.PseudoRegister ()
//					}
//				));
				
				op = new IOperand.Address (
					(IOperand.IRegisterOperand) m_assemblySection.getInput (StencilAssemblySection.INPUT_CONSTANTS_ARRAYPTR),
					nConstParamIdx * AssemblySection.getTypeSize (specType) * nSIMDVectorLength); 
			}
		}
		
		for (int i = 0; i < nUnrollFactor; i++)
			rgResult[i] = op;
		return rgResult;
	}
	
	/**
	 * Determines whether <code>opReg</code> is a register reserved for stencil node
	 * reuse or for constants.
	 * @param opReg The register operand to test
	 * @return <code>true</code> iff <code>opReg</code> is a reserved register
	 */
	private boolean isReservedRegister (IOperand opReg)
	{
		if (m_mapReuseNodesToRegisters.containsValue (opReg))
			return true;
		if (m_mapConstantsAndParams.containsValue (opReg))
			return true;
		
		return false;
	}
}
