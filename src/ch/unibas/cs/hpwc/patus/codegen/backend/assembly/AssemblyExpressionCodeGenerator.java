package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.StencilAssemblySection.OperandWithInstructions;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
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


	/**
	 * TODO: check correctness if add-sub chains are broken
	 * TODO: make into code-gen option / expose to auto-tuner
	 * TODO: test whether balanced trees give better performance
	 */
	private static final boolean BREAK_ADDSUB_CHAINS = false;

	
	///////////////////////////////////////////////////////////////////
	// Member Variables	

	private CodeGeneratorSharedObjects m_data;
	
	private StencilAssemblySection m_assemblySection;
	
	private Queue<IInstruction[]> m_quInstructionsToAdd;
	
	private Map<StencilNode, IOperand.IRegisterOperand> m_mapReuseNodesToRegisters;
	
	private Map<NameID, IOperand.IRegisterOperand[]> m_mapTempoararies;
		
	private RegisterAllocator m_allocator;

	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public AssemblyExpressionCodeGenerator (StencilAssemblySection as, CodeGeneratorSharedObjects data,
		Map<StencilNode, IOperand.IRegisterOperand> mapReuseNodesToRegisters,
		Map<NameID, IOperand.IRegisterOperand[]> mapTemporaries)
	{
		m_data = data;
		m_assemblySection = as;
		m_mapReuseNodesToRegisters = mapReuseNodesToRegisters;
		m_mapTempoararies = mapTemporaries;
		
		m_quInstructionsToAdd = new LinkedList<> ();
				
		Map<StencilNode, Boolean> mapReuse = new HashMap<> ();
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
		
		// apply fused multiply adds/subs if corresponding intrinsics are defined
		Expression exprFMAed = m_data.getCodeGenerators ().getFMACodeGenerator ().applyFMAs (expr, m_assemblySection.getDatatype (), false);
		
		// generate the inline assembly code
		m_allocator.countRegistersNeeded (exprFMAed);
		IOperand[] rgResult = traverse (exprFMAed, m_assemblySection.getDatatype (), nUnrollFactor, il);
		
		// write the result back
		if (nodeOutput.isScalar ())
		{
			IOperand.IRegisterOperand[] rgDest = new IOperand.IRegisterOperand[nUnrollFactor];
			for (int i = 0; i < nUnrollFactor; i++)
			{
				if (rgResult[i] instanceof IOperand.IRegisterOperand)
					rgDest[i] = (IOperand.IRegisterOperand) rgResult[i];
				else
				{
					rgDest[i] = new IOperand.PseudoRegister (TypeRegisterType.SIMD);
					addInstructions (il, i == nUnrollFactor - 1, new Instruction (TypeBaseIntrinsicEnum.STORE_FPR_ALIGNED, rgResult[i], rgDest[i]));
				}
			}
			
			m_mapTempoararies.put (new NameID (nodeOutput.getName ()), rgDest);
		}
		else
		{
			IOperand[] rgDest = processStencilNode (nodeOutput, nUnrollFactor, il);
			for (int i = 0; i < nUnrollFactor; i++)
				addInstructions (il, i == nUnrollFactor - 1, new Instruction (TypeBaseIntrinsicEnum.STORE_FPR_ALIGNED, rgResult[i], rgDest[i]));
		}
	}
	
	/**
	 * Recursively traverse the expression <code>expr</code>.
	 * 
	 * @param expr
	 *            The expression to traverse
	 * @param specDatatype
	 *            The datatype of the expression
	 * @param nUnrollFactor
	 *            The unrolling factor
	 * @param il
	 *            The instruction list, to which instructions are added during
	 *            traversing the expression
	 * @return The array of operands which hold the result of the computation of
	 *         <code>expr</code>
	 */
	private IOperand[] traverse (Expression expr, Specifier specDatatype, int nUnrollFactor, InstructionList il)
	{
		if (expr instanceof StencilNode)
			return processStencilNode ((StencilNode) expr, nUnrollFactor, il);
		if (expr instanceof Literal || expr instanceof IDExpression)
			return processConstantOrIdentifier (expr, nUnrollFactor, il);
		
		if (expr instanceof UnaryExpression)
			return processUnaryExpression ((UnaryExpression) expr, specDatatype, nUnrollFactor, il);
		if (RegisterAllocator.isAddSubSubtree (expr))
			return processAddSubSubtree (expr, specDatatype, nUnrollFactor, il);
		if (expr instanceof BinaryExpression)
			return processBinaryExpression ((BinaryExpression) expr, specDatatype, nUnrollFactor, il);
		if (expr instanceof FunctionCall)
			return processFunctionCall ((FunctionCall) expr, specDatatype, nUnrollFactor, il);
		
		return null;
	}
	
	/**
	 * Adds the single instruction <code>instr</code> to the instruction list,
	 * followed by the instructions that have been accumulated in the
	 * {@link AssemblyExpressionCodeGenerator#m_quInstructionsToAdd} queue.
	 * 
	 * @param il
	 *            The instruction list to which to add the instruction
	 * @param instr
	 *            the instruction to add
	 */
	private void addInstructions (InstructionList il, boolean bAddAccumulatedInstructions, IInstruction... instr)
	{
		// add the instruction proper
		il.addInstructions (instr);

		// add any accumulated post-instruction instructions
		while (bAddAccumulatedInstructions && !m_quInstructionsToAdd.isEmpty ())
			il.addInstructions (m_quInstructionsToAdd.poll ());
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
	private IOperand[] addInstruction (InstructionList il, int nUnrollFactor, TypeBaseIntrinsicEnum intrinsic, IOperand[] rgResult, IOperand[]... rgArguments)
	{
		IOperand[] rgResultLocal = rgResult;
		if (rgResultLocal == null)
			rgResultLocal = new IOperand[nUnrollFactor];
		
		boolean bHasResult = rgResult != null;
		for (int i = 0; i < nUnrollFactor; i++)
		{
			if (!bHasResult)
				rgResultLocal[i] = new IOperand.PseudoRegister (TypeRegisterType.SIMD);
			
			IOperand[] rgArgs = new IOperand[rgArguments.length + 1];
			for (int j = 0; j < rgArguments.length; j++)
				rgArgs[j] = rgArguments[j][i];
			rgArgs[rgArguments.length] = rgResultLocal[i];
			
			addInstructions (il, i == nUnrollFactor - 1, new Instruction (intrinsic, rgArgs));
		}
		
		return rgResultLocal;
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
		if (!BREAK_ADDSUB_CHAINS)
		{
			IOperand[] rgResult = null;
			
			int i = 0;
			for (AddSub addsub : RegisterAllocator.linearizeAddSubSubtree (expr))
			{
				if (i == 0)
					rgResult = traverse (addsub.getExpression (), specDatatype, nUnrollFactor, il);
				else
				{
					boolean bIsAddition = addsub.getOperator ().equals (BinaryOperator.ADD);
					IOperand[] op1 = traverse (addsub.getExpression (), specDatatype, nUnrollFactor, il);
					
					rgResult = addInstruction (il, nUnrollFactor,
						addsub.getBaseIntrinsic (),
						i == 1 ? null : rgResult,
						bIsAddition ? op1 : rgResult,	// swap the operands if this is an addition (which is commutative...)
						bIsAddition ? rgResult : op1
					);
				}
				
				i++;
			}
			
			return rgResult;
		}
		else
		{
			IOperand[] rgResult0 = null;
			IOperand[] rgResult1 = null;
			AddSub addsub1 = null;
			
			int i = 0;
			for (AddSub addsub : RegisterAllocator.linearizeAddSubSubtree (expr))
			{
				if (i == 0)
					rgResult0 = traverse (addsub.getExpression (), specDatatype, nUnrollFactor, il);
				else if (i == 1)
				{
					addsub1 = addsub;
					rgResult1 = traverse (addsub.getExpression (), specDatatype, nUnrollFactor, il);
				}
				else
				{
					boolean bIsAddition = addsub.getOperator ().equals (BinaryOperator.ADD);
					IOperand[] op1 = traverse (addsub.getExpression (), specDatatype, nUnrollFactor, il);
					
					IOperand[] rgResult = ((i % 2) == 0) ? rgResult0 : rgResult1;
					
					rgResult = addInstruction (il, nUnrollFactor,
						addsub.getBaseIntrinsic (),
						i == 2 || i == 3 ? null : rgResult,
						bIsAddition ? op1 : rgResult,	// swap the operands if this is an addition (which is commutative...)
						bIsAddition ? rgResult : op1
					);
					
					if ((i % 2) == 0)
						rgResult0 = rgResult;
					else
						rgResult1 = rgResult;
				}
				
				i++;
			}
	
			if (rgResult1 == null)
				return rgResult0;
			return addInstruction (il, nUnrollFactor, addsub1.getBaseIntrinsic (), rgResult0, rgResult0, rgResult1);
		}
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
		List<int[]> listRegsCount = new ArrayList<> (rgExpressions.length);
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
		return addInstruction (
			il, nUnrollFactor, Globals.getIntrinsicBase (expr.getOperator ()),
			null, processArguments (specDatatype, nUnrollFactor, il, expr.getLHS (), expr.getRHS ()));
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

		// only fma and fms supported for now...
		if (exprFuncName.equals (Globals.FNX_FMA))
			return processFusedMultiplyAddSub (TypeBaseIntrinsicEnum.FMA, fnxCall.getArgument (0), fnxCall.getArgument (1), fnxCall.getArgument (2), specDatatype, nUnrollFactor, il);
		else if (exprFuncName.equals (Globals.FNX_FMS))
			return processFusedMultiplyAddSub (TypeBaseIntrinsicEnum.FMS, fnxCall.getArgument (0), fnxCall.getArgument (1), fnxCall.getArgument (2), specDatatype, nUnrollFactor, il);
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
	private IOperand[] processFusedMultiplyAddSub (TypeBaseIntrinsicEnum intrinsic, Expression exprSummand, Expression exprFactor1, Expression exprFactor2,
		Specifier specDatatype, int nUnrollFactor, InstructionList il)
	{
		return addInstruction (
			il, nUnrollFactor, intrinsic, null,
			processArguments (specDatatype, nUnrollFactor, il, exprSummand, exprFactor1, exprFactor2)
		);
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
		IOperand[] rgOps = traverse (expr.getExpression (), specDatatype, nUnrollFactor, il);
		
		if (expr.getOperator ().equals (UnaryOperator.MINUS))
			rgOps = addInstruction (il, nUnrollFactor, TypeBaseIntrinsicEnum.UNARY_MINUS, null, rgOps);

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
//			nodeTest.getSpaceIndex ()[0] = node.getSpaceIndex ()[0] + i;
			nodeTest.getIndex ().setSpaceIndex (0, ExpressionUtil.add (node.getIndex ().getSpaceIndex (0), new IntegerLiteral (i)));
			
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
				StencilAssemblySection.OperandWithInstructions op = m_assemblySection.getGrid (node, i);
				
				if (op.getInstrPre () != null && op.getInstrPost () != null)
				{
					// if there are both pre and post instructions when accessing a stencil node,
					// move the stencil node data into a temporary pseudo register and surround the
					// move command with the pre and post instructions
					// (if this isn't done, the generated code could be wrong if the instruction accesses
					// data which would both require and not require the pre and post instructions, e.g.
					// 		vsubps (%1,%13,2), (%1,%13)*, %%ymm3
					// where the operand * requires negating %13.)
					
					if (i == 0)
						il.addInstructions (op.getInstrPre ());
					
					rgOpResults[i] = new IOperand.PseudoRegister (TypeRegisterType.SIMD);
					il.addInstruction (new Instruction (TypeBaseIntrinsicEnum.LOAD_FPR_ALIGNED, op.getOp (), rgOpResults[i]));
					
					if (i == nUnrollFactor - 1)
						il.addInstructions (op.getInstrPost ());
				}
				else
				{
					rgOpResults[i] = op.getOp ();

					// the pre- and post-instructions will be the same for each node access
					// (since in the unrolling direction only the displacement changes), but
					// we need to add the additional instructions only once
					
					if (i == 0)
					{
						il.addInstructions (op.getInstrPre ());
						if (op.getInstrPost () != null)
							m_quInstructionsToAdd.add (op.getInstrPost ());
					}
				}
			}
		}
		
		return rgOpResults;
	}
	
	private IOperand[] processConstantOrIdentifier (Expression exprConstantOrIdentifier, int nUnrollFactor, InstructionList il)
	{
		// try to find the NameID in the temporaries map
		IOperand[] rgResult = null;
		if (exprConstantOrIdentifier instanceof NameID)
		{
			rgResult = m_mapTempoararies.get (exprConstantOrIdentifier);
			if (rgResult != null)
				return rgResult;
		}		
		
		// find the register the constant is saved in or load the constant into a register (if too many registers)
		OperandWithInstructions owi = m_assemblySection.getConstantOrParam (exprConstantOrIdentifier);
		IOperand op = owi.getOp ();
		il.addInstructions (owi.getInstrPre ());
		if (owi.getInstrPost () != null)
			m_quInstructionsToAdd.add (owi.getInstrPost ());
		
		if (op == null)
			throw new RuntimeException (StringUtil.concat ("Could not find or resolve ", exprConstantOrIdentifier.toString ()));
				
		rgResult = new IOperand[nUnrollFactor];
		for (int i = 0; i < nUnrollFactor; i++)
			rgResult[i] = op;
		return rgResult;
	}	
}
