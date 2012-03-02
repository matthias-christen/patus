package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.backend.IBackendAssemblyCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.representation.Index;
import ch.unibas.cs.hpwc.patus.representation.StencilNode;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * 
 * @author Matthias-M. Christen
 */
public class AssemblyExpressionCodeGenerator
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;
	private IBackendAssemblyCodeGenerator m_cg;
	
	private StencilAssemblySection m_assemblySection;
	
	private Map<StencilNode, IOperand.IRegisterOperand> m_mapReuseNodesToRegisters;
	
	private Map<Double, IOperand.IRegisterOperand> m_mapConstants;
	
	private RegisterAllocator m_allocator;
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public AssemblyExpressionCodeGenerator (StencilAssemblySection as, CodeGeneratorSharedObjects data,
		Map<StencilNode, IOperand.IRegisterOperand> mapReuseNodesToRegisters,
		Map<Double, IOperand.IRegisterOperand> mapConstants)
	{
		m_data = data;
		m_assemblySection = as;
		m_mapReuseNodesToRegisters = mapReuseNodesToRegisters;
		m_mapConstants = mapConstants;
		
		m_cg = m_data.getCodeGenerators ().getBackendAssemblyCodeGenerator ();
		
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
	public InstructionList generate (Expression expr, CodeGeneratorRuntimeOptions options)
	{		
		// generate the inline assembly code
		int nUnrollFactor = options.getIntValue (IBackendAssemblyCodeGenerator.OPTION_ASSEMBLY_UNROLLFACTOR);
		
		m_allocator.countRegistersNeeded (expr);

		InstructionList il = new InstructionList ();
		traverse (expr, null, nUnrollFactor, il);
		
		return il;
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
		if (RegisterAllocator.isAddSubSubtree (expr))
			return processAddSubSubtree (expr, specDatatype, nUnrollFactor, il);
		if (expr instanceof BinaryExpression)
			return processBinaryExpression ((BinaryExpression) expr, specDatatype, nUnrollFactor, il);
		if (expr instanceof FunctionCall)
			return processFunctionCall ((FunctionCall) expr, specDatatype, nUnrollFactor, il);
		if (expr instanceof UnaryExpression)
			return processUnaryExpression ((UnaryExpression) expr, specDatatype, nUnrollFactor, il);
		
		if (expr instanceof StencilNode)
			return processStencilNode ((StencilNode) expr, nUnrollFactor);
		if (expr instanceof Literal)
		{
			// find the register the constant is saved in or load the constant into a register (if too many registers)
			
		}
		
		return null;
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
		RegisterAllocator.linearizeAddSubSubtree (expr, list, BinaryOperator.ADD);
		
		IOperand.IRegisterOperand regSum = m_assemblySection.getFreeRegister (TypeRegisterType.SIMD);
		
		boolean bIsFirst = true;
		for (AddSub addsub : list)
		{
			IOperand[] rgOpSummand = traverse (addsub.getExpression (), specDatatype, nUnrollFactor, il);
			
			if (bIsFirst)
			{
				/*
				m_assemblySection.addInstruction (new Instruction (), specDatatype);
				issueInstruction (
					IBackendAssemblyCodeGenerator.INSTR_MOV_FPR,
					new String[] { strSummand, regSum.getName () },
					specDatatype,
					sb
				);
				*/
			}
			else
			{
				/*
				issueInstruction (
					addsub.getInstruction (),
					new String[] { regSum.getName (), strSummand, regSum.getName () },
					specDatatype,
					sb
				);
				*/
			}
			
			bIsFirst = false;
		}
		
		return null;//regSum;
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
		IOperand[] rgOpResult = new IOperand[nUnrollFactor];

		// get the operands for the binary expression
		int nNumRegsLHS = m_allocator.getNumRegistersUsed (expr.getLHS ());
		int nNumRegsRHS = m_allocator.getNumRegistersUsed (expr.getRHS ());
		
		// Sethi-Ullman: traverse the left tree first if it requires more registers than the right
		IOperand[] rgOpLHS = null;
		if (nNumRegsLHS >= nNumRegsRHS)
			rgOpLHS = traverse (expr.getLHS (), specDatatype, nUnrollFactor, il);
		IOperand[] rgOpRHS = traverse (expr.getRHS (), specDatatype, nUnrollFactor, il);
		if (nNumRegsLHS < nNumRegsRHS)
			rgOpLHS = traverse (expr.getLHS (), specDatatype, nUnrollFactor, il);

		// get the intrinsic corresponding to the operator of the binary expression
		Intrinsic intrinsic = m_data.getArchitectureDescription ().getIntrinsic (expr.getOperator (), specDatatype);		
		Argument[] rgArgs = Arguments.parseArguments (intrinsic.getArguments ());
		
		// is there an output argument?
		boolean bHasOutput = Arguments.hasOutput (rgArgs);
		int nOutputArgNum = -1;
		if (bHasOutput)
			nOutputArgNum = Arguments.getOutput (rgArgs).getNumber ();
		
		// create the instructions and add them to the list
		for (int i = 0; i < nUnrollFactor; i++)
		{
			IOperand[] rgOperands = new IOperand[rgArgs.length];
			
			rgOperands[Arguments.getLHS (rgArgs).getNumber ()] = rgOpLHS[i];
			rgOperands[Arguments.getRHS (rgArgs).getNumber ()] = rgOpRHS[i];
			
			IOperand opResult = isReservedRegister (rgOpRHS[i]) ? m_assemblySection.getFreeRegister (TypeRegisterType.SIMD) : rgOpRHS[i];
			
			if (bHasOutput && rgArgs.length > 2)
			{
				// the instruction requires an output operand distinct from the LHS and RHS operands
				rgOperands[nOutputArgNum] = opResult;
			}
			
			il.addInstruction (new Instruction (intrinsic.getName (), rgOperands));			
			rgOpResult[i] = bHasOutput ? rgOperands[nOutputArgNum] : rgOpRHS[i];
		}
		
		return rgOpResult;
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

		if (exprFuncName.equals (Globals.FNX_FMA) || exprFuncName.equals (Globals.FNX_FMS))
			return processFusedMultiplyAddSub (i, specDatatype, nUnrollFactor, il);
		else
			throw new RuntimeException (StringUtil.concat ("The function '", exprFuncName.toString (), "' is currently not supported."));
	}
	
	private IOperand[] processFusedMultiplyAddSub (Intrinsic intrinsic, Specifier specDatatype, int nUnrollFactor, InstructionList il)
	{
		return null;
	}

	/**
	 * 
	 * @param expr
	 * @param specDatatype
	 * @param nUnrollFactor
	 * @param il
	 * @return
	 */
	private IOperand[] processUnaryExpression (UnaryExpression expr, Specifier specDatatype, int nUnrollFactor, InstructionList il)
	{
		IOperand[] rgOps = traverse (expr, specDatatype, nUnrollFactor, il);
		
		if (((UnaryExpression) expr).getOperator ().equals (UnaryOperator.MINUS))
		{
			Intrinsic intrinsic = m_data.getArchitectureDescription ().getIntrinsic (expr.getOperator (), specDatatype);
			if (intrinsic == null)
				throw new RuntimeException (StringUtil.concat ("Unary operator intrinsic not found for operator ", expr.getOperator ().toString ()));
			
			for (int i = 0; i < nUnrollFactor; i++)
				il.addInstruction (new Instruction (intrinsic.getName (), new IOperand[] { rgOps[i] }));
		}

		return rgOps;
	}
	
	/**
	 * 
	 * @param node
	 * @param nUnrollFactor
	 * @return
	 */
	private IOperand[] processStencilNode (StencilNode node, int nUnrollFactor)
	{
		IOperand[] rgOpResults = new IOperand[nUnrollFactor];
		boolean bIsReuse = true;

		// check whether the stencil node is saved in a reuse register
		for (int i = 0; i < nUnrollFactor; i++)
		{
			Index idx = new Index (node.getIndex ());
			idx.getSpaceIndex ()[0] += i;
			IOperand.IRegisterOperand op = m_mapReuseNodesToRegisters.get (idx);
			if (op == null)
			{
				bIsReuse = false;
				break;
			}
			
			rgOpResults[i] = op;
		}
		
		// load the value of the stencil node into a register
		if (!bIsReuse)
			for (int i = 0; i < nUnrollFactor; i++)
				rgOpResults[i] = m_assemblySection.getGrid (node, i);
		
		return rgOpResults;
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
		if (m_mapConstants.containsValue (opReg))
			return true;
		
		return false;
	}
}
