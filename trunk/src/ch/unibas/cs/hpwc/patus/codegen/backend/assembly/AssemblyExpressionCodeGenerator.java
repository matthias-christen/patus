package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.LinkedList;
import java.util.List;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.Literal;
import cetus.hir.NameID;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.codegen.backend.IBackendAssemblyCodeGenerator;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
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
		
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public AssemblyExpressionCodeGenerator (StencilAssemblySection as, CodeGeneratorSharedObjects data)
	{
		m_data = data;
		m_assemblySection = as;
		
		m_cg = m_data.getCodeGenerators ().getBackendAssemblyCodeGenerator ();
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
		if (isAddSubSubtree (expr))
			return processAddSubSubtree (expr, specDatatype, nUnrollFactor, il);
		if (expr instanceof BinaryExpression)
			return processBinaryExpression ((BinaryExpression) expr, specDatatype, nUnrollFactor, il);
		if (expr instanceof FunctionCall)
			return processFunctionCall ((FunctionCall) expr, specDatatype, nUnrollFactor, il);
		if (expr instanceof UnaryExpression)
			return processUnaryExpression ((UnaryExpression) expr, specDatatype, nUnrollFactor, il);
		
		if (expr instanceof StencilNode)
		{
			// load the value of the stencil node into a register
			IOperand[] rgOpResults = new IOperand[nUnrollFactor];
			for (int i = 0; i < nUnrollFactor; i++)
				rgOpResults[i] = m_assemblySection.getGrid ((StencilNode) expr, i);
			return rgOpResults;
		}
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
		linearizeAddSubSubtree (expr, list, BinaryOperator.ADD);
		
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
		IOperand[] rgOpLHS = traverse (expr.getLHS (), specDatatype, nUnrollFactor, il);
		IOperand[] rgOpRHS = traverse (expr.getRHS (), specDatatype, nUnrollFactor, il);

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
			if (bHasOutput && rgArgs.length > 2)
			{
				// the instruction requires an output operand distinct from the LHS and RHS operands
				
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
	 * Detects whether only additions and subtractions are done in the expression <code>expr</code>.
	 * @param expr The expression to check
	 * @return <code>true</code> iff only additions and subtractions are performed in <code>expr</code>
	 */
	private boolean isAddSubSubtree (Expression expr)
	{
		for (DepthFirstIterator it = new DepthFirstIterator (expr); it.hasNext (); )
		{
			Object oChild = it.next ();
			
			if (oChild instanceof BinaryExpression)
			{
				BinaryOperator op = ((BinaryExpression) oChild).getOperator ();
				if (!op.equals (BinaryOperator.ADD) && !op.equals (BinaryOperator.SUBTRACT))
					return false;
			}
			else if (oChild instanceof FunctionCall)
				return false;
		}
		
		return true;
	}
	
	/**
	 * Assuming that <code>expr</code> is a binary expression tree with only additions and
	 * subtractions, this method creates a flat list of (operator, expression) pairs
	 * equivalent to <code>expr</code>. 
	 * @param expr The expression to linearize
	 * @param list An empty list that will contain the linearized version on output
	 * @param op Set to {@link BinaryOperator#ADD} when calling the method
	 */
	private void linearizeAddSubSubtree (Expression expr, List<AddSub> list, BinaryOperator op)
	{
		if (expr instanceof BinaryExpression)
		{
			BinaryExpression bexpr = (BinaryExpression) expr;
			linearizeAddSubSubtree (bexpr.getLHS (), list, op);
			linearizeAddSubSubtree (bexpr.getRHS (), list,
				op.equals (BinaryOperator.SUBTRACT) ?
					(bexpr.getOperator ().equals (BinaryOperator.ADD) ? BinaryOperator.SUBTRACT : BinaryOperator.ADD) :
					bexpr.getOperator ()
			);
		}
		else
			list.add (new AddSub (op, expr));
	}
	
	
	public static void main (String[] args)
	{
		AssemblyExpressionCodeGenerator acg = new AssemblyExpressionCodeGenerator (null, null);
		
		List<AddSub> list = new LinkedList<AddSub> ();
		acg.linearizeAddSubSubtree (
			new BinaryExpression (
				new BinaryExpression (
					new BinaryExpression (new NameID ("a1"), BinaryOperator.ADD, new NameID ("a2")),
					BinaryOperator.SUBTRACT,
					new BinaryExpression (new NameID ("a3"), BinaryOperator.SUBTRACT, new NameID ("a4"))),
				BinaryOperator.SUBTRACT,
				new BinaryExpression (new NameID ("a5"), BinaryOperator.ADD, new NameID ("a6"))
			),
			list, BinaryOperator.ADD);
		
		System.out.println (list);
	}
}
