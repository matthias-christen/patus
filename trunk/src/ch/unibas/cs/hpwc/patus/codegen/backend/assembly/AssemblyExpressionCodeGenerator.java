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

public class AssemblyExpressionCodeGenerator
{
	private CodeGeneratorSharedObjects m_data;
	private IBackendAssemblyCodeGenerator m_cg;
	
	private AssemblySection m_assemblySection;
	
	
	public AssemblyExpressionCodeGenerator (AssemblySection as, CodeGeneratorSharedObjects data)
	{
		m_data = data;
		m_assemblySection = as;
		
		m_cg = m_data.getCodeGenerators ().getBackendAssemblyCodeGenerator ();
	}
	
	public Statement generate (Expression expr, CodeGeneratorRuntimeOptions options)
	{
		// initialize
		//m_cg.startAsm ();
		
		// generate the inline assembly code
		int nUnrollFactor = options.getIntValue (IBackendAssemblyCodeGenerator.OPTION_ASSEMBLY_UNROLLFACTOR);
		StringBuilder sb = new StringBuilder ();
		traverse (expr, null, nUnrollFactor, sb);
		
		return m_cg.generate (options);
	}
	
	private IOperand traverse (Expression expr, Specifier specDatatype, int nUnrollFactor, StringBuilder sb)
	{
		if (isAddSubSubtree (expr))
			return processAddSubSubtree (expr, specDatatype, nUnrollFactor, sb);
		if (expr instanceof BinaryExpression)
			return processBinaryExpression ((BinaryExpression) expr, specDatatype, nUnrollFactor, sb);
		else if (expr instanceof FunctionCall)
			return processFunctionCall ((FunctionCall) expr, specDatatype, nUnrollFactor, sb);
		else if (expr instanceof UnaryExpression)
			return processUnaryExpression ((UnaryExpression) expr, specDatatype, nUnrollFactor, sb);
		
		else if (expr instanceof StencilNode)
		{
			
		}
		else if (expr instanceof Literal)
		{
			
		}
		
		return null;
	}
	
	/**
	 * Generates the code for a add-sub subtree.
	 * @param expr
	 * @param specDatatype
	 * @param nUnrollFactor
	 * @param sb
	 * @return
	 */
	private IOperand processAddSubSubtree (Expression expr, Specifier specDatatype, int nUnrollFactor, StringBuilder sb)
	{
		List<AddSub> list = new LinkedList<AddSub> ();
		linearizeAddSubSubtree (expr, list, BinaryOperator.ADD);
		
		IOperand.IRegisterOperand regSum = m_assemblySection.getFreeRegister (TypeRegisterType.SIMD);
		
		boolean bIsFirst = true;
		for (AddSub addsub : list)
		{
			IOperand opSummand = traverse (addsub.getExpression (), specDatatype, nUnrollFactor, sb);
			
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
		
		return regSum;		
	}

	/**
	 * 
	 * @param expr
	 * @param specDatatype
	 * @param nUnrollFactor
	 * @param sb
	 * @return
	 */
	private IOperand processBinaryExpression (BinaryExpression expr, Specifier specDatatype, int nUnrollFactor, StringBuilder sb)
	{
		Intrinsic i = m_data.getArchitectureDescription ().getIntrinsic (expr.getOperator (), specDatatype);
		i.getArguments ();

		/*
		issueInstruction (
			i.getName (),
			new String[] {},
			specDatatype,
			sb
		);
		*/
		
		return null;
	}
	
	/**
	 * 
	 * @param fnxCall
	 * @param specDatatype
	 * @param nUnrollFactor
	 * @param sb
	 * @return
	 */
	private IOperand processFunctionCall (FunctionCall fnxCall, Specifier specDatatype, int nUnrollFactor, StringBuilder sb)
	{
		Expression exprFuncName = fnxCall.getName ();
		Intrinsic i = m_data.getArchitectureDescription ().getIntrinsic (exprFuncName.toString (), specDatatype);
		i.getArguments ();

		if (exprFuncName.equals (Globals.FNX_FMA))
		{
			
		}
		else if (exprFuncName.equals (Globals.FNX_FMS))
		{
			
		}

		/*
		issueInstruction (
			i.getName (),
			new String[] {},
			specDatatype,
			sb
		);
		*/
		
		return null;
	}

	/**
	 * 
	 * @param expr
	 * @param specDatatype
	 * @param nUnrollFactor
	 * @param sb
	 * @return
	 */
	private IOperand processUnaryExpression (UnaryExpression expr, Specifier specDatatype, int nUnrollFactor, StringBuilder sb)
	{
		if (((UnaryExpression) expr).getOperator ().equals (UnaryOperator.MINUS))
		{
			
		}

		return null;
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
