package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import cetus.hir.Specifier;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.arch.TypeBaseIntrinsicEnum;
import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * This class encapsulates a single inline assembly instruction (mnemonic + operands).
 * 
 * @author Matthias-M. Christen
 */
public class Instruction implements IInstruction
{
	private String m_strIntrinsicBaseName;
	private IOperand[] m_rgOperands;
	
	public Instruction (String strIntrinsicBaseName, IOperand... rgOperands)
	{
		m_strIntrinsicBaseName = strIntrinsicBaseName;
		m_rgOperands = rgOperands;
	}
	
	public Instruction (TypeBaseIntrinsicEnum t, IOperand... rgOperands)
	{
		this (t.value (), rgOperands);
	}
	
	/**
	 * 
	 * @return
	 */
	public String getIntrinsicBaseName ()
	{
		return m_strIntrinsicBaseName;
	}

	/**
	 * 
	 * @return
	 */
	public IOperand[] getOperands ()
	{
		return m_rgOperands;
	}
	
	/**
	 * 
	 * @param rgArgs
	 * @param strArgName
	 * @return
	 */
	private int getArgNum (Argument[] rgArgs, String strArgName)
	{
		if (Globals.ARGNAME_LHS.equals (strArgName))
			return Arguments.getLHS (rgArgs).getNumber ();
		if (Globals.ARGNAME_RHS.equals (strArgName))
			return Arguments.getRHS (rgArgs).getNumber ();
		return Arguments.getNamedArgument (rgArgs, strArgName).getNumber ();
	}
	
	/**
	 * 
	 * @param intrinsic
	 * @return
	 */
	private IOperand[] getOperandsForIntrinsic (Intrinsic intrinsic)
	{
		if (intrinsic == null)
			return m_rgOperands;
		
		// get the intrinsic corresponding to the operator of the binary expression
		Argument[] rgArgs = Arguments.parseArguments (intrinsic.getArguments ());
		
		// is there an output argument?
		boolean bHasOutput = Arguments.hasOutput (rgArgs);
		
		IOperand[] rgRealOperands = new IOperand[rgArgs.length];
		
		TypeBaseIntrinsicEnum type = Globals.getIntrinsicBase (intrinsic.getBaseName ());
		if (type == null)
			throw new RuntimeException (StringUtil.concat ("No TypeBaseIntrinsicEnum for ", intrinsic.getBaseName ()));

		// convert the base intrinsic arguments to the arguments of the actual intrinsic as defined in the
		// architecture description
		String[] rgIntrinsicArgNames = Globals.getIntrinsicArguments (type);
		for (int i = 0; i < m_rgOperands.length; i++)
			rgRealOperands[getArgNum (rgArgs, rgIntrinsicArgNames[i])] = m_rgOperands[i];
			
		IOperand opResult = new IOperand.PseudoRegister (); //isReservedRegister (rgOpRHS[i]) ? new IOperand.PseudoRegister () : rgOpRHS[i];
			
		if (bHasOutput && rgArgs.length > m_rgOperands.length)
		{
			// the instruction requires an output operand distinct from the input operands
			int nOutputArgNum = Arguments.getOutput (rgArgs).getNumber ();
			if (nOutputArgNum != -1)
				rgRealOperands[nOutputArgNum] = opResult;
		}
		
		return rgRealOperands;
	}

	public void issue (Specifier specDatatype, IArchitectureDescription arch, StringBuilder sbResult)
	{
		// TODO: !!!!!
		// TODO: Requires additional instruction in case the instruction has no distinct output register!?
		
		
		// try to find the intrinsic corresponding to m_strIntrinsicBaseName
		Intrinsic intrinsic = arch.getIntrinsic (m_strIntrinsicBaseName, specDatatype);
		
		// if the base name doesn't correspond to an intrinsic defined in the architecture description,
		// use m_strIntrinsicBaseName as instruction mnemonic
		String strInstruction = intrinsic == null ? m_strIntrinsicBaseName : intrinsic.getName ();
		//boolean bIsVectorInstruction = arch.getSIMDVectorLength (specDatatype) > 1;

		sbResult.append (strInstruction);
		sbResult.append (" ");
		
		boolean bFirst = true;
		for (IOperand op : getOperandsForIntrinsic (intrinsic))
		{
			if (!bFirst)
				sbResult.append (", ");
			sbResult.append (op.toString ());
			bFirst = false;
		}
		
		sbResult.append ("\\n\\t");
	}
	
	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder (m_strIntrinsicBaseName);
		sb.append (' ');
		
		boolean bFirst = true;
		for (IOperand op : m_rgOperands)
		{
			if (!bFirst)
				sb.append (", ");
			
			sb.append (op.toString ());
			bFirst = false;
		}
		
		return sb.toString ();
	}
}
