package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.x86_64;

import java.util.HashMap;
import java.util.Map;

import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * 
 * http://1-800-magic.blogspot.ch/2007/12/integer-multiplication-and-dynamic.html
 * 
 * @author Matthias-M. Christen
 *
 */
public class MultiplyByConstantCodeGenerator
{
	private enum EOpType
	{
		/**
		 * Uninitialized type
		 */
		NONE (false),
		
		/**
		 * Left shift operation (multiply by 2^k) 
		 */
		SHIFT (false),
		
		/**
		 * Use &quot;lea&quot; to multiply by (2^k+1)
		 */
		MULTIPLY (false),
		
		/**
		 * Adds or subtracts one from the constant factor (by adding or
		 * subtracting the value (saved in the temporary register
		 * <code>m_reg2</code> from the result), or uses &quot;lea&quot; to add
		 * another value to the constant factor via the temporary register
		 */
		ADDCONST (true),
		
		/**
		 * Increases the temporary register by a multiple of the factor by using
		 * &quot;lea&quot;
		 */
		INCRMUL (true);
		
		
		private boolean m_bNeedsTempReg;
		
		private EOpType (boolean bNeedsTempReg)
		{
			m_bNeedsTempReg = bNeedsTempReg;
		}
		
		public boolean needsTempRegister ()
		{
			return m_bNeedsTempReg;
		}
	}
	
	private class Operation
	{
		private EOpType m_type;
		private int m_nArgument;
		
		public Operation (EOpType type, int nArgument)
		{
			m_type = type;
			m_nArgument = nArgument;
		}

		public EOpType getType ()
		{
			return m_type;
		}

		public int compute (int nNumber)
		{
			switch (m_type)
			{
			case SHIFT:
				return nNumber << m_nArgument;
			case MULTIPLY:
				return nNumber * (m_nArgument + 1);
			case ADDCONST:
				return nNumber + m_nArgument;
			case INCRMUL:
				return nNumber * m_nArgument + 1;
			}
			
			return 0;
		}
		
		public int computeInverse (int nNumber)
		{
			int nVal = 0;
			
			switch (m_type)
			{
			case SHIFT:
				nVal = nNumber >> m_nArgument;
				break;
			case MULTIPLY:
				nVal = nNumber / (m_nArgument + 1);
				break;
			case ADDCONST:
				nVal = nNumber - m_nArgument;
				break;
			case INCRMUL:
				nVal = (nNumber - 1) / m_nArgument;
				break;
			}
			
			return (nVal > 0 && compute (nVal) == nNumber) ? nVal : 0;
		}
		
		public Operation getNextOperation ()
		{
			if (m_type.equals (EOpType.NONE))
				return new Operation (EOpType.SHIFT, 0);
			if (m_type.equals (EOpType.SHIFT))
				return new Operation (EOpType.MULTIPLY, 0);
			if (canAddConst () && m_type.equals (EOpType.MULTIPLY))
				return new Operation (EOpType.ADDCONST, 0);
			if (canIncrMul () && m_type.equals (EOpType.ADDCONST))
				return new Operation (EOpType.INCRMUL, 0);
			return null;
		}
		
		private boolean canAddConst ()
		{
			return m_reg2 != null;
		}
		
		private boolean canIncrMul ()
		{
			if (m_reg2 == null)
				return false;
			return m_reg2 instanceof IOperand.Register;
		}
		
		public Operation getNextArgument ()
		{
			switch (m_type)
			{
			case SHIFT:
				return new Operation (m_type, m_nArgument + 1);
			case MULTIPLY:
				return getNextInSequence (0, 2, 4, 8);
			case ADDCONST:
				if (m_reg2 instanceof IOperand.Register)
					return getNextInSequence (0, -1, 1, 2, 4, 8);
				else if (m_reg2 instanceof IOperand.Address)
					return getNextInSequence (0, -1, 1);
				return null;
			case INCRMUL:
				return getNextInSequence (0, 2, 4, 8);
			}
			
			return null;
		}
		
		private Operation getNextInSequence (int... rgSequence)
		{
			for (int i = 0; i < rgSequence.length; i++)
				if (m_nArgument == rgSequence[i])
					return i < rgSequence.length - 1 ? new Operation (m_type, rgSequence[i + 1]) : null;
			return null;
		}
		
		public boolean needsTempRegister ()
		{
			return m_type.needsTempRegister ();
		}
		
		public void generate (InstructionList il)
		{
			switch (m_type)
			{
			case SHIFT:
				il.addInstruction (new Instruction ("shl", new IOperand.Immediate (m_nArgument), m_reg1));
				break;
			case MULTIPLY:
				il.addInstruction (new Instruction ("lea", new IOperand.Address (m_reg1, m_reg1, m_nArgument), m_reg1));
				break;
			case ADDCONST:
				if (m_nArgument == 1)
					il.addInstruction (new Instruction ("add", m_reg2, m_reg1));
				else if (m_nArgument == -1)
					il.addInstruction (new Instruction ("sub", m_reg2, m_reg1));
				else
					// if we get here, m_reg2 is a register
					il.addInstruction (new Instruction ("lea", new IOperand.Address (m_reg1, (IOperand.IRegisterOperand) m_reg2, m_nArgument), m_reg1));
				break;
			case INCRMUL:
				// if we get here, m_reg2 is a register
				il.addInstruction (new Instruction ("lea", new IOperand.Address ((IOperand.IRegisterOperand) m_reg2, m_reg1, m_nArgument), m_reg1));
				break;
			}
		}
		
		@Override
		public String toString ()
		{
			return StringUtil.concat ("<", m_type.toString (), ", ", m_nArgument, ">");
		}
	}
	
	private static class Decomposition
	{
		private int m_nCost;
		private Operation m_operation;
		
		private Decomposition m_next;

		
		public Decomposition (int nNumber, Operation op, int nCost, Decomposition decompNext)
		{
			m_operation = op;
			m_nCost = nCost;
			m_next = decompNext;
		}

		public int getCost ()
		{
			return m_nCost;
		}

		public Operation getOperation ()
		{
			return m_operation;
		}
		
		public boolean needsTempRegister ()
		{
			if (m_operation.needsTempRegister ())
				return true;
			if (m_next != null)
				return m_next.needsTempRegister ();
			return false;
		}

		public void generate (InstructionList il)
		{
			if (m_next != null)
				m_next.generate (il);
			m_operation.generate (il);
		}
		
		@Override
		public String toString ()
		{
			StringBuilder sb = new StringBuilder ("Decomp{ ");
			sb.append (m_operation == null ? "(no op)" : m_operation.toString ());
			sb.append (", $");
			sb.append (m_nCost);
			sb.append (" }");
			
			if (m_next != null)
			{
				sb.append (" -> ");
				sb.append (m_next.toString ());
			}
			
			return sb.toString ();
		}
	}


	private static final int MAX_RECURSION_DEPTH = 5;
	
	
	private IOperand.IRegisterOperand m_reg1;
	private IOperand m_reg2;
	
	private Map<Integer, Decomposition> m_mapCache;
	
	
	/**
	 * Constructs the code generator. The constructor is provided the registers
	 * to work with.
	 * 
	 * @param reg1
	 *            The register that is to be multiplied by a constant integer.
	 * @param reg2
	 *            A temporary register that may be used for temporary
	 *            computations. Can be <code>null</code>, in which case only
	 *            multiplication code is generated that can be expressed using
	 *            one register
	 */
	public MultiplyByConstantCodeGenerator (IOperand.IRegisterOperand reg1, IOperand reg2)
	{
		m_reg1 = reg1;
		m_reg2 = reg2;
		
		m_mapCache = new HashMap<> ();
	}
	
	public MultiplyByConstantCodeGenerator (IOperand.IRegisterOperand reg)
	{
		this (reg, null);
	}
	
	/**
	 * 
	 * @param il
	 * @param nFactor
	 */
	public Decomposition generate (InstructionList il, int nFactor, boolean bGenerateTemporaryMove)
	{
		Decomposition d = null;
		
		int nAbsFactor = Math.abs (nFactor);
		if (nAbsFactor == 1)
			;	// nothing to do
		else
		{
			d = decompose (nFactor);
			if (d != null)
			{
				if (bGenerateTemporaryMove && d.needsTempRegister ())
					il.addInstruction (new Instruction ("mov", m_reg1, m_reg2));

				d.generate (il);
			}
			else
			{
				// not able to decompose; use "imul" instruction instead
				il.addInstruction (new Instruction ("imul", new IOperand.Immediate (nFactor), m_reg1));
			}
		}
		
		if (nFactor < 0)
			il.addInstruction (new Instruction ("neg", m_reg1));
		
		return d;
	}
	
	/**
	 * 
	 * @param nNumber
	 * @param bHasTempReg
	 * @param nLevel
	 * @return
	 */
	private Decomposition decompose (int nNumber, int nLevel)
	{
		Decomposition d = m_mapCache.get (nNumber);
		if (d != null)
			return d;
		
		if (nLevel > MAX_RECURSION_DEPTH)
			return null;
		
		Operation opTmp = new Operation (EOpType.NONE, 0);
		Decomposition decompTmp = new Decomposition (nNumber, opTmp, 0, null);
		
		for (boolean bOptimalFound = false; !bOptimalFound; )
		{
			Operation o = opTmp.getNextOperation ();
			if (o == null)
				break;
			opTmp = o;
			
			for ( ; ; )
			{
				o = opTmp.getNextArgument ();
				if (o == null)
					break;
				opTmp = o;
				
				int nNextVal = opTmp.computeInverse (nNumber);
				if (nNextVal == 0)
				{
					if (opTmp.getType ().equals (EOpType.SHIFT))
						break;
					continue;
				}
				
				if (nNextVal == 1)
				{
					// we're done
					decompTmp = new Decomposition (nNumber, opTmp, 1, null);
					bOptimalFound = true;
					break;
				}
				
				Decomposition decompNext = decompose (nNextVal, nLevel + 1);
				if (decompNext == null)
					continue;
				
				if (decompTmp.getOperation ().getType ().equals (EOpType.NONE) || decompNext.getCost () + 1 < decompTmp.getCost ())
					decompTmp = new Decomposition (nNumber, opTmp, decompNext.getCost () + 1, decompNext);
			}
		}
		
		if (decompTmp.getOperation ().getType ().equals (EOpType.NONE))
			return null;

		// check cache again; maybe meanwhile a better solution has been found
		d = m_mapCache.get (nNumber);
		if (d == null || d.getCost () > decompTmp.getCost ())
		{
			m_mapCache.put (nNumber, decompTmp);
			return decompTmp;
		}
		
		return d;
	}
	
	private Decomposition decompose (int nNumber)
	{
		if (nNumber <= 1)
			return null;		
		return decompose (nNumber, 0);
	}
}
