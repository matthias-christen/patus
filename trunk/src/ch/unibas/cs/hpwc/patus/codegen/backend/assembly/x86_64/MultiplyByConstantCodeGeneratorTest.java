package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.x86_64;

import java.util.HashMap;
import java.util.Map;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import ch.unibas.cs.hpwc.patus.arch.TypeRegister;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IInstruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class MultiplyByConstantCodeGeneratorTest
{
	private final static String REG1 = "eax";	
	private final static String REG2 = "ebx";
	
	private IOperand.Register m_reg1;
	private IOperand.Register m_reg2;
	
	private MultiplyByConstantCodeGenerator m_cg1;
	private MultiplyByConstantCodeGenerator m_cg2;
	
	
	@Before
	public void setUp () throws Exception
	{
		TypeRegister rt1 = new TypeRegister ();
		TypeRegister rt2 = new TypeRegister ();
		rt1.setName (REG1);
		rt2.setName (REG2);
		
		m_reg1 = new IOperand.Register (rt1);
		m_reg2 = new IOperand.Register (rt2);		
		
		m_cg1 = new MultiplyByConstantCodeGenerator (m_reg1, m_reg2);
		m_cg2 = new MultiplyByConstantCodeGenerator (m_reg1, new IOperand.Address (m_reg2));
	}
	
	private void verifyComputation (InstructionList il, int nNum)
	{
		Map<IOperand.Register, Integer> mem = new HashMap<> ();
		mem.put (m_reg1, 1);
		mem.put (m_reg2, 1);
		
		// interpret instructions
		for (IInstruction i : il)
		{
			if (i instanceof Instruction)
			{
				Instruction instr = (Instruction) i;
				
				String strInstr = instr.getInstructionName ();
				IOperand.Register regDest = (IOperand.Register) instr.getOperands ()[instr.getOperands ().length - 1];
				
				switch (strInstr)
				{
				case "shl":
					mem.put (regDest, getValue (regDest, mem) << getValue (instr.getOperands ()[0], mem));
					break;
					
				case "add":
					mem.put (regDest, getValue (regDest, mem) + getValue (instr.getOperands ()[0], mem));
					break;
					
				case "sub":
					mem.put (regDest, getValue (regDest, mem) - getValue (instr.getOperands ()[0], mem));
					break;
					
				case "lea":
					IOperand.Address addr = (IOperand.Address) instr.getOperands ()[0];
					mem.put (regDest, getValue (addr.getRegBase (), mem) + getValue (addr.getRegIndex (), mem) * addr.getScale () + (int) addr.getDisplacement ());
					break;
					
				case "imul":
					mem.put (regDest, getValue (instr.getOperands ()[0], mem) * getValue (regDest, mem));
					break;
					
				case "mov":
					mem.put (regDest, getValue (instr.getOperands ()[0], mem));
					break;
					
				default:
					Assert.fail (StringUtil.concat ("Instruction ", strInstr, " not supported"));
				}
			}
		}
		
		Assert.assertEquals (nNum, (int) mem.get (m_reg1));
	}
	
	private static int getValue (IOperand op, Map<IOperand.Register, Integer> mem)
	{
		if (op instanceof IOperand.Immediate)
			return (int) ((IOperand.Immediate) op).getValue ();
		if (op instanceof IOperand.Register)
			return mem.get (op);
		if (op instanceof IOperand.Address)
			return mem.get (((IOperand.Address) op).getRegBase ());	// we assume that the base register points to the value
		
		Assert.fail (StringUtil.concat ("Value not found for ", op.toString ()));
		return 0;
	}

	@Test
	public void testGenerate ()
	{
		for (int i = 4; i < 1000; i++)
		{
			InstructionList il = new InstructionList ();
			m_cg1.generate (il, i, true);
			verifyComputation (il, i);

			System.out.println (i + " (allow temp reg):");
			System.out.println (il.toString ());

			//*
			il = new InstructionList ();
			m_cg2.generate (il, i, false);
			verifyComputation (il, i);
			
			System.out.println (i + " (no temp reg):");
			System.out.println (il.toString ());
			//*/
		}
	}
}
