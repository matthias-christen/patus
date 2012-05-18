package ch.unibas.cs.hpwc.patus.codegen.backend.assembly.test;

import java.math.BigInteger;

import org.junit.Before;
import org.junit.Test;

import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.arch.TypeArchitectureType.Intrinsics.Intrinsic;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.Address;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.InputRef;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.PseudoRegister;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.Instruction;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionList;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionRegionScheduler;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.InstructionScheduler;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DAGraph;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.analyze.DependenceAnalysis;

public class InstructionSchedulerTest
{
	private static class ArchDesc extends DummyArchitectureDescription
	{
		@Override
		public int getIssueRate ()
		{
			return 4;
		}
		
		@Override
		public Intrinsic getIntrinsicByIntrinsicName (String strIntrinsicName)
		{
			Intrinsic intrinsic = new Intrinsic ();
			
			intrinsic.setBaseName (strIntrinsicName);
			intrinsic.setName (strIntrinsicName);
			intrinsic.setDatatype ("float");
			
			if (strIntrinsicName.indexOf ("mov") >= 0)
				intrinsic.setLatency (new BigInteger ("1"));
			else if (strIntrinsicName.indexOf ("add") >= 0 || strIntrinsicName.indexOf ("sub") >= 0)
				intrinsic.setLatency (new BigInteger ("3"));
			else if (strIntrinsicName.indexOf ("mul") >= 0)
				intrinsic.setLatency (new BigInteger ("5"));
			else if (strIntrinsicName.indexOf ("div") >= 0)
				intrinsic.setLatency (new BigInteger ("29"));
			else
				intrinsic.setLatency (new BigInteger ("1"));
			
			return intrinsic;
		}
	}
	
	
	private InstructionList m_il;
	
	@Before
	public void setUp () throws Exception
	{
		m_il = new InstructionList ();
		InstructionList il = m_il;
		
		Address op0 = new Address (new InputRef ("_constants_"), null, 1, 32);
		PseudoRegister op1 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op0, op1));
		Address op2 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 4);
		PseudoRegister op3 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op2, op3));
		Address op4 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, -4);
		PseudoRegister op5 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op4, op3, op5));
		Address op6 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op6, op5, op5));
		Address op7 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op7, op5, op5));
		Address op8 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op8, op5, op5));
		Address op9 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 1, 0);
		il.addInstruction (new Instruction ("vaddps", op9, op5, op5));
		il.addInstruction (new Instruction ("vmulps", op1, op5, op1));
		Address op10 = new Address (new InputRef ("_constants_"), null, 1, 64);
		PseudoRegister op11 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op10, op11));
		Address op12 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 8);
		PseudoRegister op13 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovups", op12, op13));
		Address op14 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, -8);
		PseudoRegister op15 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vaddps", op14, op13, op15));
		Address op16 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("(x_max*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op16, op15, op15));
		Address op17 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*x_max)*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op17, op15, op15));
		Address op18 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((x_max*y_max)*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op18, op15, op15));
		Address op19 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), new InputRef ("((-1*(x_max*y_max))*sizeof (float))"), 2, 0);
		il.addInstruction (new Instruction ("vaddps", op19, op15, op15));
		il.addInstruction (new Instruction ("vmulps", op11, op15, op11));
		il.addInstruction (new Instruction ("vaddps", op1, op11, op1));
		Address op20 = new Address (new InputRef ("_constants_"), null, 1, 0);
		PseudoRegister op21 = new PseudoRegister (TypeRegisterType.SIMD);
		il.addInstruction (new Instruction ("vmovaps", op20, op21));
		Address op22 = new Address (new InputRef ("u[t=0^, s=(0, 0, 0)][0]"), null, 1, 0);
		il.addInstruction (new Instruction ("vmulps", op22, op21, op21));
		Address op23 = new Address (new InputRef ("u[t=-1^, s=(0, 0, 0)][0]"), null, 1, 0);
		il.addInstruction (new Instruction ("vsubps", op23, op21, op21));
		il.addInstruction (new Instruction ("vaddps", op21, op1, op21));
		Address op24 = new Address (new InputRef ("u[t=1^, s=(0, 0, 0)][0]"), null, 1, 0);
		il.addInstruction (new Instruction ("vmovaps", op21, op24));
	}

	@Test
	public void testSchedule ()
	{
		InstructionRegionScheduler.DEBUG = true;
		IArchitectureDescription arch = new ArchDesc ();
		DAGraph graph = new DependenceAnalysis (m_il, arch).run ();
		graph.graphviz ();
		InstructionScheduler is = new InstructionScheduler (graph, arch);
		System.out.println (is.schedule ());
	}
}
