package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cetus.hir.ExpressionStatement;
import cetus.hir.SomeExpression;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeRegister;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.IRegisterOperand;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

public class AssemblySection
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private CodeGeneratorSharedObjects m_data;
	
	/**
	 * The list of instructions in the inline assembly section
	 */
	private List<TypedInstruction> m_listInstructions;
	
	/**
	 * The set of registers which got clobbered during the inline assembly section
	 */
	private Set<IOperand.Register> m_setClobberedRegisters;
	
	/**
	 * Data structure identifying which registers are currently in use
	 */
	private Map<IOperand.Register, Boolean> m_mapRegisterUsage;
	
	private Map<String, IOperand> m_mapInputs;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public AssemblySection (CodeGeneratorSharedObjects data)
	{
		m_data = data;
		
		m_listInstructions = new ArrayList<TypedInstruction> ();

		m_setClobberedRegisters = new HashSet<IOperand.Register> ();
		m_mapRegisterUsage = new HashMap<IOperand.Register, Boolean> ();
		m_mapInputs = new HashMap<String, IOperand> ();
	}
	
	public void addInput (String strInputName)
	{
		if (m_mapInputs.containsKey (strInputName))
			return;
		
		m_mapInputs.put (strInputName, new IOperand.InputRef (m_mapInputs.size ()));
	}
	
	public IOperand getInput (String strInputName)
	{
		return m_mapInputs.get (strInputName);
	}
	
	public void addInstruction (Instruction instruction, Specifier specDatatype)
	{
		m_listInstructions.add (new TypedInstruction (instruction, specDatatype));
	}
	
	public void addInstructions (InstructionList instructions, Specifier specDatatype)
	{
		for (IInstruction i : instructions)
			m_listInstructions.add (new TypedInstruction (i, specDatatype));
	}

	/**
	 * Returns the next free register of type <code>regtype</code>.
	 * @param regtype The desired type of the register
	 * @return The next free register
	 */
	public IRegisterOperand getFreeRegister (TypeRegisterType regtype)
	{
		for (TypeRegister reg : m_data.getArchitectureDescription ().getAssemblySpec ().getRegisters ().getRegister ())
		{
			if (!reg.getType ().equals (regtype))
				continue;
			
			IOperand.Register register = new IOperand.Register (reg);
			Boolean bIsRegUsed = m_mapRegisterUsage.get (register);
			if (bIsRegUsed == null || bIsRegUsed == false)
			{
				m_mapRegisterUsage.put (register, true);
				m_setClobberedRegisters.add (register);
				return register;
			}
		}
		
		// no free registers
		return null;
	}
	
	public void killRegister (IOperand.Register register)
	{
		m_mapRegisterUsage.put (register, false);
	}

	public Statement generate (IArchitectureDescription arch, CodeGeneratorRuntimeOptions options)
	{
		// create a C statement wrapping the inline assembly
		
		StringBuilder sbInstructions = new StringBuilder ();
		for (TypedInstruction instruction : m_listInstructions)
			instruction.issue (arch, sbInstructions);

		StringBuilder sbClobberedRegisters = new StringBuilder ();
		for (IOperand.Register reg : m_setClobberedRegisters)
		{
			if (sbClobberedRegisters.length () > 0)
				sbClobberedRegisters.append (",");
			sbClobberedRegisters.append (reg.getBaseName ());
		}
				
		return new ExpressionStatement (new SomeExpression (
			StringUtil.concat (
				"__asm__ __volatile__ (\n",
				sbInstructions.toString (),
				":\n",
				":\n",
				":", sbClobberedRegisters.toString (), "\n",
				")"
			), null)
		);
	}
}
