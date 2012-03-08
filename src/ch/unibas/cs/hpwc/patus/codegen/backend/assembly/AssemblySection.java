package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.SomeExpression;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import ch.unibas.cs.hpwc.patus.arch.TypeRegister;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.IRegisterOperand;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * 
 * @author Matthias-M. Christen
 */
public class AssemblySection
{
	///////////////////////////////////////////////////////////////////
	// Inner Types
	
	public static class AssemblySectionInput
	{
		private Object m_objKey;
		private IOperand.IRegisterOperand m_operand;
		private Expression m_exprValue;
		

		public AssemblySectionInput (Object objKey, IOperand.IRegisterOperand op, Expression exprValue)
		{
			m_objKey = objKey;
			m_operand = op;
			m_exprValue = exprValue;
		}

		public Object getKey ()
		{
			return m_objKey;
		}

		public IOperand.IRegisterOperand getOperand ()
		{
			return m_operand;
		}

		public Expression getValue ()
		{
			return m_exprValue;
		}
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables

	protected CodeGeneratorSharedObjects m_data;
	
	/**
	 * The list of instructions in the inline assembly section
	 */
	protected List<TypedInstruction> m_listInstructions;
	
	/**
	 * The set of registers which got clobbered during the inline assembly section
	 */
	protected Set<IOperand.Register> m_setClobberedRegisters;
	
	/**
	 * Data structure identifying which registers are currently in use
	 */
	protected Map<IOperand.Register, Boolean> m_mapRegisterUsage;
	
	/**
	 * The list of inputs to the assembly section
	 */
	protected List<AssemblySectionInput> m_listInputs;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation

	public AssemblySection (CodeGeneratorSharedObjects data)
	{
		m_data = data;
		
		m_listInstructions = new ArrayList<TypedInstruction> ();

		m_setClobberedRegisters = new HashSet<IOperand.Register> ();
		m_mapRegisterUsage = new HashMap<IOperand.Register, Boolean> ();
		
		m_listInputs = new ArrayList<AssemblySectionInput> ();
	}
	
	/**
	 * Adds an input to the assembly section.
	 * @param input The input key with which the corresponding generated operand can be retrieved
	 * @param exprValue The expression to assign to the operand (register) on entry into the assembly section
	 * @return The operand generated for the input
	 */
	public IOperand addInput (Object input, Expression exprValue)
	{
		IOperand.InputRef op = new IOperand.InputRef (m_listInputs.size ());
		m_listInputs.add (new AssemblySectionInput (input, op, exprValue));

		return op;
	}
	
	/**
	 * Retrieves the operand corresponding to the assembly section input <code>input</code>.
	 * @param input The input for which the retrieve the corresponding operand
	 * @return The operand corresponding to <code>input</code>
	 */
	public IOperand.IRegisterOperand getInput (Object input)
	{
		for (AssemblySectionInput asi : m_listInputs)
			if (asi.getKey ().equals (input))
				return asi.getOperand ();
		return null;
	}
	
	/**
	 * Adds one instruction to the assembly section.
	 * @param instruction The instruction to add
	 * @param specDatatype The data type used for the floating point instructions
	 */
	public void addInstruction (Instruction instruction, Specifier specDatatype)
	{
		m_listInstructions.add (new TypedInstruction (instruction, specDatatype));
	}
	
	/**
	 * Adds a list of instructions to the assembly section.
	 * @param instructions The instructions to add
	 * @param specDatatype The data type used for the floating point instructions
	 */
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
	
	/**
	 * Removes the register <code>register</code> from the list of currently used registers.
	 * @param register The register to remove from the list of currently used registers
	 */
	public void killRegister (IOperand.Register register)
	{
		m_mapRegisterUsage.put (register, false);
	}
	
	/**
	 * 
	 * @param options
	 * @return
	 */
	public Statement generate (CodeGeneratorRuntimeOptions options)
	{
		// create a C statement wrapping the inline assembly
		
		// create the string of instructions
		StringBuilder sbInstructions = new StringBuilder ();
		for (TypedInstruction instruction : m_listInstructions)
			instruction.issue (m_data.getArchitectureDescription (), sbInstructions);
		
		// create the inputs string
		StringBuilder sbInputs = new StringBuilder ();
		for (AssemblySectionInput asi : m_listInputs)
		{
			if (sbInputs.length () > 0)
				sbInputs.append (", ");
			sbInputs.append ("r(");
			sbInputs.append (asi.getValue ().toString ());
			sbInputs.append (")");
		}

		// create the clobbered registers string
		StringBuilder sbClobberedRegisters = new StringBuilder ();
		for (IOperand.Register reg : m_setClobberedRegisters)
		{
			if (sbClobberedRegisters.length () > 0)
				sbClobberedRegisters.append (", ");
			sbClobberedRegisters.append (reg.getBaseName ());
		}
		
		// build the IR object
		return new ExpressionStatement (new SomeExpression (
			StringUtil.concat (
				"__asm__ __volatile__ (\n",
				sbInstructions.toString (),
				":\n",
				": ", sbInputs.toString (), "\n",
				":", sbClobberedRegisters.toString (), "\n",
				")"
			), null)
		);
	}
	
	/**
	 * Returns the size of a floating point data type.
	 * @param specDatatype
	 * @return
	 */
	public static int getTypeSize (Specifier specDatatype)
	{
		if (specDatatype.equals (Specifier.FLOAT))
			return Float.SIZE / 8;
		if (specDatatype.equals (Specifier.DOUBLE))
			return Double.SIZE / 8;
		return 0;
	}
}
