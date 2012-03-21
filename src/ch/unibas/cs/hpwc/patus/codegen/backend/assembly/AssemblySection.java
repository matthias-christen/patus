package ch.unibas.cs.hpwc.patus.codegen.backend.assembly;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.SomeExpression;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import ch.unibas.cs.hpwc.patus.arch.TypeRegister;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterClass;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.IRegisterOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.Register;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize.IInstructionListOptimizer;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * 
 * @author Matthias-M. Christen
 */
public class AssemblySection
{
	// /////////////////////////////////////////////////////////////////
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


	// /////////////////////////////////////////////////////////////////
	// Member Variables

	protected CodeGeneratorSharedObjects m_data;

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


	// /////////////////////////////////////////////////////////////////
	// Implementation

	public AssemblySection (CodeGeneratorSharedObjects data)
	{
		m_data = data;

		m_setClobberedRegisters = new TreeSet<> (new Comparator<IOperand.Register> ()
		{
			@Override
			public int compare (Register r1, Register r2)
			{
				return r1.getBaseName ().compareTo (r2.getBaseName ());
			}
		});

		m_mapRegisterUsage = new HashMap<> ();

		m_listInputs = new ArrayList<> ();

		Label.reset ();
	}

	/**
	 * Adds an input to the assembly section.
	 * 
	 * @param input
	 *            The input key with which the corresponding generated operand can be retrieved
	 * @param exprValue
	 *            The expression to assign to the operand (register) on entry into the assembly section
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
	 * 
	 * @param input
	 *            The input for which the retrieve the corresponding operand
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
	 * 
	 * @param ilInstructions
	 * @param specDatatype
	 * @return
	 */
	public InstructionList translate (InstructionList ilInstructions, Specifier specDatatype, IInstructionListOptimizer... rgOptimizers)
	{
		// translate the generic instruction list to the architecture-specific one
		InstructionList ilTmp = InstructionListTranslator.translate (
			m_data.getArchitectureDescription (), ilInstructions, specDatatype);

		// apply peep hole optimizations
		for (IInstructionListOptimizer optimizer : rgOptimizers)
			ilTmp = optimizer.optimize (ilTmp);

		// allocate registers
		return ilTmp.allocateRegisters (this);
	}

	/**
	 * Returns the next free register of type <code>regtype</code>.
	 * 
	 * @param regtype
	 *            The desired type of the register
	 * @return The next free register
	 */
	public IRegisterOperand getFreeRegister (TypeRegisterType regtype)
	{
		for (TypeRegister reg : m_data.getArchitectureDescription ().getAssemblySpec ().getRegisters ().getRegister ())
		{
			if (!((TypeRegisterClass) reg.getClazz ()).getType ().equals (regtype))
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
	 * 
	 * @param register
	 *            The register to remove from the list of currently used registers
	 */
	public void killRegister (IOperand.Register register)
	{
		m_mapRegisterUsage.put (register, false);
	}
	
	/**
	 * Returns the list of inputs as a string.
	 * @return The list of inputs as a string
	 */
	private String getInputsAsString ()
	{
		// create the inputs string
		StringBuilder sbInputs = new StringBuilder ();
		
		// get the GPR register class
		// TODO: check whether in all cases the widest register should be used
		Iterator<TypeRegisterClass> it = m_data.getArchitectureDescription ().getRegisterClasses (TypeRegisterType.GPR).iterator ();
		TypeRegisterClass cls = null;
		if (it.hasNext ())
			cls = it.next ();
		
		for (AssemblySectionInput asi : m_listInputs)
		{
			if (sbInputs.length () > 0)
				sbInputs.append (", ");
			sbInputs.append ("\"r\"(");
			
			// cast to data type of the register
			if (cls != null)
			{
				sbInputs.append ('(');
				sbInputs.append (cls.getDatatype ());
				sbInputs.append (')');
			}
			
			sbInputs.append (asi.getValue ().toString ());
			sbInputs.append (")");
		}

		return sbInputs.toString ();
	}
	
	/**
	 * Returns the list of clobbered registers as a string.
	 * @return The list of clobbered registers as a string
	 */
	private String getClobberedRegistersAsString ()
	{
		// create the clobbered registers string
		StringBuilder sbClobberedRegisters = new StringBuilder ();
		for (IOperand.Register reg : m_setClobberedRegisters)
		{
			if (sbClobberedRegisters.length () > 0)
				sbClobberedRegisters.append (", ");
			
			sbClobberedRegisters.append ('"');
			
			// HACK
			String strRegName = reg.getBaseName ();
			if (strRegName.startsWith ("ymm"))
			{
				// GNU: error: unknown register name ‘ymmX’ in ‘asm’
				// replace "ymmX" by "xmmX"
				sbClobberedRegisters.append ('x');
				sbClobberedRegisters.append (strRegName.substring (1));
			}
			else
				sbClobberedRegisters.append (strRegName);
			
			sbClobberedRegisters.append ('"');
		}

		return sbClobberedRegisters.toString ();
	}
		
	/**
	 * 
	 * @param options
	 * @return
	 */
	public Statement generate (InstructionList ilInstructions, CodeGeneratorRuntimeOptions options)
	{
		// create a C statement wrapping the inline assembly
		
		// create the string of instructions
		StringBuilder sbInstructions = new StringBuilder ();
		for (IInstruction instruction : ilInstructions)
		{
			sbInstructions.append ('"');
			instruction.issue (sbInstructions);
			sbInstructions.append ("\"\n");
		}
		
		// build the list of child expressions (the inputs)
		// (if no children are provided to SomeExpression, the constarrs will be remove when checking whether
		// variables are referenced)
		List<Traversable> listChildren = new ArrayList<> (m_listInputs.size ());
		for (AssemblySectionInput asi : m_listInputs)
			listChildren.add (asi.getValue ().clone ());
		
		// build the IR object
		return new ExpressionStatement (new SomeExpression (
			StringUtil.concat (
				"__asm__ __volatile__ (\n",
				sbInstructions.toString (),
				":\n",
				": ", getInputsAsString (), "\n",
				": ", getClobberedRegistersAsString (), "\n",
				")"
			),
			listChildren
		));
	}

	/**
	 * Returns the size of a floating point data type.
	 * 
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
