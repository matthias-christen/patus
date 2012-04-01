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
import cetus.hir.Identifier;
import cetus.hir.Initializer;
import cetus.hir.SomeExpression;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ch.unibas.cs.hpwc.patus.arch.IArchitectureDescription;
import ch.unibas.cs.hpwc.patus.arch.TypeRegister;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterClass;
import ch.unibas.cs.hpwc.patus.arch.TypeRegisterType;
import ch.unibas.cs.hpwc.patus.codegen.CodeGeneratorSharedObjects;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.IRegisterOperand;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.IOperand.Register;
import ch.unibas.cs.hpwc.patus.codegen.backend.assembly.optimize.IInstructionListOptimizer;
import ch.unibas.cs.hpwc.patus.codegen.options.CodeGeneratorRuntimeOptions;
import ch.unibas.cs.hpwc.patus.util.ExpressionUtil;
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


	public static final String INPUT_CONSTANTS_ARRAYPTR = "_constants_";


	///////////////////////////////////////////////////////////////////
	// Member Variables

	protected CodeGeneratorSharedObjects m_data;

	/**
	 * The set of registers which got clobbered during the inline assembly section
	 */
	private Set<IOperand.Register> m_setClobberedRegisters;
	
	/**
	 * Flag indicating whether the memory is clobbered in the inline assembly section.
	 * The flag needs to be set using {@link AssemblySection#setMemoryClobbered(boolean)}.
	 */
	private boolean m_bIsMemoryClobbered;

	/**
	 * Data structure identifying which registers are currently in use
	 */
	private Map<IOperand.Register, Boolean> m_mapRegisterUsage;

	/**
	 * The list of inputs to the assembly section
	 */
	private List<AssemblySectionInput> m_listInputs;


	///////////////////////////////////////////////////////////////////
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
	
	private AssemblySectionInput getASInput (Object objInput)
	{
		for (AssemblySectionInput asi : m_listInputs)
			if (asi.getKey ().equals (objInput))
				return asi;
		return null;		
	}

	/**
	 * Retrieves the operand corresponding to the assembly section input <code>input</code>.
	 * 
	 * @param input
	 *            The input for which the retrieve the corresponding operand
	 * @return The operand corresponding to <code>input</code>
	 */
	public IOperand.IRegisterOperand getInput (Object objInput)
	{
		AssemblySectionInput asi = getASInput (objInput);
		return asi == null ? null : asi.getOperand ();
	}
	
	public int getInputsCount ()
	{
		return m_listInputs.size ();
	}
	
	public IArchitectureDescription getArchitectureDescription ()
	{
		return m_data.getArchitectureDescription ();
	}
	
	public CodeGeneratorSharedObjects getSharedObjects ()
	{
		return m_data;
	}
	
	public InstructionList translate (InstructionList ilInstructions, Specifier specDatatype)
	{
		return translate (ilInstructions, specDatatype, new IInstructionListOptimizer[] { }, new IInstructionListOptimizer[] { });
	}

	/**
	 * 
	 * @param ilInstructions
	 * @param specDatatype
	 * @return
	 */
	public InstructionList translate (InstructionList ilInstructions, Specifier specDatatype,
		IInstructionListOptimizer[] rgPreTranslateOptimizers,
		IInstructionListOptimizer[] rgPostTranslateOptimizers)
	{
		if (ilInstructions.isEmpty ())
			return ilInstructions;
		
		InstructionList ilTmp = ilInstructions;
		
		// apply pre-translate peep hole optimizations
		for (IInstructionListOptimizer optimizer : rgPreTranslateOptimizers)
			ilTmp = optimizer.optimize (ilTmp);

		// translate the generic instruction list to the architecture-specific one
		ilTmp = InstructionListTranslator.translate (
			m_data.getArchitectureDescription (), ilTmp, specDatatype);

		// apply peep hole optimizations
		for (IInstructionListOptimizer optimizer : rgPostTranslateOptimizers)
			ilTmp = optimizer.optimize (ilTmp);

		// allocate registers
		Iterable<IOperand.Register> itUsedRegs = getUsedRegisters ();
		ilTmp = ilTmp.allocateRegisters (this);
		restoreUsedRegisters (itUsedRegs);
		
		return ilTmp;
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

	public void killRegisters (Iterable<IOperand.Register> itRegisters)
	{
		for (IOperand.Register op : itRegisters)
			m_mapRegisterUsage.put (op, false);
	}
	
	public void killAllRegisters ()
	{
		m_mapRegisterUsage.clear ();
	}

	public Iterable<IOperand.Register> getUsedRegisters ()
	{
		List<IOperand.Register> listUsedRegisters = new ArrayList<> (m_mapRegisterUsage.size ());
		for (IOperand.Register op : m_mapRegisterUsage.keySet ())
			if (m_mapRegisterUsage.get (op))
				listUsedRegisters.add (op);
		return listUsedRegisters;
	}

	/**
	 *
	 */
	public void restoreUsedRegisters (Iterable<IOperand.Register> itUsedRegisters)
	{
		m_mapRegisterUsage.clear ();
		for (IOperand.Register op : itUsedRegisters)
			m_mapRegisterUsage.put (op, true);
	}
	
	/**
	 * Returns the number of registers of type <code>type</code>.
	 * 
	 * @param type
	 *            The register type
	 * @return The number of registers of type <code>type</code>
	 * @see IArchitectureDescription#getRegistersCount(TypeRegisterType)
	 */
	public int getRegistersCount (TypeRegisterType type)
	{
		return m_data.getArchitectureDescription ().getRegistersCount (type);
	}
	
	/**
	 * Counts how many registers of type <code>type</code> are currently free.
	 * 
	 * @param type
	 *            The register type
	 * @return The number of registers that are currently available
	 */
	public int getFreeRegistersCount (TypeRegisterType type)
	{
		int nRegistersCount = getRegistersCount (type);
		
		for (Register reg : m_mapRegisterUsage.keySet ())
		{
			if (m_mapRegisterUsage.get (reg) && (((TypeRegisterClass) reg.getRegister ().getClazz ()).getType ().equals (type)))
				nRegistersCount--;
		}
		
		return nRegistersCount;
	}

	/**
	 * Sets the &quot;memory clobbered&quot; flag, i.e., tells the compiler that
	 * if <code>bMemoryClobbered == true</code> memory locations are written to
	 * within the inline assembly section.
	 * 
	 * @param bMemoryClobbered
	 *            The &quot;memory clobbered&quot; flag
	 */
	public void setMemoryClobbered (boolean bMemoryClobbered)
	{
		m_bIsMemoryClobbered = bMemoryClobbered;
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
				// GNU: error: unknown register name ‘ymm?’ in ‘asm’
				// replace "ymm?" by "xmm?"
				sbClobberedRegisters.append ('x');
				sbClobberedRegisters.append (strRegName.substring (1));
			}
			else
				sbClobberedRegisters.append (strRegName);
			
			sbClobberedRegisters.append ('"');
		}
		
		if (m_bIsMemoryClobbered)
			sbClobberedRegisters.append (", \"memory\"");

		return sbClobberedRegisters.toString ();
	}
		
	/**
	 * Creates a Cetus statement from the list of assembly instructions
	 * <code>ilInstructions</code>.
	 * 
	 * @param options
	 *            Runtime code generation options
	 * @return The statement containing the list of assembly instructions as an
	 *         inline assembly section
	 */
	public Statement generate (InstructionList ilInstructions, CodeGeneratorRuntimeOptions options)
	{
		// create a C statement wrapping the inline assembly
		
		// create the string of instructions
		StringBuilder sbInstructions = new StringBuilder ();
		for (IInstruction instruction : ilInstructions)
		{
			if (!(instruction instanceof Comment))
				sbInstructions.append ('"');
			instruction.issue (sbInstructions);
			if (!(instruction instanceof Comment))
				sbInstructions.append ('"');
			sbInstructions.append ('\n');
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

	public int getConstantsAndParamsCount ()
	{
		return 0;
	}
	
	public void addSpillMemorySpace (int nMemoryPlacesCount, Specifier specDatatype)
	{
		AssemblySectionInput asi = getASInput (INPUT_CONSTANTS_ARRAYPTR);
		if (asi == null)
		{
			// TODO: add input
			throw new RuntimeException ("not implemented");
		}
		
		if (!(asi.getValue () instanceof Identifier))
			throw new RuntimeException ("Identifier expected for INPUT_CONSTANTS_ARRAYPTR");
		
		VariableDeclaration decl = (VariableDeclaration) ((Identifier) asi.getValue ()).getSymbol ().getDeclaration ();
		Initializer initializer = ((VariableDeclarator) decl.getDeclarator (0)).getInitializer ();

		int nCount = nMemoryPlacesCount * m_data.getArchitectureDescription ().getSIMDVectorLength (specDatatype);
		for (int i = 0; i < nCount; i++)
			initializer.getChildren ().add (ExpressionUtil.createFloatLiteral (0, specDatatype));
	}
}
