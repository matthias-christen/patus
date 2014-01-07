package ch.unibas.cs.hpwc.patus.ilp;

import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * 
 * @author Matthias-M. Christen
 */
public class ILPModel
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	public enum EOperator
	{
		/**
		 * Less or equal
		 */
		LE,
		
		/**
		 * Equal
		 */
		EQ,
		
		/**
		 * Greater or equal
		 */
		GE
	}

	
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	/**
	 * The number of variables in the model
	 */
	protected int m_nVariablesCount;
	
	/**
	 * The number of constraints in the model
	 */
	protected int m_nConstraintsCount;
	
	
	/**
	 * The internal, native pointer to the CoinModel
	 */
	private /* native */ transient long m_ptrModel;

	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	/**
	 * 
	 * @param nVariablesCount
	 */
	public ILPModel (int nVariablesCount)
	{
		// make sure the native library is loaded
		ILPSolver.getInstance ();
		
		m_nVariablesCount = nVariablesCount;
		m_nConstraintsCount = 0;
		
		if (m_nVariablesCount <= 0)
			throw new RuntimeException ("The ILP model must have at least one variable.");
		
		createModelInternal (m_nVariablesCount);
	}
	
	/**
	 * 
	 * @param nVariablesCount
	 */
	private native void createModelInternal (int nVariablesCount);

	/**
	 * Sets the type of the variable at index <code>nVariableIdx</code>.
	 * 
	 * @param nVariableIdx
	 *            The index of the variable
	 * @param fLowerBound
	 *            The lower bound or <code>null</code> if no lower bound
	 * @param fUpperBound
	 *            The upper bound or <code>null</code> if no upper bound
	 * @param bIsInteger
	 *            Specifies whether the variable is an integer variable
	 */
	public void setVariableType (int nVariableIdx, Double fLowerBound, Double fUpperBound, boolean bIsInteger)
	{
		setVariableTypeInternal (nVariableIdx,
			fLowerBound != null, fLowerBound == null ? 0.0 : fLowerBound,
			fUpperBound != null, fUpperBound == null ? 0.0 : fUpperBound,
			bIsInteger
		);
	}
	
	private native void setVariableTypeInternal (int nVariableIdx,
		boolean bIsLowerBoundSet, double fLowerBound, boolean bIsUpperBoundSet, double fUpperBound, boolean bIsInteger);

	/**
	 * 
	 * @param nVariableIdx
	 * @param bIsBinary
	 */
	public void setVariableBinary (int nVariableIdx)
	{
		setVariableType (nVariableIdx, 0.0, 1.0, true);
	}
	
	/**
	 * Adds a constraint to the model.
	 * 
	 * @param rgCoeffs
	 *            A dense array of coefficients, one for each of the model's
	 *            variables
	 * @param op
	 *            The operator (equal, less-or-equal, greater-or-equal)
	 * @param fRhs
	 *            The right hand side
	 */
	public void addConstraint (double[] rgCoeffs, EOperator op, double fRhs)
	{
		switch (op)
		{
		case LE:
			addConstraint (rgCoeffs, (Double) null, fRhs);
			break;
			
		case EQ:
			addConstraint (rgCoeffs, fRhs, fRhs);
			break;
			
		case GE:
			addConstraint (rgCoeffs, fRhs, (Double) null);
			break;
			
		default:
			throw new RuntimeException (StringUtil.concat ("Not implemented for operator ", op.toString ()));
		}
	}

	/**
	 * Adds a constraint to the model.
	 * 
	 * @param rgCoeffs
	 *            A dense array of coefficients, one for each of the model's
	 *            variables
	 * @param fLowerBound
	 *            The lower bound of the constraint or <code>null</code> if no
	 *            lower bound
	 * @param fUpperBound
	 *            The upper bound of the constraint or <code>null</code> if no
	 *            upper bound
	 */
	public synchronized void addConstraint (double[] rgCoeffs, Double fLowerBound, Double fUpperBound)
	{
		addConstraintInternal (rgCoeffs,
			fLowerBound != null, fLowerBound == null ? 0.0 : fLowerBound,
			fUpperBound != null, fUpperBound == null ? 0.0 : fUpperBound
		);
		
		m_nConstraintsCount++;
	}
	
	private native void addConstraintInternal (double[] rgCoeffs, boolean bIsLowerBoundSet, double fLowerBound, boolean bIsUpperBoundSet, double fUpperBound);
	
	/**
	 * 
	 * @param rgObjectiveCoeffs
	 */
	public native void setObjective (double[] rgObjectiveCoeffs);
	
	/**
	 * Writes the model to an MPS file.
	 * 
	 * @param strFilename
	 *            The output filename
	 */
	public native void writeMPS (String strFilename);
			
	/**
	 * Returns the number of variables in the model.
	 * 
	 * @return The model's number of variables
	 */
	public int getVariablesCount ()
	{
		return m_nVariablesCount;
	}
	
	/**
	 * Returns the number of constraints in the model.
	 * 
	 * @return The model's number of constraints
	 */
	public int getConstraintsCount ()
	{
		return m_nConstraintsCount;
	}

	/**
	 * Deletes the model. Call this method when the model is no longer needed to
	 * free any used resources.
	 */
	public native void delete ();
}
