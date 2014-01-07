/*******************************************************************************
 * Copyright (c) 2011 Matthias-M. Christen, University of Basel, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Matthias-M. Christen, University of Basel, Switzerland - initial API and implementation
 ******************************************************************************/
package ch.unibas.cs.hpwc.patus.codegen;

import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryOperator;

/**
 * Helper class dealing with Cetus's operators.
 * @author Matthias-M. Christen
 */
public class Operators
{
	/**
	 * Creates an assignment operator from a corresponding binary operator, e.g.
	 * + becomes +=, etc.
	 * @param op The binary operator to convert to an assignment operator
	 * @return The assignment operator corresponding to the binary operator <code>op</code>
	 */
	public static AssignmentOperator getAssignmentOperatorFromBinaryOperator (BinaryOperator op)
	{
		if (BinaryOperator.ADD.equals (op))
			return AssignmentOperator.ADD;
		if (BinaryOperator.SUBTRACT.equals (op))
			return AssignmentOperator.SUBTRACT;
		if (BinaryOperator.MULTIPLY.equals (op))
			return AssignmentOperator.MULTIPLY;
		if (BinaryOperator.DIVIDE.equals (op))
			return AssignmentOperator.DIVIDE;
		if (BinaryOperator.MODULUS.equals (op))
			return AssignmentOperator.MODULUS;
		if (BinaryOperator.SHIFT_LEFT.equals (op))
			return AssignmentOperator.SHIFT_LEFT;
		if (BinaryOperator.SHIFT_RIGHT.equals (op))
			return AssignmentOperator.SHIFT_RIGHT;
		if (BinaryOperator.BITWISE_AND.equals (op))
			return AssignmentOperator.BITWISE_AND;
		if (BinaryOperator.BITWISE_INCLUSIVE_OR.equals (op))
			return AssignmentOperator.BITWISE_INCLUSIVE_OR;
		if (BinaryOperator.BITWISE_EXCLUSIVE_OR.equals (op))
			return AssignmentOperator.BITWISE_EXCLUSIVE_OR;
		
		return null;
	}
}
