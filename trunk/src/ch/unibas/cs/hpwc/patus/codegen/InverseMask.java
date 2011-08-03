package ch.unibas.cs.hpwc.patus.codegen;

import java.lang.reflect.InvocationTargetException;

import cetus.hir.Expression;

/**
 *
 * @author Matthias-M. Christen
 */
public class InverseMask extends AbstractMask
{
	///////////////////////////////////////////////////////////////////
	// Member Variables

	private AbstractMask m_mask;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public InverseMask (Expression[] rgExpressions, Class<? extends AbstractMask> clsMask)
	{
		super (rgExpressions);

		try
		{
			m_mask = clsMask.getConstructor (Expression[].class).newInstance ((Object) rgExpressions);
		}
		catch (IllegalArgumentException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		catch (SecurityException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		catch (NegativeArraySizeException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		catch (InstantiationException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		catch (IllegalAccessException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		catch (InvocationTargetException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		catch (NoSuchMethodException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	protected int[] createMask (Expression[] rgExpressions)
	{
		int[] rgMask = m_mask.createMask (rgExpressions);

		// invert the mask
		int[] rgMaskInverse = new int[rgMask.length];
		for (int i = 0; i < rgMaskInverse.length; i++)
			rgMaskInverse[i] = rgMask[i] == 0 ? 1 : 0;

		return rgMaskInverse;
	}
}
