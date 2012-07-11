package ch.unibas.cs.hpwc.patus.graph;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import ch.unibas.cs.hpwc.patus.codegen.Globals;
import ch.unibas.cs.hpwc.patus.graph.algorithm.GraphUtil;
import ch.unibas.cs.hpwc.patus.util.IParallelOperation;
import ch.unibas.cs.hpwc.patus.util.MathUtil;
import ch.unibas.cs.hpwc.patus.util.StringUtil;

/**
 * A default graph implementation.
 * @author Matthias-M. Christen
 *
 * @param <V> The vertex type
 * @param <E> The edge type
 */
public class DefaultGraph<V extends IVertex, E extends IEdge<V>> implements IGraph<V, E>
{
	///////////////////////////////////////////////////////////////////
	// Member Variables
	
	protected Map<V, V> m_mapVertices;
	protected Set<E> m_setEdges;
	
	
	///////////////////////////////////////////////////////////////////
	// Implementation
	
	public DefaultGraph ()
	{
		m_mapVertices = new HashMap<> ();
		m_setEdges = new HashSet<> ();
	}
	
	public void addVertex (V v)
	{
		if (!m_mapVertices.containsKey (v))
			m_mapVertices.put (v, v);
	}
	
	public boolean containsVertex (V v)
	{
		return m_mapVertices.containsKey (v);
	}
	
	public V findVertex (V v)
	{
		V vRes = m_mapVertices.get (v);
		if (vRes == null)
		{
			addVertex (v);
			return v;
		}
		
		return vRes;
	}
	
	@Override
	public void addEdge (E edge)
	{
		m_setEdges.add (edge);
	}

	@Override
	public Iterable<V> getVertices ()
	{
		return m_mapVertices.keySet ();
	}

	@Override
	public Iterable<E> getEdges ()
	{
		return m_setEdges;
	}

	@Override
	public int getVerticesCount ()
	{
		return m_mapVertices.size ();
	}

	@Override
	public int getEdgesCount ()
	{
		return m_setEdges.size ();
	}
	
	@Override
	public void removeAllVertices ()
	{
		m_mapVertices.clear ();
		removeAllEdges ();
	}
	
	public void removeEdge (E edge)
	{
		m_setEdges.remove (edge);
	}
	
	@Override
	public void removeAllEdges ()
	{
		m_setEdges.clear ();
	}
	
	protected <T> void forAllElements (final IParallelOperation<T> op, final Collection<T> elements)
	{
		// runnable class used to pass the number of the thread
		abstract class RunnableWithThreadNum implements Runnable
		{
			protected int m_nThreadNum;
			
			public RunnableWithThreadNum (int nThreadNum)
			{
				m_nThreadNum = nThreadNum;
			}
		}
		
		// submit a runnable for each thread
		List<Future<?>> listFutures = new ArrayList<> (Globals.NUM_THREADS);
		for (int i = 0; i < Globals.NUM_THREADS; i++)
		{
			listFutures.add (Globals.EXECUTOR_SERVICE.submit (new RunnableWithThreadNum (i)
			{
				@Override
				public void run ()
				{
					// perform max-min+1 operations on contiguous collection elements per thread
					int nChunk = MathUtil.divCeil (elements.size (), Globals.NUM_THREADS);
					int nMin = m_nThreadNum * nChunk;
					int nMax = Math.min ((m_nThreadNum + 1) * nChunk - 1, elements.size () - 1);
					
					int j = 0;
					for (T element : elements)
					{
						if (nMin <= j && j <= nMax)
							op.perform (element);
						j++;
					}
				}
			}));
		}
		
		// synchronize
		try
		{
			for (Future<?> f : listFutures)
				f.get ();
		}
		catch (InterruptedException e)
		{			
		}
		catch (ExecutionException e)
		{			
		}		
	}
	
	@Override
	public void forAllVertices (final IParallelOperation<V> op)
	{
		forAllElements (op, m_mapVertices.keySet ());
	}
	
	@Override
	public void forAllEdges (IParallelOperation<E> op)
	{
		forAllElements (op, m_setEdges);
	}
	
	private String getShortVertexString (V vertex)
	{
		try
		{
			Method m = vertex.getClass ().getMethod ("toShortString");
			if (m != null)
				return (String) m.invoke (vertex);
		}
		catch (Exception e)
		{
		}
		
		return vertex.toString ();
	}
	
	@Override
	public String toString ()
	{
		StringBuilder sb = new StringBuilder (getClass ().getSimpleName ());
		sb.append (" {\n");
		
		for (V v : m_mapVertices.keySet ())
		{
			sb.append ('\t');
			sb.append (v.toString ());
			sb.append ("  --->  { ");

			boolean bFirst = true;
			for (V v1 : GraphUtil.getSuccessors (this, v))
			{
				if (!bFirst)
					sb.append (", ");
				sb.append (getShortVertexString (v1));					
				bFirst = false;
			}
			
			sb.append (" }\n");
		}
		
		sb.append ('}');

		return sb.toString ();
	}
	
	/**
	 * <p>
	 * Creates a graphviz representation and starts graphviz to produce a
	 * rendering of the graph.
	 * </p>
	 * <p>
	 * Requires that graphviz is installed on the system.
	 * </p>
	 */
	public void graphviz ()
	{
		StringBuilder sb = new StringBuilder ("digraph ");
		sb.append (getClass ().getSimpleName ());
		sb.append (" {\n");
		
		Map<V, String> mapVertexIDs = new HashMap<> ();
		Map<String, String> mapLegend = new TreeMap<> (new Comparator<String> ()
		{
			@Override
			public int compare (String str1, String str2)
			{
				if (str1.length () >= 2 && str2.length () >= 2)
					if (str1.charAt (0) == 'V' && Character.isDigit (str1.charAt (1)) && str2.charAt (0) == 'V' && Character.isDigit (str2.charAt (1)))
					{
						int nEnd1 = str1.indexOf (' ');
						if (nEnd1 == -1)
							nEnd1 = str1.length ();
						
						int nEnd2 = str2.indexOf (' ');
						if (nEnd2 == -1)
							nEnd2 = str2.length ();

						return Integer.parseInt (str1.substring (1, nEnd1)) - Integer.parseInt (str2.substring (1, nEnd2));
					}
				
				return str1.compareTo (str2);
			}
		});
		
		// add vertices
		int i = 0;
		for (V v : m_mapVertices.keySet ())
		{
			String strID = StringUtil.concat ("V", i++);
			mapVertexIDs.put (v, strID);
			
			sb.append ('\t');
			sb.append (strID);
			sb.append (" [label=\"");
			
			String strLabel = getShortVertexString (v);
			String strLegend = v.toString ().trim ();
			
			sb.append (strLabel);
			
			if (!strLabel.equals (strLegend))
				mapLegend.put (strLabel, strLegend);
			
			sb.append ("\"];\n");
		}
		
		// add edges
		for (E e : m_setEdges)
		{
			sb.append ('\t');
			sb.append (mapVertexIDs.get (e.getTailVertex ()));
			sb.append (" -> ");
			sb.append (mapVertexIDs.get (e.getHeadVertex ()));
			
			if (e instanceof IParametrizedEdge<?, ?>)
			{
				sb.append (" [label=\"");
				sb.append (((IParametrizedEdge<?, ?>) e).getData ());
				sb.append ("\"]");
			}
			
			sb.append (";\n");
		}
		
		// add legend (if necesary)
		if (mapLegend.size () > 0)
		{
			sb.append ("\t{\n\trank=sink;\n\tlegend [shape=none, margin=0, label=<\n\t\t<TABLE>\n");
			for (String strLabel : mapLegend.keySet ())
			{
				sb.append ("\t\t\t<TR><TD>");
				sb.append (strLabel);
				sb.append ("</TD><TD>");
				sb.append (mapLegend.get (strLabel));
				sb.append ("</TD></TR>\n");
			}
			sb.append ("\t\t</TABLE>\n\t>];\t}\n");
		}
		
		sb.append ("}\n");
		
		// visualize the structure
		try
		{
			ProcessBuilder pb = new ProcessBuilder ("dot", "-Tpng");
			Process p = pb.start ();
			OutputStream osInput = p.getOutputStream ();
			InputStream isResult = p.getInputStream ();
			
			osInput.write (sb.toString ().getBytes ());
			osInput.flush ();
			
			final BufferedImage img = ImageIO.read (isResult);
			
			JFrame wnd = new JFrame (getClass ().getSimpleName ());
			wnd.setDefaultCloseOperation (WindowConstants.DISPOSE_ON_CLOSE);
			JPanel pnlImage = new JPanel ()
			{
				private static final long serialVersionUID = 1L;

				@Override
				protected void paintComponent (Graphics g)
				{
					super.paintComponent (g);
					((Graphics2D) g).setRenderingHint (RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
					g.drawImage (img, 0, 0, getWidth (), getHeight (), this);
				}
			};
			pnlImage.setPreferredSize (new Dimension (img.getWidth (), img.getHeight ()));
			wnd.getContentPane ().setLayout (new BorderLayout ());
			wnd.getContentPane ().add (pnlImage, BorderLayout.CENTER);
			wnd.pack ();
			wnd.setVisible (true);
			
			osInput.close ();
			isResult.close ();
			p.destroy ();
		}
		catch (Exception e)
		{
			e.printStackTrace ();
		}
	}
}
