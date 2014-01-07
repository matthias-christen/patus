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
package ch.unibas.cs.hpwc.patus.config;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import javax.swing.AbstractAction;
import javax.swing.InputVerifier;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;

import layout.TableLayout;
import layout.TableLayoutConstraints;

public class ConfigUI //extends JDialog
{
	///////////////////////////////////////////////////////////////////
	// Inner Types

	private class PropertyUI
	{
		private ConfigurationProperty m_property;
		private JComponent m_component;


		/**
		 * Constructs a property UI object from a {@link ConfigurationProperty} object.
		 * @param property The configuration property
		 */
		public PropertyUI (ConfigurationProperty property)
		{
			m_property = property;
		}

//		/**
//		 * Returns the underlying property object.
//		 * @return
//		 */
//		public ConfigurationProperty getProperty ()
//		{
//			return m_property;
//		}

		/**
		 * Returns the caption of the property.
		 * @return
		 */
		public String getCaption ()
		{
			return m_property.getName ();
		}

		/**
		 * Creates a UI component corresponding to the property.
		 * @return The UI component
		 */
		@SuppressWarnings("unchecked")
		public JComponent createComponent ()
		{
			JComponent cmp = null;
			m_component = null;

			switch (m_property.getType ())
			{
			case INTEGER:
				int nValue = 0;
				try
				{
					nValue = Integer.parseInt (m_property.getDisplayValue ());
				}
				catch (NumberFormatException e)
				{
					// leave default value
				}

				if (m_property.getValues ().size () > 1)
				{
					int nMin = (Integer) m_property.getValues ().get (0);
					int nMax = (Integer) m_property.getValues ().get (1);
					m_component = new JSpinner (new SpinnerNumberModel (nValue, nMin, nMax, 1));
				}
				else
				{
					m_component = new JTextField (String.valueOf (nValue));
					((JTextField) m_component).setInputVerifier (new InputVerifier ()
					{
						@Override
						public boolean verify (JComponent cmpInput)
						{
							if (!(cmpInput instanceof JTextField))
								return true;
							String strText = ((JTextField) cmpInput).getText ();
							try
							{
								Integer.parseInt (strText);
								return true;
							}
							catch (NumberFormatException e)
							{
								JOptionPane.showMessageDialog (m_component, "The value in '" + m_property.getName () + "' must be an integer.", "Invalid Input", JOptionPane.ERROR_MESSAGE);
								return false;
							}
						}
					});
				}
				cmp = m_component;
				break;

			case STRING:
				m_component = new JTextField (m_property.getDisplayValue ());
				cmp = m_component;
				break;

			case LIST:
				Object[] rgValues = new Object[m_property.getValues ().size ()];
				m_property.getValues ().toArray (rgValues);
				m_component = new JComboBox<> (rgValues);
				((JComboBox<String>) m_component).setSelectedItem (m_property.getDisplayValue ());
				cmp = m_component;
				break;

			case FILE:
				cmp = new JPanel ();
				cmp.setLayout (new TableLayout (new double[][] {
					{ TableLayout.FILL, TableLayout.PREFERRED },
					{ TableLayout.PREFERRED }
				}));

				// create and add the text field
				m_component = new JTextField (m_property.getDisplayValue ());
				m_component.setPreferredSize (new Dimension (600, 10));
				cmp.add (m_component, "0,0");

				// create and add the browse button
				JButton btnBrowse = new JButton (new AbstractAction ("...")
				{
					private static final long serialVersionUID = 1L;

					@Override
					public void actionPerformed (ActionEvent e)
					{
						JFileChooser dlg = new JFileChooser (((JTextField) m_component).getText ());
						if (dlg.showOpenDialog (m_component) == JFileChooser.APPROVE_OPTION)
							((JTextField) m_component).setText (dlg.getSelectedFile ().getAbsolutePath ());
					}
				});
				cmp.add (btnBrowse, "1,0");
				break;
			}

			return cmp;
		}

		/**
		 * Saves the value in the UI component into the <code>Properties</code> object.
		 */
		@SuppressWarnings("unchecked")
		public void save ()
		{
			if (m_component instanceof JTextField)
				m_property.setValue (((JTextField) m_component).getText ());
			else if (m_component instanceof JSpinner)
				m_property.setValue (((JSpinner) m_component).getValue ().toString ());
			else if (m_component instanceof JComboBox)
				m_property.setValue (((JComboBox<String>) m_component).getSelectedItem ().toString ());
		}
	}


	///////////////////////////////////////////////////////////////////
	// Constants

	@SuppressWarnings("unused")
	private static final long serialVersionUID = 1L;

	private static final int VERTICAL_GAP = 4;
	private static final int HORIZONTAL_GAP = 4;
	private static final int BORDER = 8;


	///////////////////////////////////////////////////////////////////
	// Member Variables

	/**
	 * Configuration properties
	 */
	private Map<String, ConfigurationProperty> m_mapProperties;

	/**
	 * List of UI property objects
	 */
	private List<PropertyUI> m_listProperties;


	///////////////////////////////////////////////////////////////////
	// Implementation

	public ConfigUI (Map<String, ConfigurationProperty> mapProperties)
	{
		m_listProperties = new LinkedList<> ();
		m_mapProperties = mapProperties;
	}

	/**
	 * Shows the configuration dialog and returns {@link JOptionPane#OK_OPTION} if the
	 * &quot;OK&quot; button has been clicked.
	 * @return {@link JOptionPane#OK_OPTION} or {@link JOptionPane#CANCEL_OPTION}
	 */
	public int showDialog ()
	{
		int nResult = JOptionPane.showConfirmDialog (null, createTabs (), "Patus Configuration", JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);
		if (nResult == JOptionPane.OK_OPTION)
		{
			for (PropertyUI p : m_listProperties)
				p.save ();
		}

		return nResult;
	}

	private JTabbedPane createTabs ()
	{
		// build a map of property lists
		Map<String, List<PropertyUI>> mapTabs = new TreeMap<> ();
		for (ConfigurationProperty cp : m_mapProperties.values ())
		{
			// create a property object
			PropertyUI p = new PropertyUI (cp);
			m_listProperties.add (p);

			// add the property to the list of the category it belongs to
			List<PropertyUI> list = mapTabs.get (cp.getCategory ());
			if (list == null)
				mapTabs.put (cp.getCategory (), list = new LinkedList<> ());
			list.add (p);
		}

		// create the tabbed pane
		JTabbedPane pnlTabs = new JTabbedPane ();
		for (String strCategory : mapTabs.keySet ())
		{
			// create the contents of the tab
			JPanel pnlTab = new JPanel ();

			// get the list of properties in this category
			List<PropertyUI> listProps = mapTabs.get (strCategory);

			// build the layout
			double[] rgRows = new double[listProps.size () * 2 + 1];
			rgRows[0] = BORDER;
			int k = 1;
			for (int i = 0; i < listProps.size (); i++)
			{
				if (i != 0)
					rgRows[k++] = VERTICAL_GAP;
				rgRows[k++] = TableLayout.PREFERRED;
			}
			rgRows[k] = BORDER;
			pnlTab.setLayout (new TableLayout (new double[][] {
				{ BORDER, TableLayout.PREFERRED, HORIZONTAL_GAP, TableLayout.FILL, BORDER },
				rgRows
			}));

			// add the components
			int i = 1;
			for (PropertyUI p : listProps)
			{
				// add the caption
				pnlTab.add (
					new JLabel (p.getCaption ()),
					new TableLayoutConstraints (1, i, 1, i, TableLayoutConstraints.LEFT, TableLayoutConstraints.LEFT));

				// add the component
				pnlTab.add (
					p.createComponent (),
					new TableLayoutConstraints (3, i, 3, i, TableLayoutConstraints.LEFT, TableLayoutConstraints.LEFT));

				// next row
				i += 2;
			}

			// add a new tab
			pnlTabs.addTab (strCategory, pnlTab);
		}

		return pnlTabs;
	}
}
