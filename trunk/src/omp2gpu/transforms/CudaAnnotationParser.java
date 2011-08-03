package omp2gpu.transforms;

import java.io.*;
import java.util.*;

import cetus.analysis.*;
import cetus.hir.*;
import cetus.exec.*;
import cetus.transforms.*;
import omp2gpu.hir.*;
import omp2gpu.analysis.*;

/**
 * @author Seyong Lee <lee222@purdue.edu>
 *         ParaMount Group 
 *         School of ECE, Purdue University
 *
 * This pass is used to parse cuda annotations that might be
 * added in the C source code input to Cetus and convert them to
 * internal CudaAnnotations. 
 * This pass should be called after cetus.transforms.AnnotationParser
 * is executed.
 */
public class CudaAnnotationParser extends TransformPass
{

	public CudaAnnotationParser(Program program)
	{
		super(program);
	}

	public String getPassName()
	{
		return new String("[CudaAnnotParser]");
	}

	public void start()
	{
		Annotation new_annot = null;
		boolean attach_to_next_annotatable = false;
		HashMap<String, Object> new_map = null;
		LinkedList<Annotation> annots_to_be_attached = new LinkedList<Annotation>();
		
		/* Iterate over the program in Depth First order and search for Annotations */
		DepthFirstIterator iter = new DepthFirstIterator(program);

		while(iter.hasNext())
		{
			Object obj = iter.next();

			/////////////////////////////////////////////////////////////////
			// Currently, Cuda annotations exist only in functions.        //
			// Therefore, AnnotationParser store them as PragmaAnnotations //
			// in AnnotationStatements.                                    //
			/////////////////////////////////////////////////////////////////
			if (obj instanceof AnnotationStatement)
			{
				AnnotationStatement annot_container = (AnnotationStatement)obj;
				List<PragmaAnnotation> annot_list = 
					annot_container.getAnnotations(PragmaAnnotation.class);
				if( (annot_list == null) || (annot_list.size() == 0) ) {
					continue;
				}
				/////////////////////////////////////////////////////////////////////////////////
				// AnnotationParser creates one AnnotationStatement for each PragmaAnnotation. //
				/////////////////////////////////////////////////////////////////////////////////
				PragmaAnnotation pAnnot = annot_list.get(0);
				/////////////////////////////////////////////////////////////////
				// DEBUG: The above pAnnot may be a standalone CutusAnnotation //
				// or OmpAnnotation. If so, skip it.                           //
				// (Both CetusAnnotation and OmpAnnotations are child classes  //
				//  of PragmaAnnotation.)                                      //
				/////////////////////////////////////////////////////////////////
				if( pAnnot instanceof CetusAnnotation || pAnnot instanceof OmpAnnotation ) {
					continue;
				}
				String old_annot = pAnnot.getName();
				if( old_annot == null ) {
					PrintTools.println("[WARNING in CudaAnnotationParser] Pragma annotation, "
							+ pAnnot +", does not have name", 0);
					continue;
				}

				/* -------------------------------------------------------------------------
				 * STEP 1:
				 * Find the annotation type by parsing the text in the input annotation and
				 * create a new Annotation of the corresponding type
				 * -------------------------------------------------------------------------
				 */
				String[] token_array = old_annot.split("\\s+");
				// old_annot string has a leading space, and thus the 2nd token should be checked.
				//String old_annot_key = token_array[1];
				String old_annot_key = token_array[0];
				/* Check for Cuda annotations */
				if (old_annot_key.compareTo("cuda")==0) {
					/* ---------------------------------------------------------------------
					 * Parse the contents:
					 * CudaParser puts the Cuda directive parsing results into new_map
					 * ---------------------------------------------------------------------
					 */
					new_map = new HashMap<String, Object>();
					attach_to_next_annotatable = CudaParser.parse_cuda_pragma(new_map, token_array);
					/* Create an CudaAnnotation and copy the parsed contents from new_map
					 * into a new CudaAnnotation */
					new_annot = new CudaAnnotation();
					for (String key : new_map.keySet())
						new_annot.put(key, new_map.get(key));
				}
				else {
					continue;
				}
				
				/* ----------------------------------------------------------------------------------
				 * STEP 2:
				 * Based on whether the newly created annotation needs to be attached to an Annotatable
				 * object or needs to be inserted as a standalone Annotation contained within
				 * AnnotationStatement or AnnotationDeclaration, perform the following IR
				 * insertion and deletion operations
				 * ----------------------------------------------------------------------------------
				 */
				/* If the annotation doesn't need to be attached to an existing Annotatable object,
				 * remove old PragmaAnnotation and insert the new CudaAnnotation into the existing
				 * container.
				 */
				if (!attach_to_next_annotatable)
				{
					annot_container.removeAnnotations(PragmaAnnotation.class);
					annot_container.annotate(new_annot);

					/* In order to allow non-attached annotations mixed with attached annotations,
					 * check if the to_be_attached list is not empty. If it isn't, some annotations still
					 * exist that need to attached to the very next Annotatable. Hence, ... */
					if ( !annots_to_be_attached.isEmpty() )
						attach_to_next_annotatable = true;
					
				}
				else 
				{
					/* Add the newly created Annotation to a list of Annotations that will be attached
					 * to the required Annotatable object in the IR
					 */
					annots_to_be_attached.add(new_annot);
					/* Remove the old annotation container from the IR */
					CompoundStatement parent_stmt = (CompoundStatement)annot_container.getParent();
					parent_stmt.removeChild(annot_container);
				}
			}
			/* -----------------------------------------------------------------------------------
			 * STEP 3:
			 * A list of newly created Annotations to be attached has been created. Attach it to
			 * the instance of Annotatable object that does not already contain an input Annotation, 
			 * this is encountered next
			 * -----------------------------------------------------------------------------------
			 */
			else if ((obj instanceof DeclarationStatement) &&
					 (IRTools.containsClass((Traversable)obj, PreAnnotation.class))) {
				continue;
			}
			else if ((attach_to_next_annotatable) && (obj instanceof Annotatable))
			{
				Annotatable container = (Annotatable)obj;
				if (!annots_to_be_attached.isEmpty() && container != null)
				{
					/* Attach all the new annotations to this container */
					for (Annotation annot_to_be_attached : annots_to_be_attached)
						container.annotate(annot_to_be_attached);
				} 
				else
				{
					System.out.println("Error");
					System.exit(0);
				}
				/* reset the flag to false, we've attached all annotations */
				attach_to_next_annotatable = false;
				/* Clear the list of annotations to be attached, we're done with them */
				annots_to_be_attached.clear();
			}
		}
	}
	
}


