package cetus.hir;

import java.util.*;

/**
 * Annotatable interface defines methods for Cetus IR that contains a list
 * of annotations. Declaration and Statement implement Annotatable now.
 */
public interface Annotatable
{
	/**
	 * Annotates with the given annotation.
	 */
	void annotate(Annotation annotation);
	void annotateAfter(Annotation annotation);
	void annotateBefore(Annotation annotation);

	/**
	 * Returns the list of annotations.
	 */
	List<Annotation> getAnnotations();

	/**
	 * Returns the list of annotations with the given type.
	 */
	<T extends Annotation> List<T> getAnnotations(Class<T> type);

	/**
	 * Checks if this annotatable contains the specified annotation type and key.
	 */
	boolean containsAnnotation(Class<? extends Annotation> type, String key);

	/**
	 * Returns the annotation with the specified type and key.
	 */
	<T extends Annotation> T getAnnotation(Class<T> type, String key);

	/**
	 * Returns the list of annotations with the given relative position.
	 */
	List<Annotation> getAnnotations(int position);

	/**
	 * Remove all annotations.
	 */
	void removeAnnotations();

	/**
	 * Remove all annotations of the given type.
	 */
	void removeAnnotations(Class<?> type);
}
