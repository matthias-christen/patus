//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.1-b02-fcs 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2011.08.04 at 12:21:04 PM MESZ 
//


package ch.unibas.cs.hpwc.patus.arch;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for typeBaseIntrinsicName.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="typeBaseIntrinsicName">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="barrier"/>
 *     &lt;enumeration value="threadid"/>
 *     &lt;enumeration value="numthreads"/>
 *     &lt;enumeration value="unary_plus"/>
 *     &lt;enumeration value="unary_minus"/>
 *     &lt;enumeration value="plus"/>
 *     &lt;enumeration value="minus"/>
 *     &lt;enumeration value="multiply"/>
 *     &lt;enumeration value="divide"/>
 *     &lt;enumeration value="fma"/>
 *     &lt;enumeration value="sqrt"/>
 *     &lt;enumeration value="splat"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum TypeBaseIntrinsicName {

    @XmlEnumValue("barrier")
    BARRIER("barrier"),
    @XmlEnumValue("threadid")
    THREADID("threadid"),
    @XmlEnumValue("numthreads")
    NUMTHREADS("numthreads"),
    @XmlEnumValue("unary_plus")
    UNARY_PLUS("unary_plus"),
    @XmlEnumValue("unary_minus")
    UNARY_MINUS("unary_minus"),
    @XmlEnumValue("plus")
    PLUS("plus"),
    @XmlEnumValue("minus")
    MINUS("minus"),
    @XmlEnumValue("multiply")
    MULTIPLY("multiply"),
    @XmlEnumValue("divide")
    DIVIDE("divide"),
    @XmlEnumValue("fma")
    FMA("fma"),
    @XmlEnumValue("sqrt")
    SQRT("sqrt"),
    @XmlEnumValue("splat")
    SPLAT("splat");
    private final String value;

    TypeBaseIntrinsicName(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static TypeBaseIntrinsicName fromValue(String v) {
        for (TypeBaseIntrinsicName c: TypeBaseIntrinsicName.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
