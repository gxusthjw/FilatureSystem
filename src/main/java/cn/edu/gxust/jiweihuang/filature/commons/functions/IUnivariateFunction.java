/*
 * Copyright (c) 2018-2019, Jiwei Huang. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package cn.edu.gxust.jiweihuang.filature.commons.functions;

import org.hipparchus.analysis.UnivariateFunction;
import org.hipparchus.analysis.integration.MidPointIntegrator;
import org.hipparchus.analysis.integration.RombergIntegrator;
import org.hipparchus.analysis.integration.SimpsonIntegrator;
import org.hipparchus.analysis.integration.TrapezoidIntegrator;

import java.io.Serializable;

/**
 * <p>The interface {@code IUnivariateFunction} is used for
 * representing univariate function.</p>
 *
 * <p>it inherits from {@code UnivariateFunction}
 * which in {@code org.hipparchus.analysis} package of
 * {@code hipparchus} library.</p>
 *
 * <p>The library {@code hipparchus} is a math library,
 * it has many feature,include the differential (derivative) with any order,
 * numerical integration and interpolation,etc.</p>
 *
 * <p>Create date:2018-11-20.</p>
 *
 * @author JiweiHuang
 * @version 1.0.0_build-20181120
 * @see UnivariateFunction
 * @see Serializable
 */
public interface IUnivariateFunction extends UnivariateFunction,
        Serializable {
    /**
     * <p>The method {@code lowerX()} is used to get
     * the lower limit of independent variable range. </p>
     * <p>The implement class (or subinterface) of this interface
     * should override this method for changing the the lower limit value.</p>
     * <p>The default value of this method:{@code Double.NEGATIVE_INFINITY}.</p>
     * <p>Create date:2018-11-20.</p>
     *
     * @return the lower limit of independent variable range.
     * @author JiweiHuang
     * @since 1.0.0_build-20181120
     */
    default double lowerX() {
        return Double.NEGATIVE_INFINITY;
    }

    /**
     * <p>The method {@code upperX()} is used to get
     * the upper limit of independent variable range. </p>
     * <p>The implement class (or subinterface) of this interface
     * should override this method for changing the upper limit value.</p>
     * <p>The default value of this method:{@code Double.POSITIVE_INFINITY}.</p>
     * <p>Create date:2018-11-20.</p>
     *
     * @return the upper limit of independent variable range.
     * @author JiweiHuang
     * @since 1.0.0_build-20181120
     */
    default double upperX() {
        return Double.POSITIVE_INFINITY;
    }


    /**
     * <p>The method {checkX(double x)} is used to
     * check {@code x} whether which in independent variable range [lowerX,upperX].</p>
     * <p>Create date:2018-11-20.</p>
     *
     * @param x the x-value,independent variable range.
     * @author JiweiHuang
     * @since 1.0.0_build-20181120
     */
    default void checkX(double x) {
        if (x < lowerX() || x > upperX()) {
            throw new IllegalArgumentException(String.format(
                    "Expected the parameter {%f <= x <= %f},but got {@code x=%f}", lowerX(), upperX(), x));
        }
    }

    /**
     * <p>The method {@code formula()} is used to get
     * the string form of analytical expression of univariate function,
     * for instance, {@code f(x)=a*x^2+c}.</p>
     * <p>Create date:2018-11-20.</p>
     *
     * @return the string form of analytical expression of univariate function.
     * @author JiweiHuang
     * @since 1.0.0_build-20181120
     */
    String formula();

    /**
     * <p>The method {@code integrateMidPoint(double lowerX, double upperX)} is used to
     * get the numerical integral value with midpoint method (中点积分).</p>
     *
     * @param lowerX the lower limit of integral
     * @param upperX the upper limit of integral
     * @return the numerical integral value with midpoint method.
     */
    default double integrateMidPoint(double lowerX, double upperX) {
        MidPointIntegrator integrator = new MidPointIntegrator();
        return integrator.integrate(Integer.MAX_VALUE, this, lowerX, upperX);
    }

    /**
     * <p>The method {@code integrateRomberg(double lowerX, double upperX)} is used to
     * get the numerical integral value with romberg method (龙贝格积分).</p>
     *
     * @param lowerX the lower limit of integral
     * @param upperX the upper limit of integral
     * @return the numerical integral value with romberg method.
     */
    default double integrateRomberg(double lowerX, double upperX) {
        RombergIntegrator integrator = new RombergIntegrator();
        return integrator.integrate(Integer.MAX_VALUE, this, lowerX, upperX);
    }

    /**
     * <p>The method {@code integrateSimpson(double lowerX, double upperX)} is used to
     * get the numerical integral value with simpson method (辛普森积分).</p>
     *
     * @param lowerX the lower limit of integral
     * @param upperX the upper limit of integral
     * @return the numerical integral value with simpson method.
     */
    default double integrateSimpson(double lowerX, double upperX) {
        SimpsonIntegrator integrator = new SimpsonIntegrator();
        return integrator.integrate(Integer.MAX_VALUE, this, lowerX, upperX);
    }

    /**
     * <p>The method {@code integrateTrapezoid(double lowerX, double upperX)} is used to
     * get the numerical integral value with trapezoid method (梯形积分).</p>
     *
     * @param lowerX the lower limit of integral
     * @param upperX the upper limit of integral
     * @return the numerical integral value with trapezoid method.
     */
    default double integrateTrapezoid(double lowerX, double upperX) {
        TrapezoidIntegrator integrator = new TrapezoidIntegrator();
        return integrator.integrate(Integer.MAX_VALUE, this, lowerX, upperX);
    }

    /**
     * <p>The method {@code integrateMidPoint(double lowerX, double upperX)} is used to
     * get the numerical integral value with midpoint method (中点积分).</p>
     *
     * @return the numerical integral value with midpoint method.
     */
    default double integrateMidPoint() {
        MidPointIntegrator integrator = new MidPointIntegrator();
        return integrator.integrate(Integer.MAX_VALUE, this, lowerX(), upperX());
    }

    /**
     * <p>The method {@code integrateRomberg(double lowerX, double upperX)} is used to
     * get the numerical integral value with romberg method (龙贝格积分).</p>
     *
     * @return the numerical integral value with romberg method.
     */
    default double integrateRomberg() {
        RombergIntegrator integrator = new RombergIntegrator();
        return integrator.integrate(Integer.MAX_VALUE, this, lowerX(), upperX());
    }

    /**
     * <p>The method {@code integrateSimpson(double lowerX, double upperX)} is used to
     * get the numerical integral value with simpson method (辛普森积分).</p>
     *
     * @return the numerical integral value with simpson method.
     */
    default double integrateSimpson() {
        SimpsonIntegrator integrator = new SimpsonIntegrator();
        return integrator.integrate(Integer.MAX_VALUE, this, lowerX(), upperX());
    }

    /**
     * <p>The method {@code integrateTrapezoid(double lowerX, double upperX)} is used to
     * get the numerical integral value with trapezoid method (梯形积分).</p>
     *
     * @return the numerical integral value with trapezoid method.
     */
    default double integrateTrapezoid() {
        TrapezoidIntegrator integrator = new TrapezoidIntegrator();
        return integrator.integrate(Integer.MAX_VALUE, this, lowerX(), upperX());
    }
}
