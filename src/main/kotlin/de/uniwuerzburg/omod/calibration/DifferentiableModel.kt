package de.uniwuerzburg.omod.calibration

import kotlin.math.exp
import kotlin.math.pow

fun main () {
    println("Backwards test")
    bbLinearTest1()
    bbQuadraticTest1()
    bbQuadraticTest2()
    bbExponTest1()
    bbExponTest2()
    bbDivisionTermTest1()
    bbDivisionTermTest2()

    println("Forwards test")
    linearTest1()
    quadraticTest1()
    quadraticTest2()
    exponTest1()
    exponTest2()
    divisionTermTest1()
    divisionTermTest2()
    cacheTest1()
    linearRELUTest1()
    linearRELUTest2()
}

fun bbLinearTest1() {
    // f: 3x + 1y
    // value: x=1, y=1
    // expected gradient: d/dx = 3, d/dy = 1
    // expected value: 4
    val vals = doubleArrayOf(1.0, 1.0)

    val term = LinearBaseTerm(2)

    term.addTerm(0, 3.0)
    term.addTerm(1, 1.0)

    val gradient = DoubleArray(2) { 0.0 }
    val x = term.evaluate(vals)
    term.chainBackward(vals, gradient, 1.0)

    val gx = gradient[0]
    val gy = gradient[1]

    require(gx == 3.0)
    require(gy == 1.0)
    require(x == 4.0)
    println("linearTest1 OK")
}

fun bbQuadraticTest1() {
    // f: 3xy
    // value: x=2, y=5
    // expected gradient: d/dx = 15, d/dy = 6
    // expected value: 30
    // expected value: 30
    val vals = doubleArrayOf(2.0, 5.0)

    val xTerm = LinearBaseTerm(2)
    xTerm.addTerm(0, 1.0)

    val yTerm = LinearBaseTerm(2)
    yTerm.addTerm(1, 1.0)

    val qTerm = QuadraticTerm(2, xTerm, yTerm, 3.0)

    val gradient = DoubleArray(2) { 0.0 }
    val x = qTerm.evaluate(vals)
    qTerm.chainBackward(vals, gradient, 1.0)

    val gx = gradient[0]
    val gy = gradient[1]

    require(gx == 15.0)
    require(gy == 6.0)
    require(x == 30.0)
    println("quadraticTest1 OK")
}

fun bbQuadraticTest2() {
    // f: 1.5xx
    // value: x=0.3
    // expected gradient: d/dx = 0.9, d/dy = 0.0
    // expected value: 0.13499999999999998
    val vals = doubleArrayOf(0.3, 100.0)

    val xTerm = LinearBaseTerm(2)
    xTerm.addTerm(0, 1.0)

    val qTerm = QuadraticTerm(2, xTerm, xTerm, 1.5)

    val gradient = DoubleArray(2) { 0.0 }
    val x = qTerm.evaluate(vals)
    qTerm.chainBackward(vals, gradient, 1.0)

    val gx = gradient[0]
    val gy = gradient[1]

    require(gx == 3 * 0.3)
    require(gy == 0.0)
    require(x == 0.3 * 0.3 * 1.5)
    println("quadraticTest2 OK")
}


fun bbExponTest1() {
    // f: 2e(xy)
    // value: x=1, y=5
    // expected gradient: d/dx = 10e(5) = 1484.131591025766, d/dy = 2e(5) = 296.8263182051532
    // expected value: 2e(5) = 296.8263182051532
    val vals = doubleArrayOf(1.0, 5.0)

    val xTerm = LinearBaseTerm(2)
    xTerm.addTerm(0, 1.0)

    val yTerm = LinearBaseTerm(2)
    yTerm.addTerm(1, 1.0)

    val qTerm = QuadraticTerm(2, xTerm, yTerm, 1.0)

    val eTerm = ExponentialTerm(2, qTerm)

    val fTerm = LinearTerm(2)
    fTerm.addTerm(eTerm, 2.0)

    val gradient = DoubleArray(2) { 0.0 }
    val x = fTerm.evaluate(vals)
    fTerm.chainBackward(vals, gradient, 1.0)

    val gx = gradient[0]
    val gy = gradient[1]

    require(gx == 1484.131591025766)
    require(gy == 296.8263182051532)
    require(x == 296.8263182051532)
    println("exponTest1 OK")
}



fun bbExponTest2() {
    // f: 3.3e(xy + x + 2y - 1)
    // value: x=3, y=-2
    // expected gradient: d/dx = 3.3(y + 1)e(xy + x + 2y - 1) = -3.3e(-8) = -0.001107026672078289
    //                    d/dy = 3.3(x + 2)e(-8) = 16.5e(-8) = 0.005535133360391445
    // expected value: 3.3e(-6 + 3 -4 - 1) = 3.3e(-8) = 0.001107026672078289
    val vals = doubleArrayOf(3.0, -2.0)

    val xTerm = LinearBaseTerm(2)
    xTerm.addTerm(0, 1.0)

    val yTerm = LinearBaseTerm(2)
    yTerm.addTerm(1, 1.0)

    val qTerm = QuadraticTerm(2, xTerm, yTerm, 1.0)

    val exponent = LinearTerm(2)
    exponent.addTerm(qTerm, 1.0)
    exponent.addTerm(xTerm, 1.0)
    exponent.addTerm(yTerm, 2.0)
    exponent.addConstant(-1.0)

    val eTerm = ExponentialTerm(2, exponent)

    val fTerm = LinearTerm(2)
    fTerm.addTerm(eTerm, 3.3)

    val gradient = DoubleArray(2) { 0.0 }
    val x = fTerm.evaluate(vals)
    fTerm.chainBackward(vals, gradient, 1.0)

    val gx = gradient[0]
    val gy = gradient[1]

    require(gx == -0.001107026672078289)
    require(gy == 0.005535133360391445)
    require(x == 0.001107026672078289)
    println("exponTest2 OK")
}


fun bbDivisionTermTest1() {
    // f: x/y
    // value: x=3.3, y=-1.2
    // expected gradient: d/dx = 1/-1.2 = -0.8333333333333334
    //                    d/dy = -x/(y*y) = -3.3/(1.44) = -2.2916666666666665
    // expected value: -2.75
    val vals = doubleArrayOf(3.3, -1.2)

    val xTerm = LinearBaseTerm(2)
    xTerm.addTerm(0, 1.0)

    val yTerm = LinearBaseTerm(2)
    yTerm.addTerm(1, 1.0)

    val dTerm = DivisionTerm(2, xTerm, yTerm)

    val gradient = DoubleArray(2) { 0.0 }
    val x = dTerm.evaluate(vals)
    dTerm.chainBackward(vals, gradient, 1.0)

    val gx = gradient[0]
    val gy = gradient[1]

    require(gx == -0.8333333333333334)
    require(gy == -2.2916666666666665)
    require(x == -2.75)
    println("divisionTermTest1 OK")
}

fun bbDivisionTermTest2() {
    // f: x/(1x+2y)
    // value: x=2, y=-3
    // expected gradient: d/dx = (x + 2y - x) / (1x+2y)**2 = -6 / 16
    //                    d/dy = -2x / (1x+2y)**2 = 4 / 16 = -1/4 = -0.25
    // expected value: 2 / -4 = -0.5
    val vals = doubleArrayOf(2.0, -3.0)

    val dividend = LinearBaseTerm(2)
    dividend.addTerm(0, 1.0)

    val divisor = LinearBaseTerm(2)
    divisor.addTerm(0, 1.0)
    divisor.addTerm(1, 2.0)

    val dTerm = DivisionTerm(2, dividend, divisor)

    val gradient = DoubleArray(2) { 0.0 }
    val x = dTerm.evaluate(vals)
    dTerm.chainBackward(vals, gradient, 1.0)

    val gx = gradient[0]
    val gy = gradient[1]

    require(gx == (-6.0 / 16.0))
    require(gy == -0.25)
    require(x ==  -0.5)
    println("divisionTermTest2 OK")
}


fun linearTest1() {
    // f: 3x + 1y
    // value: x=1, y=1
    // expected gradient: d/dx = 3, d/dy = 1
    // expected value: 4
    val vals = doubleArrayOf(1.0, 1.0)

    val term = LinearBaseTerm(2)

    term.addTerm(0, 3.0)
    term.addTerm(1, 1.0)

    val gx = term.gradient(0, vals)
    term.clearGradientCache()
    val gy = term.gradient(1, vals)
    term.clearGradientCache()
    val x = term.evaluate(vals)

    require(gx == 3.0)
    require(gy == 1.0)
    require(x == 4.0)
    println("linearTest1 OK")
}

fun quadraticTest1() {
    // f: 3xy
    // value: x=2, y=5
    // expected gradient: d/dx = 15, d/dy = 6
    // expected value: 30
    val vals = doubleArrayOf(2.0, 5.0)

    val xTerm = LinearBaseTerm(2)
    xTerm.addTerm(0, 1.0)

    val yTerm = LinearBaseTerm(2)
    yTerm.addTerm(1, 1.0)

    val qTerm = QuadraticTerm(2, xTerm, yTerm, 3.0)

    val gx = qTerm.gradient(0, vals)
    qTerm.clearGradientCache()
    val gy = qTerm.gradient(1, vals)
    qTerm.clearGradientCache()
    val x = qTerm.evaluate(vals)

    require(gx == 15.0)
    require(gy == 6.0)
    require(x == 30.0)
    println("quadraticTest1 OK")
}

fun quadraticTest2() {
    // f: 1.5xx
    // value: x=0.3
    // expected gradient: d/dx = 0.9, d/dy = 0.0
    // expected value: 0.13499999999999998
    val vals = doubleArrayOf(0.3, 100.0)

    val xTerm = LinearBaseTerm(2)
    xTerm.addTerm(0, 1.0)

    val qTerm = QuadraticTerm(2, xTerm, xTerm, 1.5)

    val gx = qTerm.gradient(0, vals)
    qTerm.clearGradientCache()
    val gy = qTerm.gradient(1, vals)
    qTerm.clearGradientCache()
    val x = qTerm.evaluate(vals)

    require(gx == 3 * 0.3)
    require(gy == 0.0)
    require(x == 0.3 * 0.3 * 1.5)
    println("quadraticTest2 OK")
}

fun exponTest1() {
    // f: 2e(xy)
    // value: x=1, y=5
    // expected gradient: d/dx = 10e(5) = 1484.131591025766, d/dy = 2e(5) = 296.8263182051532
    // expected value: 2e(5) = 296.8263182051532
    val vals = doubleArrayOf(1.0, 5.0)

    val xTerm = LinearBaseTerm(2)
    xTerm.addTerm(0, 1.0)

    val yTerm = LinearBaseTerm(2)
    yTerm.addTerm(1, 1.0)

    val qTerm = QuadraticTerm(2, xTerm, yTerm, 1.0)

    val eTerm = ExponentialTerm(2, qTerm)

    val fTerm = LinearTerm(2)
    fTerm.addTerm(eTerm, 2.0)

    val gx = fTerm.gradient(0, vals)
    fTerm.clearGradientCache()
    val gy = fTerm.gradient(1, vals)
    fTerm.clearGradientCache()
    val x = fTerm.evaluate(vals)

    require(gx == 1484.131591025766)
    require(gy == 296.8263182051532)
    require(x == 296.8263182051532)
    println("exponTest1 OK")
}


fun exponTest2() {
    // f: 3.3e(xy + x + 2y - 1)
    // value: x=3, y=-2
    // expected gradient: d/dx = 3.3(y + 1)e(xy + x + 2y - 1) = -3.3e(-8) = -0.001107026672078289
    //                    d/dy = 3.3(x + 2)e(-8) = 16.5e(-8) = 0.005535133360391445
    // expected value: 3.3e(-6 + 3 -4 - 1) = 3.3e(-8) = 0.001107026672078289
    val vals = doubleArrayOf(3.0, -2.0)

    val xTerm = LinearBaseTerm(2)
    xTerm.addTerm(0, 1.0)

    val yTerm = LinearBaseTerm(2)
    yTerm.addTerm(1, 1.0)

    val qTerm = QuadraticTerm(2, xTerm, yTerm, 1.0)

    val exponent = LinearTerm(2)
    exponent.addTerm(qTerm, 1.0)
    exponent.addTerm(xTerm, 1.0)
    exponent.addTerm(yTerm, 2.0)
    exponent.addConstant(-1.0)

    val eTerm = ExponentialTerm(2, exponent)

    val fTerm = LinearTerm(2)
    fTerm.addTerm(eTerm, 3.3)

    val gx = fTerm.gradient(0, vals)
    fTerm.clearGradientCache()
    val gy = fTerm.gradient(1, vals)
    fTerm.clearGradientCache()
    val x = fTerm.evaluate(vals)

    require(gx == -0.001107026672078289)
    require(gy == 0.005535133360391445)
    require(x == 0.001107026672078289)
    println("exponTest2 OK")
}

fun divisionTermTest1() {
    // f: x/y
    // value: x=3.3, y=-1.2
    // expected gradient: d/dx = 1/-1.2 = -0.8333333333333334
    //                    d/dy = -x/(y*y) = -3.3/(1.44) = -2.2916666666666665
    // expected value: -2.75
    val vals = doubleArrayOf(3.3, -1.2)

    val xTerm = LinearBaseTerm(2)
    xTerm.addTerm(0, 1.0)

    val yTerm = LinearBaseTerm(2)
    yTerm.addTerm(1, 1.0)

    val dTerm = DivisionTerm(2, xTerm, yTerm)

    val gx = dTerm.gradient(0, vals)
    dTerm.clearGradientCache()
    val gy = dTerm.gradient(1, vals)
    dTerm.clearGradientCache()
    val x = dTerm.evaluate(vals)

    require(gx == -0.8333333333333334)
    require(gy == -2.2916666666666665)
    require(x == -2.75)
    println("divisionTermTest1 OK")
}

fun divisionTermTest2() {
    // f: x/(1x+2y)
    // value: x=2, y=-3
    // expected gradient: d/dx = (x + 2y - x) / (1x+2y)**2 = -6 / 16
    //                    d/dy = -2x / (1x+2y)**2 = 4 / 16 = -1/4 = -0.25
    // expected value: 2 / -4 = -0.5
    val vals = doubleArrayOf(2.0, -3.0)

    val dividend = LinearBaseTerm(2)
    dividend.addTerm(0, 1.0)

    val divisor = LinearBaseTerm(2)
    divisor.addTerm(0, 1.0)
    divisor.addTerm(1, 2.0)

    val dTerm = DivisionTerm(2, dividend, divisor)

    val gx = dTerm.gradient(0, vals)
    dTerm.clearGradientCache()
    val gy = dTerm.gradient(1, vals)
    dTerm.clearGradientCache()
    val x = dTerm.evaluate(vals)

    require(gx == (-6.0 / 16.0))
    require(gy == -0.25)
    require(x ==  -0.5)
    println("divisionTermTest2 OK")
}

fun cacheTest1() {
    // f: 2e(xy)
    // value: x=1, y=5
    // expected gradient: d/dx = 10e(5) = 1484.131591025766, d/dy = 2e(5) = 296.8263182051532
    // expected value: 2e(5) = 296.8263182051532
    val vals = doubleArrayOf(1.0, 5.0)

    val xTerm = LinearBaseTerm(2)
    xTerm.addTerm(0, 1.0)

    val yTerm = LinearBaseTerm(2)
    yTerm.addTerm(1, 1.0)

    val qTerm = QuadraticTerm(2, xTerm, yTerm, 1.0)

    val eTerm = ExponentialTerm(2, qTerm)

    val fTerm = LinearTerm(2)
    fTerm.addTerm(eTerm, 2.0)

    val gx = fTerm.gradient(0, vals)
    fTerm.clearGradientCache()
    val gy = fTerm.gradient(1, vals)
    fTerm.clearGradientCache()
    val x = fTerm.evaluate(vals)
    fTerm.clearEvalCache()

    require(gx == 1484.131591025766)
    require(gy == 296.8263182051532)
    require(x == 296.8263182051532)

    // TEST 2

    // f: 2e(xy)
    // value: x=1, y=-1
    // expected gradient: d/dx = -2e(-1) = -0.7357588823428847
    //                    d/dy = 2e(-1) = 0.7357588823428847
    // expected value: 2e(-1) = 0.7357588823428847
    val vals2 = doubleArrayOf(1.0, -1.0)

    val gx2 = fTerm.gradient(0, vals2)
    fTerm.clearGradientCache()
    val gy2 = fTerm.gradient(1, vals2)
    fTerm.clearGradientCache()
    val x2 = fTerm.evaluate(vals)
    fTerm.clearEvalCache()

    require(gx2 == -0.7357588823428847)
    require(gy2 == 0.7357588823428847)
    require(x2  == 0.7357588823428847)

    println("cacheTest1 OK")
}

fun linearRELUTest1() {
    // f: 3 RELU(x) + 1 RELU(y)
    // value: x=1, y=1
    // expected gradient: d/dx = 3, d/dy = 1
    // expected value: 4
    val vals = doubleArrayOf(1.0, 1.0)

    val term = LinearBaseRELUTerm(2)

    term.addTerm(0, 3.0)
    term.addTerm(1, 1.0)

    val gx = term.gradient(0, vals)
    term.clearGradientCache()
    val gy = term.gradient(1, vals)
    term.clearGradientCache()
    val x = term.evaluate(vals)

    require(gx == 3.0)
    require(gy == 1.0)
    require(x == 4.0)
    println("linearRELUTest1 OK")
}

fun linearRELUTest2() {
    // f: 2.2 RELU(x) + 1 RELU(y)
    // value: x=-1.2, y=1
    // expected gradient: d/dx = 0, d/dy = 1
    // expected value: 1
    val vals = doubleArrayOf(-1.2, 1.0)

    val term = LinearBaseRELUTerm(2)

    term.addTerm(0, 3.0)
    term.addTerm(1, 1.0)

    val gx = term.gradient(0, vals)
    term.clearGradientCache()
    val gy = term.gradient(1, vals)
    term.clearGradientCache()
    val x = term.evaluate(vals)

    require(gx == 0.0)
    require(gy == 1.0)
    require(x == 1.0)
    println("linearRELUTest2 OK")
}

interface Term {
    fun gradient(variable: Int, vals: DoubleArray) : Double
    fun evaluate(vals: DoubleArray) : Double
    fun clearEvalCache()
    fun clearGradientCache()
    fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) // Determine gradients with backpropagation. This version is slower than the forward approach.
}

class DifferentiableModel (
    val nVars: Int
) : Term, smile.util.function.DifferentiableMultivariateFunction {
    var root: Term = LinearBaseTerm(nVars)

    fun setRootTerm(term: Term) {
        root = term
    }

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        root.chainBackward(vals, partials, seed)
    }

    override fun gradient(variable: Int, vals: DoubleArray): Double {
        val result = root.gradient(variable, vals)
        root.clearGradientCache()
        return result
    }

    override fun evaluate(vals: DoubleArray): Double {
        val result = root.evaluate(vals)
        root.clearEvalCache()
        return result
    }

    override fun clearEvalCache() {
        root.clearEvalCache()
    }

    override fun clearGradientCache() {
        root.clearGradientCache()
    }

    // SMILE interface
    override fun f(p0: DoubleArray?): Double {
        return evaluate(p0!!)
    }

    override fun g(x: DoubleArray?, gradient: DoubleArray?): Double {
        val result = evaluate(x!!)
        for (i in 0 until nVars) {
            gradient!![i] = gradient(i, x)
        }
        clearGradientCache()
        clearEvalCache()
        return result
    }
}

class LinearTerm(nVars: Int): Term {
    val terms = mutableListOf<Term>()
    val coefficients = mutableListOf<Double>()
    var intercept = 0.0

    var evalCacheHot = false
    var evalCache: Double = 0.0

    var gradientCacheHot = false
    var gradientCache = 0.0

    fun addTerm(term: Term, coefficient: Double) {
        terms.add(term)
        coefficients.add(coefficient)
    }

    fun addConstant(value: Double) {
        intercept += value
    }

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        for (i in terms.indices) {
            terms[i].chainBackward(vals, partials, seed * coefficients[i])
        }
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        if (gradientCacheHot) {
            return gradientCache
        }
        var result = 0.0
        for (i in terms.indices) {
            result += terms[i].gradient(variable, vals) * coefficients[i]
        }
        gradientCacheHot = true
        gradientCache = result
        return result
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCacheHot) {
            return evalCache
        }
        var result = intercept
        for (i in terms.indices) {
            result += terms[i].evaluate(vals) * coefficients[i]
        }
        evalCache = result
        evalCacheHot = true
        return result
    }

    override fun clearEvalCache() {
        if (evalCacheHot) {
            evalCacheHot = false
            for (term in terms) {
                term.clearEvalCache()
            }
        }
    }

    override fun clearGradientCache() {
        if (gradientCacheHot) {
            gradientCacheHot = false
            for (term in terms) {
                term.clearGradientCache()
            }
        }
    }
}

class QuadraticTerm(
    nVars: Int,
    val termA: Term,
    val termB: Term,
    val coefficient: Double,
): Term {
    var evalCacheHot = false
    var evalCache: Double = 0.0

    var gradientCacheHot = false
    var gradientCache = 0.0

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        if (gradientCacheHot) {
            return gradientCache
        }
        val result = termA.evaluate(vals) * termB.gradient(variable, vals) * coefficient +
                     termB.evaluate(vals) * termA.gradient(variable, vals) * coefficient
        gradientCache = result
        gradientCacheHot = true
        return result
    }

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        termA.chainBackward(vals, partials, seed * termB.evaluate(vals)* coefficient)
        termB.chainBackward(vals, partials, seed * termA.evaluate(vals)* coefficient)
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCacheHot) {
            return evalCache
        }
        val result = termA.evaluate(vals) * termB.evaluate(vals) * coefficient
        evalCacheHot = true
        evalCache = result
        return result
    }

    override fun clearEvalCache() {
        if (evalCacheHot) {
           evalCacheHot = false
           termA.clearEvalCache()
           termB.clearEvalCache()
        }
    }

    override fun clearGradientCache() {
        if (gradientCacheHot) {
            gradientCacheHot = false
            termA.clearGradientCache()
            termB.clearGradientCache()
        }
    }
}

class ExponentialTerm(
    nVars: Int,
    val exponent: Term
): Term {
    var evalCacheHot = false
    var evalCache: Double = 0.0

    var gradientCacheHot = false
    var gradientCache = 0.0

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        exponent.chainBackward(vals, partials, seed * exp(exponent.evaluate(vals)))
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        if (gradientCacheHot) {
            return gradientCache
        }
        val result = exponent.gradient(variable, vals) * exp(exponent.evaluate(vals))
        gradientCache = result
        gradientCacheHot = true
        return result
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCacheHot) {
            return evalCache
        }
        val result = exp(exponent.evaluate(vals))
        evalCacheHot = true
        evalCache = result
        return result
    }

    override fun clearEvalCache() {
        if (evalCacheHot) {
            evalCacheHot = false
            exponent.clearEvalCache()
        }
    }

    override fun clearGradientCache() {
        if (gradientCacheHot) {
            gradientCacheHot = false
            exponent.clearGradientCache()
        }
    }
}

class DivisionTerm(
    nVars: Int,
    val dividend: Term,
    val divisor: Term
): Term {
    var evalCacheHot = false
    var evalCache: Double = 0.0

    var gradientCacheHot = false
    var gradientCache = 0.0

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        dividend.chainBackward(vals, partials, seed / divisor.evaluate(vals))
        divisor.chainBackward(vals, partials, seed * dividend.evaluate(vals) / - divisor.evaluate(vals).pow(2.0))
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        if (gradientCacheHot) {
            return gradientCache
        }
        val divisorEval = divisor.evaluate(vals)
        val result = (dividend.gradient(variable, vals) * divisorEval - dividend.evaluate(vals) * divisor.gradient(variable, vals)) / (divisorEval * divisorEval)
        gradientCache = result
        gradientCacheHot = true
        return result
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCacheHot) {
            return evalCache
        }
        val result = dividend.evaluate(vals) / divisor.evaluate(vals)
        evalCacheHot = true
        evalCache = result
        return result
    }

    override fun clearEvalCache() {
        if (evalCacheHot) {
            evalCacheHot = false
            dividend.clearEvalCache()
            divisor.clearEvalCache()
        }
    }

    override fun clearGradientCache() {
        if (gradientCacheHot) {
            gradientCacheHot = false
            dividend.clearGradientCache()
            divisor.clearGradientCache()
        }
    }
}


class LinearBaseTerm(nVars: Int): Term {
    val coefficients = DoubleArray(nVars) {0.0}
    var intercept = 0.0
    var evalCacheHot = false
    var evalCache: Double = 0.0


    fun addTerm(variable: Int, coefficient: Double) {
        coefficients[variable] += coefficient
    }

    fun addConstant(value: Double) {
        intercept += value
    }

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        for (i in partials.indices) {
            partials[i] += seed * coefficients[i]
        }
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        return coefficients[variable]
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCacheHot) {
            return evalCache
        }
        var result = intercept
        for (i in coefficients.indices) {
            result += coefficients[i] * vals[i]
        }
        evalCacheHot = true
        evalCache = result
        return result
    }

    override fun clearEvalCache() {
        evalCacheHot = false
    }

    override fun clearGradientCache() { }
}

class LinearBaseRELUTerm(nVars: Int): Term {
    val coefficients = DoubleArray(nVars) {0.0}
    var intercept = 0.0
    var evalCacheHot = false
    var evalCache: Double = 0.0

    fun addTerm(variable: Int, coefficient: Double) {
        coefficients[variable] += coefficient
    }

    fun addConstant(value: Double) {
        intercept += value
    }

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        throw NotImplementedError()
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        return if(vals[variable] <= 0.0) {
            0.0
        } else {
            coefficients[variable]
        }
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCacheHot) {
            return evalCache
        }
        var result = intercept
        for (i in coefficients.indices) {
            if (vals[i] > 0.0) {
                result += coefficients[i] * vals[i]
            }
        }
        evalCacheHot = true
        evalCache = result
        return result
    }

    override fun clearEvalCache() {
        evalCacheHot = false
    }

    override fun clearGradientCache() { }
}


/*
class QuadraticBaseTerm(nVars: Int): Term {
    val lTermCoefficients = DoubleArray(nVars) {0.0}
    val qTermCoefficients = Array(nVars) {
        DoubleArray(nVars) {0.0}
    }
    var intercept = 0.0

    fun addQTerm(varA: Int, varB: Int, coefficient: Double) {
        qTermCoefficients[varA][varB] += coefficient
    }

    fun addLTerm(variable: Int, coefficient: Double) {
        lTermCoefficients[variable] += coefficient
    }

    fun addConstant(value: Double) {
        intercept += value
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        var result = lTermCoefficients[variable]

        for ((i, value) in vals.withIndex()) {
            if (i != variable) {
                result += qTermCoefficients[i][variable] * value
            } else {
                for (j in qTermCoefficients.indices) {
                    if (j != variable) {
                        result += qTermCoefficients[i][j] * vals[j]
                    } else {
                        result += qTermCoefficients[i][j] * value * 2
                    }
                }
            }
        }
        return result
    }

    override fun evaluate(vals: DoubleArray) : Double {
        var result = intercept
        for ((i, valueA) in vals.withIndex()) {
            for ((j, valueB) in vals.withIndex()) {
                result += qTermCoefficients[i][j] * valueA * valueB
            }
        }
        return result
    }
}*/
