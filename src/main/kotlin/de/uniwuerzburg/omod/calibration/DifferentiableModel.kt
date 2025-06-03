package de.uniwuerzburg.omod.calibration

import javax.sound.sampled.Line
import kotlin.math.exp

fun main () {
    /*val test = DifferentiableModel(2)

    val lTerm = LinearTerm(2)

    val lbTerm = LinearBaseTerm(2)
    lbTerm.addTerm(0, 2.0)
    lbTerm.addTerm(1, 3.0)

    lTerm.addTerm(lbTerm, 2.0)

    val vals = doubleArrayOf(1.0, 1.0)
    println(lTerm.evaluate(vals))
    println(lTerm.gradient(0, vals))
    println(lTerm.gradient(1, vals))*/

    linearTest1()
    quadraticTest1()
    exponTest1()
    exponTest2()
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
    val gy = term.gradient(1, vals)
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


    val qTerm = QuadraticTerm(xTerm, yTerm, 3.0)

    val gx = qTerm.gradient(0, vals)
    val gy = qTerm.gradient(1, vals)
    val x = qTerm.evaluate(vals)

    require(gx == 15.0)
    require(gy == 6.0)
    require(x == 30.0)
    println("quadraticTest1 OK")
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

    val qTerm = QuadraticTerm(xTerm, yTerm, 1.0)

    val eTerm = ExponentialTerm(qTerm)

    val fTerm = LinearTerm(2)
    fTerm.addTerm(eTerm, 2.0)

    val gx = fTerm.gradient(0, vals)
    val gy = fTerm.gradient(1, vals)
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

    val qTerm = QuadraticTerm(xTerm, yTerm, 1.0)

    val exponent = LinearTerm(2)
    exponent.addTerm(qTerm, 1.0)
    exponent.addTerm(xTerm, 1.0)
    exponent.addTerm(yTerm, 2.0)
    exponent.addConstant(-1.0)

    val eTerm = ExponentialTerm(exponent)

    val fTerm = LinearTerm(2)
    fTerm.addTerm(eTerm, 3.3)

    val gx = fTerm.gradient(0, vals)
    val gy = fTerm.gradient(1, vals)
    val x = fTerm.evaluate(vals)

    require(gx == -0.001107026672078289)
    require(gy == 0.005535133360391445)
    require(x == 0.001107026672078289)
    println("exponTest2 OK")
}

interface Term {
    fun gradient(variable: Int, vals: DoubleArray) : Double
    fun evaluate(vals: DoubleArray) : Double
}

class DifferentiableModel (
    val nVars: Int
) : Term {
    var root: Term = LinearBaseTerm(nVars)

    fun setRootTerm(term: Term) {
        root = term
    }

    override fun gradient(variable: Int, vals: DoubleArray): Double {
        return root.gradient(variable, vals)
    }

    override fun evaluate(vals: DoubleArray): Double {
        return root.evaluate(vals)
    }
}

class LinearTerm(nVars: Int): Term {
    val terms = mutableListOf<Term>()
    val coefficients = mutableListOf<Double>()
    var intercept = 0.0

    fun addTerm(term: Term, coefficient: Double) {
        terms.add(term)
        coefficients.add(coefficient)
    }

    fun addConstant(value: Double) {
        intercept += value
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        var result = 0.0
        for (i in terms.indices) {
            result += terms[i].gradient(variable, vals) * coefficients[i]
        }
        return result
    }

    override fun evaluate(vals: DoubleArray) : Double {
        var result = intercept
        for (i in terms.indices) {
            result += terms[i].evaluate(vals) * coefficients[i]
        }
        return result
    }
}

class QuadraticTerm(
    val termA: Term,
    val termB: Term,
    val coefficient: Double,
): Term {
    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        return termA.evaluate(vals) * termB.gradient(variable, vals) * coefficient +
               termB.evaluate(vals) * termA.gradient(variable, vals) * coefficient
    }

    override fun evaluate(vals: DoubleArray) : Double {
        return termA.evaluate(vals) * termB.evaluate(vals) * coefficient
    }
}

class ExponentialTerm(
    val exponent: Term
): Term {
    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        return exponent.gradient(variable, vals) * exp(exponent.evaluate(vals))
    }

    override fun evaluate(vals: DoubleArray) : Double {
        return exp(exponent.evaluate(vals))
    }
}


class LinearBaseTerm(nVars: Int): Term {
    val coefficients = DoubleArray(nVars) {0.0}
    var intercept = 0.0

    fun addTerm(variable: Int, coefficient: Double) {
        coefficients[variable] += coefficient
    }

    fun addConstant(value: Double) {
        intercept += value
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        return coefficients[variable]
    }

    override fun evaluate(vals: DoubleArray) : Double {
        var result = intercept
        for (i in coefficients.indices) {
            result += coefficients[i] * vals[i]
        }
        return result
    }
}

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
}
