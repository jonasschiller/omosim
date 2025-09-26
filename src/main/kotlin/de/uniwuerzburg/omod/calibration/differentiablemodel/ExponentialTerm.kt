package de.uniwuerzburg.omod.calibration.differentiablemodel

import kotlin.math.exp

class ExponentialTerm(
    override val nVars: Int,
    val exponent: Term
): Term {
    var evalCache = ThreadLocal<Double>()
    var gradientCache = ThreadLocal<Double>()

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        exponent.chainBackward(vals, partials, seed * exp(exponent.evaluate(vals)))
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        if (gradientCache.get() != null) {
            return gradientCache.get()
        }
        val result = exponent.gradient(variable, vals) * exp(exponent.evaluate(vals))
        gradientCache.set(result)
        return result
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCache.get() != null) {
            return evalCache.get()
        }
        val result = exp(exponent.evaluate(vals))
        evalCache.set(result)
        return result
    }

    override fun clearEvalCache() {
        if (evalCache.get() != null) {
            evalCache.set(null)
            exponent.clearEvalCache()
        }
    }

    override fun clearGradientCache() {
        if (gradientCache.get() != null) {
            gradientCache.set(null)
            exponent.clearGradientCache()
        }
    }
}