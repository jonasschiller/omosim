package de.uniwuerzburg.omod.calibration.differentiablemodel

import kotlin.math.exp

class ExponentialTerm(
    override val nVars: Int,
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