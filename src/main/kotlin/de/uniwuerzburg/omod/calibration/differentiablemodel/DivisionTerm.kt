package de.uniwuerzburg.omod.calibration.differentiablemodel

import kotlin.math.pow

class DivisionTerm(
    override val nVars: Int,
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
        val result = (dividend.gradient(variable, vals) * divisorEval -
                      dividend.evaluate(vals) * divisor.gradient(variable, vals)) / (divisorEval * divisorEval)
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