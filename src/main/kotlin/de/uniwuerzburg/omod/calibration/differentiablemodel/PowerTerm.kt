package de.uniwuerzburg.omod.calibration.differentiablemodel

import kotlin.math.pow

class PowerTerm(
    override val nVars: Int,
    val base: Term,
    val power: Int
): Term {
    var evalCacheHot = false
    var evalCache: Double = 0.0

    var gradientCacheHot = false
    var gradientCache = 0.0

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        throw NotImplementedError()
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        if (gradientCacheHot) {
            return gradientCache
        }
        val result = power * base.evaluate(vals).pow(power-1) * base.gradient(variable, vals)
        gradientCache = result
        gradientCacheHot = true
        return result
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCacheHot) {
            return evalCache
        }
        val result = base.evaluate(vals).pow(power)
        evalCacheHot = true
        evalCache = result
        return result
    }

    override fun clearEvalCache() {
        if (evalCacheHot) {
            evalCacheHot = false
            base.clearEvalCache()
        }
    }

    override fun clearGradientCache() {
        if (gradientCacheHot) {
            gradientCacheHot = false
            base.clearGradientCache()
        }
    }
}