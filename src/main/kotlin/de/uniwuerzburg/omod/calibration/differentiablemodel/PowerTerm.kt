package de.uniwuerzburg.omod.calibration.differentiablemodel

import kotlin.math.pow

class PowerTerm(
    override val nVars: Int,
    val base: Term,
    val power: Int
): Term {
    var evalCache = ThreadLocal<Double>()
    var gradientCache = ThreadLocal<Double>()

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        throw NotImplementedError()
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        if (gradientCache.get() != null) {
            return gradientCache.get()
        }
        val result = power * base.evaluate(vals).pow(power-1) * base.gradient(variable, vals)
        gradientCache.set(result)
        return result
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCache.get() != null) {
            return evalCache.get()
        }
        val result = base.evaluate(vals).pow(power)
        evalCache.set(result)
        return result
    }

    override fun clearEvalCache() {
        if (evalCache.get() != null) {
            evalCache.set(null)
            base.clearEvalCache()
        }
    }

    override fun clearGradientCache() {
        if (gradientCache.get() != null) {
            gradientCache.set(null)
            base.clearGradientCache()
        }
    }
}