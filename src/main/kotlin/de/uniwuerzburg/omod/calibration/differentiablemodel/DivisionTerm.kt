package de.uniwuerzburg.omod.calibration.differentiablemodel

import kotlin.math.pow

class DivisionTerm(
    override val nVars: Int,
    val dividend: Term,
    val divisor: Term
): Term {
    var evalCache = ThreadLocal<Double>()
    var gradientCache = ThreadLocal<Double>()
    override var nReceivers = 0
    var received = 0
    var adjoint = 0.0

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        if (received != nReceivers) {
            adjoint += seed
            received += 1
        }

        if (received == nReceivers) {
            dividend.chainBackward(vals, partials, adjoint / divisor.evaluate(vals))
            divisor.chainBackward(vals, partials, adjoint * dividend.evaluate(vals) / - divisor.evaluate(vals).pow(2.0))
        }
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        if (gradientCache.get() != null) {
            return gradientCache.get()
        }
        val divisorEval = divisor.evaluate(vals)
        val result = (dividend.gradient(variable, vals) * divisorEval -
                      dividend.evaluate(vals) * divisor.gradient(variable, vals)) / (divisorEval * divisorEval)
        gradientCache.set(result)
        return result
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCache.get() != null) {
            return evalCache.get()
        }
        val result = dividend.evaluate(vals) / divisor.evaluate(vals)
        evalCache.set(result)
        return result
    }

    override fun clearEvalCache() {
        if (evalCache.get() != null) {
            evalCache.set(null)
            dividend.clearEvalCache()
            divisor.clearEvalCache()
        }
    }

    override fun clearGradientCache(caller:Term?) {
        if (gradientCache.get() != null) {
            gradientCache.set(null)
            dividend.clearGradientCache(this)
            divisor.clearGradientCache(this)
        }
        if (received != 0) {
            received = 0
            adjoint = 0.0
            dividend.clearGradientCache(this)
            divisor.clearGradientCache(this)
        }
    }

    override fun countReceivers(caller:Term?) {
        nReceivers += 1
        if (nReceivers == 1) {
            dividend.countReceivers(this)
            divisor.countReceivers(this)
        }
    }
}