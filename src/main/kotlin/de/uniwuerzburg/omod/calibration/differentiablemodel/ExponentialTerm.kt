package de.uniwuerzburg.omod.calibration.differentiablemodel

import kotlin.math.exp

class ExponentialTerm(
    override val nVars: Int,
    val exponent: Term
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
            exponent.chainBackward(vals, partials, adjoint * exp(exponent.evaluate(vals)))
        }
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

    override fun clearGradientCache(caller:Term?) {
        if (gradientCache.get() != null) {
            gradientCache.set(null)
            exponent.clearGradientCache(this)
        }
        if ((received != 0) || (adjoint != 0.0)) {
            received = 0
            adjoint = 0.0
            exponent.clearGradientCache(this)
        }
    }

    override fun countReceivers(caller:Term?) {
        nReceivers += 1
        if (nReceivers == 1) {
            exponent.countReceivers(this)

            if (gradientCache.get() != null) {
                print("exp errorA")
            }
            if (evalCache.get() != null) {
                print("exp errorB")
            }
        }
    }

    override fun clearReceivers() {
        if (nReceivers != 0) {
            nReceivers = 0
            exponent.clearReceivers()
        }
        nReceivers = 0
    }
}