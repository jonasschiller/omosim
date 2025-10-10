package de.uniwuerzburg.omod.calibration.differentiablemodel

import kotlin.math.exp

class ExponentialTerm(
    override val nVars: Int,
    val exponent: Term
): Term {
    private var evalCache = ThreadLocal<Double>()
    private var gradientCache = ThreadLocal<Double>()
    override var nReceivers = 0
    private var received = 0
    private var adjoint = 0.0

    override fun gradientReverse(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        // Accumulate adjoint variable
        if (received != nReceivers) {
            adjoint += seed
            received += 1
        }

        // If finalized continue
        if (received == nReceivers) {
            exponent.gradientReverse(vals, partials, adjoint * exp(exponent.evaluate(vals)))
        }
    }

    override fun gradientForward(variable: Int, vals: DoubleArray) : Double {
        if (gradientCache.get() != null) {
            return gradientCache.get()
        }
        val result = exponent.gradientForward(variable, vals) * exp(exponent.evaluate(vals))
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
        if ((received != 0) || (adjoint != 0.0)) {
            received = 0
            adjoint = 0.0
            exponent.clearGradientCache()
        }
    }

    override fun countReceivers() {
        nReceivers += 1
        if (nReceivers == 1) {
            exponent.countReceivers()
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