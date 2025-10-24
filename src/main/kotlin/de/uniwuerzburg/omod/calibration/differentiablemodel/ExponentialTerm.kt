package de.uniwuerzburg.omod.calibration.differentiablemodel

import kotlin.math.exp

class ExponentialTerm(
    override val nVars: Int,
    val exponent: Term
): Term {
    private var evalCache: Double? = null
    private var gradientCache = ThreadLocal<Double>()
    override var nReceivers = 0
    private var received = 0
    private var adjoint = 0.0
    override var visited: Boolean = false

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
        if (evalCache != null) {
            return evalCache!!
        }
        val result = exp(exponent.evaluate(vals))
        evalCache = result
        return result
    }

    override fun clearEvalCache() {
        if (!visited) {
            evalCache = null
            exponent.clearEvalCache()
        }
    }

    override fun clearGradientCache() {
        if (!visited) {
            received = 0
            adjoint = 0.0
            gradientCache.set(null)
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

    override fun clearSearchMarkers() {
        if (visited) {
            visited = false
            exponent.clearSearchMarkers()
        }
    }
}