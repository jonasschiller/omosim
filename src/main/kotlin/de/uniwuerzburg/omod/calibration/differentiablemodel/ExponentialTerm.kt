package de.uniwuerzburg.omod.calibration.differentiablemodel

import kotlin.math.exp

class ExponentialTerm(
    override val nVars: Int,
    val exponent: Term
): Term {
    var evalCache = ThreadLocal<Double>()
    var gradientCache = ThreadLocal<Double>()
    var nReceivers = ThreadLocal<Int>()
    private var received = ThreadLocal<Int>()
    private var adjoint = ThreadLocal<Double>()
    override var visited = ThreadLocal<Boolean>()

    override fun gradientReverse(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        var adjoint = adjoint.get() ?: 0.0
        var received = received.get() ?: 0

        // Accumulate adjoint variable
        if (received != nReceivers.get()) {
            adjoint += seed
            received += 1

            this.adjoint.set(adjoint)
            this.received.set(received)
        }

        // If finalized continue
        if (received == nReceivers.get()) {
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
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            evalCache.set(null)
            exponent.clearEvalCache()
        }
    }

    override fun clearGradientCache() {
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            received.set(0)
            adjoint.set(0.0)
            gradientCache.set(null)
            exponent.clearGradientCache()
        }
    }

    override fun countReceivers() {
        nReceivers.set( (nReceivers.get() ?: 0) + 1)
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            exponent.countReceivers()
        }
    }

    override fun clearReceivers() {
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            nReceivers.set(0)
            exponent.clearReceivers()
        }
    }

    override fun clearSearchMarkers() {
        if ((visited.get() == null) || (visited.get())) {
            visited.set(false)
            exponent.clearSearchMarkers()
        }
    }
}