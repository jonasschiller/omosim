package de.uniwuerzburg.omod.calibration.differentiablemodel

import kotlin.math.pow

class PowerTerm(
    override val nVars: Int,
    val base: Term,
    val power: Int
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

            this.adjoint.set( adjoint)
            this.received.set( received)
        }

        // If finalized continue
        if (received == nReceivers.get()) {
            base.gradientReverse(vals, partials, adjoint * power * base.evaluate(vals).pow(power-1))
        }
    }

    override fun gradientForward(variable: Int, vals: DoubleArray) : Double {
        if (gradientCache.get() != null) {
            return gradientCache.get()
        }
        val result = power * base.evaluate(vals).pow(power-1) * base.gradientForward(variable, vals)
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
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            evalCache.set(null)
            base.clearEvalCache()
        }
    }

    override fun clearGradientCache() {
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            received.set(0)
            adjoint.set(0.0)
            gradientCache.set(null)
            base.clearGradientCache()
        }
    }

    override fun countReceivers() {
        nReceivers.set( (nReceivers.get() ?: 0) + 1)
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            base.countReceivers()
        }
    }

    override fun clearReceivers() {
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            nReceivers.set(0)
            base.clearReceivers()
        }
    }

    override fun clearSearchMarkers() {
        if ((visited.get() == null) || (visited.get())) {
            visited.set(false)
            base.clearSearchMarkers()
        }
    }
}