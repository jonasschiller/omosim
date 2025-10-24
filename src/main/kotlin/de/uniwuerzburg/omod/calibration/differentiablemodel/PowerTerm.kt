package de.uniwuerzburg.omod.calibration.differentiablemodel

import kotlin.math.pow

class PowerTerm(
    override val nVars: Int,
    val base: Term,
    val power: Int
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
        if (evalCache != null) {
            return evalCache!!
        }
        val result = base.evaluate(vals).pow(power)
        evalCache = result
        return result
    }

    override fun clearEvalCache() {
        if (!visited) {
            evalCache = null
            base.clearEvalCache()
        }
    }

    override fun clearGradientCache() {
        if (!visited) {
            received = 0
            adjoint = 0.0
            gradientCache.set(null)
            base.clearGradientCache()
        }
    }

    override fun countReceivers() {
        nReceivers += 1
        if (nReceivers == 1) {
            base.countReceivers()
        }
    }

    override fun clearReceivers() {
        if (nReceivers != 0) {
            nReceivers = 0
            base.clearReceivers()
        }
        nReceivers = 0
    }

    override fun clearSearchMarkers() {
        if (visited) {
            visited = false
            base.clearSearchMarkers()
        }
    }
}