package de.uniwuerzburg.omod.calibration.differentiablemodel

import kotlin.math.pow

class DivisionTerm(
    override val nVars: Int,
    val dividend: Term,
    val divisor: Term
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
            dividend.gradientReverse(vals, partials, adjoint / divisor.evaluate(vals))
            divisor.gradientReverse(vals, partials, adjoint * dividend.evaluate(vals) / - divisor.evaluate(vals).pow(2.0))
        }
    }

    override fun gradientForward(variable: Int, vals: DoubleArray) : Double {
        if (gradientCache.get() != null) {
            return gradientCache.get()
        }
        val divisorEval = divisor.evaluate(vals)
        val result = (dividend.gradientForward(variable, vals) * divisorEval -
                      dividend.evaluate(vals) * divisor.gradientForward(variable, vals)) / (divisorEval * divisorEval)
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
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            evalCache.set(null)
            dividend.clearEvalCache()
            divisor.clearEvalCache()
        }
    }

    override fun clearGradientCache() {
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            received.set(0)
            adjoint.set(0.0)
            gradientCache.set(null)
            dividend.clearGradientCache()
            divisor.clearGradientCache()
        }
    }

    override fun countReceivers() {
        nReceivers.set( (nReceivers.get() ?: 0) + 1)
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            dividend.countReceivers()
            divisor.countReceivers()
        }
    }

    override fun clearReceivers() {
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            nReceivers.set(0)
            dividend.clearReceivers()
            divisor.clearReceivers()
        }
    }

    override fun clearSearchMarkers() {
        if ((visited.get() == null) || (visited.get())) {
            visited.set(false)
            dividend.clearSearchMarkers()
            divisor.clearSearchMarkers()
        }
    }
}