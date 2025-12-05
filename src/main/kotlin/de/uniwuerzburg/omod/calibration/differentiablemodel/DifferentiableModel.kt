package de.uniwuerzburg.omod.calibration.differentiablemodel

import smile.util.function.DifferentiableMultivariateFunction

/**
 * Warning: Don't use within coroutines!! ThreadLocal cache will be unstable. Use an ExecutorService instead.
 */
class DifferentiableModel (
    val nVars: Int
) : DifferentiableMultivariateFunction {
    private var root: Term = LinearBaseTerm(nVars)
    private var visited = ThreadLocal<Boolean>()

    fun setRootTerm(term: Term) {
        root = term

        // Determine value receivers for Reverse mode
        clearReceivers()
        countReceivers()
    }

    fun gradientReverse(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        root.gradientReverse(vals, partials, seed)
        clearGradientCache()
        clearEvalCache()
    }

    fun gradientForward(variable: Int, vals: DoubleArray): Double {
        val result = root.gradientForward(variable, vals)
        clearGradientCache()
        clearEvalCache()
        return result
    }

    fun evaluate(vals: DoubleArray): Double {
        val result = root.evaluate(vals)
        clearEvalCache() // Safer, but slows down reverse mode a bit.
        return result
    }

    fun clearEvalCache() {
        root.clearEvalCache()
        clearSearchMarkers()
    }

    fun clearGradientCache() {
        root.clearGradientCache()
        clearSearchMarkers()
    }

    fun countReceivers() {
        root.countReceivers()
        clearSearchMarkers()
    }

    fun clearReceivers() {
        root.clearReceivers()
        clearSearchMarkers()
    }

    // SMILE interface
    override fun f(p0: DoubleArray?): Double {
        return evaluate(p0!!)
    }

    override fun g(x: DoubleArray?, gradient: DoubleArray?): Double {
        val result = evaluate(x!!)
        gradientReverse(x, gradient!!, 1.0)
        return result
    }

    fun clearSearchMarkers() {
        visited.set(false)
        root.clearSearchMarkers()
    }
}