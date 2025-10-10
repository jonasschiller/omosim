package de.uniwuerzburg.omod.calibration.differentiablemodel

import smile.util.function.DifferentiableMultivariateFunction

/**
 * Warning: Don't use within coroutines!! ThreadLocal cache will be unstable. Use an ExecutorService instead.
 */
class DifferentiableModel (
    override val nVars: Int
) : Term, DifferentiableMultivariateFunction {
    var root: Term = LinearBaseTerm(nVars)
    override var nReceivers = 0

    fun setRootTerm(term: Term) {
        root = term

        // Determine value receivers for Reverse mode
        clearReceivers()
        countReceivers()
    }

    override fun gradientReverse(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        root.gradientReverse(vals, partials, seed)
        root.clearGradientCache()
        root.clearEvalCache()
    }

    override fun gradientForward(variable: Int, vals: DoubleArray): Double {
        val result = root.gradientForward(variable, vals)
        root.clearGradientCache()
        root.clearEvalCache()
        return result
    }

    override fun evaluate(vals: DoubleArray): Double {
        val result = root.evaluate(vals)
        root.clearEvalCache() // Safer, but slows down reverse mode a bit.
        return result
    }

    override fun clearEvalCache() {
        root.clearEvalCache()
    }

    override fun clearGradientCache() {
        root.clearGradientCache()
    }

    override fun countReceivers() {
        root.countReceivers()
    }

    override fun clearReceivers() {
        root.clearReceivers()
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
}