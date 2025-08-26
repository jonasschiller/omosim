package de.uniwuerzburg.omod.calibration.differentiablemodel

import smile.util.function.DifferentiableMultivariateFunction

class DifferentiableModel (
    override val nVars: Int
) : Term, DifferentiableMultivariateFunction {
    var root: Term = LinearBaseTerm(nVars)

    fun setRootTerm(term: Term) {
        root = term
    }

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        root.chainBackward(vals, partials, seed)
    }

    override fun gradient(variable: Int, vals: DoubleArray): Double {
        val result = root.gradient(variable, vals)
        root.clearGradientCache()
        return result
    }

    override fun evaluate(vals: DoubleArray): Double {
        val result = root.evaluate(vals)
        root.clearEvalCache()
        return result
    }

    override fun clearEvalCache() {
        root.clearEvalCache()
    }

    override fun clearGradientCache() {
        root.clearGradientCache()
    }

    // SMILE interface
    override fun f(p0: DoubleArray?): Double {
        return evaluate(p0!!)
    }

    override fun g(x: DoubleArray?, gradient: DoubleArray?): Double {
        val result = evaluate(x!!)
        for (i in 0 until nVars) {
            gradient!![i] = gradient(i, x)
        }
        clearGradientCache()
        clearEvalCache()
        return result
    }
}