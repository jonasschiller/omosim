package de.uniwuerzburg.omod.calibration.differentiablemodel

class LinearBaseTerm(
    override val nVars: Int
): Term {
    val coefficients = DoubleArray(nVars) {0.0}
    var intercept = 0.0
    var evalCacheHot = false
    var evalCache: Double = 0.0

    fun addTerm(variable: Int, coefficient: Double) {
        coefficients[variable] += coefficient
    }

    fun addConstant(value: Double) {
        intercept += value
    }

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        for (i in partials.indices) {
            partials[i] += seed * coefficients[i]
        }
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        return coefficients[variable]
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCacheHot) {
            return evalCache
        }
        var result = intercept
        for (i in coefficients.indices) {
            result += coefficients[i] * vals[i]
        }
        evalCacheHot = true
        evalCache = result
        return result
    }

    override fun clearEvalCache() {
        evalCacheHot = false
    }

    override fun clearGradientCache() { }
}