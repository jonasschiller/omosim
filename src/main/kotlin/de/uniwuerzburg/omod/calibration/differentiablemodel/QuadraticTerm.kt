package de.uniwuerzburg.omod.calibration.differentiablemodel

class QuadraticTerm(
    override val nVars: Int,
    val termA: Term,
    val termB: Term,
    val coefficient: Double,
): Term {
    var evalCacheHot = false
    var evalCache: Double = 0.0

    var gradientCacheHot = false
    var gradientCache = 0.0

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        if (gradientCacheHot) {
            return gradientCache
        }
        val result = termA.evaluate(vals) * termB.gradient(variable, vals) * coefficient +
                     termB.evaluate(vals) * termA.gradient(variable, vals) * coefficient
        gradientCache = result
        gradientCacheHot = true
        return result
    }

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        termA.chainBackward(vals, partials, seed * termB.evaluate(vals)* coefficient)
        termB.chainBackward(vals, partials, seed * termA.evaluate(vals)* coefficient)
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCacheHot) {
            return evalCache
        }
        val result = termA.evaluate(vals) * termB.evaluate(vals) * coefficient
        evalCacheHot = true
        evalCache = result
        return result
    }

    override fun clearEvalCache() {
        if (evalCacheHot) {
           evalCacheHot = false
           termA.clearEvalCache()
           termB.clearEvalCache()
        }
    }

    override fun clearGradientCache() {
        if (gradientCacheHot) {
            gradientCacheHot = false
            termA.clearGradientCache()
            termB.clearGradientCache()
        }
    }
}