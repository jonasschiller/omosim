package de.uniwuerzburg.omod.calibration.differentiablemodel

class QuadraticTerm(
    override val nVars: Int,
    val termA: Term,
    val termB: Term,
    val coefficient: Double,
): Term {
    var evalCache = ThreadLocal<Double>()
    var gradientCache = ThreadLocal<Double>()

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        if (gradientCache.get() != null) {
            return gradientCache.get()
        }
        val result = termA.evaluate(vals) * termB.gradient(variable, vals) * coefficient +
                     termB.evaluate(vals) * termA.gradient(variable, vals) * coefficient
        gradientCache.set(result)
        return result
    }

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        termA.chainBackward(vals, partials, seed * termB.evaluate(vals)* coefficient)
        termB.chainBackward(vals, partials, seed * termA.evaluate(vals)* coefficient)
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCache.get() != null) {
            return evalCache.get()
        }
        val result = termA.evaluate(vals) * termB.evaluate(vals) * coefficient
        evalCache.set(result)
        return result
    }

    override fun clearEvalCache() {
        if (evalCache.get() != null) {
           evalCache.set(null)
           termA.clearEvalCache()
           termB.clearEvalCache()
        }
    }

    override fun clearGradientCache() {
        if (gradientCache.get() != null) {
            gradientCache.set(null)
            termA.clearGradientCache()
            termB.clearGradientCache()
        }
    }
}