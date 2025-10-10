package de.uniwuerzburg.omod.calibration.differentiablemodel

class LinearBaseTerm(
    override val nVars: Int
): Term {
    val coefficients = DoubleArray(nVars) {0.0}
    var intercept = 0.0
    var evalCache = ThreadLocal<Double>()
    override var nReceivers = 0

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
        if (evalCache.get() != null) {
            return evalCache.get()
        }
        var result = intercept
        for (i in coefficients.indices) {
            result += coefficients[i] * vals[i]
        }
        evalCache.set(result)
        return result
    }

    override fun clearEvalCache() {
        evalCache.set(null)
    }

    override fun clearGradientCache(caller:Term?) { }

    override fun countReceivers(caller:Term?) {
        if (evalCache.get() != null) {
            print("errorB")
        }
    }

    override fun clearReceivers() { }
}