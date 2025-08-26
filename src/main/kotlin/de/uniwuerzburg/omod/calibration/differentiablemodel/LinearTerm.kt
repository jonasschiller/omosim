package de.uniwuerzburg.omod.calibration.differentiablemodel

class LinearTerm(
    override val nVars: Int
): Term {
    val terms = mutableListOf<Term>()
    val coefficients = mutableListOf<Double>()
    var intercept = 0.0

    var evalCacheHot = false
    var evalCache: Double = 0.0

    var gradientCacheHot = false
    var gradientCache = 0.0

    fun addTerm(term: Term, coefficient: Double) {
        terms.add(term)
        coefficients.add(coefficient)
    }

    fun addConstant(value: Double) {
        intercept += value
    }

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        for (i in terms.indices) {
            terms[i].chainBackward(vals, partials, seed * coefficients[i])
        }
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        if (gradientCacheHot) {
            return gradientCache
        }
        var result = 0.0
        for (i in terms.indices) {
            result += terms[i].gradient(variable, vals) * coefficients[i]
        }
        gradientCacheHot = true
        gradientCache = result
        return result
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCacheHot) {
            return evalCache
        }
        var result = intercept
        for (i in terms.indices) {
            result += terms[i].evaluate(vals) * coefficients[i]
        }
        evalCache = result
        evalCacheHot = true
        return result
    }

    override fun clearEvalCache() {
        if (evalCacheHot) {
            evalCacheHot = false
            for (term in terms) {
                term.clearEvalCache()
            }
        }
    }

    override fun clearGradientCache() {
        if (gradientCacheHot) {
            gradientCacheHot = false
            for (term in terms) {
                term.clearGradientCache()
            }
        }
    }
}