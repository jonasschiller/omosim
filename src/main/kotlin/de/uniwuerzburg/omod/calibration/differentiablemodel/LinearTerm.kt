package de.uniwuerzburg.omod.calibration.differentiablemodel

class LinearTerm(
    override val nVars: Int
): Term {
    val terms = mutableListOf<Term>()
    val coefficients = mutableListOf<Double>()
    var intercept = 0.0
    override var nReceivers = 0
    var received = 0
    var adjoint = 0.0

    var evalCache = ThreadLocal<Double>()
    var gradientCache = ThreadLocal<Double>()
    val receivers = mutableListOf<Term?>()

    fun addTerm(term: Term, coefficient: Double) {
        terms.add(term)
        coefficients.add(coefficient)
    }

    fun addConstant(value: Double) {
        intercept += value
    }

    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        if (received != nReceivers) {
            adjoint += seed
            received += 1
        }

        if (received == nReceivers) {
            for (i in terms.indices) {
                terms[i].chainBackward(vals, partials, adjoint * coefficients[i])
            }
        }
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        if (gradientCache.get() != null) {
            return gradientCache.get()
        }
        var result = 0.0
        for (i in terms.indices) {
            result += terms[i].gradient(variable, vals) * coefficients[i]
        }
        gradientCache.set(result)
        return result
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCache.get() != null) {
            return evalCache.get()
        }
        var result = intercept
        for (i in terms.indices) {
            result += terms[i].evaluate(vals) * coefficients[i]
        }
        evalCache.set(result)
        return result
    }

    override fun clearEvalCache() {
        if (evalCache.get() != null) {
            evalCache.set(null)
            for (term in terms) {
                term.clearEvalCache()
            }
        }
    }

    override fun clearGradientCache(caller:Term?) {
        if (gradientCache.get() != null) {
            gradientCache.set(null)
            for (term in terms) {
                term.clearGradientCache(this)
            }
        }
        if (received != 0) {
            received = 0
            adjoint = 0.0
            for (term in terms) {
                term.clearGradientCache(this)
            }
        }
    }

    override fun countReceivers(caller:Term?) {
        receivers.add(caller)
        nReceivers += 1

        if (nReceivers == 1) {
            for (term in terms) {
                term.countReceivers(this)
            }
        }
    }
}