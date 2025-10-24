package de.uniwuerzburg.omod.calibration.differentiablemodel

class LinearTerm(
    override val nVars: Int
): Term {
    private val terms = mutableListOf<Term>()
    private val coefficients = mutableListOf<Double>()
    private var intercept = 0.0
    override var nReceivers = 0
    private var received = 0
    private var adjoint = 0.0

    private var evalCache: Double? = null
    private var gradientCache = ThreadLocal<Double>()
    override var visited: Boolean = false

    fun addTerm(term: Term, coefficient: Double) {
        terms.add(term)
        coefficients.add(coefficient)
    }

    fun addConstant(value: Double) {
        intercept += value
    }

    override fun gradientReverse(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        // Accumulate adjoint variable
        if (received != nReceivers) {
            adjoint += seed
            received += 1
        }

        // If finalized continue
        if (received == nReceivers) {
            for (i in terms.indices) {
                terms[i].gradientReverse(vals, partials, adjoint * coefficients[i])
            }
        }
    }

    override fun gradientForward(variable: Int, vals: DoubleArray) : Double {
        if (gradientCache.get() != null) {
            return gradientCache.get()
        }
        var result = 0.0
        for (i in terms.indices) {
            result += terms[i].gradientForward(variable, vals) * coefficients[i]
        }
        gradientCache.set(result)
        return result
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCache != null) {
            return evalCache!!
        }
        var result = intercept
        for (i in terms.indices) {
            result += terms[i].evaluate(vals) * coefficients[i]
        }
        evalCache = result
        return result
    }

    override fun clearEvalCache() {
        if (!visited) {
            evalCache = null
            for (term in terms) {
                term.clearEvalCache()
            }
        }
    }

    override fun clearGradientCache() {
        if (!visited) {
            received = 0
            adjoint = 0.0
            gradientCache.set(null)
            for (term in terms) {
                term.clearGradientCache()
            }
        }
    }

    override fun countReceivers() {
        nReceivers += 1
        if (nReceivers == 1) {
            for (term in terms) {
                term.countReceivers()
            }
        }
    }

    override fun clearReceivers() {
        if (nReceivers != 0) {
            nReceivers = 0
            for (term in terms) {
                term.clearReceivers()
            }
        }
        nReceivers = 0
    }

    override fun clearSearchMarkers() {
        if (visited) {
            visited = false
            for (term in terms) {
                term.clearSearchMarkers()
            }
        }
    }
}