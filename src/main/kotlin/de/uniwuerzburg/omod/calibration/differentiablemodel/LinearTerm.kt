package de.uniwuerzburg.omod.calibration.differentiablemodel

class LinearTerm(
    override val nVars: Int
): Term {
    private val terms = mutableListOf<Term>()
    private val coefficients = mutableListOf<Double>()
    private var intercept = 0.0

    var evalCache = ThreadLocal<Double>()
    var gradientCache = ThreadLocal<Double>()
    var nReceivers = ThreadLocal<Int>()
    private var received = ThreadLocal<Int>()
    private var adjoint = ThreadLocal<Double>()
    override var visited = ThreadLocal<Boolean>()

    fun addTerm(term: Term, coefficient: Double) {
        terms.add(term)
        coefficients.add(coefficient)
    }

    fun addConstant(value: Double) {
        intercept += value
    }

    override fun gradientReverse(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        var adjoint = adjoint.get() ?: 0.0
        var received = received.get() ?: 0

        // Accumulate adjoint variable
        if (received != nReceivers.get()) {
            adjoint += seed
            received += 1

            this.adjoint.set( adjoint)
            this.received.set( received)
        }

        // If finalized continue
        if (received == nReceivers.get()) {
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
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            evalCache.set(null)
            for (term in terms) {
                term.clearEvalCache()
            }
        }
    }

    override fun clearGradientCache() {
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            received.set(0)
            adjoint.set(0.0)
            gradientCache.set(null)
            for (term in terms) {
                term.clearGradientCache()
            }
        }
    }

    override fun countReceivers() {
        nReceivers.set( (nReceivers.get() ?: 0) + 1)
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            for (term in terms) {
                term.countReceivers()
            }
        }
    }

    override fun clearReceivers() {
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            nReceivers.set(0)
            for (term in terms) {
                term.clearReceivers()
            }
        }
    }

    override fun clearSearchMarkers() {
        if ((visited.get() == null) || (visited.get())) {
            visited.set(false)
            for (term in terms) {
                term.clearSearchMarkers()
            }
        }
    }
}