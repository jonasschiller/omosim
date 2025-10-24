package de.uniwuerzburg.omod.calibration.differentiablemodel

class QuadraticTerm(
    override val nVars: Int,
    val termA: Term,
    val termB: Term,
    val coefficient: Double,
): Term {
    private var evalCache: Double? = null
    private var gradientCache = ThreadLocal<Double>()
    override var nReceivers = 0
    private var received = 0
    private var adjoint = 0.0
    override var visited: Boolean = false

    override fun gradientReverse(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        // Accumulate adjoint variable
        if (received != nReceivers) {
            adjoint += seed
            received += 1
        }

        // If finalized continue
        if (received == nReceivers) {
            termA.gradientReverse(vals, partials, adjoint * termB.evaluate(vals) * coefficient)
            termB.gradientReverse(vals, partials, adjoint * termA.evaluate(vals) * coefficient)
        }
    }

    override fun gradientForward(variable: Int, vals: DoubleArray) : Double {
        if (gradientCache.get() != null) {
            return gradientCache.get()
        }

        val result = termA.evaluate(vals) * termB.gradientForward(variable, vals) * coefficient +
                     termB.evaluate(vals) * termA.gradientForward(variable, vals) * coefficient
        gradientCache.set(result)
        return result
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCache != null) {
            return evalCache!!
        }
        val result = termA.evaluate(vals) * termB.evaluate(vals) * coefficient
        evalCache = result
        return result
    }

    override fun clearEvalCache() {
       if (!visited) {
           evalCache = null
           termA.clearEvalCache()
           termB.clearEvalCache()
       }
    }

    override fun clearGradientCache() {
        if (!visited) {
            received = 0
            adjoint = 0.0
            gradientCache.set(null)
            termA.clearGradientCache()
            termB.clearGradientCache()
        }
    }

    override fun countReceivers() {
        nReceivers += 1
        if (nReceivers == 1) {
            termA.countReceivers()
            termB.countReceivers()
        }
    }

    override fun clearReceivers() {
        if (nReceivers != 0) {
            nReceivers = 0
            termA.clearReceivers()
            termB.clearReceivers()
        }
        nReceivers = 0
    }

    override fun clearSearchMarkers() {
        if (visited) {
            visited = false
            termA.clearSearchMarkers()
            termB.clearSearchMarkers()
        }
    }
}