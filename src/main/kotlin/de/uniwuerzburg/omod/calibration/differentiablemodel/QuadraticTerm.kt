package de.uniwuerzburg.omod.calibration.differentiablemodel

class QuadraticTerm(
    override val nVars: Int,
    val termA: Term,
    val termB: Term,
    val coefficient: Double,
): Term {
    var evalCache = ThreadLocal<Double>()
    var gradientCache = ThreadLocal<Double>()
    var nReceivers = ThreadLocal<Int>()
    private var received = ThreadLocal<Int>()
    private var adjoint = ThreadLocal<Double>()
    override var visited = ThreadLocal<Boolean>()

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
        if (evalCache.get() != null) {
            return evalCache.get()
        }
        val result = termA.evaluate(vals) * termB.evaluate(vals) * coefficient
        evalCache.set(result)
        return result
    }

    override fun clearEvalCache() {
        if ((visited.get() == null) || (visited.get() == false)) {
           visited.set(true)
           evalCache.set(null)
           termA.clearEvalCache()
           termB.clearEvalCache()
       }
    }

    override fun clearGradientCache() {
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            received.set(0)
            adjoint.set(0.0)
            gradientCache.set(null)
            termA.clearGradientCache()
            termB.clearGradientCache()
        }
    }

    override fun countReceivers() {
        nReceivers.set( (nReceivers.get() ?: 0) + 1)
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            termA.countReceivers()
            termB.countReceivers()
        }
    }

    override fun clearReceivers() {
        if ((visited.get() == null) || (visited.get() == false)) {
            visited.set(true)
            nReceivers.set(0)
            termA.clearReceivers()
            termB.clearReceivers()
        }
    }

    override fun clearSearchMarkers() {
        if ((visited.get() == null) || (visited.get())) {
            visited.set(false)
            termA.clearSearchMarkers()
            termB.clearSearchMarkers()
        }
    }
}