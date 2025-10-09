package de.uniwuerzburg.omod.calibration.differentiablemodel

class QuadraticTerm(
    override val nVars: Int,
    val termA: Term,
    val termB: Term,
    val coefficient: Double,
): Term {
    var evalCache = ThreadLocal<Double>()
    var gradientCache = ThreadLocal<Double>()
    override var nReceivers = 0
    var received = 0
    var adjoint = 0.0

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
        if (received != nReceivers) {
            adjoint += seed
            received += 1
        }

        if (received == nReceivers) {
            termA.chainBackward(vals, partials, adjoint * termB.evaluate(vals) * coefficient)
            termB.chainBackward(vals, partials, adjoint * termA.evaluate(vals) * coefficient)
        }
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

    override fun clearGradientCache(caller:Term?) {
        if (gradientCache.get() != null) {
            gradientCache.set(null)
            termA.clearGradientCache(this)
            termB.clearGradientCache(this)
        }
        if (received != 0) {
            received = 0
            adjoint = 0.0
            termA.clearGradientCache(this)
            termB.clearGradientCache(this)
        }
    }

    override fun countReceivers(caller:Term?) {
        nReceivers += 1
        if (nReceivers == 1) {
            termA.countReceivers(this)
            termB.countReceivers(this)
        }
    }
}