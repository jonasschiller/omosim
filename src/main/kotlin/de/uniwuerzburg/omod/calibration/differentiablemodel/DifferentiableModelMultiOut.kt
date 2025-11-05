package de.uniwuerzburg.omod.calibration.differentiablemodel

import smile.util.function.DifferentiableMultivariateFunction

/**
 * Warning: Don't use within coroutines!! ThreadLocal cache will be unstable. Use an ExecutorService instead.
 */
class DifferentiableModelMultiOut (
    val nVars: Int
) {
    var roots: List<Term> = listOf()

    fun setRootTerms(terms: List<Term>) {
        roots = terms
    }

    fun jacobian(vals: DoubleArray) : Array<DoubleArray> {
        val jac = Array(roots.size) { DoubleArray(nVars) { 0.0 } }
        for ((i, root) in roots.withIndex()) {
            root.clearReceivers()
            root.clearSearchMarkers()
            root.countReceivers()
            root.clearSearchMarkers()
            root.gradientReverse(vals, jac[i], 1.0)
            clearGradientCache()
        }
        clearEvalCache()
        return jac
    }

    fun evaluate(vals: DoubleArray): DoubleArray {
        val result = DoubleArray(roots.size) { 0.0 }
        for ((i, root) in roots.withIndex()) {
            result[i] = root.evaluate(vals)
        }
        clearEvalCache()
        return result
    }

    fun clearEvalCache() {
        for (root in roots) {
            root.clearEvalCache()
        }
        clearSearchMarkers()
    }

    fun clearGradientCache() {
        for (root in roots) {
            root.clearGradientCache()
        }
        clearSearchMarkers()
    }

    fun clearSearchMarkers() {
        for (root in roots) {
            root.clearSearchMarkers()
        }
    }
}