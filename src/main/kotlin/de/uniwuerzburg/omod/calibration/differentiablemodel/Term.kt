package de.uniwuerzburg.omod.calibration.differentiablemodel

interface Term {
    val nVars: Int

    fun gradient(variable: Int, vals: DoubleArray) : Double
    fun evaluate(vals: DoubleArray) : Double
    fun clearEvalCache()
    fun clearGradientCache()
    fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) // Determine gradients with backpropagation. This version is slower than the forward approach.
}