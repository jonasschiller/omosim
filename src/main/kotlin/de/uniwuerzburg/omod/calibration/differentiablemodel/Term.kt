package de.uniwuerzburg.omod.calibration.differentiablemodel

interface Term {
    val nVars: Int
    var nReceivers: Int

    fun gradient(variable: Int, vals: DoubleArray) : Double
    fun evaluate(vals: DoubleArray) : Double
    fun clearEvalCache()
    fun clearGradientCache(caller:Term? = null)
    fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) // Determine gradients with backpropagation. This version is slower than the forward approach.
    fun countReceivers(caller:Term?)
}