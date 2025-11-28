package de.uniwuerzburg.omod.calibration.algorithms

// Bound Projection
fun DoubleArray.project(lb: Double, ub: Double) {
    for (j in this.indices) {
        if (this[j] < lb) {
            this[j] = lb
        }
        if (this[j] > ub) {
            this[j] = ub
        }
    }
}