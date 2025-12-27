package de.uniwuerzburg.omosim.utils

fun List<Double>.normalize() : List<Double>? {
    val s = this.sum()
    return if (s == 0.0) {
        null
    } else {
        this.map { it / s }
    }
}

fun DoubleArray.normalize() : DoubleArray? {
    val s = this.sum()
    return if (s == 0.0) {
        null
    } else {
        this.map { it / s }.toDoubleArray()
    }
}