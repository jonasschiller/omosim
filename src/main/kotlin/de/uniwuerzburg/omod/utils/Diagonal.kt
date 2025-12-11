package de.uniwuerzburg.omod.utils

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set

/**
 * Create diagonal matrix from vector.
 */
inline fun <reified T : Any> D2Array<T>.diagonal() : D2Array<T> {
    require(this.shape[0] == 1)

    val diagonal  = mk.zeros<T>(this.size, this.size)
    for ( i in 0 until this.size) {
        diagonal[i, i] = this[0,i]
    }
    return diagonal
}