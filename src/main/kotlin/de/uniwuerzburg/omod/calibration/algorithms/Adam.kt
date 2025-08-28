package de.uniwuerzburg.omod.calibration.algorithms

import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.time.measureTime

object Adam {
    fun run(
        model: DifferentiableModel,
        x0: DoubleArray,
        iterations: Int = 1000,
        lr: Double = 1.0e-9,
        b1: Double = 0.9,
        b2: Double = 0.999,
        eps: Double = 1.0e-8,
        out: File? = null
    ) : DoubleArray {
        val writer = if (out != null) {
            BufferedWriter(FileWriter(out))
        } else {
            null
        }

        val x = x0.copyOf()
        val g = DoubleArray(x0.size) { 0.0 }
        val m = DoubleArray(x0.size) { 0.0 }
        val mHat = DoubleArray(x0.size) { 0.0 }
        val v = DoubleArray(x0.size) { 0.0 }
        val vHat = DoubleArray(x0.size) { 0.0 }

        for (i in 0 until iterations) {
            val time = measureTime {
                // Determine gradients
                for (j in x.indices) {
                    g[j] = model.gradient(j, x)
                }

                // Update momentum
                for (j in x.indices) {
                    m[j] = b1 * m[j] + (1 - b1) * g[j]
                    mHat[j] = m[j] / (1-b1.pow(j+1))
                }

                // Update velocity
                for (j in x.indices) {
                    v[j] = b2 * v[j] + (1 - b2) * g[j] * g[j]
                    vHat[j] = v[j] / (1-b2.pow(j+1))
                }

                // Step
                for (j in x.indices) {
                    x[j] -= lr * mHat[j] / (sqrt(vHat[j]) + eps)
                }
            }
            val oval = model.evaluate(x)
            val line = "$i,$time,$oval"
            if (writer != null) {
                writer.write(line)
                writer.newLine()
            }
        }
        writer?.flush()
        writer?.close()
        return x
    }
}

