package de.uniwuerzburg.omod.calibration.algorithms

import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
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
        lr: Double = 1.0e-6,
        b1: Double = 0.9,
        b2: Double = 0.999,
        eps: Double = 1.0e-8,
        lb: Double = 0.0,
        ub: Double = 100.0,
        out: File? = null
    ) : DoubleArray {
        val writer = if (out != null) {
            BufferedWriter(FileWriter(out))
        } else {
            null
        }
        val header = "Iteration, time, Objective Value, Best"
        if (writer != null) {
            writer.write(header)
            writer.newLine()
        }

        // Init
        val x = x0.copyOf()
        var bestX = x0.copyOf()
        var bestLoss = model.evaluate(x0)

        val m = DoubleArray(x0.size) { 0.0 }
        val mHat = DoubleArray(x0.size) { 0.0 }
        val v = DoubleArray(x0.size) { 0.0 }
        val vHat = DoubleArray(x0.size) { 0.0 }

        for (i in 0 until iterations) {
            val g = DoubleArray(x0.size) { 0.0 }
            val time = measureTime {
                // Compute Gradient
                model.gradientReverse(x, g, 1.0)

                // Update momentum
                for (j in x.indices) {
                    m[j] = b1 * m[j] + (1 - b1) * g[j]
                    mHat[j] = m[j] / (1-b1.pow(i+1))
                }

                // Update velocity
                for (j in x.indices) {
                    v[j] = b2 * v[j] + (1 - b2) * g[j] * g[j]
                    vHat[j] = v[j] / (1-b2.pow(i+1))
                }

                // Step
                for (j in x.indices) {
                    x[j] -= lr * mHat[j] / (sqrt(vHat[j]) + eps)
                }

                // Bounds
                for (j in x.indices) {
                    if (x[j] < lb) {
                        x[j] = lb
                    }
                    if (x[j] > ub) {
                        x[j] = ub
                    }
                }
            }

            // Evaluate
            val loss = model.evaluate(x)

            if (loss < bestLoss) {
                bestX = x.copyOf()
                bestLoss = loss
            }

            val line = "$i,$time,$loss,$bestLoss"
            if (writer != null) {
                writer.write(line)
                writer.newLine()

                if (i % 10 == 0) {
                    writer.flush()
                }
            }
        }
        writer?.flush()
        writer?.close()
        return bestX
    }
}

