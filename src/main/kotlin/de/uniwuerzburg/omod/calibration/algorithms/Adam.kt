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
    object Defaults {
        const val lr = 1.0e-6
        const val b1 = 0.9
        const val b2 = 0.999
        const val eps = 1.0e-8
        const val lb = 0.0
        const val ub = 100.0
    }

    fun run(
        model: DifferentiableModel,
        x0: DoubleArray,
        iterations: Int = 1000,
        out: File? = null,
        parameters: Map<String, String>? = null,
    ) : DoubleArray {
        return run(
            model,
            x0,
            iterations,
            out,
            lr = parameters?.get("lr")?.toDoubleOrNull() ?: Defaults.lr,
            b1 = parameters?.get("b1")?.toDoubleOrNull() ?: Defaults.b1,
            b2 = parameters?.get("b2")?.toDoubleOrNull() ?:  Defaults.b2,
            eps = parameters?.get("eps")?.toDoubleOrNull() ?: Defaults.eps,
            lb = parameters?.get("lb")?.toDoubleOrNull() ?: Defaults.lb,
            ub = parameters?.get("ub")?.toDoubleOrNull() ?: Defaults.ub,
        )
    }

    fun run(
        model: DifferentiableModel,
        x0: DoubleArray,
        iterations: Int = 1000,
        out: File? = null,
        lr: Double = Defaults.lr,
        b1: Double = Defaults.b1,
        b2: Double = Defaults.b2,
        eps: Double = Defaults.eps,
        lb: Double = Defaults.lb,
        ub: Double = Defaults.ub,
    ) : DoubleArray {
        val writer = if (out != null) {
            BufferedWriter(FileWriter(out))
        } else {
            null
        }

        val parameterLine = "Parameters:lr=$lr:lb=$lb:ub$ub:b1$b1:b2$b2:eps$eps"
        val header = "Iteration,time,Objective Value,Best"
        if (writer != null) {
            writer.write(parameterLine)
            writer.newLine()
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

