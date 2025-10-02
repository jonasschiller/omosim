package de.uniwuerzburg.omod.calibration.algorithms

import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import kotlin.time.measureTime

object GradientDescent {
    fun run(
        model: DifferentiableModel,
        x0: DoubleArray,
        iterations: Int = 1000,
        lr: Double = 1.0e-9,
        out: File? = null
    ) : DoubleArray {
        val writer = if (out != null) {
            BufferedWriter(FileWriter(out))
        } else {
            null
        }
        val header = "Iteration, time, Objective Value"
        if (writer != null) {
            writer.write(header)
            writer.newLine()
        }

        val x = x0.copyOf()
        val g = DoubleArray(x0.size) { 0.0 }

        for (i in 0 until iterations) {
            val time = measureTime {
                // Determine gradients
                for (j in x.indices) {
                    g[j] = model.gradient(j, x)
                }

                // Step
                for (j in x.indices) {
                    x[j] -= lr * g[j]
                }
            }
            val oval = model.evaluate(x)
            val line = "$i,$time,$oval"
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
        return x
    }
}