package de.uniwuerzburg.omod.calibration.algorithms

import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.nio.file.Path
import java.nio.file.Paths
import java.util.*
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.time.measureTime

// TODO test WSPSA, MSPSA

object SPSA {
    fun run(
        x0: DoubleArray,
        objective: (DoubleArray) -> Double,
        rng: Random,
        iterations: Int = 10000,
        parameters: Map<String, String>? = null,
        lb: Double = parameters?.get("lb")?.toDoubleOrNull() ?: 0.0,
        ub: Double = parameters?.get("ub")?.toDoubleOrNull() ?: 1e3,
        a0: Double = parameters?.get("a0")?.toDoubleOrNull() ?: 1.0,
        c0: Double = parameters?.get("c0")?.toDoubleOrNull() ?: 1.0,
        gamma: Double = parameters?.get("gamma")?.toDoubleOrNull() ?: (1.0 / 3.0),
        out: File? = null
    ) : DoubleArray {
        println("SPSA...")
        // Store results
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
        for (i in 1 .. iterations) {
            val time = measureTime {
                val a = a0 / i.toDouble()
                val c = c0 / i.toDouble().pow(gamma)

                val perturbation = DoubleArray(x0.size) { (rng.nextInt(0, 2) * 2 - 1).toDouble() }

                // Gradient estimate
                val xPlus = DoubleArray(x0.size) { j -> max(lb, x[j] + c * perturbation[j]) }
                val jPlus = objective(xPlus)
                val xMinus = DoubleArray(x0.size) { j -> max(lb, x[j] - c * perturbation[j]) }
                val jMinus = objective(xMinus)
                val grad  = DoubleArray(x0.size) { j -> (jPlus - jMinus) / (2 * c * perturbation[j])}

                for (j in 0 until x0.size) {
                    // Gradient descent step
                    var xj = x[j] - a*grad[j]

                    // Bound handling
                    xj = max(lb, xj)
                    xj = min(ub, xj)
                    x[j] = xj
                }
            }
            val oval = objective(x)
            val line = "$i,$time,$oval"
            if (writer != null) {
                writer.write(line)
                writer.newLine()
                writer.flush()
            }
        }
        println("SPSA... Done!")
        writer?.flush()
        writer?.close()
        return x
    }
}