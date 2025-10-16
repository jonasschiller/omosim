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
    object Defaults {
        const val lb = 0.0
        const val ub = 100.0
        const val a0 = 1.0
        const val c0 = 1.0
        const val gamma = (1.0 / 3.0)
    }

    fun run(
        x0: DoubleArray,
        objective: (DoubleArray) -> Double,
        rng: Random,
        iterations: Int = 10000,
        out: File? = null,
        parameters: Map<String, String>? = null,
    ) : DoubleArray {
        return run(
            x0 = x0,
            objective = objective,
            rng = rng,
            iterations = iterations,
            out = out,
            lb = parameters?.get("lb")?.toDoubleOrNull() ?: Defaults.lb,
            ub = parameters?.get("ub")?.toDoubleOrNull() ?: Defaults.ub,
            a0 = parameters?.get("a0")?.toDoubleOrNull() ?: Defaults.a0,
            c0 = parameters?.get("c0")?.toDoubleOrNull() ?: Defaults.c0,
            gamma = parameters?.get("gamma")?.toDoubleOrNull() ?: Defaults.gamma,
        )
    }

    fun run(
        x0: DoubleArray,
        objective: (DoubleArray) -> Double,
        rng: Random,
        iterations: Int = 10000,
        out: File? = null,
        lb: Double = Defaults.lb,
        ub: Double = Defaults.ub,
        a0: Double = Defaults.a0,
        c0: Double = Defaults.c0,
        gamma: Double = Defaults.gamma
    ) : DoubleArray {
        println("SPSA...")
        // Store results
        val writer = if (out != null) {
            BufferedWriter(FileWriter(out))
        } else {
            null
        }
        val header = "Iteration,time,Objective Value,Best"
        if (writer != null) {
            writer.write(header)
            writer.newLine()
        }

        val x = x0.copyOf()
        var bestX = x0.copyOf()
        var bestLoss = objective(x0)

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
            val loss = objective(x)

            if (loss < bestLoss) {
                bestX = x.copyOf()
                bestLoss = loss
            }

            val line = "$i,$time,$loss,$bestLoss"
            if (writer != null) {
                writer.write(line)
                writer.newLine()
                writer.flush()
            }
        }
        println("SPSA... Done!")
        writer?.flush()
        writer?.close()
        return bestX
    }
}