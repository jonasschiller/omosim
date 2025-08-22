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
    fun hpGridSearch(
        x0: DoubleArray,
        objective: (DoubleArray) -> Double,
        rng: Random,
        outPath: Path,
        lb: Double = 0.00,
        ub: Double = 1e3,
        iterations: Int = 10000,
        a0: List<Double> = listOf(1.0),
        c0: List<Double> = listOf(1.0),
        gamma: List<Double> =  listOf(1.0 / 3.0),
    ) {
        for (a in a0) {
            for (c in c0) {
                for (g in gamma) {
                    val out = Paths.get(
                        outPath.toString(),
                        "SPSA_GS_a${a}_c${c}_gamma${g}.csv"
                    ).toFile()

                    run(
                        x0,
                        objective,
                        rng,
                        lb = lb,
                        ub = ub,
                        iterations= iterations,
                        a0 = a,
                        c0 = c,
                        gamma = g,
                        out = out
                    )
                }
            }
        }
    }

    fun run(
        x0: DoubleArray,
        objective: (DoubleArray) -> Double,
        rng: Random,
        lb: Double = 0.00,
        ub: Double = 1e3,
        iterations: Int = 10000,
        a0: Double = 1.0,
        c0: Double = 1.0,
        gamma: Double = 1.0 / 3.0,
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
            println(line)
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