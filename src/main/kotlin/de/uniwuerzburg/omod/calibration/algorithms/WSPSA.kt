package de.uniwuerzburg.omod.calibration.algorithms

import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModelMultiOut
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.nio.file.Path
import java.nio.file.Paths
import java.util.*
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.time.measureTime

// TODO test WSPSA, MSPSA

object WSPSA {
    object Defaults {
        const val lb = 0.0
        const val ub = 100.0
        const val a0 = 1.0
        const val c0 = 1.0
        const val A = 0.0
        const val gamma = 0.1
        const val alpha = 0.6
    }

    fun run(
        x0: DoubleArray,
        objective: (DoubleArray) -> Double,
        measurements: List<Double>,
        model: DifferentiableModelMultiOut,
        rng: Random,
        iterations: Int = 10000,
        out: File? = null,
        parameters: Map<String, String>? = null,
    ) : DoubleArray {
        return run(
            x0 = x0,
            objective = objective,
            measurements = measurements,
            model = model,
            rng = rng,
            iterations = iterations,
            out = out,
            lb = parameters?.get("lb")?.toDoubleOrNull() ?: Defaults.lb,
            ub = parameters?.get("ub")?.toDoubleOrNull() ?: Defaults.ub,
            a0 = parameters?.get("a0")?.toDoubleOrNull() ?: Defaults.a0,
            c0 = parameters?.get("c0")?.toDoubleOrNull() ?: Defaults.c0,
            A = parameters?.get("A")?.toDoubleOrNull() ?: Defaults.A,
            gamma = parameters?.get("gamma")?.toDoubleOrNull() ?: Defaults.gamma,
            alpha = parameters?.get("alpha")?.toDoubleOrNull() ?: Defaults.alpha,
        )
    }

    fun run(
        x0: DoubleArray,
        objective: (DoubleArray) -> Double,
        measurements: List<Double>,
        model: DifferentiableModelMultiOut,
        rng: Random,
        iterations: Int = 1000,
        out: File? = null,
        lb: Double = Defaults.lb,
        ub: Double = Defaults.ub,
        a0: Double = Defaults.a0,
        c0: Double = Defaults.c0,
        A: Double = Defaults.A,
        gamma: Double = Defaults.gamma,
        alpha: Double = Defaults.alpha
    ) : DoubleArray {
        val m = measurements.toDoubleArray()

        // Store results
        val writer = if (out != null) {
            BufferedWriter(FileWriter(out))
        } else {
            null
        }

        val parameterLine = "Parameters:lb=$lb:ub$ub:ao=$a0:c0=$c0:A=$A:gamma=$gamma:alpha=$alpha"
        val header = "Iteration,time,Objective Value,Best"
        if (writer != null) {
            writer.write(parameterLine)
            writer.newLine()
            writer.write(header)
            writer.newLine()
        }

        val x = x0.copyOf()
        var bestX = x0.copyOf()
        var bestLoss = objective(x0)

        for (i in 0 until iterations) {
            val time = measureTime {
                val a = a0 / (A + i + 1).toDouble().pow(alpha)
                val c = c0 / (i + 1).toDouble().pow(gamma)

                val perturbation = DoubleArray(x0.size) { (rng.nextInt(0, 2) * 2 - 1).toDouble() }

                // Plus
                val xPlus = DoubleArray(x0.size) { j -> x[j] + c * perturbation[j] }
                for (j in xPlus.indices) {
                    if (xPlus[j] < lb) {
                        xPlus[j] = lb
                    }
                    if (xPlus[j] > ub) {
                        xPlus[j] = ub
                    }
                }
                val jPlus = squareErrors(xPlus, model, m)

                // Minus
                val xMinus = DoubleArray(x0.size) { j -> x[j] - c * perturbation[j] }
                for (j in xMinus.indices) {
                    if (xMinus[j] < lb) {
                        xMinus[j] = lb
                    }
                    if (xMinus[j] > ub) {
                        xMinus[j] = ub
                    }
                }
                val jMinus = squareErrors(xMinus, model, m)

                // Jacobian
                val jac = model.jacobian(x0)

                // Normalize
                for (j in jac.indices) {
                    for (k in jac[j].indices) {
                        jac[j][k] = abs( jac[j][k] )
                    }
                }
                /*
                for (j in jac.indices) {
                    val sum = jac[j].sum()
                    if (sum == 0.0) { continue }

                    for (k in jac[j].indices) {
                        jac[j][k] /= sum
                    }
                }
                */

                // Gradient estimate
                val grad = gradient(jPlus, jMinus, jac, c, perturbation)

                // Step
                for (j in x.indices) {
                    x[j] -= a*grad[j]
                }

                // Bound Projection
                for (j in x.indices) {
                    if (x[j] < lb) {
                        x[j] = lb
                    }
                    if (x[j] > ub) {
                        x[j] = ub
                    }
                }
            }
            val loss = objective(x)

            if (loss < bestLoss) {
                bestX = x.copyOf()
                bestLoss = loss
            }

            val line = "$i,$time,$loss,$bestLoss"
            println(line)
            if (writer != null) {
                writer.write(line)
                writer.newLine()
                writer.flush()
            }
        }
        writer?.flush()
        writer?.close()
        return bestX
    }

    private fun squareErrors(x: DoubleArray, model: DifferentiableModelMultiOut, m: DoubleArray) : DoubleArray {
        val s = model.evaluate(x)
        var se = DoubleArray(s.size) { 0.0 }
        for (i in s.indices) {
            se[i] = (m[i] - s[i]).pow(2)
        }
        return se
    }

    private fun gradient(
        jPlus: DoubleArray,
        jMinus: DoubleArray,
        wMatrix: Array<DoubleArray>,
        c: Double,
        perturbation: DoubleArray,
    ) : DoubleArray {
        val nVars = wMatrix.first().size
        val nMeasures = wMatrix.size
        val g = DoubleArray ( nVars ) { 0.0 }

        val jDiff = DoubleArray ( nMeasures ) { 0.0 }
        for (j in 0 until nMeasures) {
            jDiff[j] = (jPlus[j] - jMinus[j])
        }

        for (i in 0 until nVars) {
            for (j in 0 until nMeasures) {
                g[i] += wMatrix[j][i] * jDiff[j] / (2 * c * perturbation[i])
            }
        }
        return g
    }
}