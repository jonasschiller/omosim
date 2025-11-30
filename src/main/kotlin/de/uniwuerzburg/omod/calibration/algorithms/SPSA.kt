package de.uniwuerzburg.omod.calibration.algorithms

import java.io.File
import java.util.*
import kotlin.math.pow
import kotlin.time.measureTime

object SPSA {
    private const val NAME = "SPSA"

    object Defaults {
        const val lb = 1e-3
        const val ub = 1e3
        const val a0 = 300.0
        const val c0 = 100.0
        const val A = 0.0
        const val gamma = 0.1
        const val alpha = 0.6
    }

    fun run(
        x0: DoubleArray,
        objective: (DoubleArray) -> Double,
        rng: Random,
        iterations: Int = 2000,
        parameters: Map<String, String>? = null,
    ) : DoubleArray {
        return run(
            x0 = x0,
            objective = objective,
            rng = rng,
            iterations = iterations,
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
        rng: Random,
        iterations: Int = 2000,
        lb: Double = Defaults.lb,
        ub: Double = Defaults.ub,
        a0: Double = Defaults.a0,
        c0: Double = Defaults.c0,
        A: Double = Defaults.A,
        gamma: Double = Defaults.gamma,
        alpha: Double = Defaults.alpha
    ) : DoubleArray {
        ProgressLogger.logParameters(this.NAME, "lb=$lb:ub$ub:ao=$a0:c0=$c0:A=$A:gamma=$gamma:alpha=$alpha")

        val x = x0.copyOf()
        var bestX = x0.copyOf()
        var bestLoss = objective(x0)
        ProgressLogger.logInitialLoss(this.NAME, bestLoss)

        ProgressLogger.logProgressHeader()
        for (i in 0 until iterations) {
            val time = measureTime {
                val a = a0 / (A + i + 1).pow(alpha)
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
                val jPlus = objective(xPlus)

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
                val jMinus = objective(xMinus)

                // Gradient estimate
                val grad  = DoubleArray(x0.size) { j -> (jPlus - jMinus) / (2 * c * perturbation[j])}

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

            ProgressLogger.logProgress(this.NAME, i, time, bestLoss)
        }
        ProgressLogger.logFinalLoss(this.NAME, bestLoss)
        return bestX
    }
}