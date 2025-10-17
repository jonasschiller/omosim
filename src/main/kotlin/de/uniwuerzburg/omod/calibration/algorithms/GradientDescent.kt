package de.uniwuerzburg.omod.calibration.algorithms

import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import kotlin.math.abs
import kotlin.time.measureTime

object GradientDescent {
    object Defaults {
        const val lr0 = 1.0e-8
        const val lb = 0.0
        const val ub = 100.0
        val lrStrategy = LearningRateUpdateStrategy.STATIC
    }

    fun run(
        model: DifferentiableModel,
        x0: DoubleArray,
        iterations: Int = 1000,
        parameters: Map<String, String>? = null,
        out: File? = null
    ) : DoubleArray {
        return run(
            model,
            x0,
            iterations,
            out = out,
            lr0 = parameters?.get("lr0")?.toDoubleOrNull() ?: Defaults.lr0,
            lb = parameters?.get("lb")?.toDoubleOrNull() ?: Defaults.lb,
            ub = parameters?.get("ub")?.toDoubleOrNull() ?: Defaults.ub,
            lrStrategy = parameters?.get("lrStrategy")?.toLearningRateUpdateStrategy() ?: Defaults.lrStrategy
        )
    }

    fun run(
        model: DifferentiableModel,
        x0: DoubleArray,
        iterations: Int = 1000,
        out: File? = null,
        lr0: Double = Defaults.lr0,
        lb: Double = Defaults.lb,
        ub: Double = Defaults.ub,
        lrStrategy: LearningRateUpdateStrategy =  Defaults.lrStrategy,
    ) : DoubleArray {
        val writer = if (out != null) {
            BufferedWriter(FileWriter(out))
        } else {
            null
        }

        val parameterLine = "Parameters:lr0=$lr0:lb=$lb:ub$ub:lrStrategy$lrStrategy"
        val header = "Iteration,time,Objective Value,Best"
        if (writer != null) {
            writer.write(parameterLine)
            writer.newLine()
            writer.write(header)
            writer.newLine()
        }

        // Init
        var lr = lr0
        val x = x0.copyOf()
        var bestX = x0.copyOf()
        var bestLoss = model.evaluate(x0)

        // For Barzilai Borwein learning rate update
        var xPrior = x0.copyOf()
        var gPrior = DoubleArray(x0.size) { 0.0 }

        for (i in 0 until iterations) {
            val g = DoubleArray(x0.size) { 0.0 }
            val time = measureTime {
                // Compute Gradient
                model.gradientReverse(x, g, 1.0)

                // Update step length
                lr = when(lrStrategy) {
                    LearningRateUpdateStrategy.STATIC -> lr
                    LearningRateUpdateStrategy.BARZILAI_BORWEIN -> {
                        val newLr = if (i == 0) {
                            lr
                        } else {
                            barBowUpdate(x, g, xPrior, gPrior)
                        }
                        xPrior = x.copyOf()
                        gPrior = g.copyOf()
                        newLr
                    }
                }

                // Step
                for (j in x.indices) {
                    x[j] -= lr * g[j]
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

    enum class LearningRateUpdateStrategy {
        STATIC, BARZILAI_BORWEIN
    }

    private fun String.toLearningRateUpdateStrategy() : LearningRateUpdateStrategy? {
        for (entry in LearningRateUpdateStrategy.entries) {
            if (this == entry.toString()) {
                return entry
            }
        }
        return null
    }

    fun barBowUpdate(x: DoubleArray, g: DoubleArray, xPrior: DoubleArray, gPrior: DoubleArray) : Double {
        var top = 0.0
        var bot = 0.0
        for (i in x.indices) {
            val dG = (g[i] - gPrior[i])
            top += (x[i] - xPrior[i]) * dG
            bot += dG * dG
        }
        return abs(top) / bot
    }
}


