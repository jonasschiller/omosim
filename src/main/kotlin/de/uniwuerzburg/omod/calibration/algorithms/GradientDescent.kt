package de.uniwuerzburg.omod.calibration.algorithms

import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import kotlin.time.measureTime

object GradientDescent {
    fun run(
        model: DifferentiableModel,
        x0: DoubleArray,
        iterations: Int = 1000,
        lr: Double = 1.0e-6,
        lb: Double = 0.0,
        ub: Double = 100.0,
        dispatcher: CoroutineDispatcher = Dispatchers.Default,
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
        val gF = DoubleArray(x0.size) { 0.0 }

        for (i in 0 until iterations) {
            println(i)
            val time = measureTime {
                // Determine gradients
                runBlocking(dispatcher) {
                    for (j in x.indices) {
                        launch {
                            gF[j] = model.gradient(j, x)
                        }
                    }
                }
                val g = DoubleArray(x0.size) { 0.0 }
                //model.evaluate(x)
                //println("TEst")
                model.clearGradientCache()
                model.clearEvalCache()
                model.chainBackward(x, g, 1.0)
                model.clearGradientCache()
                model.clearEvalCache()
                println(gF.toList().take(10))
                println(g.toList().take(10))
                // Step
                for (j in x.indices) {
                    x[j] -= lr * g[j]
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
            val oval = model.evaluate(x)
            val line = "$i,$time,$oval"
            println(line)
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