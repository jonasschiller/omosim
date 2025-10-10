package de.uniwuerzburg.omod.calibration.algorithms

import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import de.uniwuerzburg.omod.calibration.differentiablemodel.LinearTerm
import kotlinx.coroutines.*
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import kotlin.time.measureTime

object GradientDescent {
    @OptIn(ExperimentalCoroutinesApi::class)
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
        //val gF = DoubleArray(x0.size) { 0.0 }
        for (i in 0 until iterations) {
            val time = measureTime {
                // Determine gradients
                val executor = Executors.newWorkStealingPool()
                val gF = DoubleArray(x0.size) { 0.0 }
                val jobs = MutableList<Future<*>?>(x0.size) {null}
                for (j in x.indices) {
                    jobs[j] = executor.submit {
                        //model.clearEvalCache()
                        //model.clearReceivers()
                        //model.countReceivers(null)
                        gF[j] = model.gradient(j, x)
                        gF[j]
                    }
                }
                executor.shutdown()
                executor.awaitTermination(9999, TimeUnit.SECONDS)
                /*runBlocking(dispatcher.limitedParallelism(1)) {
                    for (j in x.indices) {
                        //val jx = x.copyOf()
                        launch {
                            if (j == 0) {
                                println(x[0])
                                println("${Thread.currentThread()}")
                                println("${(model.root as LinearTerm).gradientCache.get()}")
                            }

                            model.clearEvalCache()
                            model.clearReceivers()
                            //model.countReceivers(null)
                            gF[j] = model.gradient(j, x)

                            if (j == 0) {
                                println(gF[j] )
                                println("${Thread.currentThread()}")
                                println("${(model.root as LinearTerm).gradientCache.get()}")
                            }
                        }
                    }
                }*/
                //val gF = jobs.map { it!!.getCompleted() }.toDoubleArray()
                //println(model.evaluate(x))
                //println("Ye")
                val g = DoubleArray(x0.size) { 0.0 }
                //model.evaluate(x)
                //println("TEst")

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
                println(x[0])
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