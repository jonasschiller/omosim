package de.uniwuerzburg.omod.calibration.algorithms

import alglib.*
import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import kotlin.time.TimeSource
import kotlin.time.measureTime

object MinBc {
    object Defaults {
        const val lb = 0.001
        const val ub = 1e3
    }

    fun run (
        model: DifferentiableModel,
        x0: DoubleArray,
        iterations: Int = 100,
        out: File? = null,
        parameters: Map<String, String>? = null,
    ) : DoubleArray {
        return run(
            model,
            x0,
            iterations,
            out,
            lb = parameters?.get("lb")?.toDoubleOrNull() ?: Defaults.lb,
            ub = parameters?.get("ub")?.toDoubleOrNull() ?: Defaults.ub
        )
    }

    fun run(
        model: DifferentiableModel,
        x0: DoubleArray,
        iterations: Int = 100,
        out: File? = null,
        lb: Double = Defaults.lb,
        ub: Double = Defaults.ub
    ) : DoubleArray {
        val writer = if (out != null) {
            BufferedWriter(FileWriter(out))
        } else {
            null
        }

        val parameterLine = "Parameters:lb=$lb:ub$ub"
        val header = "Iteration,time,Objective Value,Best"
        if (writer != null) {
            writer.write(parameterLine)
            writer.newLine()
            writer.write(header)
            writer.newLine()
        }

        if (writer != null) {
            writer.write("0,0,${model.evaluate(x0)},${model.evaluate(x0)}")
            writer.newLine()
            writer.flush()
        }

        // Create optimizer
        val state = alglib.minbccreate(x0)

        // Bounds
        val l = DoubleArray(model.nVars){ lb }
        val u = DoubleArray(model.nVars){ ub }
        alglib.minbcsetbc(state, l, u)

        // Stopping criteria
        alglib.minbcsetcond(state, 0.0, 0.0, 0.0, iterations)

        // Gradient calculation
        val grad = { x: DoubleArray, grad: DoubleArray, _: Any? ->
            val result = model.evaluate(x)
            model.gradientReverse(x, grad, 1.0)
            result
        }

        // Progress report
        alglib.minbcsetxrep(state, true) // Turn on reporting
        val timer = TimeSource.Monotonic
        var i = 0
        var tLast = timer.markNow()
        val reporter: ( DoubleArray?, Double, Any?) -> Unit  = { xi: DoubleArray?, fi: Double, _: Any? ->
            val now = timer.markNow()
            i += 1
            writer?.write("${i},${now - tLast},${fi},${fi}")
            writer?.newLine()
            writer?.flush()
            tLast = now
        }

        // Run
        val time = measureTime {
            alglib.minbcoptimize(state, grad, reporter, null)
        }

        // Result
        val result = alglib.minbcresults(state)
        val solution = model.evaluate(result.x)
        if (writer != null) {
            writer.write("${iterations},$time,${solution},${solution}")
            writer.newLine()
        }

        writer?.flush()
        writer?.close()
        return result.x
    }
}