package de.uniwuerzburg.omod.calibration.algorithms

import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import smile.math.BFGS
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import kotlin.time.measureTime
import kotlin.time.measureTimedValue

object BFGS {
    object Defaults {
        const val m = 5
        const val gTol = 1e-5
        const val lb = 1e-3
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
            ub = parameters?.get("ub")?.toDoubleOrNull() ?: Defaults.ub,
            m = parameters?.get("m")?.toIntOrNull() ?: Defaults.m,
            gTol = parameters?.get("gTol")?.toDoubleOrNull() ?: Defaults.gTol,
        )
    }

    fun run(
        model: DifferentiableModel,
        x0: DoubleArray,
        iterations: Int = 100,
        out: File? = null,
        lb: Double = Defaults.lb,
        ub: Double = Defaults.ub,
        m: Int = Defaults.m,
        gTol: Double = Defaults.gTol,
    ) : DoubleArray {
        val writer = if (out != null) {
            BufferedWriter(FileWriter(out))
        } else {
            null
        }

        val parameterLine = "Parameters:lb=$lb:ub$ub:m$m:gTol$gTol"
        val header = "Iteration,time,Objective Value,Best"
        if (writer != null) {
            writer.write(parameterLine)
            writer.newLine()
            writer.write(header)
            writer.newLine()
        }

        val x = x0.copyOf()
        val l = DoubleArray(model.nVars){ lb }
        val u = DoubleArray(model.nVars){ ub }

        if (writer != null) {
            writer.write("0,0,${model.evaluate(x0)},${model.evaluate(x0)}")
            writer.newLine()
            writer.flush()
        }

        val (solution, time) = measureTimedValue {
            BFGS.minimize(model, m, x, l, u, gTol, iterations)
        }

        if (writer != null) {
            writer.write("${iterations},${time},${solution},${solution}")
            writer.newLine()
        }

        writer?.flush()
        writer?.close()
        return x
    }
}