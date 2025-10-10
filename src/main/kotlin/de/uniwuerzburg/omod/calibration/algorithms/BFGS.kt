package de.uniwuerzburg.omod.calibration.algorithms

import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import smile.math.BFGS
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

object BFGS {
    fun run(
        model: DifferentiableModel,
        x0: DoubleArray,
        iterations: Int = 100,
        parameters: Map<String, String>? = null,
        lb: Double = parameters?.get("lb")?.toDoubleOrNull() ?: 0.0,
        ub: Double = parameters?.get("ub")?.toDoubleOrNull() ?: 1e3,
        m: Int = parameters?.get("m")?.toIntOrNull() ?: 5,
        gTol: Double = parameters?.get("gTol")?.toDoubleOrNull() ?: 1e-5,
        out: File? = null
    ) : DoubleArray {
        val writer = if (out != null) {
            BufferedWriter(FileWriter(out))
        } else {
            null
        }
        val header = "Iteration, Objective Value"
        if (writer != null) {
            writer.write(header)
            writer.newLine()
        }

        val x = x0.copyOf()
        val l = DoubleArray(model.nVars){ lb }
        val u = DoubleArray(model.nVars){ ub }

        if (writer != null) {
            writer.write("0,${model.evaluate(x0)}")
            writer.newLine()
            writer.flush()
        }

        val solution = BFGS.minimize(model, m, x, l, u, gTol, iterations)

        if (writer != null) {
            writer.write("${iterations},${solution}")
            writer.newLine()
        }

        writer?.flush()
        writer?.close()
        return x
    }
}