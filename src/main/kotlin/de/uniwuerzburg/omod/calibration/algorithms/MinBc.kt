package de.uniwuerzburg.omod.calibration.algorithms

import alglib.alglib
import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import kotlin.time.TimeSource

object MinBc {
    private const val NAME = "MinBc"

    object Defaults {
        const val lb = 1e-3
        const val ub = 1e3
    }

    fun run (
        model: DifferentiableModel,
        x0: DoubleArray,
        iterations: Int = 100,
        parameters: Map<String, String>? = null,
    ) : DoubleArray {
        return run(
            model,
            x0,
            iterations,
            lb = parameters?.get("lb")?.toDoubleOrNull() ?: Defaults.lb,
            ub = parameters?.get("ub")?.toDoubleOrNull() ?: Defaults.ub
        )
    }

    fun run(
        model: DifferentiableModel,
        x0: DoubleArray,
        iterations: Int = 100,
        lb: Double = Defaults.lb,
        ub: Double = Defaults.ub
    ) : DoubleArray {
        ProgressLogger.logParameters(this.NAME,"Parameters:lb=$lb:ub$ub")

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
        val reporter: ( DoubleArray?, Double, Any?) -> Unit  = { _: DoubleArray?, fi: Double, _: Any? ->
            val now = timer.markNow()
            i += 1
            ProgressLogger.logProgress(this.NAME, i, now - tLast, fi)
            tLast = now
        }

        ProgressLogger.logInitialLoss(this.NAME, model.evaluate(x0))
        ProgressLogger.logProgressHeader()

        // Run
        alglib.minbcoptimize(state, grad, reporter, null)

        // Result
        val result = alglib.minbcresults(state)
        val solution = model.evaluate(result.x)
        ProgressLogger.logFinalLoss(this.NAME, solution)
        return result.x
    }
}