package de.uniwuerzburg.omod.calibration.algorithms

import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import smile.math.BFGS

object BFGS {
    fun run(
        model: DifferentiableModel,
        x0: DoubleArray,
        lb: Double = 0.0,
        ub: Double = 1e3,
        m: Int = 5,
        iterations: Int = 30,
        gTol: Double = 1e-5
    ) : DoubleArray {
        val x = x0.copyOf()
        val l = DoubleArray(model.nVars){ lb }
        val u = DoubleArray(model.nVars){ ub }

        println("BFGS: start oval: ${model.evaluate(x0)}")
        val solution = BFGS.minimize(model, m, x, l, u, gTol, iterations)
        println("Solution oval: $solution")
        return x
    }
}