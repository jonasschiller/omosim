package de.uniwuerzburg.omod.calibration.algorithms

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.util.Random
import kotlin.math.abs

class GATest {
    val tol = 1e-3

    @Test
    fun testQuadraticFunction() {
        val objective: (DoubleArray) -> Double = { x: DoubleArray ->
            x[0] * x[0]
        }
        val xOpt = GA.run(1, objective, Random(), lb=-100.0, ub=100.0, iterations = 1000, generationSize = 1000)
        assert(abs(0.0 - objective(xOpt)) <= tol)
    }

    @Test
    fun testSphere() {
        val nDimensions = 10
        val objective: (DoubleArray) -> Double = { x: DoubleArray ->
            var oval = 0.0
            for (i in 0 until nDimensions) {
                oval += x[i] * x[i]
            }
            oval
        }

        val xOpt = GA.run(nDimensions, objective, Random(), lb=-100.0, ub=100.0, iterations = 1000, generationSize = 1000)
        assert(abs(0.0 - objective(xOpt)) <= tol)
    }

    @Test
    fun testShiftedSphere() {
        val nDimensions = 10
        val shift = 10
        val objective: (DoubleArray) -> Double = { x: DoubleArray ->
            var oval = 0.0
            for (i in 0 until nDimensions) {
                oval += (x[i] - shift) * (x[i] - shift)
            }
            oval
        }

        val xOpt = GA.run(nDimensions, objective, Random(), lb=-100.0, ub=100.0, iterations = 1000, generationSize = 1000)
        assert(abs(0.0 - objective(xOpt)) <= tol)
    }
}