package de.uniwuerzburg.omod.calibration.algorithms

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.util.Random
import kotlin.math.abs

class SPSATest {
    val tol = 1e-5

    @Test
    fun testQuadraticFunction() {
        val objective: (DoubleArray) -> Double = { x: DoubleArray ->
            x[0] * x[0]
        }
        val x0 = DoubleArray(1) { 0.1 }
        val xOpt = SPSA.run(x0, objective, Random(), lb=-100.0, ub=100.0, iterations = 10000)
        println(xOpt.toList())
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
        val x0 = DoubleArray(nDimensions) { 0.1 }
        val xOpt = SPSA.run(x0, objective, Random(), lb=-100.0, ub=100.0, iterations = 10000)
        println(xOpt.toList())
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
        val x0 = DoubleArray(nDimensions) { 0.1 }
        val xOpt = SPSA.run(x0, objective, Random(), lb=0.1, ub=100.0, iterations = 10000)
        println(xOpt.toList())
        assert(abs(0.0 - objective(xOpt)) <= tol)
    }
}