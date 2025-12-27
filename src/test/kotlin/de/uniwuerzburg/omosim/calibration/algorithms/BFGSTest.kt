package de.uniwuerzburg.omosim.calibration.algorithms

import org.junit.jupiter.api.Test
import kotlin.math.abs

class BFGSTest {
    val tol = 2e-5

    @Test
    fun testDiffModel() {
        val (objective, model) = TestObjectives.diffModel()
        val x0 = DoubleArray(1) { 1.0 }
        val xOpt = BFGS.run(model, x0, lb=-100.0, ub=100.0)
        assert(abs(0.0 - objective(xOpt)) <= tol)
    }

    @Test
    fun testQuadraticFunction() {
        val (objective, model) = TestObjectives.sphere(1, 0.0)
        val x0 = DoubleArray(1) { 1.0 }
        val xOpt = BFGS.run(model, x0, lb=-100.0, ub=100.0)
        assert(abs(0.0 - objective(xOpt)) <= tol)
    }

    @Test
    fun testSphere() {
        val nDimensions = 10
        val (objective, model) = TestObjectives.sphere(nDimensions, 0.0)
        val x0 = DoubleArray(nDimensions) { 1.0 }
        val xOpt = BFGS.run(model, x0, lb=-100.0, ub=100.0)
        assert(abs(0.0 - objective(xOpt)) <= tol)
    }

    @Test
    fun testShiftedSphere() {
        val nDimensions = 10
        val (objective, model) = TestObjectives.sphere(nDimensions, 10.0)
        val x0 = DoubleArray(nDimensions) { 1.0 }
        val xOpt = BFGS.run(model, x0, lb=-100.0, ub=100.0)
        assert(abs(0.0 - objective(xOpt)) <= tol)
    }
}
