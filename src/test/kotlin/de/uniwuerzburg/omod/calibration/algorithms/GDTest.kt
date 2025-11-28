package de.uniwuerzburg.omod.calibration.algorithms

import org.junit.jupiter.api.Test
import kotlin.math.abs

class GDTest {
    val tol = 2e-5

    @Test
    fun testDiffModel() {
        val (objective, model) = TestObjectives.diffModel()
        val x0 = DoubleArray(1) { 1.0 }
        val xOpt = GradientDescent.run(model, x0, parameters=mapOf("lr0" to "1e-2"))
        assert(abs(0.0 - objective(xOpt)) <= tol)
    }

    @Test
    fun testQuadraticFunction() {
        val (objective, model) = TestObjectives.sphere(1, 0.0)
        val x0 = DoubleArray(1) { 1.0 }
        val xOpt = GradientDescent.run(model, x0, parameters=mapOf("lr0" to "1e-2"))
        assert(abs(0.0 - objective(xOpt)) <= tol)
    }

    @Test
    fun testSphere() {
        val nDimensions = 10
        val (objective, model) = TestObjectives.sphere(nDimensions, 0.0)
        val x0 = DoubleArray(nDimensions) { 1.0 }
        val xOpt = GradientDescent.run(model, x0, parameters=mapOf("lr0" to "1e-2"))
        assert(abs(0.0 - objective(xOpt)) <= tol)
    }

    @Test
    fun testShiftedSphere() {
        val nDimensions = 10
        val (objective, model) = TestObjectives.sphere(nDimensions, 10.0)
        val x0 = DoubleArray(nDimensions) { 1.0 }
        val xOpt = GradientDescent.run(model, x0, parameters=mapOf("lr0" to "1e-2"))
        assert(abs(0.0 - objective(xOpt)) <= tol)
    }
}