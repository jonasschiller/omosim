package de.uniwuerzburg.omod.calibration.algorithms;

import org.openjdk.jmh.annotations.*
import java.lang.Thread.sleep
import java.util.*
import java.util.concurrent.*

@BenchmarkMode(Mode.AverageTime)
@Warmup(iterations = 1)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@State(Scope.Benchmark)
class PSOBenchmark {
    val tol = 1e-5
    val slowQObjective: (DoubleArray) -> Double =  { x: DoubleArray ->
        sleep(1)
        x[0] * x[0]
    }

    @Benchmark
    fun testSlowQuadraticFunction() : DoubleArray {
        val xOpt = PSO.run(1, slowQObjective, Random(), lb = -100.0, ub = 100.0, iterations = 10, nParticles = 40)
        return xOpt
    }
}