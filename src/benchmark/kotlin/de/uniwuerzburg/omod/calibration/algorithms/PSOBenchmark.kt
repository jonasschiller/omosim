package de.uniwuerzburg.omod.calibration.algorithms;

import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import de.uniwuerzburg.omod.calibration.differentiablemodel.LinearTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.Term
import kotlinx.benchmark.Blackhole
import org.openjdk.jmh.annotations.*
import java.lang.Thread.sleep
import java.util.*
import java.util.concurrent.*

@BenchmarkMode(Mode.AverageTime)
@Fork(value = 2)
@Warmup(iterations = 1)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@State(Scope.Benchmark)
class PSOBenchmark {
    var slowQObjective: (DoubleArray) -> Double =  { x: DoubleArray -> 0.0 }

    @Setup
    fun setup() {
        val objective = DifferentiableModel(1)
        val slowTerm = SlowTerm(1)
        val topTerm = LinearTerm(1)
        for (i in 0 until 10) {
            topTerm.addTerm(slowTerm, 1.0)
        }
        objective.setRootTerm(topTerm)
        slowQObjective = { x: DoubleArray ->
            objective.evaluate(x)
        }
    }

    @Benchmark
    fun psoBench(bh: Blackhole) {
        val xOpt = PSO.run(1, slowQObjective, Random(), iterations = 10, nParticles = 40)
        bh.consume(xOpt)
    }
}

class SlowTerm(
    override val nVars: Int
): Term {
    var evalCacheHot = false
    var gradientCacheHot = false
    override var nReceivers: Int = 0


    override fun chainBackward(vals: DoubleArray, partials: DoubleArray, seed: Double) {
        throw NotImplementedError()
    }

    override fun gradient(variable: Int, vals: DoubleArray) : Double {
        if (gradientCacheHot) {
            return 0.0
        }
        sleep(1)
        gradientCacheHot = true
        return 0.0
    }

    override fun evaluate(vals: DoubleArray) : Double {
        if (evalCacheHot) {
            return 0.0
        }
        sleep(1)
        evalCacheHot = true
        return 0.0
    }

    override fun clearEvalCache() {
        if (evalCacheHot) {
            evalCacheHot = false
        }
    }

    override fun clearGradientCache() {
        if (gradientCacheHot) {
            gradientCacheHot = false
        }
    }
}