package de.uniwuerzburg.omod.calibration.algorithms

import de.uniwuerzburg.omod.calibration.algorithms.GradientDescent.LearningRateUpdateStrategy
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.nio.file.Path
import java.nio.file.Paths
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.math.min
import kotlin.time.measureTime

// Based on: https://ieeexplore.ieee.org/abstract/document/488968
// Hyper-parameter: https://ieeexplore.ieee.org/abstract/document/6163405
// Velocity clamping: https://ieeexplore.ieee.org/abstract/document/9680690
// Bound handling: https://ieeexplore.ieee.org/abstract/document/6163405

object PSO {
    object Defaults {
        const val lb = 0.0
        const val ub = 100.0
        const val nParticles = 20
        const val w = 0.8
        const val phiP = 1.6
        const val phiG = 1.6
        const val vClamp = 1.0
        val boundStrategy = BoundStrategy.REFLECT_Z
    }

    fun run (
        nDimensions: Int,
        objective: (DoubleArray) -> Double,
        rng: Random,
        iterations: Int = 1000,
        nWorker: Int? = null,
        out: File? = null,
        parameters: Map<String, String>? = null,
    ) : DoubleArray {
        return run(
            nDimensions,
            objective,
            rng,
            iterations,
            nWorker,
            out,
            lb = parameters?.get("lb")?.toDoubleOrNull() ?: Defaults.lb,
            ub = parameters?.get("ub")?.toDoubleOrNull() ?: Defaults.ub,
            nParticles = parameters?.get("nParticles")?.toIntOrNull() ?: Defaults.nParticles,
            w = parameters?.get("w")?.toDoubleOrNull() ?: Defaults.w,
            phiP = parameters?.get("phiP")?.toDoubleOrNull() ?: Defaults.phiP,
            phiG = parameters?.get("phiG")?.toDoubleOrNull() ?: Defaults.phiG,
            vClamp = parameters?.get("vClamp")?.toDoubleOrNull() ?: Defaults.vClamp,
            boundStrategy = parameters?.get("boundStrategy")?.toBoundStrategy() ?: Defaults.boundStrategy,
        )
    }

    fun run(
        nDimensions: Int,
        objective: (DoubleArray) -> Double,
        rng: Random,
        iterations: Int = 1000,
        nWorker: Int? = null,
        out: File? = null,
        lb: Double = Defaults.lb,
        ub: Double = Defaults.ub,
        nParticles: Int = Defaults.nParticles,
        w: Double = Defaults.w,
        phiP: Double = Defaults.phiP,
        phiG: Double = Defaults.phiG,
        vClamp: Double = Defaults.vClamp,
        boundStrategy: BoundStrategy = Defaults.boundStrategy,
    ) : DoubleArray {
        val writer = if (out != null) {
            BufferedWriter(FileWriter(out))
        } else {
            null
        }

        //println("Initializing PSO...\r")
        val maxVelocity = vClamp * (ub - lb)

        // Initial mse
        var globalBestPosition = DoubleArray(nDimensions) { 1.0 }
        var globalBest = objective(globalBestPosition)

        // Initialize particles
        val particles = List(nParticles) {
            val x = DoubleArray(nDimensions) { rng.nextDouble(lb, ub) }
            val v = DoubleArray(nDimensions) { rng.nextDouble(-maxVelocity, maxVelocity) }
            val oval = objective(x)
            if (oval < globalBest) {
                globalBest = oval
                globalBestPosition = x.copyOf()
            }
            PSOParticle(v, x, x, oval)
        }

        val parameterLine = "Parameters:b=$lb:ub$ub:nParticles$nParticles:w$w:phiP" +
                "$phiP:phiG$phiG:vClamp$vClamp:boundStrategy$boundStrategy"
        val header = "Iteration,time,Objective Value,Best"
        if (writer != null) {
            writer.write(parameterLine)
            writer.newLine()
            writer.write(header)
            writer.newLine()
        }

        for(iteration in 0 until iterations ) {
            val time = measureTime {
                val executor = if (nWorker == null)
                    Executors.newWorkStealingPool()
                else {
                    Executors.newWorkStealingPool(nWorker)
                }
                for (particle in particles) {
                    executor.submit {
                        var inBound = true
                        for (i in 0 until nDimensions) {
                            val rp = rng.nextDouble()
                            val rg = rng.nextDouble()

                            // Update velocity
                            var velocity =
                                w * particle.velocity[i] +
                                        phiP * rp * (particle.bestPosition[i] - particle.position[i]) +
                                        phiG * rg * (globalBestPosition[i] - particle.position[i])

                            // Clamp
                            velocity = min(velocity, maxVelocity)
                            velocity = max(velocity, -maxVelocity)
                            particle.velocity[i] = velocity

                            // Update position
                            particle.position[i] += particle.velocity[i]

                            // Bound handling
                            val dInBound = when(boundStrategy) {
                                BoundStrategy.REFLECT_Z -> bhReflectZ(particle, i, lb, ub, rng)
                                BoundStrategy.INFINITY -> bhInfinity(particle, i, lb, ub)
                            }
                            inBound = inBound && dInBound
                        }

                        if (inBound) {
                            // Check performance
                            val oval = objective(particle.position)

                            if (oval < particle.best) {
                                particle.bestPosition = particle.position.copyOf()
                                particle.best = oval
                            }

                        }
                    }
                }
                executor.shutdown()
                executor.awaitTermination(5, TimeUnit.HOURS) // Wait as long as necessary.

                for (particle in particles) {
                    if (particle.best < globalBest) {
                        globalBestPosition = particle.bestPosition.copyOf()
                        globalBest = particle.best
                    }
                }
            }

            val line = "$iteration,$time,$globalBest,$globalBest"
            if (writer != null) {
                writer.write(line)
                writer.newLine()

                if (iteration % 10 == 0) {
                    writer.flush()
                }
            }
        }
        writer?.flush()
        writer?.close()
        return globalBestPosition
    }

    /**
        Based on:
        doi: 10.1109/TEVC.2012.2189404
    */
    enum class BoundStrategy {
        INFINITY, REFLECT_Z
    }

    private fun String.toBoundStrategy() : BoundStrategy? {
        for (entry in BoundStrategy.entries) {
            if (this == entry.toString()) {
                return entry
            }
        }
        return null
    }

    /**
     * Allow movement out of bounds but assign objective function value of INF
     */
    private fun bhInfinity(
        particle: PSOParticle,
        i: Int,
        lb: Double,
        ub: Double
    ) : Boolean {
        if (particle.position[i] < lb) { return false }
        if (particle.position[i] >= ub) { return false }
        return true
    }

    /**
     * Reflect particle at bound and set velocity to zero.
     */
    private fun bhReflectZ(
        particle: PSOParticle,
        i: Int,
        lb: Double,
        ub: Double,
        rng: Random
    ) : Boolean {
        if (particle.position[i] < lb) {
            particle.position[i] = lb + (lb - particle.position[i])
            particle.velocity[i] = 0.0
        }
        if (particle.position[i] >= ub) {
            particle.position[i] = ub - (particle.position[i] - ub)
            particle.velocity[i] = 0.0
        }
        // If multiple reflections would be necessary randomly place particle
        if ((particle.position[i] < lb) ||  (particle.position[i] > ub)) {
            particle.position[i] = rng.nextDouble(lb, ub)
        }
        return true
    }

    private class PSOParticle(
        var velocity: DoubleArray,
        var position: DoubleArray,
        var bestPosition: DoubleArray,
        var best: Double
    )
}
