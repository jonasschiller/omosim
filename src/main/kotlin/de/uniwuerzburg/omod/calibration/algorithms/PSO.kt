package de.uniwuerzburg.omod.calibration.algorithms

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
import kotlin.math.max
import kotlin.math.min
import kotlin.time.measureTime

// Based on: https://ieeexplore.ieee.org/abstract/document/488968
// Hyper-parameter: https://ieeexplore.ieee.org/abstract/document/6163405
// Velocity clamping: https://ieeexplore.ieee.org/abstract/document/9680690
// Bound handling: https://ieeexplore.ieee.org/abstract/document/6163405

object PSO {
    // Defaults
    const val dNParticles = 20
    const val dW = 0.7
    const val dPhiP = 1.4
    const val dPhiG = 1.4
    const val dVClamp = 0.5
    val dBoundStrategy = BoundStrategy.REFLECT_Z

    fun hpGridSearch(
        nDimensions: Int,
        objective: (DoubleArray) -> Double,
        rng: Random,
        outPath: Path,
        lb: Double = 0.0,
        ub: Double = 1e3,
        nParticles: List<Int> = listOf(dNParticles),
        chi: List<Double> = listOf(dW),
        vClamp: List<Double> = listOf(dVClamp),
        boundStrategy: List<BoundStrategy> = listOf(dBoundStrategy)
    ) {
        for (iVClamp in vClamp) {
            for (iBoundStrategy in boundStrategy) {
                for (iChi in chi) {
                    for (iNParticles in nParticles) {
                        val out = Paths.get(
                            outPath.toString(),
                            "PSO_GS_Clamp${iVClamp}_bs${iBoundStrategy}_chi${iChi}_n${iNParticles}.csv"
                        ).toFile()

                        run(
                            nDimensions,
                            objective,
                            rng,
                            lb = lb,
                            ub = ub,
                            iterations = 1000,
                            nParticles = iNParticles,
                            w = iChi,
                            phiP = iChi * 2,
                            phiG = iChi * 2,
                            vClamp = iVClamp,
                            boundStrategy = iBoundStrategy,
                            out = out
                        )
                    }
                }
            }
        }
    }

    fun run(
        nDimensions: Int,
        objective: (DoubleArray) -> Double,
        rng: Random,
        lb: Double = 0.0,
        ub: Double = 1e3,
        iterations: Int = 1000,
        nParticles: Int = dNParticles,
        w: Double = dW,
        phiP: Double = dPhiP,
        phiG: Double = dPhiG,
        vClamp: Double = dVClamp,
        boundStrategy: BoundStrategy = dBoundStrategy,
        dispatcher: CoroutineDispatcher = Dispatchers.Default,
        out: File? = null
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
        //println("Initializing PSO... done!")

        val header = "Iteration, time, Objective Value, inbound"
        if (writer != null) {
            writer.write(header)
            writer.newLine()
        }

        for(iteration in 0 until iterations ) {
            var nInbound = 0 // Performance indicator for INFINITY bound handling
            val time = measureTime {
                //runBlocking(dispatcher) {
                    for (particle in particles) {
                        //launch {
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
                                    BoundStrategy.INFINITY -> bhInfinity(particle, i, lb, ub, rng)
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

                                nInbound += 1
                            }
                        //}
                    //}
                }

                for (particle in particles) {
                    if (particle.best < globalBest) {
                        globalBestPosition = particle.bestPosition.copyOf()
                        globalBest = particle.best
                        println(globalBestPosition.toList())
                    }
                }
            }

            val oval = objective(globalBestPosition)
            val line = "$iteration,$time,$oval,$nInbound"
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

    /**
     * Allow movement out of bounds but assign objective function value of INF
     */
    private fun bhInfinity(
        particle: PSOParticle,
        i: Int,
        lb: Double,
        ub: Double,
        rng: Random
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
