package de.uniwuerzburg.omod.calibration.algorithms

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.time.measureTime

object GA {
    object Defaults {
        const val lb = 0.0
        const val ub = 100.0
        const val generationSize = 100
        const val shareSurvivors = 0.4
        const val nEliteOffspring = 5
        const val pR = 0.8
        const val pM = 0.8
        const val pGM = 0.02
        val sigGM = null
    }

    fun run(
        nDimensions: Int,
        objective: (DoubleArray) -> Double,
        rng: Random,
        iterations: Int = 1000,
        nWorker: Int? = null,
        out: File? = null,
        parameters: Map<String, String>? = null,
    ) : DoubleArray {
        return run (
            nDimensions,
            objective,
            rng, iterations,
            nWorker,
            out,
            lb = parameters?.get("lb")?.toDoubleOrNull() ?: Defaults.lb,
            ub = parameters?.get("ub")?.toDoubleOrNull() ?: Defaults.ub,
            generationSize = parameters?.get("generationSize")?.toIntOrNull() ?: Defaults.generationSize,
            shareSurvivors = parameters?.get("shareSurvivors")?.toDoubleOrNull() ?: Defaults.shareSurvivors,
            nEliteOffspring = parameters?.get("nEliteOffspring")?.toIntOrNull() ?: Defaults.nEliteOffspring,
            pR = parameters?.get("pR")?.toDoubleOrNull() ?: Defaults.pR,
            pM = parameters?.get("pM")?.toDoubleOrNull() ?: Defaults.pM,
            pGM = parameters?.get("pGM")?.toDoubleOrNull() ?: Defaults.pGM,
            sigGMarg = parameters?.get("sigGM")?.toDoubleOrNull() ?: Defaults.sigGM,
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
        generationSize: Int = Defaults.generationSize,
        shareSurvivors: Double = Defaults.shareSurvivors,
        nEliteOffspring: Int = Defaults.nEliteOffspring,
        pR: Double = Defaults.pR,
        pM: Double = Defaults.pM,
        pGM: Double = Defaults.pGM,
        sigGMarg: Double? = Defaults.sigGM
    ) : DoubleArray {
        val sigGM = sigGMarg ?: ((ub - lb) / 6.0) // Computed default

        val writer = if (out != null) {
            BufferedWriter(FileWriter(out))
        } else {
            null
        }

        println("Initializing GA...\r")

        var bestX = DoubleArray(nDimensions) { 1.0 }
        var bestLoss = objective(bestX)

        // Initialize candidates
        var currentGeneration = MutableList(generationSize) {
            val x = DoubleArray(nDimensions) { rng.nextDouble(lb, ub) }
            val loss = objective(x)
            Candidate(x, loss)
        }
        println("Initializing GA... done!")

        val parameterLine = "Parameters:lb=$lb:ub$ub:generationSize$generationSize:shareSurvivors" +
                "$shareSurvivors:nEliteOffspring$nEliteOffspring:pR$pR:pM$pM:pGM$pGM:sigGMarg$sigGM"
        val header = "Iteration, time, Objective Value"
        if (writer != null) {
            writer.write(parameterLine)
            writer.newLine()
            writer.write(header)
            writer.newLine()
        }

        val nSurvivors = (shareSurvivors * generationSize).toInt()
        for(iteration in 0 until iterations ) {
            val time = measureTime {
                currentGeneration.sortBy { it.loss }
                val nextGeneration = currentGeneration.take(nSurvivors).toMutableList()
                val fathers = currentGeneration.takeLast(generationSize - nSurvivors)
                val offspring = Array<Candidate?>(fathers.size) { null }

                val executor = if (nWorker == null)
                    Executors.newWorkStealingPool()
                else {
                    Executors.newWorkStealingPool(nWorker)
                }
                for ((i, father) in fathers.withIndex()) {
                    executor.submit{
                        var x = father.x.copyOf()

                        // Recombination
                        if (rng.nextDouble() < pR) {
                            val mother = currentGeneration[rng.nextInt(generationSize)]

                            // Crossovers
                            val cPoint = rng.nextInt(x.size)
                            for (j in cPoint until x.size) {
                                x[j] = mother.x[j]
                            }
                        }

                        // Mutation
                        if (rng.nextDouble() < pM) {
                            for (j in x.indices) {
                                if (rng.nextDouble() < pGM) {
                                    x[j] = x[j] + rng.nextGaussian(0.0, sigGM)
                                }
                            }
                        }

                        // Check if feasible
                        var feasible = true
                        for (j in x.indices) {
                            if (x[j] < lb) {
                                feasible = false
                                break
                            }
                            if (x[j] >= ub) {
                                feasible = false
                                break
                            }
                        }

                        // If not feasible replace with random
                        if (!feasible) {
                            x = DoubleArray(nDimensions) { rng.nextDouble(lb, ub) }
                        }

                        // Evaluate loss
                        val loss = objective(x)

                        val newborn = Candidate(x, loss)
                        offspring[i] = newborn
                    }
                }
                executor.shutdown()
                executor.awaitTermination(5, TimeUnit.HOURS) // Wait as long as necessary.

                val newCandidates = (fathers + offspring.map {it!!}).toMutableList()

                // Elite selection
                newCandidates.sortBy{ it.loss }
                nextGeneration.addAll(newCandidates.take(nEliteOffspring))

                // Tournament
                val tournamentParticipants = newCandidates.takeLast(newCandidates.size - nEliteOffspring).toMutableList()
                while(nextGeneration.size < generationSize) {
                    tournamentParticipants.shuffle(rng)
                    val round = tournamentParticipants.take(3)
                    val winner = round.minBy { it.loss }
                    nextGeneration.add(winner)
                    tournamentParticipants.remove(winner)
                }

                // Generation change
                currentGeneration = nextGeneration

                // Check if new best solution is found
                for (candidate in currentGeneration) {
                    if (candidate.loss < bestLoss) {
                        bestLoss = candidate.loss
                        bestX = candidate.x
                    }
                }
            }

            val line = "$iteration,$time,$bestLoss"
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
        return bestX
    }

    class Candidate(
        val x: DoubleArray,
        var loss: Double
    )
}
