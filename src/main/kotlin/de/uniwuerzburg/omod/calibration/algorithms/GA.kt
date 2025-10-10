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
    fun run(
        nDimensions: Int,
        objective: (DoubleArray) -> Double,
        rng: Random,
        iterations: Int = 1000,
        parameters: Map<String, String>? = null,
        lb: Double = parameters?.get("lb")?.toDoubleOrNull() ?: 0.0,
        ub: Double = parameters?.get("ub")?.toDoubleOrNull() ?: 1e3,
        generationSize: Int = parameters?.get("generationSize")?.toIntOrNull() ?: 100,
        shareSurvivors: Double = parameters?.get("shareSurvivors")?.toDoubleOrNull() ?: 0.4,
        nEliteOffspring: Int = parameters?.get("nEliteOffspring")?.toIntOrNull() ?: 5,
        pR: Double = parameters?.get("pR")?.toDoubleOrNull() ?: 0.8,
        pM: Double = parameters?.get("pM")?.toDoubleOrNull() ?: 0.8,
        pGM: Double = parameters?.get("pGM")?.toDoubleOrNull() ?: 0.02,
        sigGM: Double = parameters?.get("sigGM")?.toDoubleOrNull() ?: ((ub - lb) / 6.0),
        nWorker: Int? = null,
        out: File? = null
    ) : DoubleArray {
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

        val header = "Iteration, time, Objective Value"
        if (writer != null) {
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
