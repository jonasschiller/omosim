package de.uniwuerzburg.omod.calibration.algorithms

import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.util.*
import kotlin.time.measureTime

object GA {
    fun run(
        nDimensions: Int,
        objective: (DoubleArray) -> Double,
        rng: Random,
        lb: Double = 0.0,
        ub: Double = 1e3,
        iterations: Int = 1000,
        generationSize: Int = 100,
        shareSurvivors: Double = 0.4,
        nEliteOffspring: Int = 5,
        pR: Double = 0.8,
        pM: Double = 0.8,
        pGM: Double = 0.02,
        sigGM: Double = (ub - lb) / 6.0,
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

        val header = "Iteration, time, Objective Value, inbound"
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
                val offspring = mutableListOf<Candidate>()

                for (father in fathers) {
                    var x = father.x.copyOf()

                    // Recombination
                    if (rng.nextDouble() < pR) {
                        val mother = currentGeneration[rng.nextInt(generationSize)]

                        // Crossovers
                        val cPoint = rng.nextInt(x.size)
                        for (i in cPoint until x.size) {
                            x[i] = mother.x[i]
                        }
                    }

                    // Mutation
                    if (rng.nextDouble() < pM) {
                        for (i in x.indices) {
                            if (rng.nextDouble() < pGM) {
                                x[i] =  x[i] + rng.nextGaussian(0.0, sigGM)
                            }
                        }
                    }

                    // Check if feasible
                    var feasible = true
                    for (i in x.indices) {
                        if (x[i] < lb) {
                            feasible = false
                            break
                        }
                        if (x[i] >= ub) {
                            feasible = false
                            break
                        }
                    }

                    // If not feasible replace with random
                    if(!feasible) {
                        x = DoubleArray(nDimensions) { rng.nextDouble(lb, ub) }
                    }

                    // Evaluate loss
                    val loss = objective(x)

                    val newborn = Candidate(x, loss)
                    offspring.add(newborn)
                }

                val newCandidates = (fathers + offspring).toMutableList()

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
