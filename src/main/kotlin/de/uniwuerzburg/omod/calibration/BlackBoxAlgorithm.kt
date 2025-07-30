package de.uniwuerzburg.omod.calibration

import de.uniwuerzburg.omod.core.*
import de.uniwuerzburg.omod.core.models.Cell
import de.uniwuerzburg.omod.core.models.PopStratum
import de.uniwuerzburg.omod.core.models.RealLocation
import java.util.*
import kotlin.math.absoluteValue
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.time.measureTime

object BlackBoxAlgorithm {
    private fun runIPF(
        omod: Omod,
        sensors: List<TrafficSensor>,
        modeChoiceCalibration: ModeChoiceCalibration,
        popStrata: List<PopStratum>,
        carOwnership: CarOwnership,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        iterations: Int = 10
    ){
        de.uniwuerzburg.omod.core.logger.on = false // Switch off logger for iterative calibration runs
        println("Start IPF")
        val nDimensions = omod.grid.size
        // Set Parameters
        val finder = omod.destinationFinder as DestinationFinderDefault
        //finder.updateCellCValues(parameters, omod.grid)
        val fullPopulation = omod.buildings.sumOf { it.population }

        // Initial mse
        val (mses, startFlow, _) = determineJointOD(
                omod,
                sensors,
                modeChoiceCalibration,
                popStrata,
                carOwnership,
                affectedLinks,
                Array(nDimensions) { 1.0 }
        )

        println("MSE start: $mses")
        println("------------------------------")
        println("Sensor | \t Flow OMOD | \t Flow Measured")
        for ((i, flow) in startFlow.values.withIndex()) {
            println("${sensors[i].name} | \t $flow | \t ${sensors[i].measuredFlow }")
        }

        val x: MutableMap<Cell, Double> =  omod.grid.zip(Array(nDimensions) { 1.0 }).toMap().toMutableMap()
        println("Start iterations")
        for(iteration in 0 until iterations ) {
            var cnt = 0
            for (sensor in sensors) {
                // Pair probability
                val od = finder.determinePairProbabilities(
                    omod.grid, omod.activityGenerator as ActivityGeneratorDefault,
                    modeChoiceCalibration, x,
                    popStrata, carOwnership
                )
                // TODO find sensor sets that can be seen as a single dimension together

                // Determine affected sensors
                var simCount = 0.0
                val cells = mutableSetOf<Cell>()
                for (origin in omod.grid) {
                    for (destination in omod.grid) {
                        val odPair = Pair(origin, destination)
                        if (odPair in affectedLinks) {
                            val thisSensors = affectedLinks[odPair]!!
                            if (sensor in thisSensors) {
                                simCount += od[odPair]!! * fullPopulation
                                //cells.add(origin)
                                cells.add(destination)
                            }
                        }
                    }
                }

                for (cell in cells) {
                    x[cell] = (x[cell]!! * sensor.measuredFlow) / simCount
                }

                println("Sensor $cnt done")

                cnt += 1
            }
            val tst = Array(nDimensions) { 1.0 }
            for ((i, cell) in omod.grid.withIndex()) {
                tst[i] = x[cell]!!
            }

            val (mse, sFlow, _) = determineJointOD(
                omod,
                sensors,
                modeChoiceCalibration,
                popStrata,
                carOwnership,
                affectedLinks,
                tst
            )
            println("______________________________")
            println("MSE iteration $iteration: $mse")
            println("------------------------------")
            println("Sensor | \t Flow OMOD | \t Flow Measured")
            for ((i, flow) in sFlow.values.withIndex()) {
                println("${sensors[i].name} | \t $flow | \t ${sensors[i].measuredFlow }")
            }
        }
        de.uniwuerzburg.omod.core.logger.on = true
    }

    private fun spoa(
        omod: Omod,
        sensors: List<TrafficSensor>,
        modeChoiceCalibration: ModeChoiceCalibration,
        popStrata: List<PopStratum>,
        carOwnership: CarOwnership,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        iterations: Int = 1000,
        pertbound: Double = 0.1
    ) {
        val rng = Random()
        val nDimensions = omod.grid.size
        val x = Array(nDimensions) { 1.0 }

        val (mse, sFlow, _) = determineJointOD(
            omod,
            sensors,
            modeChoiceCalibration,
            popStrata,
            carOwnership,
            affectedLinks,
            x
        )
        println("______________________________")
        println("MSE iteration $-1: $mse")
        println("------------------------------")
        println("Sensor | \t Flow OMOD | \t Flow Measured")
        for ((k, flow) in sFlow.values.withIndex()) {
            println("${sensors[k].name} | \t $flow | \t ${sensors[k].measuredFlow }")
        }

        for (i in 1 .. iterations) {
            val alpha =  1e-7 / i.toDouble() //
            val beta  =  1e-9 / i.toDouble().pow(1.0 / 3.0) //

            val perturbation = Array(nDimensions) { rng.nextDouble() * pertbound }

            // Check performance
            val xplus = Array(nDimensions) { j -> max(0.0, x[j] + beta * perturbation[j]) }
            val (jplus, _, _) =  determineJointOD(
                omod,
                sensors,
                modeChoiceCalibration,
                popStrata,
                carOwnership,
                affectedLinks,
                xplus
            )

            val xminus = Array(nDimensions) { j -> max(0.0, x[j] - beta * perturbation[j]) }
            val (jminus, _, _) =  determineJointOD(
                omod,
                sensors,
                modeChoiceCalibration,
                popStrata,
                carOwnership,
                affectedLinks,
                xminus
            )

            val lossdiff = jplus - jminus
            val grad  = Array(nDimensions) { j -> lossdiff / (2 * beta * perturbation[j])}

            for (j in 0 until nDimensions) {
                x[j] = max(0.0, x[j] - alpha*grad[j])
            }

            val (mse, sFlow, _) =  determineJointOD(
                omod,
                sensors,
                modeChoiceCalibration,
                popStrata,
                carOwnership,
                affectedLinks,
                x
            )
            println("______________________________")
            println("MSE iteration $i: $mse")
            println("------------------------------")
            println("Sensor | \t Flow OMOD | \t Flow Measured")
            for ((k, flow) in sFlow.values.withIndex()) {
                println("${sensors[k].name} | \t $flow | \t ${sensors[k].measuredFlow }")
            }
        }
    }

    private fun testNShops(
        omod: Omod,
        sensors: List<TrafficSensor>,
        modeChoiceCalibration: ModeChoiceCalibration,
        popStrata: List<PopStratum>,
        carOwnership: CarOwnership,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
    ) : Array<Double> {
        de.uniwuerzburg.omod.core.logger.on = false // Switch off logger for iterative calibration runs
        println("Start GD")
        val nDimensions = omod.grid.size

        val x = Array(nDimensions) { 1.0 }
        val (mse, sFlow, _) =  determineJointOD(
            omod,
            sensors,
            modeChoiceCalibration,
            popStrata,
            carOwnership,
            affectedLinks,
            x
        )
        println("______________________________")
        println("Baseline")
        println("MSE iteration: $mse")
        println("------------------------------")
        println("Sensor | \t Flow OMOD | \t Flow Measured")
        for ((i, flow) in sFlow.values.withIndex()) {
            println("${sensors[i].name} | \t $flow | \t ${sensors[i].measuredFlow }")
        }

        val values = listOf<Double>(-0.03512838, -0.000001,  -0.1)

        for(value in values ) {
            (omod.destinationFinder as DestinationFinderDefault).dCoeffImpact(value)
            val (mse, sFlow, _) =  determineJointOD(
                omod,
                sensors,
                modeChoiceCalibration,
                popStrata,
                carOwnership,
                affectedLinks,
                x
            )
            println("______________________________")
            println("MSE $value: $mse")
            println("------------------------------")
            println("Sensor | \t Flow OMOD | \t Flow Measured")
            for ((i, flow) in sFlow.values.withIndex()) {
                println("${sensors[i].name} | \t $flow | \t ${sensors[i].measuredFlow }")
            }
        }
        de.uniwuerzburg.omod.core.logger.on = true
        return x
    }


    private fun runGradientDescent(
        omod: Omod,
        sensors: List<TrafficSensor>,
        modeChoiceCalibration: ModeChoiceCalibration,
        popStrata: List<PopStratum>,
        carOwnership: CarOwnership,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        iterations: Int = 1000
    ) : Array<Double> {
        de.uniwuerzburg.omod.core.logger.on = false // Switch off logger for iterative calibration runs
        println("Start GD")
        val nDimensions = omod.grid.size

        val x = Array(nDimensions) { 1.0 }

        for(iteration in 0 until iterations ) {
            val time = measureTime {
                val gradient = twoPointGrad(
                    omod,
                    sensors,
                    modeChoiceCalibration,
                    popStrata,
                    carOwnership,
                    affectedLinks,
                    x
                ) // Takes forever because we have so many parameters

                for (i in x.indices) {
                    x[i] = x[i] - 0.1 * gradient[i]
                }
            }

            val (mse, sFlow, _) =  determineJointOD(
                omod,
                sensors,
                modeChoiceCalibration,
                popStrata,
                carOwnership,
                affectedLinks,
                x
            )
            println("______________________________")
            println("Iteration $iteration took : $time")
            println("MSE iteration $iteration: $mse")
            println("------------------------------")
            println("Sensor | \t Flow OMOD | \t Flow Measured")
            for ((i, flow) in sFlow.values.withIndex()) {
                println("${sensors[i].name} | \t $flow | \t ${sensors[i].measuredFlow }")
            }
        }
        de.uniwuerzburg.omod.core.logger.on = true
        return x
    }

    private fun runPSO(
        omod: Omod,
        sensors: List<TrafficSensor>,
        modeChoiceCalibration: ModeChoiceCalibration,
        popStrata: List<PopStratum>,
        carOwnership: CarOwnership,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        iterations: Int = 10, nParticles: Int = 10,
        blo: Double = 0.1, bup: Double = 5.0,
        w: Double = 0.9, phiP: Double = 0.5,
        phiG: Double = 0.3
    ) : Array<Double> {
        de.uniwuerzburg.omod.core.logger.on = false // Switch off logger for iterative calibration runs
        println("Start PSO")
        val nDimensions = omod.grid.size

        // Initial mse
        var globalBestPosition = Array(nDimensions) { 1.0 }
        var (globalBestMSE, _, _) =  determineJointOD(
            omod,
            sensors,
            modeChoiceCalibration,
            popStrata,
            carOwnership,
            affectedLinks,
            globalBestPosition
        )

        // Initialize particles
        val vLow = -(bup - blo).absoluteValue
        val vUp = (bup - blo).absoluteValue
        val particles = List(nParticles) {
            val x = Array(nDimensions) { omod.mainRng.nextDouble(blo, bup) }
            val v = Array(nDimensions) { omod.mainRng.nextDouble(vLow, vUp) }
            val (mse, _, _) =  determineJointOD(
                omod,
                sensors,
                modeChoiceCalibration,
                popStrata,
                carOwnership,
                affectedLinks,
                x
            )
            if (mse < globalBestMSE) {
                globalBestMSE = mse
                globalBestPosition = x.copyOf()
            }
            PSOParticle(v, x, x, mse)
        }
        println("Start iterations")
        for(iteration in 0 until iterations ) {
            val time = measureTime {
                //runBlocking(omod.dispatcher) { // TODO slows things down? RAM issue?
                for (particle in particles) {
                    //launch {
                    for (i in 0 until nDimensions) {
                        val rp = omod.mainRng.nextDouble()
                        val rg = omod.mainRng.nextDouble()

                        // Update velocity
                        particle.velocity[i] =
                            w * particle.velocity[i] +
                                    phiP * rp * (particle.bestPosition[i] - particle.position[i]) +
                                    phiG * rg * (globalBestPosition[i] - particle.position[i])

                        // Update position
                        particle.position[i] += particle.velocity[i]

                        // Clip position
                        if (particle.position[i] < 0) {
                            particle.position[i] = 0.0
                        }
                    }

                    // Check performance
                    val (mse, _, _) =  determineJointOD(
                        omod,
                        sensors,
                        modeChoiceCalibration,
                        popStrata,
                        carOwnership,
                        affectedLinks,
                        particle.position
                    )

                    if (mse < particle.bestMse) {
                        particle.bestPosition = particle.position.copyOf()
                        particle.bestMse = mse
                    }

                }

                for (particle in particles) {
                    if (particle.bestMse < globalBestMSE) {
                        globalBestPosition = particle.bestPosition.copyOf()
                        globalBestMSE = particle.bestMse
                    }
                }
            }

            val (mse, sFlow, _) =  determineJointOD(
                omod,
                sensors,
                modeChoiceCalibration,
                popStrata,
                carOwnership,
                affectedLinks,
                globalBestPosition
            )
            println("______________________________")
            println("Iteration $iteration took : $time")
            println("MSE iteration $iteration: $mse")
            println("------------------------------")
            println("Sensor | \t Flow OMOD | \t Flow Measured")
            for ((i, flow) in sFlow.values.withIndex()) {
                println("${sensors[i].name} | \t $flow | \t ${sensors[i].measuredFlow }")
            }
        }
        de.uniwuerzburg.omod.core.logger.on = true
        return globalBestPosition
    }


    private fun determineJointOD(
        omod: Omod,
        sensors: List<TrafficSensor>,
        modeChoiceCalibration: ModeChoiceCalibration,
        popStrata: List<PopStratum>,
        carOwnership: CarOwnership,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        parameters: Array<Double>
    ) : Triple<Double, Map<TrafficSensor, Double>, Map<Pair<Cell, Cell>, Double>> {
        // Set Parameters
        val finder = omod.destinationFinder as DestinationFinderDefault
        //finder.updateCellCValues(parameters, omod.grid)
        val fullPopulation = omod.buildings.sumOf { it.population }

        // Pair probability
        val od = finder.determinePairProbabilities(
            omod.grid, omod.activityGenerator as ActivityGeneratorDefault,
            modeChoiceCalibration, omod.grid.zip(parameters).toMap(),
            popStrata, carOwnership
        )

        // Determine affected sensors
        // TODO temporal check
        val staticCount = sensors.associateWith { 0.0 }.toMutableMap()
        for (origin in omod.grid) {
            for (destination in omod.grid) {
                val odPair = Pair(origin, destination)
                if (odPair in affectedLinks) {
                    val sensors = affectedLinks[odPair]!!
                    for (sensor in sensors) {
                        staticCount[sensor] = staticCount[sensor]!! + od[odPair]!! * fullPopulation
                    }
                }
            }
        }

        var mse = 0.0
        val allFlows = mutableMapOf<TrafficSensor, Double>()
        for (sensor in sensors) {
            val simFlow = staticCount[sensor]!!
            mse += (sensor.measuredFlow - simFlow).pow(2)

            allFlows[sensor] = simFlow
        }
        mse /= sensors.size

        return Triple(mse, staticCount, od)
    }


    fun twoPointGrad(
        omod: Omod,
        sensors: List<TrafficSensor>,
        modeChoiceCalibration: ModeChoiceCalibration,
        popStrata: List<PopStratum>,
        carOwnership: CarOwnership,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        parameters: Array<Double>,
        stepsize: Double =  sqrt(Double.MIN_VALUE)
    ) : DoubleArray {
        val gradient = Array(parameters.size) { 0.0 }
        val fx = determineJointOD(
            omod,
            sensors,
            modeChoiceCalibration,
            popStrata,
            carOwnership,
            affectedLinks,
            parameters
        ).first

        for (i in parameters.indices) {
            val pdash = parameters.copyOf()
            pdash[i] = parameters[i] + stepsize
            val fdash =  determineJointOD(
                omod,
                sensors,
                modeChoiceCalibration,
                popStrata,
                carOwnership,
                affectedLinks,
                pdash
            ).first
            gradient[i] = (fdash - fx) / stepsize
        }
        return gradient.toDoubleArray()
    }


    class PSOParticle(
        var velocity: Array<Double>,
        var position: Array<Double>,
        var bestPosition: Array<Double>,
        var bestMse: Double
    )
}