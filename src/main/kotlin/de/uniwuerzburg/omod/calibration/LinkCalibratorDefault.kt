package de.uniwuerzburg.omod.calibration

import com.graphhopper.GraphHopper
import de.uniwuerzburg.omod.core.DestinationFinderDefault
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.logger
import de.uniwuerzburg.omod.core.models.Cell
import de.uniwuerzburg.omod.core.models.RealLocation
import de.uniwuerzburg.omod.routing.routeWith
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.geom.GeometryFactory
import org.locationtech.jts.io.WKTReader
import java.io.File
import kotlin.math.absoluteValue
import kotlin.math.pow

class LinkCalibratorDefault(linkDataFile: File, val omod: Omod) : LinkCalibrator {
    private val sensors: List<TrafficSensor>
    private val affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>

    init {
        sensors = readSensorData(linkDataFile)
        affectedLinks = determineAffectedLinks(omod.grid, sensors, omod.hopper!!)
        runPSO()
    }

    private fun runPSO(
        iterations: Int = 10, nParticles: Int = 10,
        blo: Double = 0.0, bup: Double = 10.0,
        w: Double = 0.9, phiP: Double = 0.5,
        phiG: Double = 0.3
    ) {
        logger.on = false // Switch off logger for iterative calibration runs

        // Initial mse
        val finder = omod.destinationFinder as DestinationFinderDefault
        var globalBestPosition = finder.getCalibrationPosition()
        var (globalBestMSE, _) = runBatch(globalBestPosition)

        val nDimensions = globalBestPosition.size

        // Initialize particles
        val vLow = -(bup - blo).absoluteValue
        val vUp = (bup - blo).absoluteValue
        val particles = List(nParticles) {
            val x = Array(nDimensions) { omod.mainRng.nextDouble(blo, bup) }
            val v = Array(nDimensions) { omod.mainRng.nextDouble(vLow, vUp) }
            val (mse, _) = runBatch(x)
            if (mse < globalBestMSE) {
                globalBestMSE = mse
                globalBestPosition = x
            }
            PSOParticle(v, x, x, mse)
        }

        for(iteration in 0 until iterations ) {
            for (particle in particles) {
                for (i in 0 until  nDimensions) {
                    val rp = omod.mainRng.nextDouble()
                    val rg = omod.mainRng.nextDouble()

                    // Update velocity
                    particle.velocity[i] =
                        w         * particle.velocity[i] +
                        phiP * rp * ( particle.bestPosition[i] - particle.position[i] ) +
                        phiG * rg * ( globalBestPosition[i] - particle.position[i] )

                    // Update position
                    particle.position[i] += particle.velocity[i]
                }

                // Check performance
                val (mse, _) = runBatch(particle.position)

                if (mse < particle.bestMse) {
                    particle.bestPosition = particle.position
                    particle.bestMse = mse
                }

                if (mse < globalBestMSE) {
                    globalBestPosition = particle.position
                    globalBestMSE = mse
                }
            }

            val (mse, sFlow) = runBatch(globalBestPosition)
            println("______________________________")
            println("MSE iteration $iteration: $mse")
            println("------------------------------")
            for ((i, flow) in sFlow.values.withIndex()) {
                println("Sensor $i $iteration: $flow / ${sensors[i].measuredFlow }")
            }
        }
        logger.on = true
    }

    private fun runBatch(parameters: Array<Double>) : Pair<Double, Map<TrafficSensor, Double>> {
        // Set Parameters
        val finder = omod.destinationFinder as DestinationFinderDefault
        finder.updateCalibrationPosition(parameters, omod.grid)
        omod.mainRng.setSeed(0)

        // Run Simulation
        val agents = omod.run(10_000)

        // Determine affected sensors
        val simCount = sensors.associateWith { 0.0 }.toMutableMap()
        for (agent in agents) {
            var origin = agent.mobilityDemand.first().activities.first()
            for (activity in agent.mobilityDemand.first().activities.drop(1)) {
                if ((origin.location.getAggLoc() is Cell) and (activity.location.getAggLoc() is Cell)) {
                    val od = Pair(origin.location.getAggLoc() as Cell, activity.location.getAggLoc() as Cell)
                    if (od in affectedLinks) {
                        val sensors = affectedLinks[od]!!
                        for (sensor in sensors) {
                            simCount[sensor] = simCount[sensor]!! + 1
                        }
                    }
                }
                origin = activity
            }
        }

        val fullPopulation = omod.buildings.sumOf { it.population }

        var mse = 0.0
        val allFlows = mutableMapOf<TrafficSensor, Double>()
        for (sensor in sensors) {
            val simFlow = simCount[sensor]!! * fullPopulation / agents.size
            mse += (sensor.measuredFlow - simFlow).pow(2)

            allFlows[sensor] = simFlow
        }
        mse /= sensors.size

        return Pair(mse, allFlows)
    }

    private fun readSensorData(linkData: File) : List<TrafficSensor> {
        val sensors = mutableListOf<TrafficSensor>()
        val reader = linkData.bufferedReader()

        val delimiter = ";"

        val wktReader = WKTReader()

        // Parse header
        val header = reader.readLine()
        val idxMap = header.split(delimiter).withIndex().associate { (i, v) -> v to i }

        // Index of cols to extract
        val flowCol =  idxMap["dailyFlow"]
        val geometryCol = idxMap["Geometry"]

        // Read data
        for(line in reader.lines()) {
            val values = line.split(delimiter)
            val flow = values[flowCol!!].toDouble()
            val wkt = values[geometryCol!!]
            val geometry = omod.transformer.toModelCRS(wktReader.read(wkt))

            val sensor = TrafficSensor(flow, geometry)
            sensors.add(sensor)
        }

        reader.close()
        return sensors
    }

    private fun determineAffectedLinks(
        grid: List<Cell>,
        sensors: List<TrafficSensor>,
        hopper: GraphHopper
    ) : Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>> {
        val affectedLinks = mutableMapOf<Pair<RealLocation, RealLocation>, List<TrafficSensor>>()
        val geometryFactory = GeometryFactory()

        var cntr = -1
        for (origin in grid) {
            cntr += 1

            for (destination in grid) {
                val response = routeWith("car", origin, destination, hopper)
                val coords = response.best.points.map { Coordinate(it.lat, it.lon) }.toTypedArray()

                if (coords.size <= 2) {
                    continue
                }

                val routeLine = omod.transformer.toModelCRS( geometryFactory.createLineString(coords) )

                val thisAffected = mutableListOf<TrafficSensor>()
                for (sensor in sensors) {
                    if (routeLine.intersects(sensor.field)) {
                        thisAffected.add(sensor)
                    }
                }

                if(thisAffected.isNotEmpty()) {
                    affectedLinks[Pair(origin, destination)] = thisAffected
                }
            }
            if(cntr % 100 == 0) {
                println(cntr.toDouble()/grid.size * 100)
            }

        }
        return affectedLinks
    }
}

class TrafficSensor(
    val measuredFlow: Double,
    val field: Geometry
)

class PSOParticle(
   var velocity: Array<Double>,
   var position: Array<Double>,
   var bestPosition: Array<Double>,
   var bestMse: Double
)
