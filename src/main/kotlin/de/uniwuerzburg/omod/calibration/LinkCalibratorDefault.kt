package de.uniwuerzburg.omod.calibration

import com.github.ajalt.mordant.table.grid
import com.graphhopper.GraphHopper
import de.uniwuerzburg.omod.core.ActivityGeneratorDefault
import de.uniwuerzburg.omod.core.DestinationFinderDefault
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.logger
import de.uniwuerzburg.omod.core.models.ActivityType
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
import kotlin.time.measureTimedValue

/*
//TODO Try
Non-stochastic function evaluation:
1. Get Home probs by cell
2. Get Work probs dependent on home prob
3. For each chain(simplify with common chain Segements):
    3.1. If H<->W: Add prob to relevant link (H: easy, W: Kinda easy)
    3.2  If H,W->X: Add prob to link (Also only ok difficult)
    3.3  If X->X: Same as last
    3.4  If X->H,W: Very Hard. Maybe I can shortcut this with symmetry. I.e. if I have H->W double it same with H->X etc.

// TODO Try
PSO with attraction coefficient adjustments for each zone
*/
class LinkCalibratorDefault(linkDataFile: File, val omod: Omod) : LinkCalibrator {
    private val sensors: List<TrafficSensor>
    private val affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>

    init {
        sensors = readSensorData(linkDataFile)
        val finder = omod.destinationFinder as DestinationFinderDefault
        val (test, time) = measureTimedValue {
            omod.destinationFinder.determinePairProbabilities(
                omod.grid, omod.activityGenerator as ActivityGeneratorDefault
            )
        }
        println("$time")
        println("Expected trips:  ${test.values.sum()}")
        /*println(test.values.max())
        for ((i, o) in omod.grid.withIndex()) {
            for ((j, d) in omod.grid.withIndex()) {
                val v = test[Pair(o, d)]!!
                if((v * 10_000.0) > 1.3) {
                    println(i)
                    println(j)
                }
            }
        }*/
        affectedLinks = determineAffectedLinks(omod.grid, sensors, omod.hopper!!)

        val (mse, sFlow, nAgents) = runBatch( Array( omod.grid.size) { 1.0 })

        // Determine affected sensors
        val shouldCount = sensors.associateWith { 0.0 }.toMutableMap()
        for (origin in omod.grid) {
            for (destination in omod.grid) {
                val od = Pair(origin, destination)
                if (od in affectedLinks) {
                    val sensors = affectedLinks[od]!!
                    for (sensor in sensors) {
                        shouldCount[sensor] = shouldCount[sensor]!! + test[od]!! * nAgents
                    }
                }
            }
        }

        val testOrigin = omod.grid[306]
        val testDestination = omod.grid[307]
        println("On test count in static map: ${test[Pair(testOrigin, testDestination)]!! * nAgents}")
        println("Total trips in static map:  ${test.values.sum() * nAgents}")

        println("______________________________")
        for ((i, flow) in sFlow.values.withIndex()) {
            println("Sensor $i: $flow  / ${shouldCount[sensors[i]]} / ${sensors[i].measuredFlow }")
        }

        //runPSO()
    }

    private fun runPSO(
        iterations: Int = 10, nParticles: Int = 10,
        blo: Double = 0.1, bup: Double = 5.0,
        w: Double = 0.9, phiP: Double = 0.5,
        phiG: Double = 0.3
    ) {
        logger.on = false // Switch off logger for iterative calibration runs

        val nDimensions = omod.grid.size

        // Initial mse
        var globalBestPosition = Array(nDimensions) { 1.0 }
        var (globalBestMSE, _) = runBatch(globalBestPosition)

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

    private fun runBatch(parameters: Array<Double>) : Triple<Double, Map<TrafficSensor, Double>, Int> {
        // Set Parameters
        val finder = omod.destinationFinder as DestinationFinderDefault
        finder.updateCellCValues(parameters, omod.grid)
        omod.mainRng.setSeed(0)

        // Run Simulation
        val agents = omod.run(1.0)

        // Determine affected sensors
        var totalTripCount = 0
        val testOrigin = omod.grid[306]
        val testDestination = omod.grid[307]
        var testCount = 0.0
        val simCount = sensors.associateWith { 0.0 }.toMutableMap()
        for (agent in agents) {
            var origin = agent.mobilityDemand.first().activities.first()
            for (activity in agent.mobilityDemand.first().activities.drop(1)) {
                totalTripCount += 1
                if ((origin.location.getAggLoc() == testOrigin) and (activity.location.getAggLoc() == testDestination)) {
                    testCount += 1
                }
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
            val simFlow = simCount[sensor]!! //TODO: Redo * fullPopulation / agents.size
            mse += (sensor.measuredFlow - simFlow).pow(2)

            allFlows[sensor] = simFlow
        }
        mse /= sensors.size

        println("On test od: ${testCount}")
        println("Total trip count: ${totalTripCount}")
        return Triple(mse, allFlows, agents.size)
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
