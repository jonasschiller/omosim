package de.uniwuerzburg.omod.calibration

import com.graphhopper.GraphHopper
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.Cell
import de.uniwuerzburg.omod.core.models.RealLocation
import de.uniwuerzburg.omod.routing.routeWith
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.geom.GeometryFactory
import java.io.File
import kotlin.math.pow

// TODO
// 1. Determine if link which links are affected by a transition
// 2. Simulate a batch of agents
// 3. Check transitions
// 4. Do particle swarm
class LinkCalibratorDefault(linkDataFile: File, val omod: Omod) : LinkCalibrator {
    private val sensors: List<TrafficSensor>
    private val affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>

    init {
        sensors = readSensorData(linkDataFile)
        affectedLinks = determineAffectedLinks(omod.grid, sensors, omod.hopper!!)
        val fullPopulation = omod.buildings.sumOf { it.population }
        val mse = runBatch(fullPopulation)
    }

    fun runBatch(fullPopulation: Double) : Double {
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

        var mse = 0.0
        for (sensor in sensors) {
            val simFlow = simCount[sensor]!! * fullPopulation / agents.size
            mse += (sensor.measuredFlow - simFlow).pow(2)
        }
        mse /= sensors.size

        return mse
    }

    fun readSensorData(linkData: File) : List<TrafficSensor> {
        val geometryFactory = GeometryFactory()
        val sensors = mutableListOf<TrafficSensor>()
        val reader = linkData.bufferedReader()

        val delimiter = ","

        // Parse header
        val header = reader.readLine()
        val idxMap = header.split(delimiter).withIndex().associate { (i, v) -> v to i }

        // Index of cols to extract
        val flowCol =  idxMap["dailyFlow"]
        val startLatCol = idxMap["startLat"]
        val startLonCol = idxMap["startLon"]
        val stopLatCol = idxMap["stopLat"]
        val stopLonCol = idxMap["stopLon"]
        val widthCol = idxMap["width"]
        val directionalCol = idxMap["directional"]

        // Read data
        for(line in reader.lines()) {
            val values = line.split(delimiter)
            val flow = values[flowCol!!].toDouble()
            val startLat = values[startLatCol!!].toDouble()
            val startLon = values[startLonCol!!].toDouble()
            val stopLat = values[stopLatCol!!].toDouble()
            val stopLon = values[stopLonCol!!].toDouble()
            val width = values[widthCol!!].toDouble()
            val directional = values[directionalCol!!].toBoolean()

            val coordinates = arrayOf(Coordinate(startLat, startLon), Coordinate(stopLat, stopLon))
            val lineGeom = geometryFactory.createLineString(coordinates)
            val field = omod.transformer.toModelCRS(lineGeom).buffer(width/2)

            val sensor = TrafficSensor(flow, field, directional)
            sensors.add(sensor)
        }

        reader.close()
        return sensors
    }

    /**
     * Returns OSM-ID of affected links
     */
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
        println("Grid size: ${grid.size}")
        println("Affected pairs: ${affectedLinks.size} / ${grid.size * grid.size}")
        return affectedLinks
    }
}

data class TrafficSensor(
    val measuredFlow: Double,
    val field: Geometry,
    val directional: Boolean
)