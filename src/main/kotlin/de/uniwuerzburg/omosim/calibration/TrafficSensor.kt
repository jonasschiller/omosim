package de.uniwuerzburg.omosim.calibration

import de.uniwuerzburg.omosim.core.Omosim
import de.uniwuerzburg.omosim.core.models.RealLocation
import de.uniwuerzburg.omosim.routing.routeAltCar
import de.uniwuerzburg.omosim.routing.routeWith
import de.uniwuerzburg.omosim.utils.CRSTransformer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.geotools.filter.function.StaticGeometry.intersection
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.geom.GeometryFactory
import org.locationtech.jts.geom.LineString
import org.locationtech.jts.geom.MultiLineString
import org.locationtech.jts.index.hprtree.HPRtree
import org.locationtech.jts.io.WKTReader
import java.io.File
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

class TrafficSensor(
    val name: String,
    val measuredFlow: DoubleArray,
    val flowDirection: Direction,
    val field: Geometry
) {
    companion object {
        fun readSensorData(linkData: File, omosim: Omosim) : List<TrafficSensor> {
            val sensors = mutableListOf<TrafficSensor>()
            val reader = linkData.bufferedReader()

            val delimiter = ";"

            val wktReader = WKTReader()

            // Parse header
            val header = reader.readLine()
            val idxMap = header.split(delimiter).withIndex().associate { (i, v) -> v to i }

            // Index of cols to extract
            val nameCol =  idxMap["name"]
            val flowCol =  idxMap["dailyFlow"]
            val dirCol =  idxMap["flowDirection"]
            val geometryCol = idxMap["Geometry"]

            // Read data
            for(line in reader.lines()) {
                val values = line.split(delimiter)
                val name = nameCol?.let { values[it] }
                val flow = values[flowCol!!]
                    .replace("[", "")
                    .replace("]", "")
                    .split(",")
                    .map { it.toDouble() }
                    .toDoubleArray()

                // Get sensor field
                val wkt = values[geometryCol!!]
                val latlonGeom = wktReader.read(wkt)
                val geometry = omosim.transformer.toModelCRS(latlonGeom)

                // Get sensor direction
                val angle = values[dirCol!!].toDouble()
                val centroid = latlonGeom.centroid.coordinates.first()
                val direction = Direction(angle, centroid, omosim.transformer)

                val sensor = TrafficSensor(name ?: sensors.size.toString(), flow, direction, geometry)
                sensors.add(sensor)
            }

            reader.close()
            return sensors
        }

        fun affectedSensors(
            sensors: List<TrafficSensor>,
            omosim: Omosim
        ) : Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>> {
            val geometryFactory = GeometryFactory()

            val sensorTree = HPRtree()
            for (sensor in sensors) {
                sensorTree.insert(sensor.field.envelopeInternal, sensor)
            }

            val affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>> = runBlocking(omosim.dispatcher) {
                channelFlow {
                    for (origin in omosim.grid) {
                        launch {
                            for (destination in omosim.grid) {
                                val response = routeWith("car", origin, destination, omosim.hopper!!)
                                val coords = response.best.points.map { Coordinate(it.lat, it.lon) }.toTypedArray()

                                if (coords.size >= 2) {
                                    val routeLine = omosim.transformer.toModelCRS( geometryFactory.createLineString(coords) )

                                    val thisAffected = sensorTree.query(routeLine.envelopeInternal)
                                        .map { it as TrafficSensor }
                                        .filter { it.field.envelope.intersects(routeLine) && it.field.intersects(routeLine) }
                                        .filter {
                                            val inters = intersection(it.field, routeLine)
                                            if (inters is LineString) {
                                                it.flowDirection.isSameDirection(inters, 30.0)
                                            } else if (inters is MultiLineString) {
                                                var sameDir = false
                                                for (n in 0 until inters.numGeometries)
                                                    if (it.flowDirection.isSameDirection(inters.getGeometryN(n) as LineString, 30.0)) {
                                                        sameDir = true
                                                        break
                                                    }
                                                sameDir
                                            }else {
                                                false
                                            }
                                        }

                                    if(thisAffected.isNotEmpty()) {
                                        send(Pair(Pair(origin, destination), thisAffected))
                                    }
                                }
                            }
                        }
                    }
                }.toList()
            }.toMap()

            return affectedSensors
        }

        fun altAffectedSensors(
            sensors: List<TrafficSensor>,
            omosim: Omosim
        ) : Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>> {
            val geometryFactory = GeometryFactory()

            val sensorTree = HPRtree()
            for (sensor in sensors) {
                sensorTree.insert(sensor.field.envelopeInternal, sensor)
            }

            val affectedSensors: Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>> = runBlocking(omosim.dispatcher) {
                channelFlow {
                    for (origin in omosim.grid) {
                        launch {
                            for (destination in omosim.grid) {
                                val response = routeAltCar(origin, destination, omosim.hopper!!)
                                val thisAffected = mutableListOf<List<TrafficSensor>>()

                                for (path in response.all) {
                                    val coords = path.points.map { Coordinate(it.lat, it.lon) }.toTypedArray()

                                    if (coords.size >= 2) {
                                        val routeLine = omosim.transformer.toModelCRS( geometryFactory.createLineString(coords) )

                                        val thisAltAffected = sensorTree.query(routeLine.envelopeInternal)
                                            .map { it as TrafficSensor }
                                            .filter { it.field.envelope.intersects(routeLine) && it.field.intersects(routeLine) }
                                            .filter {
                                                val inters = intersection(it.field, routeLine)
                                                if (inters is LineString) {
                                                    it.flowDirection.isSameDirection(inters, 30.0)
                                                } else if (inters is MultiLineString) {
                                                    var sameDir = false
                                                    for (n in 0 until inters.numGeometries)
                                                        if (it.flowDirection.isSameDirection(inters.getGeometryN(n) as LineString, 30.0)) {
                                                            sameDir = true
                                                            break
                                                        }
                                                    sameDir
                                                }else {
                                                    false
                                                }
                                            }

                                        // Also add paths that do not affect any sensors
                                        thisAffected.add(thisAltAffected)
                                    }
                                }

                                if(thisAffected.isNotEmpty()) {
                                    send(Pair(Pair(origin, destination), thisAffected))
                                }
                            }
                        }
                    }
                }.toList()
            }.toMap()

            return affectedSensors
        }
    }

    class Direction(
        azimuth: Double,
        centroid: Coordinate,
        transformer: CRSTransformer
    ) {
        private val vector: DoubleArray

        init {
            // Convert angle
            var polarAngle = 360.0 - azimuth + 90.0
            if (polarAngle >= 360.0) {
                polarAngle -= 360.0
            }

            // Create vector
            val dx = 0.01 * cos(polarAngle / 180.0 * PI)
            val dy = 0.01 * sin(polarAngle / 180.0 * PI)
            val vCoordsLatLon = GeometryFactory().createLineString(
                listOf(
                    centroid,
                    Coordinate(centroid.x + dy, centroid.y + dx) // Swap dx and dy because of lat-lon crs
                ).toTypedArray()
            )

            val vCoordsUTM = transformer
                .toModelCRS(vCoordsLatLon)
                .coordinates

            vector = vecFromCoords(vCoordsUTM)
        }

        private fun vecFromCoords(coords: Array<Coordinate>) : DoubleArray {
            require(coords.size == 2)

            val vector = DoubleArray(2) {0.0}
            vector[0] = coords[1].x - coords[0].x
            vector[1] = coords[1].y - coords[0].y
            return vector
        }

        private fun angleBetween(other: LineString): Double {
            val oVec = vecFromCoords(arrayOf(other.coordinates.first(), other.coordinates.last()))

            val dot = vector[0] * oVec[0] + vector[1] * oVec[1]
            val det = vector[0] * oVec[1] - vector[1] * oVec[0]
            return atan2(det, dot)
        }

        fun isSameDirection(other: LineString, leeway: Double) : Boolean {
            val leewayRad = leeway / 180.0 * PI
            return abs(angleBetween(other)) < leewayRad
        }
    }
}