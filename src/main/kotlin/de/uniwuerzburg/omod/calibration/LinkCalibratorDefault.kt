package de.uniwuerzburg.omod.calibration

import com.graphhopper.GraphHopper
import de.uniwuerzburg.omod.core.*
import de.uniwuerzburg.omod.core.models.*
import de.uniwuerzburg.omod.routing.routeWith
import de.uniwuerzburg.omod.utils.CRSTransformer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.geotools.filter.function.StaticGeometry.intersection
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.geom.GeometryFactory
import org.locationtech.jts.geom.LineString
import org.locationtech.jts.index.hprtree.HPRtree
import org.locationtech.jts.io.WKTReader
import java.io.File
import kotlin.math.*
import kotlin.time.measureTime

class LinkCalibratorDefault(
    linkDataFile: File,
    val omod: Omod,
    val popStrata: List<PopStratum>,
    val carOwnership: CarOwnership
) : LinkCalibrator {
    private val sensors: List<TrafficSensor>
    private val affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    private val modeChoiceCalibration = ModeChoiceCalibration()

    init {
        sensors = readSensorData(linkDataFile)
        affectedLinks = determineAffectedLinks(omod.grid, sensors, omod.hopper!!)

        //val bestPosition = runPSO()
        val bestPosition = Array<Double>(omod.grid.size) {1.0}
        //val bestPosition = runGradientDescent(10)

        val (_, sFlow, nAgents) = runBatch( bestPosition )
        val (_, staticFlow, staticMap) = determineJointOD( bestPosition )

        val testOrigin = omod.grid[306]
        val testDestination = omod.grid[307]
        println("On test count in static map: ${staticMap[Pair(testOrigin, testDestination)]!! * nAgents}")
        println("Total trips in static map:  ${staticMap.values.sum() * nAgents}")
        println("_".repeat(20*4 + 5*3))
        println("${"Sensor".padEnd(20)} | \t" +
                "${"Flow Simulated".padEnd(20)} | \t" +
                "${"Flow Deterministic".padEnd(20)} | \t" +
                "Flow Measured".padEnd(20)
        )
        println("_".repeat(20) +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20))
        for ((i, flow) in sFlow.values.withIndex()) {
            println(
                "${sensors[i].name.padEnd(20)} | \t" +
                "${flow.toString().padEnd(20)} | \t" +
                "${staticFlow[sensors[i]].toString().padEnd(20)} | \t" +
                sensors[i].measuredFlow.toString().padEnd(20)
            )
        }
    }

    private fun runGradientDescent(
        iterations: Int = 10
    ) : Array<Double> {
        logger.on = false // Switch off logger for iterative calibration runs
        println("Start GD")
        val nDimensions = omod.grid.size

        val x = Array(nDimensions) { 1.0 }
        
        for(iteration in 0 until iterations ) {
            val time = measureTime {
                val gradient = twoPointGrad(x) // Takes forever because we have so many parameters

                for (i in x.indices) {
                    x[i] = x[i] - 0.1 * gradient[i]
                }
            }

            val (mse, sFlow, _) = determineJointOD(x)
            println("______________________________")
            println("Iteration $iteration took : $time")
            println("MSE iteration $iteration: $mse")
            println("------------------------------")
            println("Sensor | \t Flow OMOD | \t Flow Measured")
            for ((i, flow) in sFlow.values.withIndex()) {
                println("${sensors[i].name} | \t $flow | \t ${sensors[i].measuredFlow }")
            }
        }
        logger.on = true
        return x
    }

    private fun runPSO(
        iterations: Int = 10, nParticles: Int = 10,
        blo: Double = 0.1, bup: Double = 5.0,
        w: Double = 0.9, phiP: Double = 0.5,
        phiG: Double = 0.3
    ) : Array<Double> {
        logger.on = false // Switch off logger for iterative calibration runs
        println("Start PSO")
        val nDimensions = omod.grid.size

        // Initial mse
        var globalBestPosition = Array(nDimensions) { 1.0 }
        var (globalBestMSE, _, _) = determineJointOD(globalBestPosition)

        // Initialize particles
        val vLow = -(bup - blo).absoluteValue
        val vUp = (bup - blo).absoluteValue
        val particles = List(nParticles) {
            val x = Array(nDimensions) { omod.mainRng.nextDouble(blo, bup) }
            val v = Array(nDimensions) { omod.mainRng.nextDouble(vLow, vUp) }
            val (mse, _, _) = determineJointOD(x)
            if (mse < globalBestMSE) {
                globalBestMSE = mse
                globalBestPosition = x.copyOf()
            }
            PSOParticle(v, x, x, mse)
        }
        println("Start iterations")
        for(iteration in 0 until iterations ) {
            val time = measureTime {
                runBlocking {
                    for (particle in particles) {
                        launch {
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
                            val (mse, _, _) = determineJointOD(particle.position)

                            if (mse < particle.bestMse) {
                                particle.bestPosition = particle.position.copyOf()
                                particle.bestMse = mse
                            }
                        }
                    }
                }

                for (particle in particles) {
                    if (particle.bestMse < globalBestMSE) {
                        globalBestPosition = particle.bestPosition.copyOf()
                        globalBestMSE = particle.bestMse
                    }
                }
            }

            val (mse, sFlow, _) = determineJointOD(globalBestPosition)
            println("______________________________")
            println("Iteration $iteration took : $time")
            println("MSE iteration $iteration: $mse")
            println("------------------------------")
            println("Sensor | \t Flow OMOD | \t Flow Measured")
            for ((i, flow) in sFlow.values.withIndex()) {
                println("${sensors[i].name} | \t $flow | \t ${sensors[i].measuredFlow }")
            }
        }
        logger.on = true
        return globalBestPosition
    }

    private fun determineJointOD(
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

    private fun runBatch(parameters: Array<Double>) : Triple<Double, Map<TrafficSensor, Double>, Int> {
        // Set Parameters
        val finder = omod.destinationFinder as DestinationFinderDefault
        finder.updateCellCValues(parameters, omod.grid)
        omod.mainRng.setSeed(0) // Seed impact low with 100% of agents

        // Run Simulation
        val agents = omod.run(1.0)
        omod.doModeChoice(agents, ModeChoiceOption.FAST, false)

        // Determine affected sensors
        var totalTripCount = 0
        val testOrigin = omod.grid[306]
        val testDestination = omod.grid[307]
        var testCount = 0.0
        val simCount = sensors.associateWith { 0.0 }.toMutableMap()
        for (agent in agents) {
            var origin = agent.mobilityDemand.first().activities.first()
            val activities = agent.mobilityDemand.first().activities.drop(1)

            /* For tests
            for (activity in activities) {
                var mode = Mode.UNDEFINED
                if ( omod.mainRng.nextDouble() < 0.75) { // Car availability
                    val distance = omod.routingCache.getDistances(
                        origin.location.getAggLoc()!!, listOf(activity.location.getAggLoc()!!)
                    ).first().toDouble() / 1000.0
                    val weights = modeChoiceCalibration.utilitiesForCalibration(distance, agent, activity.type, Weekday.UNDEFINED)
                    val distr = createCumDist(weights)
                    mode = modeChoiceCalibration.tripModeOptions[sampleCumDist(distr, omod.mainRng)].mode
                }

                if (mode == Mode.CAR_DRIVER) {
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
                }
                origin = activity
            }
            */
            // Real version
            val trips = agent.mobilityDemand.first().trips
            for ((activity, trip) in activities.zip(trips)) {
                if (trip.mode == Mode.CAR_DRIVER) {
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

        println("On test od: $testCount")
        println("Total trip count: $totalTripCount")
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
        val nameCol =  idxMap["name"]
        val flowCol =  idxMap["dailyFlow"]
        val dirCol =  idxMap["flowDirection"]
        val geometryCol = idxMap["Geometry"]

        // Read data
        for(line in reader.lines()) {
            val values = line.split(delimiter)
            val name = nameCol?.let { values[it] }
            val flow = values[flowCol!!].toDouble()

            // Get sensor field
            val wkt = values[geometryCol!!]
            val latlonGeom = wktReader.read(wkt)
            val geometry = omod.transformer.toModelCRS(latlonGeom)

            // Get sensor direction
            val angle = values[dirCol!!].toDouble()
            val centroid = latlonGeom.centroid.coordinates.first()
            val direction = Direction(angle, centroid, omod.transformer)

            val sensor = TrafficSensor(name ?: sensors.size.toString(), flow, direction, geometry)
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
        val geometryFactory = GeometryFactory()

        val sensorTree = HPRtree()
        for (sensor in sensors) {
            sensorTree.insert(sensor.field.envelopeInternal, sensor)
        }

        val affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>> = runBlocking(omod.dispatcher) {
            channelFlow {
                for (origin in grid) {
                    launch {
                        for (destination in grid) {
                            val response = routeWith("car", origin, destination, hopper)
                            val coords = response.best.points.map { Coordinate(it.lat, it.lon) }.toTypedArray()

                            if (coords.size >= 2) {
                                val routeLine = omod.transformer.toModelCRS( geometryFactory.createLineString(coords) )

                                val thisAffected = sensorTree.query(routeLine.envelopeInternal)
                                    .map { it as TrafficSensor }
                                    .filter { it.field.envelope.intersects(routeLine) && it.field.intersects(routeLine) }
                                    .filter {
                                        val inters = intersection(it.field, routeLine) as LineString
                                        it.flowDirection.isSameDirection(inters, 30.0)
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

        return affectedLinks
    }

    fun twoPointGrad(parameters: Array<Double>, stepsize: Double =  sqrt(Double.MIN_VALUE)) : DoubleArray {
        val gradient = Array(parameters.size) { 0.0 }
        val fx = determineJointOD(parameters).first

        for (i in parameters.indices) {
            val pdash = parameters.copyOf()
            pdash[i] = parameters[i] + stepsize
            val fdash = determineJointOD(pdash).first
            gradient[i] = (fdash - fx) / stepsize
        }
        return gradient.toDoubleArray()
    }
}

class TrafficSensor(
    val name: String,
    val measuredFlow: Double,
    val flowDirection: Direction,
    val field: Geometry
)

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
        val angleBe = angleBetween(other)
        return abs(angleBetween(other)) < leewayRad
    }
}

class PSOParticle(
   var velocity: Array<Double>,
   var position: Array<Double>,
   var bestPosition: Array<Double>,
   var bestMse: Double
)



