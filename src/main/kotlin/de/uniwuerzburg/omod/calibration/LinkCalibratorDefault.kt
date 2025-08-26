package de.uniwuerzburg.omod.calibration

import com.graphhopper.GraphHopper
import de.uniwuerzburg.omod.calibration.algorithms.PSO
import de.uniwuerzburg.omod.calibration.algorithms.SPSA
import de.uniwuerzburg.omod.core.*
import de.uniwuerzburg.omod.core.models.*
import de.uniwuerzburg.omod.routing.routeAltCar
import de.uniwuerzburg.omod.routing.routeWith
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.descriptors.setSerialDescriptor
import org.geotools.filter.function.StaticGeometry.intersection
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.GeometryFactory
import org.locationtech.jts.geom.LineString
import org.locationtech.jts.geom.MultiLineString
import org.locationtech.jts.index.hprtree.HPRtree
import org.locationtech.jts.io.WKTReader
import java.io.File
import java.nio.file.Path
import java.nio.file.Paths
import java.util.Random
import kotlin.math.*

enum class CalibrationOption {
    PSO, MM_LBFGS, SPSA
}

class LinkCalibratorDefault(
    linkDataFile: File,
    val omod: Omod,
    val popStrata: List<PopStratum>,
    val carOwnership: CarOwnership
) {
    private val sensors: List<TrafficSensor>
    private val affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    private val modeChoiceCalibration = ModeChoiceCalibration()
    val altPercentages: Map<Pair<RealLocation, RealLocation>, List<Double>> = mapOf()

    init {
        sensors = readSensorData(linkDataFile)
        affectedLinks = determineAffectedLinks(omod.grid, sensors, omod.hopper!!)
    }

    fun matrixTestRun() {
        val model = DefaultMetaModel(omod)
        val wm = model.calibrateMatrix(ActivityType.OTHER, sensors, affectedLinks)

        val totalPop = omod.buildings.sumOf { it.population }
        val finder = omod.destinationFinder as DestinationFinderDefault
        val parameters = Array<Double>(omod.grid.size) {1.0}
        val wcl = OACalClean.run(
           omod.grid,  omod.activityGenerator as ActivityGeneratorDefault,
           modeChoiceCalibration, mapOf(ActivityType.OTHER to omod.grid.zip(parameters).toMap()),
           popStrata, carOwnership, finder, totalPop, affectedLinks,
           sensors
        )

        val force = mutableMapOf<Cell, DoubleArray>()
        for ((i, cell) in omod.grid.withIndex()) {
            force[cell] = wm!!.toArray()[i]
        }
        finder.forceOMatrix = force

        evaluate(DoubleArray(omod.grid.size) {1.0})
    }

    fun hpTune(option: CalibrationOption) {
        val model = DefaultMetaModel(omod).getDiffModel(ActivityType.OTHER, sensors, affectedLinks)
        val objective: (DoubleArray) -> Double = { x: DoubleArray ->
            model.evaluate(x)
        }
        val outPath =  Paths.get("output/")

        when(option) {
            CalibrationOption.PSO -> PSO.hpGridSearch(
                omod.grid.size - 1, objective, Random(), outPath,
                nParticles = listOf(20, 40),
                chi = listOf(0.6, 0.8, 0.9),
                vClamp = listOf(0.5, 1.0),
                boundStrategy = listOf(PSO.BoundStrategy.INFINITY, PSO.BoundStrategy.REFLECT_Z)
            )
            CalibrationOption.SPSA -> {
                val d0 = DoubleArray(omod.grid.size - 1) { 1.0 }
                SPSA.hpGridSearch(
                    d0,
                    objective,
                    Random(),
                    outPath,
                    a0 = listOf(2.0, 1.0, 0.5),
                    c0 = listOf(5.0, 2.0, 1.0, 0.5),
                    gamma =  listOf(1.0 / 3.0, 1.0 / 5.0, 1.0 / 10.0)
                )
            }
            else -> throw NotImplementedError()
        }
    }

    fun calibrate(option: CalibrationOption) {
        var d = when (option) {
            CalibrationOption.PSO -> calibratePSO()
            CalibrationOption.MM_LBFGS -> calibrateMetaModelLBFGS()
            CalibrationOption.SPSA -> calibrateSPSA()
        }
        d = (d.toList() + listOf(1.0)).toDoubleArray()
        evaluate(d)
    }

    fun evaluate(d: DoubleArray) {
        val finder = omod.destinationFinder as DestinationFinderDefault
        val (_, sFlowBase, nAgents, _) = runBatch( arrayOf(0.0) )
        finder.updateCellCValues(ActivityType.OTHER, d.toTypedArray(), omod.grid)
        val (_, sFlow, _, _) = runBatch( arrayOf(0.0) )

        var mseSim = 0.0
        var mseSimBase = 0.0
        //var mseExpec = 0.0
        for (sensor in sensors) {
            mseSim += (sFlow[sensor]!! - sensor.measuredFlow).pow(2)
            mseSimBase += (sFlowBase[sensor]!! - sensor.measuredFlow).pow(2)
            //mseExpec += (staticCount[sensor]!! - sensor.measuredFlow).pow(2)
        }

        println("_".repeat(20*4 + 5*3))
        println("${"Sensor".padEnd(20)} | \t" +
                "${"Flow Simulated".padEnd(20)} | \t" +
                "${"Flow Simulated Base".padEnd(20)} | \t" +
                // "${"Flow Deterministic".padEnd(20)} | \t" +
                "Flow Measured".padEnd(20)
        )
        println("_".repeat(20) +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20)  +
                //" | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20))
        println(" ".repeat(20) +
                " | \t" + "%.2f e6".format(mseSim/sensors.size / 1e6).padEnd(20)  +
                " | \t" + "%.2f e6".format(mseSimBase/sensors.size / 1e6).padEnd(20)  +
                //" | \t" + "%.2f e6".format(mseExpec/sensors.size / 1e6).padEnd(20) +
                " | \t" + " ".repeat(20)
        )
        println("_".repeat(20) +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20)  +
                //" | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20))
        for ((i, flow) in sFlow.values.withIndex()) {
            println(
                "${sensors[i].name.padEnd(20)} | \t" +
                        "${flow.toString().padEnd(20)} | \t" +
                        "${sFlowBase[sensors[i]].toString().padEnd(20)} | \t" +
                        //"${staticCount[sensors[i]].toString().padEnd(20)} | \t" +
                        sensors[i].measuredFlow.toString().padEnd(20)
            )
        }
    }

    fun calibratePSO() : DoubleArray {
        val finder = omod.destinationFinder as DestinationFinderDefault
        val fullPopulation = omod.buildings.sumOf { it.population }
        val model = DefaultMetaModel(omod).getDiffModel(ActivityType.OTHER, sensors, affectedLinks)

        val objective: (DoubleArray) -> Double = { x: DoubleArray ->
            model.evaluate(x)
        }
        val d = PSO.run(
            omod.grid.size - 1, objective, Random(), out=File("TestPSO.csv"), vClamp = 1.0,  w = 0.8,
            phiP = 0.8 * 2,
            phiG = 0.8 * 2,
            nParticles = 40
        )
        return d
    }

    fun calibrateMetaModelLBFGS() : DoubleArray {
        val mModel = DefaultMetaModel(omod)
        return mModel.calibrateK1(ActivityType.OTHER, sensors, affectedLinks).toDoubleArray()
    }

    fun calibrateSPSA() : DoubleArray {
        val model = DefaultMetaModel(omod).getDiffModel(ActivityType.OTHER, sensors, affectedLinks)

        val objective: (DoubleArray) -> Double = { x: DoubleArray ->
            model.evaluate(x)
        }
        val d0 = DoubleArray(omod.grid.size - 1) { 1.0 }
        val d = SPSA.run(d0, objective, Random(), out=File("TestPSO.csv"))
        return d
    }


    fun simBatchMSE(
        x: DoubleArray, activityType: ActivityType, finder: DestinationFinderDefault, fullPopulation: Double
    ) : Double {
        de.uniwuerzburg.omod.core.logger.on = false
        finder.updateCellCValues(activityType, x.toTypedArray(), omod.grid)

        omod.mainRng.setSeed(0)
        val agents = omod.run(0.1)
        omod.doModeChoice(agents, ModeChoiceOption.FAST, false)

        // Determine affected sensors
        val simCount = sensors.associateWith {0.0}.toMutableMap()
        var locBeforeTrip = Array(omod.grid.size) {0.0}.toMutableList()
        for (agent in agents) {
            var origin = agent.mobilityDemand.first().activities.first()
            val activities = agent.mobilityDemand.first().activities.drop(1)

            val trips = agent.mobilityDemand.first().trips
            for ((activity, trip) in activities.zip(trips)) {
                if (trip.mode == Mode.CAR_DRIVER) {
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
                if (origin.location.getAggLoc() is Cell) {
                    val ocell = origin.location.getAggLoc() as Cell
                    val i = omod.grid.indexOf(ocell)
                    locBeforeTrip[i] = locBeforeTrip[i] + 1
                }
                origin = activity
            }
        }
        var mse = 0.0
        val allFlows = mutableMapOf<TrafficSensor, Double>()
        for (sensor in sensors) {
            val simFlow = simCount[sensor]!! * fullPopulation / agents.size
            mse += (sensor.measuredFlow - simFlow).pow(2)

            allFlows[sensor] = simFlow
        }
        mse /= sensors.size
        de.uniwuerzburg.omod.core.logger.on = true
        return mse
    }
    /*init {
        sensors = readSensorData(linkDataFile)
        affectedLinks = determineAffectedLinks(omod.grid, sensors, omod.hopper!!)

        // TODO Gurobi test
        // Pair probability

        val finder = omod.destinationFinder as DestinationFinderDefault
        val parameters = Array<Double>(omod.grid.size) {1.0}
        val fullPopulation = omod.buildings.sumOf { it.population }
        /*
       val od = finder.determinePairProbabilities(
           omod.grid, omod.activityGenerator as ActivityGeneratorDefault,
           modeChoiceCalibration, omod.grid.zip(parameters).toMap(),
           popStrata, carOwnership
       )
       val altAffectedLinks = determineAltAffectedLinks(omod.grid, sensors, omod.hopper!!)
       val m = altAffectedLinks.values.map { it.size }.average()
       println("Average affected links: $m")
       val altPercentages = optimize(omod.grid, od, fullPopulation, altAffectedLinks, sensors)

       val odProbs = optimize(
           omod.grid,  omod.activityGenerator as ActivityGeneratorDefault,
           modeChoiceCalibration,omod.grid.zip(parameters).toMap(),
           popStrata, carOwnership, finder,
           fullPopulation, affectedLinks, sensors
       )
       // Determine affected sensors
       // TODO temporal check
       val staticCount = sensors.associateWith { 0.0 }.toMutableMap()
       for(event in odProbs) {
           val odPair = Pair(omod.grid[event.o], omod.grid[event.d])
           if (odPair in affectedLinks) {
               val sensors = affectedLinks[odPair]!!
               for (sensor in sensors) {
                   staticCount[sensor] = staticCount[sensor]!! + event.pCar * event.pTransition * event.pChain * fullPopulation
               }
           }
       }
       val staticFlowLst = mutableMapOf<TrafficSensor, Double>()
       for (sensor in sensors) {
           val simFlow = staticCount[sensor]!!
           staticFlowLst[sensor] = simFlow
       }
       */

        val oldod = finder.determinePairProbabilities(
            omod.grid, omod.activityGenerator as ActivityGeneratorDefault,
            modeChoiceCalibration, mapOf(ActivityType.WORK to omod.grid.zip(parameters).toMap()),
            popStrata, carOwnership
        )

       /* val balalhbals = WACalClean.run(
            omod.grid,  omod.activityGenerator as ActivityGeneratorDefault,
            modeChoiceCalibration,omod.grid.zip(parameters).toMap(),
            popStrata, carOwnership, finder,fullPopulation, affectedLinks,
            sensors
        )*/

        /*val (ooptmatrix, k1) = OGradDescent.run(
            omod.grid,  omod.activityGenerator as ActivityGeneratorDefault,
            modeChoiceCalibration, mapOf(ActivityType.OTHER to omod.grid.zip(parameters).toMap()),
            popStrata, carOwnership, finder,fullPopulation, affectedLinks,
            sensors
        )*/

        /*
        val woptmatrixgg = OGradDescent.run(
            omod.grid,  omod.activityGenerator as ActivityGeneratorDefault,
            modeChoiceCalibration,omod.grid.zip(parameters).toMap(),
            popStrata, carOwnership, finder,fullPopulation, affectedLinks,
            sensors
        )*/

        //modeChoiceCal()
        val k1 = fourStepCal()

       //.times(fullPopulation)
       /*val test = SPCalibrator.determinePairProbabilities(
           omod.grid,  omod.activityGenerator as ActivityGeneratorDefault,
           modeChoiceCalibration,omod.grid.zip(parameters).toMap(),
           popStrata, carOwnership, finder
       )//.times(fullPopulation)
       println(test)*/
       val (_, sFlowBase, nAgentsVBase, sLocsBase) = runBatch( Array(omod.grid.size) { 1.0 } )

        /*val mModel = DefaultMetaModel(omod)
        for (activity in listOf(ActivityType.WORK, ActivityType.SCHOOL, ActivityType.OTHER, ActivityType.SHOPPING)) {
            val k1 = mModel.calibrateK1(activity, sensors, affectedLinks)
            finder.updateCellCValues(activity, k1.toTypedArray(), omod.grid)
        }*/

        /*
       val oforce = mutableMapOf<Cell, DoubleArray>()
       //val sforce = mutableMapOf<Cell, DoubleArray>()
       for ((i, cell) in omod.grid.withIndex()) {
           oforce[cell] = ooptmatrix.toArray()[i]
           //sforce[cell] = woptmatrixgg.second.toArray()[i]
       }
       //finder.forceWMatrix = wforce
       //finder.forceOMatrix = oforce
       finder.updateCellCValues(ActivityType.OTHER, k1.toTypedArray(), omod.grid)*/
       finder.updateCellCValues(ActivityType.OTHER, k1.toTypedArray(), omod.grid)

       val (_, sFlow, nAgents, sLocs) = runBatch(  Array(omod.grid.size) { 1.0 } )

       // Determine affected sensors
       // TODO temporal check
       //val staticCount = sensors.associateWith { 0.0 }.toMutableMap()
       /*val oldStaticCount = sensors.associateWith { 0.0 }.toMutableMap()
       for (origin in omod.grid) {
           for (destination in omod.grid) {
               val odPair = Pair(origin, destination)
               if (odPair in affectedLinks) {
                   val sensors = affectedLinks[odPair]!!
                   for (sensor in sensors) {
                       //staticCount[sensor] = staticCount[sensor]!! + test[odPair]!! * fullPopulation
                       oldStaticCount[sensor] = oldStaticCount[sensor]!! + oldod[odPair]!! * fullPopulation
                   }
               }
           }
       }*/


       //println(test.toList().take(10))
       //println(sLocs.take(10))

       // TODO Test END
       //omod.altPercentages = altPercentages

       //runIPF(10)
       //val bestPosition = runPSO(iterations = 1000)
       //val bestPosition = spoa(iterations = 1000)
       //val bestPosition = runIPF(iterations = 1000)
       //val bestPosition = Array<Double>(omod.grid.size) {1.0}
       //val bestPosition = runGradientDescent(10)

       //val (_, sFlow, nAgents, _) = runBatch( bestPosition )
       //val (_, staticFlow, staticMap) = determineJointOD( bestPosition )

       //val testOrigin = omod.grid[306]
       //val testDestination = omod.grid[307]
       //println("On test count in static map: ${staticMap[Pair(testOrigin, testDestination)]!! * nAgents}")
       // println("Total trips in static map:  ${staticMap.values.sum() * nAgents}")

        var mseSim = 0.0
        var mseSimBase = 0.0
        //var mseExpec = 0.0
        for (sensor in sensors) {
            mseSim += (sFlow[sensor]!! - sensor.measuredFlow).pow(2)
            mseSimBase += (sFlowBase[sensor]!! - sensor.measuredFlow).pow(2)
            //mseExpec += (staticCount[sensor]!! - sensor.measuredFlow).pow(2)
        }

       println("_".repeat(20*4 + 5*3))
       println("${"Sensor".padEnd(20)} | \t" +
               "${"Flow Simulated".padEnd(20)} | \t" +
               "${"Flow Simulated Base".padEnd(20)} | \t" +
              // "${"Flow Deterministic".padEnd(20)} | \t" +
               "Flow Measured".padEnd(20)
       )
        println("_".repeat(20) +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20)  +
                //" | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20))
        println(" ".repeat(20) +
                " | \t" + "%.2f e6".format(mseSim/sensors.size / 1e6).padEnd(20)  +
                " | \t" + "%.2f e6".format(mseSimBase/sensors.size / 1e6).padEnd(20)  +
                //" | \t" + "%.2f e6".format(mseExpec/sensors.size / 1e6).padEnd(20) +
                " | \t" + " ".repeat(20)
        )
       println("_".repeat(20) +
               " | \t" + "_".repeat(20)  +
               " | \t" + "_".repeat(20)  +
               //" | \t" + "_".repeat(20)  +
               " | \t" + "_".repeat(20))
       for ((i, flow) in sFlow.values.withIndex()) {
           println(
               "${sensors[i].name.padEnd(20)} | \t" +
               "${flow.toString().padEnd(20)} | \t" +
               "${sFlowBase[sensors[i]].toString().padEnd(20)} | \t" +
               //"${staticCount[sensors[i]].toString().padEnd(20)} | \t" +
               sensors[i].measuredFlow.toString().padEnd(20)
           )
       }
   }*/

    private fun fourStepCal() : List<Double> {
        omod.mainRng.setSeed(0) // Seed impact low with 100% of agents

        // Run Simulation
        val agents = omod.run(0.1)
        return calibrateK1(agents, omod, sensors, affectedLinks)
    }


    private fun modeChoiceCal() {
        omod.mainRng.setSeed(0) // Seed impact low with 100% of agents

        // Run Simulation
        val agents = omod.run(0.1)
        omod.doModeChoice(agents, ModeChoiceOption.FAST, false)

        // Mode Choice
        calibrate(agents, omod.mainRng, omod, sensors, affectedLinks)
    }


   private fun runBatch(parameters: Array<Double>) : BatchResult {
       // Set Parameters
       val finder = omod.destinationFinder as DestinationFinderDefault
       //finder.updateCellCValues(ActivityType.WORK, parameters, omod.grid)
       omod.mainRng.setSeed(0) // Seed impact low with 100% of agents

       // Run Simulation
       val agents = omod.run(0.1)
       omod.doModeChoice(agents, ModeChoiceOption.FAST, false)

       // Determine affected sensors
       var totalTripCount = 0
       //val testOrigin = omod.grid[306]
       //val testDestination = omod.grid[307]
       var testCount = 0.0
       val simCount = sensors.associateWith { 0.0 }.toMutableMap()
       var locBeforeTrip = Array(omod.grid.size) {0.0}.toMutableList()
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
                   //if ((origin.location.getAggLoc() == testOrigin) and (activity.location.getAggLoc() == testDestination)) {
                   //    testCount += 1
                   //}
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
               if (origin.location.getAggLoc() is Cell) {
                   val ocell = origin.location.getAggLoc() as Cell
                   val i = omod.grid.indexOf(ocell)
                   locBeforeTrip[i] = locBeforeTrip[i] + 1
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

       locBeforeTrip = locBeforeTrip.map { it * fullPopulation / agents.size }.toMutableList()

       val rslt = BatchResult(
           mse,
           allFlows,
           agents.size,
           locBeforeTrip
       )
       return rslt
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

       return affectedLinks
   }

   private fun determineAltAffectedLinks(
       grid: List<Cell>,
       sensors: List<TrafficSensor>,
       hopper: GraphHopper
   ) : Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>> {
       val geometryFactory = GeometryFactory()

       val sensorTree = HPRtree()
       for (sensor in sensors) {
           sensorTree.insert(sensor.field.envelopeInternal, sensor)
       }

       val affectedLinks: Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>> = runBlocking(omod.dispatcher) {
           channelFlow {
               for (origin in grid) {
                   launch {
                       for (destination in grid) {
                           val response = routeAltCar(origin, destination, hopper)
                           val thisAffected = mutableListOf<List<TrafficSensor>>()

                           for (path in response.all) {
                               val coords = path.points.map { Coordinate(it.lat, it.lon) }.toTypedArray()

                               if (coords.size >= 2) {
                                   val routeLine = omod.transformer.toModelCRS( geometryFactory.createLineString(coords) )

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

       return affectedLinks
   }
}


data class BatchResult (
   val mse: Double,
   val sensorCount: MutableMap<TrafficSensor, Double>,
   val nAgents: Int,
   val locBeforeTrip: List<Double>
)



