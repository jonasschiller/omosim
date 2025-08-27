package de.uniwuerzburg.omod.calibration

import com.graphhopper.GraphHopper
import de.uniwuerzburg.omod.calibration.algorithms.PSO
import de.uniwuerzburg.omod.calibration.algorithms.SPSA
import de.uniwuerzburg.omod.core.CarOwnership
import de.uniwuerzburg.omod.core.DestinationFinderDefault
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.*
import de.uniwuerzburg.omod.routing.routeAltCar
import de.uniwuerzburg.omod.routing.routeWith
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.geotools.filter.function.StaticGeometry.intersection
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.GeometryFactory
import org.locationtech.jts.geom.LineString
import org.locationtech.jts.geom.MultiLineString
import org.locationtech.jts.index.hprtree.HPRtree
import java.io.File
import java.nio.file.Paths
import java.util.*
import kotlin.math.pow

enum class CalibrationOption {
    PSO, MM_LBFGS, SPSA
}

class TrafficCountCalibrator(
    linkDataFile: File,
    val omod: Omod,
    val carOwnership: CarOwnership
) {
    private val sensors: List<TrafficSensor>
    private val affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>

    init {
        sensors = TrafficSensor.readSensorData(linkDataFile, omod)
        affectedSensors = TrafficSensor.affectedSensors(sensors, omod)
    }

    fun matrixTestRun() {
        val model = MetaModel.build(omod)
        val wm = model!!.calibrateMatrix(ActivityType.OTHER, sensors, affectedSensors)

        val finder = omod.destinationFinder as DestinationFinderDefault
        val force = mutableMapOf<Cell, DoubleArray>()
        for ((i, cell) in omod.grid.withIndex()) {
            force[cell] = wm!!.toArray()[i]
        }
        finder.forceOMatrix = force

        evaluate(DoubleArray(omod.grid.size) {1.0})
    }

    fun hpTune(option: CalibrationOption) {
        val model = MetaModel.build(omod)!!.getDiffModel(ActivityType.OTHER, sensors, affectedSensors)
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
        val flowBase = runBatch(0.1)
        finder.updateCellCValues(ActivityType.OTHER, d.toTypedArray(), omod.grid)
        val flowCal = runBatch(0.1)

        var mseSim = 0.0
        var mseSimBase = 0.0
        for (sensor in sensors) {
            mseSim += (flowCal[sensor]!! - sensor.measuredFlow).pow(2)
            mseSimBase += (flowBase[sensor]!! - sensor.measuredFlow).pow(2)
        }

        println("_".repeat(20*4 + 5*3))
        println("${"Sensor".padEnd(20)} | \t" +
                "${"Flow Simulated".padEnd(20)} | \t" +
                "${"Flow Simulated Base".padEnd(20)} | \t" +
                "Flow Measured".padEnd(20)
        )
        println("_".repeat(20) +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20))
        println(" ".repeat(20) +
                " | \t" + "%.2f e6".format(mseSim/sensors.size / 1e6).padEnd(20)  +
                " | \t" + "%.2f e6".format(mseSimBase/sensors.size / 1e6).padEnd(20)  +
                " | \t" + " ".repeat(20)
        )
        println("_".repeat(20) +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20))
        for ((i, flow) in flowCal.values.withIndex()) {
            println(
                "${sensors[i].name.padEnd(20)} | \t" +
                        "${flow.toString().padEnd(20)} | \t" +
                        "${flowBase[sensors[i]].toString().padEnd(20)} | \t" +
                        sensors[i].measuredFlow.toString().padEnd(20)
            )
        }
    }

    fun calibratePSO() : DoubleArray {
        val model = MetaModel.build(omod)!!.getDiffModel(ActivityType.OTHER, sensors, affectedSensors)

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
        val mModel = MetaModel.build(omod)!!
        return mModel.calibrateK1(ActivityType.OTHER, sensors, affectedSensors).toDoubleArray()
    }

    fun calibrateSPSA() : DoubleArray {
        val model = MetaModel.build(omod)!!.getDiffModel(ActivityType.OTHER, sensors, affectedSensors)

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
                        if (od in affectedSensors) {
                            val sensors = affectedSensors[od]!!
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

    private fun fourStepCal() : List<Double> {
        omod.mainRng.setSeed(0) // Seed impact low with 100% of agents

        // Run Simulation
        val agents = omod.run(0.1)
        return calibrateK1(agents, omod, sensors, affectedSensors)
    }


    private fun modeChoiceCal() {
        omod.mainRng.setSeed(0) // Seed impact low with 100% of agents

        // Run Simulation
        val agents = omod.run(0.1)
        omod.doModeChoice(agents, ModeChoiceOption.FAST, false)

        // Mode Choice
        calibrate(agents, omod.mainRng, omod, sensors, affectedSensors)
    }

   private fun runBatch(sharePop: Double) : Map<TrafficSensor, Double> {
       val fullPopulation = omod.buildings.sumOf { it.population }

       // Ensure results are deterministic
       omod.mainRng.setSeed(0)

       // Run Simulation
       val agents = omod.run(sharePop)
       omod.doModeChoice(agents, ModeChoiceOption.FAST, false)

       // Determine counts at sensors
       val simCount = sensors.associateWith { 0.0 }.toMutableMap()
       for (agent in agents) {
           var origin = agent.mobilityDemand.first().activities.first()
           val activities = agent.mobilityDemand.first().activities.drop(1)

           val trips = agent.mobilityDemand.first().trips
           for ((activity, trip) in activities.zip(trips)) {
               if (trip.mode == Mode.CAR_DRIVER) {
                   if ((origin.location.getAggLoc() is Cell) and (activity.location.getAggLoc() is Cell)) {
                       val od = Pair(origin.location.getAggLoc() as Cell, activity.location.getAggLoc() as Cell)
                       if (od in affectedSensors) {
                           val sensors = affectedSensors[od]!!
                           for (sensor in sensors) {
                               simCount[sensor] = simCount[sensor]!! + 1
                           }
                       }
                   }
               }
               origin = activity
           }
       }

       // Aggregate demand to sensors
       val allFlows = mutableMapOf<TrafficSensor, Double>()
       for (sensor in sensors) {
           val simFlow = simCount[sensor]!! * fullPopulation / agents.size
           allFlows[sensor] = simFlow
       }

       return allFlows
   }
}




