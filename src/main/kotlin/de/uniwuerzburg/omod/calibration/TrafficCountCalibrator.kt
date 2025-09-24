package de.uniwuerzburg.omod.calibration

import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.algorithms.PSO
import de.uniwuerzburg.omod.calibration.algorithms.SPSA
import de.uniwuerzburg.omod.core.CarOwnership
import de.uniwuerzburg.omod.core.DestinationFinderDefault
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray
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

    /*fun matrixTestRun() {
        val model = MetaModel.build(omod)
        val wm = model!!.calibrateMatrix(ActivityType.OTHER, sensors, affectedSensors)

        val finder = omod.destinationFinder as DestinationFinderDefault
        val force = mutableMapOf<Cell, DoubleArray>()
        for ((i, cell) in omod.grid.withIndex()) {
            force[cell] = wm!!.toArray()[i]
        }
        finder.forcedTransitionMatrix[ActivityType.OTHER] = force

        evaluate(DoubleArray(omod.grid.size) {1.0})
    }*/

    fun hpTune(activity: ActivityType, option: CalibrationOption) {
        val model = MetaModel.build(omod)!!.getDiffModel(activity, sensors, affectedSensors)
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
        val k = mutableMapOf<ActivityType, DoubleArray>()
        for (activity in listOf(ActivityType.WORK, ActivityType.SCHOOL, ActivityType.SCHOOL, ActivityType.OTHER)) {
            var d = when (option) {
                CalibrationOption.PSO -> calibratePSO(activity)
                CalibrationOption.MM_LBFGS -> calibrateMetaModelLBFGS(activity)
                CalibrationOption.SPSA -> calibrateSPSA(activity)
            }
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            k[activity] = d
        }
        evaluate(k)
    }

    fun evaluate(k: Map<ActivityType, DoubleArray>) {
        val finder = omod.destinationFinder as DestinationFinderDefault
        val flowBase = runBatch(0.1)
        //finder.updateCellCValues(ActivityType.OTHER, d.toTypedArray(), omod.grid)
        for ((activity, d) in k.entries) {
            val dcFunction = finder.locChoiceWeightFuns[activity]!!
            for ((cell, x) in omod.grid.zip(d.toTypedArray())) {
                cell.updateAttractionScaler(dcFunction, x)
            }
        }
        val flowCal = runBatch(0.1)

        var mseSim = 0.0
        var mseSimBase = 0.0
        for (sensor in sensors) {
            for ( t in 0 until T) {
                mseSim += (flowCal[sensor]!![t] - sensor.measuredFlow[t]).pow(2)
                mseSimBase += (flowBase[sensor]!![t] - sensor.measuredFlow[t]).pow(2)
            }
        }

        println("_".repeat(20*4 + 5*3))
        println("${"Sensor".padEnd(20)} | \t" +
                "${"T".padEnd(20)} | \t" +
                "${"Flow Simulated".padEnd(20)} | \t" +
                "${"Flow Simulated Base".padEnd(20)} | \t" +
                "Flow Measured".padEnd(20)
        )
        println("_".repeat(20) +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20))
        println(" ".repeat(20) +
                " | \t" + " ".repeat(20) +
                " | \t" + "%.2f e6".format(mseSim/sensors.size / 1e6).padEnd(20)  +
                " | \t" + "%.2f e6".format(mseSimBase/sensors.size / 1e6).padEnd(20)  +
                " | \t" + " ".repeat(20)
        )
        println("_".repeat(20) +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20))
        for ((i, flow) in flowCal.values.withIndex()) {
            for (seg in listOf(Pair(0, T))) {
                var optVal = 0.0
                var baseVal = 0.0
                var mVal = 0.0
                for (t in seg.first until seg.second) {
                    optVal += flow[t]
                    baseVal += flowBase[sensors[i]]!![t]
                    mVal += sensors[i].measuredFlow[t]
                }
                println(
                    "${sensors[i].name.padEnd(20)} | \t" +
                    "${(seg.first.toString() + "-" + seg.second).padEnd(20)} | \t" +
                    "${optVal.toString().padEnd(20)} | \t" +
                    "${baseVal.toString().padEnd(20)} | \t" +
                    mVal.toString().padEnd(20)
                )
            }
        }
    }

    fun calibratePSO(activityType: ActivityType) : DoubleArray {
        val model = MetaModel.build(omod)!!.getDiffModel(activityType, sensors, affectedSensors)

        val objective: (DoubleArray) -> Double = { x: DoubleArray ->
            model.evaluate(x)
        }
        val d = PSO.run(
            omod.grid.size - 1, objective, Random(), out=File("TestPSO${activityType}.csv"), vClamp = 1.0,  w = 0.8,
            phiP = 0.8 * 2,
            phiG = 0.8 * 2,
            nParticles = 40
        )
        return d
    }

    fun calibrateMetaModelLBFGS(activityType: ActivityType) : DoubleArray {
        val mModel = MetaModel.build(omod)!!
        return mModel.calibrateK1(activityType, sensors, affectedSensors).toDoubleArray() // TODO Activities
    }

    fun calibrateSPSA(activityType: ActivityType) : DoubleArray {
        val model = MetaModel.build(omod)!!.getDiffModel(activityType, sensors, affectedSensors)

        val objective: (DoubleArray) -> Double = { x: DoubleArray ->
            model.evaluate(x)
        }
        val d0 = DoubleArray(omod.grid.size - 1) { 1.0 }
        val d = SPSA.run(d0, objective, Random(), out=File("TestPSO.csv"))
        return d
    }

    /*private fun fourStepCal() : List<Double> {
        omod.mainRng.setSeed(0) // Seed impact low with 100% of agents

        // Run Simulation
        val agents = omod.run(0.1, verbose = false)
        return calibrateK1(agents, omod, sensors, affectedSensors)
    }*/


    /*private fun modeChoiceCal() {
        omod.mainRng.setSeed(0) // Seed impact low with 100% of agents

        // Run Simulation
        val agents = omod.run(0.1, verbose = false)
        omod.doModeChoice(agents, ModeChoiceOption.FAST, false)

        // Mode Choice
        calibrate(agents, omod.mainRng, omod, sensors, affectedSensors)
    }*/

   @Suppress("SameParameterValue")
   private fun runBatch(sharePop: Double) : Map<TrafficSensor, DoubleArray> {
       val fullPopulation = omod.buildings.sumOf { it.population }

       // Ensure results are deterministic
       omod.mainRng.setSeed(0)

       // Run Simulation
       val agents = omod.run(sharePop, verbose = false)
       omod.doModeChoice(agents, ModeChoiceOption.FAST, false)

       // Determine counts at sensors
       val simCount = sensors.associateWith { Array(T) {0.0} }.toMutableMap()
       val visitor: TripVisitor = { trip, originActivity, destinationActivity, departureTime, _, _ ->
           if (trip.mode == Mode.CAR_DRIVER) {
               if ((originActivity.location.getAggLoc() is Cell) and (destinationActivity.location.getAggLoc() is Cell)) {
                   val od = Pair(originActivity.location.getAggLoc() as Cell, destinationActivity.location.getAggLoc() as Cell)
                   if (od in affectedSensors) {
                       val sensors = affectedSensors[od]!!
                       for (sensor in sensors) {
                           val arr = simCount[sensor]!!
                           arr[departureTime.hour] = arr[departureTime.hour] + 1
                       }
                   }
               }
           }
       }
       for (agent in agents) {
           agent.mobilityDemand.first().visitTrips(visitor)
       }

       // Aggregate demand to sensors
       val allFlows = sensors.associateWith { DoubleArray(T) {0.0} }.toMutableMap()
       for (sensor in sensors) {
           for (t in 0 until T) {
               val simFlow = simCount[sensor]!![t] * fullPopulation / agents.size
               allFlows[sensor]!![t] = simFlow
           }
       }

       return allFlows
   }
}




