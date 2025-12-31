package de.uniwuerzburg.omosim.calibration

import de.uniwuerzburg.omosim.calibration.CalibrationConstants.T
import de.uniwuerzburg.omosim.calibration.algorithms.*
import de.uniwuerzburg.omosim.cli.CalibrationStep
import de.uniwuerzburg.omosim.core.DestinationFinderDefault
import de.uniwuerzburg.omosim.core.ModeChoiceFast
import de.uniwuerzburg.omosim.core.Omosim
import de.uniwuerzburg.omosim.core.models.*
import de.uniwuerzburg.omosim.io.json.writeJson
import java.io.File
import kotlin.math.floor
import kotlin.math.pow

/**
 * Calibrate OMoSim with traffic count data // TODO stores calibration context
 *
 * @param omosim Simulator
 */
class TrafficCountCalibrationContext(
    linkDataFile: File,
    val omosim: Omosim
) {
    val sensors: List<TrafficSensor> = TrafficSensor.readSensorData(linkDataFile, omosim) // Read traffic count data
    val finder = omosim.destinationFinder as DestinationFinderDefault
    val affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    var affectedAltSensors: Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>> = mapOf()

    init {
        T = sensors.first().measuredFlow.size // Set number of time slices
        if (!sensors.all { it.measuredFlow.size == T }) {
            throw IllegalArgumentException(
                "Sensor measurement arrays are not uniformly sized!" +
                "Validate the --cal_traffic_count_file file. "
            )
        }
        affectedSensors = TrafficSensor.affectedSensors(sensors, omosim)
    }

    /**
     * Calibration entry point.
     *
     * @param gravityCalOut File where to store the calibration result for the GRAVITY step
     * @param modeChoiceCalOut  File where to store the calibration result for the MODE_CHOICE step
     * @param steps Calibration steps to be done.
     */
    fun calibrate(
        gravityCalOut: File,
        modeChoiceCalOut: File,
        steps: List<CalibrationStep>
    ) {
        // Complete the steps in the given order
        for ((i, step) in steps.withIndex()) {
            when(step.type) {
                CalibrationType.GRAVITY -> {
                    Gravity(this).calibrate(step.alg, step.activities, step.parameters)
                    val finder = omosim.destinationFinder as DestinationFinderDefault
                    GravityCalibrationStore.write(gravityCalOut, omosim.buildings, finder.locChoiceWeightFuns)
                }
                CalibrationType.MODE_CHOICE -> {
                    modeChoiceCal(modeChoiceCalOut, ModeChoiceCalibrationObjective.FitIndividualMeasurements)
                    omosim.tourModeUtilityFn = modeChoiceCalOut
                }
                CalibrationType.ROUTE_CHOICE -> {
                    altRouteCal()
                }
                CalibrationType.EVALUATE -> {
                    evaluate(0.1)
                }
                null -> { logger.warn("Calibration step $i skipped. Step type is null.") }
            }
        }
    }

    private fun modeChoiceCal(calFile: File, objectiveType: ModeChoiceCalibrationObjective) {
        omosim.mainRng.setSeed(0) // Seed impact low with 100% of agents

        // Run Simulation
        val agents = omosim.run(0.1, verbose = false)
        omosim.doModeChoice(agents, ModeChoiceOption.FAST, false, false)

        // Create mode choice surrogate model
        val mc = ModeChoiceFast(omosim.routingCache)
        val model = ModeChoice.buildModel(agents, mc, omosim.mainRng, omosim, sensors, affectedSensors, objectiveType)

        // Get X0
        val carUtil = mc.tourModeOptions.find { it.mode == Mode.CAR_DRIVER }
        val x0 = doubleArrayOf(carUtil!!.intercept)

        // Optimize
        val x = BFGS.run(model, x0, lb=-50.0, ub=50.0)

        // Store calibration
        carUtil.intercept = x[0]
        writeJson(mc.tourModeOptions, calFile)
    }

    private fun altRouteCal() {
        omosim.mainRng.setSeed(0) // Seed impact low with 100% of agents

        // Run Simulation
        val agents = omosim.run(0.1, verbose = false)
        omosim.doModeChoice(agents, ModeChoiceOption.FAST, false, false)

        affectedAltSensors = TrafficSensor.altAffectedSensors(sensors, omosim)

        // Mode Choice
        omosim.altPercentages = RouteChoice.optimize(agents, omosim, sensors, affectedAltSensors)
    }

    fun evaluate(sharePop: Double) {
        val finder = omosim.destinationFinder as DestinationFinderDefault
        val flowCal = runBatch(sharePop)

        // Clear calibration
        for (activity in ActivityType.entries) {
            val dcFunction = finder.locChoiceWeightFuns[activity]!!
            for (cell in omosim.grid) {
                cell.updateAttractionScaler(dcFunction, 1.0)
            }
        }
        finder.forcedTransitionMatrix.clear()
        omosim.tourModeUtilityFn = null
        omosim.altPercentages = mapOf()

        // Run uncalibrated
        val flowBase = runBatch(sharePop)

        // Calculate Loss
        var mseSim = 0.0
        var mseSimBase = 0.0
        for (sensor in sensors) {
            for ( t in 0 until T) {
                mseSim += (flowCal[sensor]!![t] - sensor.measuredFlow[t]).pow(2)
                mseSimBase += (flowBase[sensor]!![t] - sensor.measuredFlow[t]).pow(2)
            }
        }

        // Print table
        println("_".repeat(20*4 + 5*3))
        println("${"Sensor".padEnd(20)} | \t" +
                "${"T".padEnd(20)} | \t" +
                "${"Flow Simulated".padEnd(20)} | \t" +
                "${"Flow Simulated Base".padEnd(20)} | \t" +
                "Flow Measured".padEnd(20)
        )
        printTabHLine()
        println(" ".repeat(20) +
                " | \t" + " ".repeat(20) +
                " | \t" + "%.2f e6".format(mseSim/sensors.size / 1e6).padEnd(20)  +
                " | \t" + "%.2f e6".format(mseSimBase/sensors.size / 1e6).padEnd(20)  +
                " | \t" + " ".repeat(20)
        )
        printTabHLine()
        // Measurement vs Simulated
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

    private fun printTabHLine() {
        println(
            "_".repeat(20) +
            " | \t" + "_".repeat(20)  +
            " | \t" + "_".repeat(20)  +
            " | \t" + "_".repeat(20)  +
            " | \t" + "_".repeat(20)
        )
    }

   @Suppress("SameParameterValue")
   fun runBatch(sharePop: Double) : Map<TrafficSensor, DoubleArray> {
       val fullPopulation = omosim.buildings.sumOf { it.population }

       // Ensure results are deterministic
       omosim.mainRng.setSeed(0)

       // Run Simulation
       val agents = omosim.run(sharePop, verbose = false)
       omosim.doModeChoice(agents, ModeChoiceOption.FAST, false, verbose = false)

       // Determine counts at sensors
       val simCount = sensors.associateWith { Array(T) {0.0} }.toMutableMap()
       val visitor: TripVisitor = { trip, originActivity, destinationActivity, departureTime, _, _ ->
           val mod = departureTime.minute + departureTime.hour * 60
           val t = floor((mod % 1440.0) / 1440.0 * T).toInt()

           if (trip.mode == Mode.CAR_DRIVER) {
               if ((originActivity.location.getAggLoc() is Cell) and (destinationActivity.location.getAggLoc() is Cell)) {
                   val od = Pair(originActivity.location.getAggLoc() as Cell, destinationActivity.location.getAggLoc() as Cell)

                   if (od in affectedSensors) {
                       val key = ODTTriple(od.first, od.second, t)
                       if (key in omosim.altPercentages) {
                           val p = omosim.altPercentages[key]!!
                           for ((i, alternative) in affectedAltSensors[od]!!.withIndex()) {
                               for (sensor in alternative) {
                                   val arr = simCount[sensor]!!
                                   arr[t] = arr[t] + p[i]
                               }
                           }
                       } else {
                           val sensors = affectedSensors[od]!!
                           for (sensor in sensors) {
                               val arr = simCount[sensor]!!

                               arr[t] = arr[t] + 1
                           }
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

    fun flowMSE(flowCal: Map<TrafficSensor, DoubleArray>) : Double {
        var mse = 0.0
        for (sensor in sensors) {
            for ( t in 0 until T) {
                mse += (flowCal[sensor]!![t] - sensor.measuredFlow[t]).pow(2)
            }
        }
        return mse
    }
}




