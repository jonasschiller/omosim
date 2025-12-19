package de.uniwuerzburg.omod.calibration

import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.algorithms.*
import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModelMultiOut
import de.uniwuerzburg.omod.calibration.surrogate.*
import de.uniwuerzburg.omod.core.DestinationFinderDefault
import de.uniwuerzburg.omod.core.ModeChoiceFast
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.*
import de.uniwuerzburg.omod.io.json.writeJson
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray
import java.io.File
import java.util.*
import kotlin.math.floor
import kotlin.math.pow

/**
 * Optimization algorithm options
 *
 * @see de.uniwuerzburg.omod.calibration.algorithms
 */
enum class CalibrationOption {
    MM_LBFGS,
    MM_MINBC,
    MM_GD,
    MM_PSO, PSO, PSO_AO,
    MM_SPSA, SPSA, SPSA_AO,
    MM_WSPSA, WSPSA,
    MM_MATRIX,
}

/**
 *  Specify what part of the model should be calibrated.
 *
 *  GRAVITY: Attraction scaling in the gravity model
 *  MODE_CHOICE: Car mode intercept in mode choice model
 *  ROUTE_CHOICE: Route choice between alternatives given by GraphHopper
 *  EVALUATE: Test the current calibration and print out a summary
 */
enum class CalibrationType {
    GRAVITY, MODE_CHOICE, ROUTE_CHOICE, EVALUATE
}

/**
 * Calibrate OMoSim with traffic count data
 *
 * @param omod Simulator
 */
class TrafficCountCalibrator(
    linkDataFile: File,
    val omod: Omod
) {
    private val sensors: List<TrafficSensor> = TrafficSensor.readSensorData(linkDataFile, omod) // Read traffic count data
    private val finder = omod.destinationFinder as DestinationFinderDefault
    private val affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    private var affectedAltSensors: Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>> = mapOf()

    init {
        T = sensors.first().measuredFlow.size // Set number of time slices
        if (!sensors.all { it.measuredFlow.size == T }) {
            throw IllegalArgumentException(
                "Sensor measurement arrays are not uniformly sized!" +
                "Validate the --cal_traffic_count_file file. "
            )
        }
        affectedSensors = TrafficSensor.affectedSensors(sensors, omod)
    }

    /**
     * Calibration entry point.
     *
     * @param gravityCalOut File where to store the calibration result for the GRAVITY step
     * @param modeChoiceCalOut  File where to store the calibration result for the MODE_CHOICE step
     * @param option Calibration option to use // TODO
     * @param activities Activities should be calibrated. Only matters for Gravity calibration
     * @param iterations Maximum number of iterations
     * @param steps Calibration steps to be done.
     */
    fun calibrate(
        gravityCalOut: File,
        modeChoiceCalOut: File,
        option: CalibrationOption,
        activities: List<ActivityType>,
        iterations: Int = 100,
        parameters: Map<String, String>? = null,
        steps: List<CalibrationType> = listOf(CalibrationType.GRAVITY, CalibrationType.EVALUATE)
    ) {
        // Complete the steps in the given order
        for (step in steps) {
            when(step) {
                CalibrationType.GRAVITY -> {
                    when (option) {
                        CalibrationOption.MM_LBFGS  -> calibrateLBFGSMM(activities, iterations, parameters)
                        CalibrationOption.MM_MINBC  -> calibrateMinBcMM(activities, iterations, parameters)
                        CalibrationOption.MM_GD     -> calibrateGGMM(activities, iterations, parameters)
                        CalibrationOption.MM_PSO    -> calibratePSOMM(activities, iterations, parameters)
                        CalibrationOption.PSO       -> calibratePSO(activities, iterations, parameters)
                        CalibrationOption.PSO_AO    -> calibratePSOAllAtOnce(activities, iterations, parameters)
                        CalibrationOption.MM_SPSA   -> calibrateSPSAMM(activities, iterations, parameters)
                        CalibrationOption.SPSA      -> calibrateSPSA(activities, iterations, parameters)
                        CalibrationOption.SPSA_AO   -> calibrateSPSAAllAtOnce(activities, iterations, parameters)
                        CalibrationOption.MM_WSPSA  -> calibrateWSPSAMM(activities, iterations, parameters)
                        CalibrationOption.WSPSA     -> calibrateWSPSA(activities, iterations, parameters)
                        CalibrationOption.MM_MATRIX -> calibrateMatrix(activities)
                    }
                    val finder = omod.destinationFinder as DestinationFinderDefault
                    GravityCalibrationStore.write(gravityCalOut, omod.buildings, finder.locChoiceWeightFuns)
                }
                CalibrationType.MODE_CHOICE -> {
                    modeChoiceCal(modeChoiceCalOut, ModeChoiceCalibrationObjective.FitIndividualMeasurements)
                    omod.tourModeUtilityFn = modeChoiceCalOut
                }
                CalibrationType.ROUTE_CHOICE -> {
                    altRouteCal()
                }
                CalibrationType.EVALUATE -> {
                    evaluate(0.1)
                }
            }
        }
    }

    private fun modeChoiceCal(calFile: File, objectiveType: ModeChoiceCalibrationObjective) {
        omod.mainRng.setSeed(0) // Seed impact low with 100% of agents

        // Run Simulation
        val agents = omod.run(0.1, verbose = false)
        omod.doModeChoice(agents, ModeChoiceOption.FAST, false, false)

        // Create mode choice surrogate model
        val mc = ModeChoiceFast(omod.routingCache)
        val model = ModeChoice.buildModel(agents, mc, omod.mainRng, omod, sensors, affectedSensors, objectiveType)

        // Get X0
        val carUtil = mc.tourModeOptions.find { it.mode == Mode.CAR_DRIVER }
        val x0 = doubleArrayOf(carUtil!!.intercept)

        // Optimize
        val x = BFGS.run(model, x0, lb=-50.0, ub=50.0, iterations=50)

        // Store calibration
        carUtil.intercept = x[0]
        writeJson(mc.tourModeOptions, calFile)
    }

    private fun altRouteCal() {
        omod.mainRng.setSeed(0) // Seed impact low with 100% of agents

        // Run Simulation
        val agents = omod.run(0.1, verbose = false)
        omod.doModeChoice(agents, ModeChoiceOption.FAST, false, false)

        affectedAltSensors = TrafficSensor.altAffectedSensors(sensors, omod)

        // Mode Choice
        omod.altPercentages = RouteChoice.optimize(agents, omod, sensors, affectedAltSensors)
    }

    fun evaluate(sharePop: Double) {
        val finder = omod.destinationFinder as DestinationFinderDefault
        val flowCal = runBatch(sharePop)

        // Clear calibration
        for (activity in ActivityType.entries) {
            val dcFunction = finder.locChoiceWeightFuns[activity]!!
            for (cell in omod.grid) {
                cell.updateAttractionScaler(dcFunction, 1.0)
            }
        }
        finder.forcedTransitionMatrix.clear()
        omod.tourModeUtilityFn = null
        omod.altPercentages = mapOf()

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

    // L-BFGS
    private fun calibrateLBFGSMM(
        activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    )  {
        for (activity in activities) {
            val model = SGGravity(omod).buildModelSSE(activity, sensors, affectedSensors)
            val x0 = DoubleArray(omod.grid.size - 1) { 1.0 }
            var d =  BFGS.run(model, x0, iterations=iterations, parameters=parameters)
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }

    // MinBC (CG)
    private fun calibrateMinBcMM(
        activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    )  {
        for (activity in activities) {
            val model = SGGravity(omod).buildModelSSE(activity, sensors, affectedSensors)
            val x0 = DoubleArray(omod.grid.size - 1) { 1.0 }
            var d =  MinBc.run(model, x0, iterations=iterations, parameters=parameters)
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }

    // Gradient Descent
    private fun calibrateGGMM(
        activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ){
        for (activity in activities) {
            val model = SGGravity(omod).buildModelSSE(activity, sensors, affectedSensors)
            val x0 = DoubleArray(omod.grid.size - 1) { 1.0 }
            var d = GradientDescent.run(model, x0, iterations=iterations, parameters=parameters)
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }

    // PSO
    private fun calibratePSOMM(
        activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ){
        for (activity in activities) {
            val objective = metaModelObj(activity)
            var d = PSO.run(
                omod.grid.size - 1, objective, Random(),
                iterations = iterations, parameters = parameters
            )
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }
    private fun calibratePSO(
        activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ) {
        for (activity in activities) {
            val objective = batchObj(activity)
            val d = PSO.run(
                omod.grid.size, objective, Random(),
                iterations = iterations, parameters = parameters
            )
            updateCalibration(d, activity)
        }
    }
    private fun calibratePSOAllAtOnce(
        activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ) {
        val objective = batchObj(activities)
        val d = PSO.run(
            omod.grid.size * activities.size, objective, Random(),
            iterations = iterations, parameters = parameters
        )
        updateCalibration(d, activities)
    }

    // SPSA
    fun calibrateSPSAMM(
        activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ) {
         for (activity in activities) {
            val objective = metaModelObj(activity)
            val x0 = DoubleArray(omod.grid.size - 1) { 1.0 }
            var d = SPSA.run(x0, objective, Random(), iterations = iterations, parameters = parameters)

            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }
    private fun calibrateSPSAAllAtOnce(
        activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ) {
        val objective = batchObj(activities)
        val x0 = DoubleArray(omod.grid.size * activities.size) { 1.0 }
        val d = SPSA.run(x0, objective, Random(), iterations = iterations, parameters = parameters)
        updateCalibration(d, activities)
    }
    private fun calibrateSPSA(
        activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ) {
        for (activity in activities) {
            val objective = batchObj(activity)
            val x0 = DoubleArray(omod.grid.size ) { 1.0 }
            val d = SPSA.run(x0, objective, Random(), iterations = iterations, parameters = parameters)
            updateCalibration(d, activity)
        }
    }

    // W-SPSA
    private fun calibrateWSPSAMM(
        activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ) {
        val measurements = sensors.map { it.measuredFlow }.flatMap { it.toList() }

        for (activity in activities) {
            val model = SGGravity(omod).buildModelSimCounts(activity, sensors, affectedSensors)
            val objective = metaModelObjWSPSA(model, sensors)
            val x0 = DoubleArray(omod.grid.size - 1) { 1.0 }
            var d = WSPSA.run(
                x0, objective, measurements, model, Random(),
                iterations = iterations, parameters = parameters
            )
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }
    private fun calibrateWSPSA(
        activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ) {
        val measurements = sensors.map { it.measuredFlow }.flatMap { it.toList() }

        for (activity in activities) {
            val model = SGGravity(omod).buildModelSimCounts(activity, sensors, affectedSensors)
            val objective = batchObjWSPSA(activity)
            val x0 = DoubleArray(omod.grid.size - 1) { 1.0 }
            var d = WSPSA.run(
                x0, objective, measurements, model, Random(),
                iterations = iterations, parameters = parameters
            )
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }

    private fun calibrateMatrix(activities: List<ActivityType>) {
        for (activity in activities) {
            val model = SGGravity(omod)
            val wm = model.optimizeTMatrix(activity, sensors, affectedSensors)

            val finder = omod.destinationFinder as DestinationFinderDefault
            val force = mutableMapOf<Cell, DoubleArray>()
            for ((i, cell) in omod.grid.withIndex()) {
                force[cell] = wm!!.toArray()[i]
            }
            finder.forcedTransitionMatrix[activity] = force
        }
    }

    // Objectives
    private fun metaModelObj(activity: ActivityType): (DoubleArray) -> Double {
        val model = SGGravity(omod).buildModelSSE(activity, sensors, affectedSensors)
        return { x: DoubleArray ->
            model.evaluate(x)
        }
    }
    private fun metaModelObjWSPSA(model: DifferentiableModelMultiOut, sensors: List<TrafficSensor>): (DoubleArray) ->
        Pair<Double, DoubleArray> {
        return { x: DoubleArray ->
            val simCounts = model.evaluate(x)

            val flows = mutableMapOf<TrafficSensor, DoubleArray>()
            var i = 0
            for (sensor in sensors) {
                val sensorCounts = DoubleArray(T) {0.0}
                for (t in 0 until T) {
                    sensorCounts[t] = simCounts[i]
                    i += 1
                }
                flows[sensor] = sensorCounts
            }
            flowMSE(flows)

            Pair(flowMSE(flows), simCounts)
        }
    }
    private fun batchObj(activities: List<ActivityType>): (DoubleArray) -> Double {
        return { x: DoubleArray ->
            for ((ai, activity) in activities.withIndex()) {
                val dcFunction = finder.locChoiceWeightFuns[activity]!!
                for ((gi, cell) in omod.grid.withIndex()) {
                    cell.updateAttractionScaler(dcFunction, x[ai * omod.grid.size + gi])
                }
            }
            val flows = runBatch(0.1)
            flowMSE(flows)
        }
    }
    private fun batchObj(activity: ActivityType): (DoubleArray) -> Double {
        return { x: DoubleArray ->
            val dcFunction = finder.locChoiceWeightFuns[activity]!!
            for ((i, cell) in omod.grid.withIndex()) {
                cell.updateAttractionScaler(dcFunction, x[i])
            }
            val flows = runBatch(0.1)
            flowMSE(flows)
        }
    }

    private fun batchObjWSPSA(activity: ActivityType): (DoubleArray) -> Pair<Double, DoubleArray> {
        return { xTmp: DoubleArray ->
            val x = (xTmp.toList() + listOf(1.0)).toDoubleArray()
            val dcFunction = finder.locChoiceWeightFuns[activity]!!
            for ((i, cell) in omod.grid.withIndex()) {
                cell.updateAttractionScaler(dcFunction, x[i])
            }
            val flows = runBatch(0.1)
            flowMSE(flows)

            val countsFlat = mutableListOf<Double>()
            for (sensor in sensors) {
                for (t in 0 until T) {
                    countsFlat.add( flows[sensor]!![t] )
                }
            }

            Pair(flowMSE(flows), countsFlat.toDoubleArray())
        }
    }

    // Calibration updates
    private fun updateCalibration(x: DoubleArray, activity: ActivityType) {
        val dcFunction = finder.locChoiceWeightFuns[activity]!!
        for ((cell, v) in omod.grid.zip(x.toTypedArray())) {
            cell.updateAttractionScaler(dcFunction, v)
        }
    }
    private fun updateCalibration(x: DoubleArray, activities: List<ActivityType>) {
        for ((ai, activity) in activities.withIndex()) {
            val dcFunction = finder.locChoiceWeightFuns[activity]!!
            for ((gi, cell) in omod.grid.withIndex()) {
                cell.updateAttractionScaler(dcFunction, x[ai * omod.grid.size + gi])
            }
        }
    }

   @Suppress("SameParameterValue")
   private fun runBatch(sharePop: Double) : Map<TrafficSensor, DoubleArray> {
       val fullPopulation = omod.buildings.sumOf { it.population }

       // Ensure results are deterministic
       omod.mainRng.setSeed(0)

       // Run Simulation
       val agents = omod.run(sharePop, verbose = false)
       omod.doModeChoice(agents, ModeChoiceOption.FAST, false, verbose = false)

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
                       if (key in omod.altPercentages) {
                           val p = omod.altPercentages[key]!!
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

    private fun flowMSE(flowCal: Map<TrafficSensor, DoubleArray>) : Double {
        var mse = 0.0
        for (sensor in sensors) {
            for ( t in 0 until T) {
                mse += (flowCal[sensor]!![t] - sensor.measuredFlow[t]).pow(2)
            }
        }
        return mse
    }
}




