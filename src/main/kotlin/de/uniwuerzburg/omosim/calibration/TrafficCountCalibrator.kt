package de.uniwuerzburg.omosim.calibration

import de.uniwuerzburg.omosim.calibration.CalibrationConstants.T
import de.uniwuerzburg.omosim.calibration.algorithms.*
import de.uniwuerzburg.omosim.calibration.differentiablemodel.DifferentiableModelMultiOut
import de.uniwuerzburg.omosim.calibration.surrogate.*
import de.uniwuerzburg.omosim.cli.CalibrationStep
import de.uniwuerzburg.omosim.core.DestinationFinderDefault
import de.uniwuerzburg.omosim.core.ModeChoiceFast
import de.uniwuerzburg.omosim.core.Omosim
import de.uniwuerzburg.omosim.core.models.*
import de.uniwuerzburg.omosim.io.json.writeJson
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray
import java.io.File
import java.util.*
import kotlin.math.floor
import kotlin.math.pow

/**
 * Optimization algorithm options
 *
 * @see de.uniwuerzburg.omosim.calibration.algorithms
 */
enum class CalibrationAlgorithm {
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
 * @param omosim Simulator
 */
class TrafficCountCalibrator(
    linkDataFile: File,
    val omosim: Omosim
) {
    private val sensors: List<TrafficSensor> = TrafficSensor.readSensorData(linkDataFile, omosim) // Read traffic count data
    private val finder = omosim.destinationFinder as DestinationFinderDefault
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
                    when (step.alg) {
                        CalibrationAlgorithm.MM_LBFGS  -> calibrateLBFGSMM(step.activities, step.parameters)
                        CalibrationAlgorithm.MM_MINBC  -> calibrateMinBcMM(step.activities, step.parameters)
                        CalibrationAlgorithm.MM_GD     -> calibrateGGMM(step.activities, step.parameters)
                        CalibrationAlgorithm.MM_PSO    -> calibratePSOMM(step.activities, step.parameters)
                        CalibrationAlgorithm.PSO       -> calibratePSO(step.activities, step.parameters)
                        CalibrationAlgorithm.PSO_AO    -> calibratePSOAllAtOnce(step.activities, step.parameters)
                        CalibrationAlgorithm.MM_SPSA   -> calibrateSPSAMM(step.activities, step.parameters)
                        CalibrationAlgorithm.SPSA      -> calibrateSPSA(step.activities, step.parameters)
                        CalibrationAlgorithm.SPSA_AO   -> calibrateSPSAAllAtOnce(step.activities, step.parameters)
                        CalibrationAlgorithm.MM_WSPSA  -> calibrateWSPSAMM(step.activities, step.parameters)
                        CalibrationAlgorithm.WSPSA     -> calibrateWSPSA(step.activities, step.parameters)
                        CalibrationAlgorithm.MM_MATRIX -> calibrateMatrix(step.activities)
                        null -> throw IllegalArgumentException("Algorithm can't be null for Gravity model calibration!")
                    }
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

    // L-BFGS
    private fun calibrateLBFGSMM(
        activities: List<ActivityType>, parameters: Map<String, String>? = null
    )  {
        for (activity in activities) {
            val model = SGGravity(omosim).buildModelSSE(activity, sensors, affectedSensors)
            val x0 = DoubleArray(omosim.grid.size - 1) { 1.0 }
            var d =  BFGS.run(model, x0, parameters=parameters)
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }

    // MinBC (CG)
    private fun calibrateMinBcMM(
        activities: List<ActivityType>, parameters: Map<String, String>? = null
    )  {
        for (activity in activities) {
            val model = SGGravity(omosim).buildModelSSE(activity, sensors, affectedSensors)
            val x0 = DoubleArray(omosim.grid.size - 1) { 1.0 }
            var d =  MinBc.run(model, x0, parameters=parameters)
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }

    // Gradient Descent
    private fun calibrateGGMM(
        activities: List<ActivityType>, parameters: Map<String, String>? = null
    ){
        for (activity in activities) {
            val model = SGGravity(omosim).buildModelSSE(activity, sensors, affectedSensors)
            val x0 = DoubleArray(omosim.grid.size - 1) { 1.0 }
            var d = GradientDescent.run(model, x0, parameters=parameters)
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }

    // PSO
    private fun calibratePSOMM(
        activities: List<ActivityType>, parameters: Map<String, String>? = null
    ){
        for (activity in activities) {
            val objective = metaModelObj(activity)
            var d = PSO.run(
                omosim.grid.size - 1, objective, Random(), parameters = parameters
            )
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }
    private fun calibratePSO(
        activities: List<ActivityType>, parameters: Map<String, String>? = null
    ) {
        for (activity in activities) {
            val objective = batchObj(activity)
            val d = PSO.run(
                omosim.grid.size, objective, Random(), parameters = parameters
            )
            updateCalibration(d, activity)
        }
    }
    private fun calibratePSOAllAtOnce(
        activities: List<ActivityType>, parameters: Map<String, String>? = null
    ) {
        val objective = batchObj(activities)
        val d = PSO.run(
            omosim.grid.size * activities.size, objective, Random(), parameters = parameters
        )
        updateCalibration(d, activities)
    }

    // SPSA
    fun calibrateSPSAMM(
        activities: List<ActivityType>, parameters: Map<String, String>? = null
    ) {
         for (activity in activities) {
            val objective = metaModelObj(activity)
            val x0 = DoubleArray(omosim.grid.size - 1) { 1.0 }
            var d = SPSA.run(x0, objective, Random(), parameters = parameters)

            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }
    private fun calibrateSPSAAllAtOnce(
        activities: List<ActivityType>, parameters: Map<String, String>? = null
    ) {
        val objective = batchObj(activities)
        val x0 = DoubleArray(omosim.grid.size * activities.size) { 1.0 }
        val d = SPSA.run(x0, objective, Random(), parameters = parameters)
        updateCalibration(d, activities)
    }
    private fun calibrateSPSA(
        activities: List<ActivityType>, parameters: Map<String, String>? = null
    ) {
        for (activity in activities) {
            val objective = batchObj(activity)
            val x0 = DoubleArray(omosim.grid.size ) { 1.0 }
            val d = SPSA.run(x0, objective, Random(), parameters = parameters)
            updateCalibration(d, activity)
        }
    }

    // W-SPSA
    private fun calibrateWSPSAMM(
        activities: List<ActivityType>, parameters: Map<String, String>? = null
    ) {
        val measurements = sensors.map { it.measuredFlow }.flatMap { it.toList() }

        for (activity in activities) {
            val model = SGGravity(omosim).buildModelSimCounts(activity, sensors, affectedSensors)
            val objective = metaModelObjWSPSA(model, sensors)
            val x0 = DoubleArray(omosim.grid.size - 1) { 1.0 }
            var d = WSPSA.run(
                x0, objective, measurements, model, Random(), parameters = parameters
            )
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }
    private fun calibrateWSPSA(
        activities: List<ActivityType>, parameters: Map<String, String>? = null
    ) {
        val measurements = sensors.map { it.measuredFlow }.flatMap { it.toList() }

        for (activity in activities) {
            val model = SGGravity(omosim).buildModelSimCounts(activity, sensors, affectedSensors)
            val objective = batchObjWSPSA(activity)
            val x0 = DoubleArray(omosim.grid.size - 1) { 1.0 }
            var d = WSPSA.run(
                x0, objective, measurements, model, Random(), parameters = parameters
            )
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }

    private fun calibrateMatrix(activities: List<ActivityType>) {
        for (activity in activities) {
            val model = SGGravity(omosim)
            val wm = model.optimizeTMatrix(activity, sensors, affectedSensors)

            val finder = omosim.destinationFinder as DestinationFinderDefault
            val force = mutableMapOf<Cell, DoubleArray>()
            for ((i, cell) in omosim.grid.withIndex()) {
                force[cell] = wm!!.toArray()[i]
            }
            finder.forcedTransitionMatrix[activity] = force
        }
    }

    // Objectives
    private fun metaModelObj(activity: ActivityType): (DoubleArray) -> Double {
        val model = SGGravity(omosim).buildModelSSE(activity, sensors, affectedSensors)
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
                for ((gi, cell) in omosim.grid.withIndex()) {
                    cell.updateAttractionScaler(dcFunction, x[ai * omosim.grid.size + gi])
                }
            }
            val flows = runBatch(0.1)
            flowMSE(flows)
        }
    }
    private fun batchObj(activity: ActivityType): (DoubleArray) -> Double {
        return { x: DoubleArray ->
            val dcFunction = finder.locChoiceWeightFuns[activity]!!
            for ((i, cell) in omosim.grid.withIndex()) {
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
            for ((i, cell) in omosim.grid.withIndex()) {
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
        for ((cell, v) in omosim.grid.zip(x.toTypedArray())) {
            cell.updateAttractionScaler(dcFunction, v)
        }
    }
    private fun updateCalibration(x: DoubleArray, activities: List<ActivityType>) {
        for ((ai, activity) in activities.withIndex()) {
            val dcFunction = finder.locChoiceWeightFuns[activity]!!
            for ((gi, cell) in omosim.grid.withIndex()) {
                cell.updateAttractionScaler(dcFunction, x[ai * omosim.grid.size + gi])
            }
        }
    }

   @Suppress("SameParameterValue")
   private fun runBatch(sharePop: Double) : Map<TrafficSensor, DoubleArray> {
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




