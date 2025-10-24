package de.uniwuerzburg.omod.calibration

import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.algorithms.*
import de.uniwuerzburg.omod.core.CarOwnership
import de.uniwuerzburg.omod.core.DestinationFinderDefault
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.*
import java.io.File
import java.util.*
import kotlin.math.floor
import kotlin.math.pow

enum class CalibrationOption {
    PSO, MM_LBFGS, SPSA, MM_PSO, PSO_OS, MM_GG, MM_SPSA, MM_ADAM, SPSA_OS, MM_GA, MM_MINBC
}

class TrafficCountCalibrator(
    linkDataFile: File,
    val omod: Omod,
    val carOwnership: CarOwnership
) {
    private val sensors: List<TrafficSensor>
    private val affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    private val finder = omod.destinationFinder as DestinationFinderDefault

    init {
        sensors = TrafficSensor.readSensorData(linkDataFile, omod)
        T = sensors.first().measuredFlow.size
        if (!sensors.all { it.measuredFlow.size == T }) {
            throw IllegalArgumentException(
                "Sensor measurement arrays are not uniformly sized!" +
                "Validate the --cal_traffic_count_file file. "
            )
        }

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

    fun calibrate(
        file: File, option: CalibrationOption, activities: List<ActivityType>,
        iterations: Int = 100, lossLog: File? = null, parameters: Map<String, String>? = null
    ) {
        when (option) {
            CalibrationOption.PSO       -> calibratePSO(lossLog, activities, iterations, parameters)
            CalibrationOption.PSO_OS    -> calibratePSOAllAtOnce(lossLog, activities, iterations, parameters)
            CalibrationOption.MM_PSO    -> calibratePSOMM(lossLog, activities, iterations, parameters)
            CalibrationOption.MM_GA     -> calibrateGAMM(lossLog, activities, iterations, parameters)
            CalibrationOption.MM_GG     -> calibrateGGMM(lossLog, activities, iterations, parameters)
            CalibrationOption.MM_ADAM   -> calibrateAdamMM(lossLog, activities, iterations, parameters)
            CalibrationOption.MM_LBFGS  -> calibrateLBFGSMM(lossLog, activities, iterations, parameters)
            CalibrationOption.SPSA      -> calibrateSPSA(lossLog, activities, iterations, parameters)
            CalibrationOption.SPSA_OS   -> calibrateSPSAAllAtOnce(lossLog, activities, iterations, parameters)
            CalibrationOption.MM_SPSA   -> calibrateSPSAMM(lossLog, activities, iterations, parameters)
            CalibrationOption.MM_MINBC  -> calibrateMinBcMM(lossLog, activities, iterations, parameters)
        }

        val finder = omod.destinationFinder as DestinationFinderDefault
        CalibrationInfo.write(file, omod.buildings, finder.locChoiceWeightFuns)
        evaluate(0.1)
    }

    fun evaluate(sharePop: Double) {
        val finder = omod.destinationFinder as DestinationFinderDefault
        val flowCal = runBatch(sharePop)

        for (activity in ActivityType.entries) {
            val dcFunction = finder.locChoiceWeightFuns[activity]!!
            for (cell in omod.grid) {
                cell.updateAttractionScaler(dcFunction, 1.0)
            }
        }
        val flowBase = runBatch(sharePop)


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

    private fun calibratePSOAllAtOnce(
        lossLog: File?, activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ) {
        val objective = batchObj(activities)
        val d = PSO.run(
            omod.grid.size * activities.size, objective, Random(), out=lossLog,
            iterations = iterations, parameters = parameters
        )
        updateCalibration(d, activities)
    }

    private fun calibratePSO(
        lossLog: File?, activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ) {
        for (activity in activities) {
            val lossLogA = activityLogFile(activity, lossLog)

            val objective = batchObj(activity)
            val d = PSO.run(
                omod.grid.size, objective, Random(), out=lossLogA,
                iterations = iterations, parameters = parameters
            )
            updateCalibration(d, activity)
        }
    }

    private fun calibratePSOMM(
        lossLog: File?, activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ){
        for (activity in activities) {
            val lossLogA = activityLogFile(activity, lossLog)
            val objective = metaModelObj(activity)

            var d = PSO.run(
                omod.grid.size - 1, objective, Random(), out=lossLogA,
                iterations = iterations, parameters = parameters
            )

            d = (d.toList() + listOf(1.0)).toDoubleArray()

            updateCalibration(d, activity)
        }
    }

    private fun calibrateGAMM(
        lossLog: File?, activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ){
        for (activity in activities) {
            val lossLogA = activityLogFile(activity, lossLog)
            val objective = metaModelObj(activity)

            var d = GA.run(
                omod.grid.size - 1, objective, Random(), out=lossLogA,
                iterations = iterations, parameters = parameters
            )

            d = (d.toList() + listOf(1.0)).toDoubleArray()

            updateCalibration(d, activity)
        }
    }

    private fun calibrateGGMM(
        lossLog: File?, activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ){
        for (activity in activities) {
            val lossLogA = activityLogFile(activity, lossLog)
            val model = MetaModel.build(omod)!!.getDiffModel(activity, sensors, affectedSensors)
            val x0 = DoubleArray(omod.grid.size - 1) { 1.0 }
            var d = GradientDescent.run(model, x0, iterations=iterations, out=lossLogA, parameters = parameters)

            d = (d.toList() + listOf(1.0)).toDoubleArray()

            updateCalibration(d, activity)
        }
    }

    private fun calibrateAdamMM(
        lossLog: File?, activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ){
         for (activity in activities) {
             val lossLogA = activityLogFile(activity, lossLog)
             val model = MetaModel.build(omod)!!.getDiffModel(activity, sensors, affectedSensors)
             val x0 = DoubleArray(omod.grid.size - 1) { 1.0 }
             var d = Adam.run(model, x0, iterations=iterations, out=lossLogA, parameters = parameters)

             d = (d.toList() + listOf(1.0)).toDoubleArray()
             updateCalibration(d, activity)
        }
    }

    fun calibrateLBFGSMM(
        lossLog: File?, activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    )  {
        for (activity in activities) {
            val lossLogA = activityLogFile(activity, lossLog)
            val model = MetaModel.build(omod)!!.getDiffModel(activity, sensors, affectedSensors)
            val x0 = DoubleArray(omod.grid.size - 1) { 1.0 }
            var d =  BFGS.run(model, x0, iterations=iterations, out=lossLogA, parameters = parameters)
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }

    fun calibrateMinBcMM(
        lossLog: File?, activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    )  {
        for (activity in activities) {
            val lossLogA = activityLogFile(activity, lossLog)
            val model = MetaModel.build(omod)!!.getDiffModel(activity, sensors, affectedSensors)
            val x0 = DoubleArray(omod.grid.size - 1) { 1.0 }
            var d =  MinBc.run(model, x0, iterations=iterations, out=lossLogA, parameters=parameters)
            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }

    fun calibrateSPSAMM(
        lossLog: File?, activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ) {
         for (activity in activities) {
            val lossLogA = activityLogFile(activity, lossLog)
            val objective = metaModelObj(activity)
            val x0 = DoubleArray(omod.grid.size - 1) { 1.0 }
            var d = SPSA.run(x0, objective, Random(), iterations = iterations, out=lossLogA, parameters = parameters)

            d = (d.toList() + listOf(1.0)).toDoubleArray()
            updateCalibration(d, activity)
        }
    }

    private fun calibrateSPSAAllAtOnce(
        lossLog: File?, activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ) {
        val objective = batchObj(activities)
        val x0 = DoubleArray(omod.grid.size * activities.size) { 1.0 }
        val d = SPSA.run(x0, objective, Random(), iterations = iterations, out=lossLog, parameters = parameters)
        updateCalibration(d, activities)
    }

    private fun calibrateSPSA(
        lossLog: File?, activities: List<ActivityType>, iterations: Int, parameters: Map<String, String>? = null
    ) {
         for (activity in activities) {
            val lossLogA = activityLogFile(activity, lossLog)
            val objective = batchObj(activity)
            val x0 = DoubleArray(omod.grid.size ) { 1.0 }
            val d = SPSA.run(x0, objective, Random(), iterations = iterations, out=lossLogA, parameters = parameters)
            updateCalibration(d, activity)
        }
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
    private fun activityLogFile(activity: ActivityType, lossLog: File?) : File? {
        return if(lossLog != null) {
            File(
                lossLog.parent + "/" +
                        lossLog.name.replace(".losslog", "") +
                        "$activity" + ".losslog")
        } else {
            null
        }
    }

    private fun metaModelObj(activity: ActivityType): (DoubleArray) -> Double {
        val model = MetaModel.build(omod)!!.getDiffModel(activity, sensors, affectedSensors)
        return { x: DoubleArray ->
            model.evaluate(x)
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
           if (trip.mode == Mode.CAR_DRIVER) {
               if ((originActivity.location.getAggLoc() is Cell) and (destinationActivity.location.getAggLoc() is Cell)) {
                   val od = Pair(originActivity.location.getAggLoc() as Cell, destinationActivity.location.getAggLoc() as Cell)
                   if (od in affectedSensors) {
                       val sensors = affectedSensors[od]!!
                       for (sensor in sensors) {
                           val arr = simCount[sensor]!!
                           val mod = departureTime.minute + departureTime.hour * 60
                           val i = floor((mod % 1440.0) / 1440.0 * T).toInt()
                           arr[i] = arr[i] + 1
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
        return mse / sensors.size
    }
}




