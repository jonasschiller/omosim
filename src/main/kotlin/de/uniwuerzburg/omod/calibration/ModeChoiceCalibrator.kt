package de.uniwuerzburg.omod.calibration

import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.algorithms.BFGS
import de.uniwuerzburg.omod.calibration.differentiablemodel.*
import de.uniwuerzburg.omod.core.ModeChoiceFast
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.*
import de.uniwuerzburg.omod.io.json.writeJson
import java.io.File
import java.util.*
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.time.measureTime

enum class ModeChoiceCalibrationObjective {
    FitTotalCarTrips, FitIndividualMeasurements
}

fun calibrate(
    agents: List<MobiAgent>, rng: Random, omod: Omod,
    sensors: List<TrafficSensor>,
    affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
    objectiveType: ModeChoiceCalibrationObjective
): DoubleArray {
    // Create mode choice model
    val mc = ModeChoiceFast(omod.routingCache)
    val model = buildModel(agents, mc, rng, omod, sensors, affectedSensors, objectiveType)

    // Get X0
    val carUtil = mc.tourModeOptions.find { it.mode == Mode.CAR_DRIVER }
    val x0 = doubleArrayOf(carUtil!!.intercept)

    // Optimize
    val x = BFGS.run(model, x0, lb=-50.0, ub=50.0, iterations=50)

    return x
}

fun buildModel(
    agents: List<MobiAgent>, mc: ModeChoiceFast, rng: Random, omod: Omod,
    sensors: List<TrafficSensor>,
    affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
    objectiveType: ModeChoiceCalibrationObjective
) : DifferentiableModel {
    val totalPop = omod.buildings.sumOf { it.population }
    val model = DifferentiableModel(1)

    // Simulated traffic counts
    val simCount = mutableMapOf<TrafficSensor, List<LinearTerm>>()
    for (sensor in sensors) {
        simCount[sensor] = List(T) { LinearTerm(model.nVars) }
    }

    // Driver utility model
    val driverUtilityModel = mc.tourModeOptions.first { it.mode == Mode.CAR_DRIVER }

    for (agent in agents) {
        val aTours = mc.getTours(agent.mobilityDemand.first(), rng)

        for (tour in aTours) {
            // Only do HOME-HOME tours as one block
            if (tour.first().fromActivity.type != ActivityType.HOME) { continue }
            if (tour.last().toActivity.type != ActivityType.HOME) { continue }

            // Aggregate distance and times
            val carDistance = tour.sumOf { it.carDistance }

            // Main purpose of tour is defined by the activity with the longest stay time
            val mainPurpose = tour
                .dropLast(1)
                .maxByOrNull { it.toActivity.stayTime!! }?.toActivity?.type ?: ActivityType.HOME

            // Weekday
            val weekday = tour.first().weekday

            // Car mode utility
            val utilityTerm = LinearTerm(model.nVars)
            val utility = driverUtilityModel.calc(null, carDistance, mainPurpose, agent.carAccess, weekday, agent)

            // Make intercept variable
            val driverUtilityVar = LinearBaseTerm(1)
            driverUtilityVar.addConstant(utility - driverUtilityModel.intercept)
            driverUtilityVar.addTerm(0, 1.0) // Variable intercept
            utilityTerm.addTerm(driverUtilityVar, -1.0)

            // Other mode weights
            val otherWeight = ln(mc.tourModeOptions
                .filter { it.mode != Mode.CAR_DRIVER }
                .sumOf { util -> exp(util.calc(null, carDistance, mainPurpose, agent.carAccess, weekday, agent)) })
            utilityTerm.addConstant(otherWeight)

            // Car mode probability
            val eTerm = ExponentialTerm(model.nVars, utilityTerm)
            val normTerm = LinearTerm(model.nVars)
            normTerm.addConstant(1.0)
            normTerm.addTerm(eTerm, 1.0)
            val pTerm = PowerTerm(model.nVars, normTerm, -1)

            // Get expected origin destination trips
            var o = tour.first().fromActivity.location.getAggLoc()!! as Cell
            for (trip in tour) {
                val mod = trip.departureTime.minute + trip.departureTime.hour * 60
                val t = floor((mod % 1440.0) / 1440.0 * T).toInt()
                val d = trip.toActivity.location.getAggLoc()!! as Cell
                val tripOD = Pair(o, d)
                if (tripOD in affectedSensors) {
                    for (sensor in affectedSensors[tripOD]!!) {
                        simCount[sensor]!![t].addTerm(pTerm, totalPop / agents.size)
                    }
                }
                o = d
            }
        }
    }

    // Objective
    val objective = when(objectiveType) {
        ModeChoiceCalibrationObjective.FitIndividualMeasurements -> {
            val obj = LinearTerm(model.nVars)
            for (sensor in sensors) {
                for (t in 0 until T) {
                    // (Sm - Ss)^2 = Sm^2 - 2SmSs + Ss^2
                    val simCountTerm = simCount[sensor]!![t]
                    obj.addConstant(sensor.measuredFlow[t] * sensor.measuredFlow[t])
                    obj.addTerm(simCountTerm, -2 * sensor.measuredFlow[t])
                    val qTerm = QuadraticTerm(
                        model.nVars,
                        simCountTerm,
                        simCountTerm,
                        1.0
                    )
                    obj.addTerm(qTerm, 1.0)
                }
            }
            obj
        }
        ModeChoiceCalibrationObjective.FitTotalCarTrips -> {
            var totalMeasured = 0.0
            val totalSim = LinearTerm(model.nVars)
            for (sensor in sensors) {
                for (t in 0 until T) {
                    totalSim.addTerm(simCount[sensor]!![t], 1.0)
                    totalMeasured += sensor.measuredFlow[t]
                }
            }
            val obj = LinearTerm(model.nVars)
            obj.addConstant(totalMeasured * totalMeasured)
            obj.addTerm(totalSim, -2 * totalMeasured)
            val qTerm = QuadraticTerm(
                model.nVars,
                totalSim,
                totalSim,
                1.0
            )
            obj.addTerm(qTerm, 1.0)
            obj
        }
    }

    model.setRootTerm(objective)
    return model
}