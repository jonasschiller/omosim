package de.uniwuerzburg.omod.calibration

import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.differentiablemodel.*
import de.uniwuerzburg.omod.core.ModeChoiceFast
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.*
import java.util.*
import kotlin.math.exp
import kotlin.math.ln

/**
 * Objectives for mode choice calibration
 *
 * FitTotalCarTrips: Calibrate the total number of car trips across all measurements: minimize (sum(M) - sum(S))**2
 * FitIndividualMeasurements: Calibrate each measurement individually (normal case): minimize (sum(m - s))**2
 */
enum class ModeChoiceCalibrationObjective {
    FitTotalCarTrips, FitIndividualMeasurements
}

/**
 * Calibrate OMoSim output by adjusting mode choice.
 *
 * Optimizes the intercept of the car mode utility such that the objective is minimized.
 */
object ModeChoice {
    /**
     * Build a differentiable model that computes the objective loss for a given intercept.
     *
     * @param agents OMoSim run result
     * @param mc Mode choice model to calibrate
     * @param rng Random number generator
     * @param omod Simulator
     * @param sensors Sensors with measurements
     * @param affectedSensors Gives all sensors that are affected by a path option for a certain origin-destination pair
     * @param objective Type of objective to use @See de.uniwuerzburg.omod.calibration.ModeChoiceCalibrationObjective
     * @return Differentiable Model
     */
    fun buildModel(
        agents: List<MobiAgent>,
        mc: ModeChoiceFast,
        rng: Random,
        omod: Omod,
        sensors: List<TrafficSensor>,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        objective: ModeChoiceCalibrationObjective
    ) : DifferentiableModel {
        val totalPop = omod.buildings.sumOf { it.population } // TODO
        val model = DifferentiableModel(1) // Only variable: Intercept of car mode

        // Initialize simulated traffic counts
        val simCount = mutableMapOf<TrafficSensor, List<LinearTerm>>()
        for (sensor in sensors) {
            simCount[sensor] = List(T) { LinearTerm(model.nVars) }
        }

        // Utility model for the car driver mode
        val driverUtilityModel = mc.tourModeOptions.first { it.mode == Mode.CAR_DRIVER }

        // Add contribution of each agent and each tour to simulated counts
        for (agent in agents) {
            val aTours = mc.getTours(agent.mobilityDemand.first(), rng)

            for (tour in aTours) {
                // Only do HOME-HOME tours
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

                // Create intercept variable
                val driverUtilityVar = LinearBaseTerm(1)
                driverUtilityVar.addConstant(utility - driverUtilityModel.intercept)
                driverUtilityVar.addTerm(0, 1.0)
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

                // Add expected origin destination trips to sim counts
                var o = tour.first().fromActivity.location.getAggLoc()!! as Cell
                for (trip in tour) {
                    val d = trip.toActivity.location.getAggLoc()!! as Cell
                    val t = trip.departureTime.determineTimeSlice()
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
        val objTerm = when(objective) {
            ModeChoiceCalibrationObjective.FitIndividualMeasurements -> {
                sseObjective(model.nVars, sensors, simCount) // Normal sum of squares objective
            }
            ModeChoiceCalibrationObjective.FitTotalCarTrips -> {
                // Objective that minimizes the differences between the sum of simulated counts across all
                // traffic counting stations and the sum of measurements.
                // Purpose: Calibrate the total car demand in the area
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
        model.setRootTerm(objTerm)
        return model
    }
}
