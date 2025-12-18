package de.uniwuerzburg.omod.calibration

import com.gurobi.gurobi.*
import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.differentiablemodel.*
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.*
import java.time.LocalTime

/**
 * Origin-destination pair at a given time t.
 * Used to define the route choice probability distributions for different Trips possibilities.
 *
 * @param origin Start location
 * @param destination End location
 * @param t Time slice
 */
data class ODTTriple(
    val origin: LocationOption,
    val destination: LocationOption,
    val t: Int
)

/**
 * Calibrate OMoSim output by adjusting route choice.
 * Each agent will choose a route between A and B from the paths found by GraphHopper when ALT_ROUTE is set.
 * The result are multiple probability vector p - one for each o-d pair - that give the probability that each path
 * is selected by an agent.
 * By default, the agents will always choose the first result p = (1.0, 0.0, 0.0, ...).
 */
object RouteChoice {
    /**
     * Find the optimal p with Gurobi.
     *
     * @param agents OMoSim run result
     * @param omod Simulator
     * @param sensors Sensors with measurements
     * @param affectedAltSensors Gives all sensors that are affected by a path option for a certain origin-destination pair
     * @return Optimal p for each trip possibility
     */
    fun optimize(
        agents: List<MobiAgent>,
        omod: Omod,
        sensors: List<TrafficSensor>,
        affectedAltSensors: Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>>
    ) : Map<ODTTriple, List<Double>> {
        logger.info("Starting route choice calibration with gurobi.")
        val odtCounts = getODTCounts(agents, omod)

        try {
            // Setup
            val env = GRBEnv()
            val model = GRBModel(env)

            // Initialize simulated traffic counts
            val simCount = mutableMapOf<TrafficSensor, List<GRBLinExpr>>()
            for (sensor in sensors) {
                simCount[sensor] = List(T) { GRBLinExpr() }
            }

            // Add contribution of each odt to simulated counts
            val result = mutableMapOf<ODTTriple, MutableList<GRBVar>>()
            for ((od, alternatives) in affectedAltSensors.entries) {
                for (t in 0 until T) {
                    val odt = ODTTriple(od.first, od.second, t)
                    if (odt in odtCounts) {
                        val count = odtCounts[odt]!!
                        val pSum = GRBLinExpr()
                        for (alternative in alternatives) {
                            // Probability of choosing that alternative
                            val pA = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "P")

                            if (odt in result) {
                                result[odt]!!.add(pA)
                            } else {
                                result[odt] = mutableListOf(pA)
                            }

                            // Add to simulated count
                            for (sensor in alternative) {
                                simCount[sensor]!![t].addTerm(count, pA)
                            }

                            // For PCondition
                            pSum.addTerm(1.0, pA)
                        }
                        // Ensure that p is a proper probability distribution
                        model.addConstr(pSum, GRB.EQUAL, 1.0, "PCondition")
                    }
                }
            }

            // Objective
            val obj = grbMseObjective(model, sensors, simCount)
            model.setObjective(obj, GRB.MINIMIZE)

            // Solve
            model.optimize()

            val success = handleGrbStatus(model)
            if (success) {
                val oval = model[GRB.DoubleAttr.ObjVal]
                logger.info("Optimization (gurobi) finished with optimal objective: $oval")
                val rVal = result.mapValues { (_, v) -> v.map { it.get(GRB.DoubleAttr.X) } }
                model.dispose()
                env.dispose()
                return rVal
            }
            model.dispose()
            env.dispose()
        } catch (e: GRBException) {
            logger.error("Gurobi Error! Error code: ${e.errorCode}. ${e.message}")
        }
        return mapOf()
    }

    /**
     * Build a differentiable model that computes the sum-of-square loss for a given p.
     *
     * @param agents OMoSim run result
     * @param omod Simulator
     * @param sensors Sensors with measurements
     * @param affectedAltSensors Gives all sensors that are affected by a path option for a certain origin-destination pair
     * @return Differentiable Model
     */
    fun buildModel(
        agents: List<MobiAgent>, omod: Omod,
        sensors: List<TrafficSensor>,
        affectedAltSensors: Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>>
    ) : DifferentiableModel {
        val odtCounts = getODTCounts(agents, omod)

        // Setup. Initialize differentiable model
        var nVar = 0
        for ((od, alternatives) in affectedAltSensors.entries) {
            for (t in 0 until T) {
                val odt = ODTTriple(od.first, od.second, t)
                if (odt in odtCounts) {
                    nVar += alternatives.size
                }
            }
        }
        val model = DifferentiableModel(nVar)

        // Initialize simulated traffic counts
        val simCount = mutableMapOf<TrafficSensor, List<LinearTerm>>()
        for (sensor in sensors) {
            simCount[sensor] = List(T) { LinearTerm(model.nVars) }
        }

        // Add contribution of each odt to simulated counts
        var iVar = 0 // Keep track of which variable represents the current route choice decision.
        for ((od, alternatives) in affectedAltSensors.entries) {
            for (t in 0 until T) {
                val odt = ODTTriple(od.first, od.second, t)
                if (odt in odtCounts) {
                    val count = odtCounts[odt]!!
                    val pAs = mutableListOf<Term>()
                    for (alternative in alternatives) {
                        // Probability of choosing that alternative
                        val exTripsAlternative = LinearBaseTerm(model.nVars)
                        exTripsAlternative.addTerm(iVar, 1.0)
                        iVar += 1
                        pAs.add(exTripsAlternative)
                    }

                    val pSum = LinearTerm(model.nVars)
                    for (pA in pAs) {
                        pSum.addTerm(pA, 1.0)
                    }

                    for ((i, alternative) in alternatives.withIndex()) {
                        // Ensure that p is a proper probability distribution
                        val pTerm = DivisionTerm(model.nVars, pAs[i], pSum)

                        // Add to simulated count
                        for (sensor in alternative) {
                            simCount[sensor]!![t].addTerm(pTerm, count)
                        }
                    }
                }
            }
        }

        // Objective
        val obj = mseObjective(model.nVars, sensors, simCount)
        model.setRootTerm(obj)
        return model
    }

    /**
     * Determine how often a specific origin-destination-time triple occurs.
     * @param agents OMoSim run result
     * @param omod Simulator
     * @return Occurrence
     */
    private fun getODTCounts(
        agents: List<MobiAgent>,
        omod: Omod
    ) : Map<ODTTriple, Double> {
        val totalPop = omod.buildings.sumOf { it.population } // TODO

        val n = mutableMapOf<ODTTriple, Double>()
        for (agent in agents) {
            val demand = agent.mobilityDemand.first() // Get demand for first day

            // Get trip start times
            val startTimes = mutableListOf<LocalTime>()
            val origins = mutableListOf<Activity>()
            val destinations = mutableListOf<Activity>()
            val visitor: TripVisitor = { _: Trip, origin: Activity, destination: Activity,
                                         departureTime: LocalTime, _: Weekday, _: Boolean ->
                startTimes.add(departureTime)
                origins.add(origin)
                destinations.add(destination)
            }
            demand.visitTrips(visitor)

            // Count car trips
            for (i in 0 until demand.trips.size) {
                val trip = demand.trips[i]
                val startTime = startTimes[i]
                val origin = origins[i].location.getAggLoc()!!
                val destination = destinations[i].location.getAggLoc()!!

                if (trip.mode != Mode.CAR_DRIVER) {
                    continue
                }

                val t =  startTime.determineTimeSlice()
                val odt = ODTTriple(origin, destination, t)
                n[odt] = (n[odt] ?: 0.0) + totalPop / agents.size
            }
        }
        return n
    }
}

