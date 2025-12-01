package de.uniwuerzburg.omod.calibration.surrogate

import com.gurobi.gurobi.*
import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.TrafficSensor
import de.uniwuerzburg.omod.calibration.differentiablemodel.*
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.*
import java.time.LocalTime
import kotlin.math.floor

object SGALtRoute {
    fun grb(
        agents: List<MobiAgent>, omod: Omod,
        sensors: List<TrafficSensor>,
        affectedAltSensors: Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>>
    ) : Map<Triple<LocationOption, LocationOption, Int>, List<Double>> {
        val altCoefs = determineCoefficients(agents, omod)

        try {
            val env = GRBEnv()
            val model = GRBModel(env)

            // Simulated traffic counts
            val simCount = mutableMapOf<TrafficSensor, List<GRBLinExpr>>()
            for (sensor in sensors) {
                simCount[sensor] = List(T) { GRBLinExpr() }
            }

            // Create sim counts based on alternative choices
            val result = mutableMapOf<Triple<LocationOption, LocationOption, Int>, MutableList<GRBVar>>()
            for ((od, alternatives) in affectedAltSensors.entries) {
                for (t in 0 until T) {
                    val key = Triple(od.first, od.second, t)
                    if (key in altCoefs) {
                        val c = altCoefs[key]!!
                        val pSum = GRBLinExpr()
                        for ((i, alternative) in alternatives.withIndex()) {
                            val v = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "P")

                            if (key in result) {
                                result[key]!!.add(v)
                            } else {
                                result[key] = mutableListOf(v)
                            }

                            pSum.addTerm(1.0, v)
                            for (sensor in alternative) {
                                simCount[sensor]!![t].addTerm(c, v)
                            }
                        }
                        model.addConstr(pSum, GRB.EQUAL, 1.0, "ProperDistr")
                    }
                }
            }

            // Objective
            val obj = grbMseObjective(model, sensors, simCount)
            model.setObjective(obj, GRB.MINIMIZE)

            model.optimize()

            var status = model[GRB.IntAttr.Status]

            if (status == GRB.Status.INF_OR_UNBD) {
                model[GRB.IntParam.Presolve] = 0
                model.optimize()
                status = model[GRB.IntAttr.Status]
            }
            if (status == GRB.Status.OPTIMAL) {
                // Print result
                val oval = model[GRB.DoubleAttr.ObjVal]
                println("Optimal objective: $oval")
                return result.mapValues { (k, v) -> v.map { it.get(GRB.DoubleAttr.X) } }
            } else if (status == GRB.Status.INFEASIBLE) {
                println("Model is infeasible")
            } else if (status == GRB.Status.UNBOUNDED) {
                println("Model is unbounded")
            } else {
                println(
                    "Optimization was stopped with status = $status"
                )
            }

        }  catch (e: GRBException) {
            println(
                ("Error code: " + e.errorCode + ". " +
                        e.message)
            )
        }
        return mapOf()
    }

    fun build(
        agents: List<MobiAgent>, omod: Omod,
        sensors: List<TrafficSensor>,
        affectedAltSensors: Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>>
    ) : DifferentiableModel {
        val altCoefs = determineCoefficients(agents, omod)

        // Create diff model
        var nVar = 0
        for ((od, alternatives) in affectedAltSensors.entries) {
            for (t in 0 until T) {
                val key = Triple(od.first, od.second, t)
                if (key in altCoefs) {
                    nVar += alternatives.size
                }
            }
        }
        val model = DifferentiableModel(nVar)

        // Simulated traffic counts
        val simCount = mutableMapOf<TrafficSensor, List<LinearTerm>>()
        for (sensor in sensors) {
            simCount[sensor] = List(T) { LinearTerm(model.nVars) }
        }

        // Create sim counts based on alternative choices
        var iVar = 0
        for ((od, alternatives) in affectedAltSensors.entries) {
            for (t in 0 until T) {
                val key = Triple(od.first, od.second, t)
                if (key in altCoefs) {
                    val c = altCoefs[key]!!
                    val altVars = mutableListOf<Term>()
                    for (alternative in alternatives) {
                        val exTripsAlternative = LinearBaseTerm(model.nVars)
                        exTripsAlternative.addTerm(iVar, 1.0)
                        iVar += 1
                        altVars.add(exTripsAlternative)
                    }

                    val sumTerm = LinearTerm(model.nVars)
                    for (v in altVars) {
                        sumTerm.addTerm(v, 1.0)
                    }

                    for ((i, alternative) in alternatives.withIndex()) {
                        val pTerm = DivisionTerm(model.nVars, altVars[i], sumTerm)
                        for (sensor in alternative) {
                            simCount[sensor]!![t].addTerm(pTerm, c)
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

    private fun determineCoefficients(
        agents: List<MobiAgent>, omod: Omod
    ) : Map<Triple<LocationOption, LocationOption, Int>, Double> {
        val totalPop = omod.buildings.sumOf { it.population }

        // Get var coefficients
        val altCoefs = mutableMapOf<Triple<LocationOption, LocationOption, Int>, Double>()

        for (agent in agents) {
            val demand = agent.mobilityDemand.first()

            // Get Start times
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

            for (i in 0 until demand.trips.size) {
                val trip = demand.trips[i]
                val startTime = startTimes[i]
                val origin = origins[i].location.getAggLoc()!!
                val destination = destinations[i].location.getAggLoc()!!

                if (trip.mode != Mode.CAR_DRIVER) {
                    continue
                }

                // Get start time interval
                val mod = startTime.minute + startTime.hour * 60
                val t = floor((mod % 1440.0) / 1440.0 * T).toInt()

                val key = Triple(origin, destination, t)
                val c = altCoefs[key] ?: 0.0
                altCoefs[key] = c + totalPop / agents.size
            }
        }
        return altCoefs
    }
}