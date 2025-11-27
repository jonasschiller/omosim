package de.uniwuerzburg.omod.calibration

import com.gurobi.gurobi.*
import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.differentiablemodel.*
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.*
import java.time.LocalTime
import kotlin.math.floor

fun calibrateAltRoute(
    agents: List<MobiAgent>, omod: Omod,
    sensors: List<TrafficSensor>,
    affectedAltSensors: Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>>
) : Map<Triple<LocationOption, LocationOption, Int>,  List<Double>> {
    return grbOptAltRoute(agents, omod, sensors, affectedAltSensors)

    /*
    // Create mode choice model
    val model = buildModelAltRoute(agents, omod, sensors, affectedAltSensors)

    // Get X0
    val x0 = DoubleArray(model.nVars) { 1.0 }

    // Optimize
    val x = BFGS.run(model, x0, lb=0.0, ub=1.0, iterations=50, out = File("test.losslog"))
    */
}


fun grbOptAltRoute(
    agents: List<MobiAgent>, omod: Omod,
    sensors: List<TrafficSensor>,
    affectedAltSensors: Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>>
) : Map<Triple<LocationOption, LocationOption, Int>, List<Double>> {
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

        val sensorSimCount = List(T) {
            model.addVars(
                DoubleArray(sensors.size) { 0.0 },
                null,
                DoubleArray(sensors.size) { 0.0 },
                CharArray(sensors.size) { GRB.CONTINUOUS },
                Array(sensors.size) { "" }
            )
        }

        for ((i, sensor) in sensors.withIndex()) {
            for (t in 0 until T) {
                model.addConstr(
                    simCount[sensor]!![t],
                    GRB.EQUAL,
                    sensorSimCount[t][i]!!,
                    "cnteq"
                )
            }
        }

        // Objective
        val obj = GRBQuadExpr()
        for ((i, sensor) in sensors.withIndex()) {
            for (t in 0 until T) {
                // (Sm - Ss)^2 = Sm^2 - 2SmSs + Ss^2
                obj.addConstant(sensor.measuredFlow[t] * sensor.measuredFlow[t])
                obj.addTerm(-2 * sensor.measuredFlow[t], sensorSimCount[t][i])
                obj.addTerm(1.0, sensorSimCount[t][i], sensorSimCount[t][i])
            }
        }
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

fun buildModelAltRoute(
    agents: List<MobiAgent>, omod: Omod,
    sensors: List<TrafficSensor>,
    affectedAltSensors: Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>>
) : DifferentiableModel {
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

    model.setRootTerm(obj)
    return model
}

/*
fun optimize(
    grid: List<Cell>,
    odMatrix: Map<Pair<Cell, Cell>, Double>,
    totalPop: Double,
    affectedAltSensors: Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>>,
    sensors: List<TrafficSensor>
) :  Map<Pair<RealLocation, RealLocation>, List<Double>> {

    try {
        val env = GRBEnv()
        val model = GRBModel(env)

        val sensorSimCount = model.addVars(
            DoubleArray(sensors.size) {0.0},
            null,
            DoubleArray(sensors.size) {0.0},
            CharArray(sensors.size) {GRB.CONTINUOUS},
            Array(sensors.size) {""}
        )

        val sensorCountExpr = mutableMapOf<TrafficSensor, GRBLinExpr>()
        for (sensor in sensors) {
            sensorCountExpr[sensor] = GRBLinExpr()
        }

        val As = mutableMapOf<Pair<RealLocation, RealLocation>, List<GRBVar>>()

        for (origin in grid) {
            for (destination in grid) {
                val od = Pair(origin, destination)
                if (od in affectedAltSensors) {
                    val x = odMatrix[od]!!
                    val altAffect = affectedAltSensors[od]!!
                    val altsize = altAffect.size
                    val a = model.addVars(
                        DoubleArray(altsize) {0.0},
                        DoubleArray(altsize) {1.0},
                        DoubleArray(altsize) {0.0},
                        CharArray(altsize) {GRB.CONTINUOUS},
                        Array(altsize) {""}
                    )
                    As[od] = a.toList()

                    for ((i, alt) in altAffect.withIndex()) {
                        for (sensor in alt) {
                            val sensorSum = sensorCountExpr[sensor]!!
                            sensorSum.addTerm(x*totalPop, a[i])
                        }
                    }

                    val sumExpr = GRBLinExpr()
                    sumExpr.addTerms(DoubleArray(a.size) {1.0}, a)
                    model.addConstr(sumExpr, GRB.EQUAL, 1.0, "altsum")
                }

            }
        }

        for ((i, sensor) in sensors.withIndex()) {
            val sensorSum = sensorCountExpr[sensor]!!
            model.addConstr(sensorSum, GRB.EQUAL, sensorSimCount[i], "cnteq")
        }

        // Objective
        val obj = GRBQuadExpr()
        for ((i, sensor) in sensors.withIndex()) {
            // (Sm - Ss)^2 = Sm^2 - 2SmSs + Ss^2
            obj.addConstant(sensor.measuredFlow * sensor.measuredFlow)
            obj.addTerm(-2 * sensor.measuredFlow, sensorSimCount[i])
            obj.addTerm(1.0, sensorSimCount[i], sensorSimCount[i])
        }
        model.setObjective(obj, GRB.MINIMIZE)

        model.optimize()

        var optimstatus = model[GRB.IntAttr.Status]

        if (optimstatus == GRB.Status.INF_OR_UNBD) {
            model[GRB.IntParam.Presolve] = 0
            model.optimize()
            optimstatus = model[GRB.IntAttr.Status]
        }

        if (optimstatus == GRB.Status.OPTIMAL) {
            val objval = model[GRB.DoubleAttr.ObjVal]
            println("Optimal objective: $objval")

            println("_".repeat(20*4 + 5*3))
            println("${"Sensor".padEnd(20)} | \t" +
                    "${"Flow AltOpt".padEnd(20)} | \t" +
                    "Flow Measured".padEnd(20)
            )
            println("_".repeat(20) +
                    " | \t" + "_".repeat(20)  +
                    " | \t" + "_".repeat(20))
            var myobjval = 0.0
            for ((i, sensor) in sensors.withIndex()) {
                val optVal = sensorSimCount[i].get(GRB.DoubleAttr.X)
                println(
                    "${sensors[i].name.padEnd(20)} | \t" +
                            "${optVal.toString().padEnd(20)} | \t" +
                            sensors[i].measuredFlow.toString().padEnd(20)
                )
                myobjval += (sensors[i].measuredFlow - optVal) * (sensors[i].measuredFlow - optVal)
            }
            println("My Obj val: $myobjval")

        } else if (optimstatus == GRB.Status.INFEASIBLE) {
            println("Model is infeasible")
        } else if (optimstatus == GRB.Status.UNBOUNDED) {
            println("Model is unbounded")
        } else {
            println(
                "Optimization was stopped with status = "
                        + optimstatus
            )
        }

        val out =  As.mapValues { it.value.map { v -> v.get(GRB.DoubleAttr.X) } }

        // Dispose of model and environment
        model.dispose()
        env.dispose()

        return out
    } catch (e: GRBException) {
        println(
            ("Error code: " + e.errorCode + ". " +
                    e.message)
        )
    }
    return mapOf()
}*/