package de.uniwuerzburg.omod.calibration

import com.gurobi.gurobi.*
import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import de.uniwuerzburg.omod.calibration.differentiablemodel.ExponentialTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.LinearTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.PowerTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.QuadraticTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.Term
import de.uniwuerzburg.omod.core.ModeChoiceFast
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.*
import smile.math.BFGS
import java.util.*
import kotlin.math.exp
import kotlin.math.ln
import kotlin.time.measureTime
import kotlin.time.measureTimedValue
/*
fun calibrate(
    agents: List<MobiAgent>, rng: Random, omod: Omod,
    sensors: List<TrafficSensor>,
    affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
): List<Double> {
    val mc  = ModeChoiceFast(omod.routingCache)
    println("Mode Choice Cal")

    println("Building diff model...")
    val (model, simCount, pTermsDebug) = buildModel(agents, mc, rng, omod, sensors, affectedSensors)

    var parameters = doubleArrayOf(0.507651, 1.440126, 0.902728, 1.332559, 0.202816,  1.259017) //DoubleArray(model.nVars) { 0.0 }

    val lbfgsTime = measureTime {
        parameters = lbfgs(model, parameters)
    }
    println("LBFGS took $lbfgsTime")
    //parameters = gradDescent(model, parameters)
    println(parameters.toList())

    val pmean = pTermsDebug.sumOf { it.evaluate(parameters) } / pTermsDebug.size.toDouble()
    println("P mean ${pmean}")

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
        val optVal = simCount[sensor]!!.evaluate(parameters)
        println(
            "${sensors[i].name.padEnd(20)} | \t" +
                    "${optVal.toString().padEnd(20)} | \t" +
                    sensors[i].measuredFlow.toString().padEnd(20)
        )
        myobjval += (sensors[i].measuredFlow - optVal) * (sensors[i].measuredFlow - optVal)
    }
    println("MSE: ${myobjval /sensors.size}")

    //optimize(agents, mc, rng, omod, sensors, affectedSensors)

    return parameters.toList()
}

fun lbfgs(model: DifferentiableModel, vals: DoubleArray) : DoubleArray {
    println("LBFGS-B")
    println("Start: ${model.evaluate(vals)}")
    val l = DoubleArray(model.nVars){-50.0}
    val u = DoubleArray(model.nVars){50.0}
    val solution = BFGS.minimize(model, 5, vals, l, u, 1e-5, 50)
    println("Solution: $solution")
    println("Confirm: ${model.evaluate(vals)}")
    return vals
}

fun gradDescent(model: DifferentiableModel, vals: DoubleArray, iterations: Int = 10000, lr: Double = 1.0e-9) :
        DoubleArray
{
    val currentValues = vals.copyOf()
    val gradients = DoubleArray(vals.size) { 0.0 }

    val evaltime = measureTime {
        val loss = model.evaluate(currentValues)
        println("Start loss: $loss")
    }
    println("Eval time: $evaltime")

    for (i in 0 until iterations) {
        val (loss, time) = measureTimedValue {
            // Determine gradients
            for ((i, value) in currentValues.withIndex()) {
                gradients[i] = model.gradient(i, currentValues)
            }

            for ((i, value) in currentValues.withIndex()) {
                currentValues[i] -= lr * gradients[i]
            }
            val loss = model.evaluate(currentValues)
            loss
        }

        print("Iteration $i, vals ${currentValues.toList().toString()}, loss: $loss \t took $time \r")
    }
    val loss = model.evaluate(currentValues)
    println("Solution loss: $loss")
    return currentValues
}

fun buildModel(
    agents: List<MobiAgent>, mc: ModeChoiceFast, rng: Random, omod: Omod,
    sensors: List<TrafficSensor>,
    affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
) : Triple<DifferentiableModel, Map<TrafficSensor, LinearTerm>, List<Term>> {
    val totalPop = omod.buildings.sumOf { it.population }

    val model = DifferentiableModel(6)

    val simcount = mutableMapOf<TrafficSensor, LinearTerm>()
    for (sensor in sensors) {
        simcount[sensor] = LinearTerm(model.nVars)
    }
    val pTermsDebug = mutableListOf<Term>()

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

            val otherWeight = ln(mc.tourModeOptions
                .filter { it.mode != Mode.CAR_DRIVER }
                .sumOf { util -> exp(util.calc(null, carDistance, mainPurpose, agent.carAccess, weekday, agent)) })
            val driverUtil = mc.tourModeOptions.first { it.mode == Mode.CAR_DRIVER }
            //val driverWeight = driverUtil.calc(null, carDistance, mainPurpose, agent.carAccess, weekday, agent)
            val driverWeight = driverUtil.calTerm(null, carDistance, mainPurpose, agent.carAccess, weekday, agent)
            val utilTerm = LinearTerm(model.nVars)
            //utilTerm.addConstant(otherWeight - driverWeight)
            utilTerm.addConstant(otherWeight)
            utilTerm.addTerm(driverWeight, -1.0)
            val eTerm = ExponentialTerm(model.nVars, utilTerm)

            val normTerm = LinearTerm(model.nVars)
            normTerm.addConstant(1.0)
            normTerm.addTerm(eTerm, 1.0)

            val pTerm = PowerTerm(model.nVars, normTerm, -1)
            pTermsDebug.add(pTerm)

            /*
            val driverUtil = mc.tourModeOptions.first { it.mode == Mode.CAR_DRIVER }
            val driverWeight = driverUtil.calc(null, carDistance, mainPurpose, agent.carAccess, weekday, agent)
            val driverTerm = LinearBaseTerm(model.nVars)
            driverTerm.addConstant(driverWeight)
            driverTerm.addTerm(0, 1.0)
            val driverETerm = ExponentialTerm(model.nVars, driverTerm)

            val otherWeight = mc.tourModeOptions
                .filter { it.mode != Mode.CAR_DRIVER }
                .sumOf { util -> exp(util.calc(null, carDistance, mainPurpose, agent.carAccess, weekday, agent)) }

            val normTerm = LinearTerm(model.nVars)
            normTerm.addConstant(otherWeight)
            normTerm.addTerm(driverETerm, 1.0)

            val pTerm = DivisionTerm(model.nVars, driverETerm, normTerm)
            pTermsDebug.add(pTerm)*/

            var o = tour.first().fromActivity.location.getAggLoc()!! as Cell
            for (trip in tour) {
                val d = trip.toActivity.location.getAggLoc()!! as Cell
                val tripOD = Pair(o, d)
                if (tripOD in affectedSensors) {
                    val affectedSensors = affectedSensors[tripOD]!!
                    for (sensor in affectedSensors) {
                        simcount[sensor]!!.addTerm(pTerm, totalPop / agents.size)
                    }
                }
                o = d
            }
        }
    }

    // Objective
    val obj = LinearTerm(model.nVars)
    for ((i, sensor) in sensors.withIndex()) {
        // (Sm - Ss)^2 = Sm^2 - 2SmSs + Ss^2
        val simCountTerm = simcount[sensor]!!
        obj.addConstant(sensor.measuredFlow * sensor.measuredFlow)
        obj.addTerm(simCountTerm, -2 * sensor.measuredFlow)
        val qTerm = QuadraticTerm(
            model.nVars,
            simCountTerm,
            simCountTerm,
            1.0
        )
        obj.addTerm(qTerm, 1.0)
    }

    model.setRootTerm(obj)
    return Triple(model, simcount, pTermsDebug)
}

fun optimize(
    agents: List<MobiAgent>, mc: ModeChoiceFast, rng: Random, omod: Omod,
    sensors: List<TrafficSensor>,
    affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
) {
    val totalPop = omod.buildings.sumOf { it.population }

    try {
        // Setup
        val env = GRBEnv()
        val model = GRBModel(env)

        val simcount = mutableMapOf<TrafficSensor, GRBLinExpr>()
        for (sensor in sensors) {
            simcount[sensor] = GRBLinExpr()
        }

        val p = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "p_car")

        for (agent in agents) {
            val aTours = mc.getTours(agent.mobilityDemand.first(), rng)

            for (tour in aTours) {
                var o = tour.first().fromActivity.location.getAggLoc()!! as Cell
                for (trip in tour) {
                    val d = trip.toActivity.location.getAggLoc()!! as Cell
                    val tripOD = Pair(o, d)
                    if (tripOD in affectedSensors) {
                        val affectedSensors = affectedSensors[tripOD]!!
                        for (sensor in affectedSensors) {
                            simcount[sensor]!!.addTerm(totalPop / agents.size, p)
                        }
                    }
                    o = d
                }
            }
        }

        val simcountVar = model.addVars(
            DoubleArray(sensors.size) {0.0},
            null,
            DoubleArray(sensors.size) {0.0},
            CharArray(sensors.size) { GRB.CONTINUOUS},
            Array(sensors.size) {""}
        )
        for ((i, sensor) in sensors.withIndex()) {
            model.addConstr( simcount[sensor]!!, GRB.EQUAL, simcountVar[i], "cnteq")
        }

        // Objective
        val obj = GRBQuadExpr()
        for ((i, sensor) in sensors.withIndex()) {
            // (Sm - Ss)^2 = Sm^2 - 2SmSs + Ss^2
            obj.addConstant(sensor.measuredFlow * sensor.measuredFlow)
            obj.addTerm(-2 * sensor.measuredFlow, simcountVar[i])
            obj.addTerm(1.0, simcountVar[i], simcountVar[i])
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
            // TODO print gurobi P
            println("Optimal p car: ${p.get(GRB.DoubleAttr.X)}")

            val objval = model[GRB.DoubleAttr.ObjVal]
            println("Optimal objective: $objval")

            println("_".repeat(20*4 + 5*3))
            println("MSE: ${objval /sensors.size}")
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
                val optVal = simcountVar[i].get(GRB.DoubleAttr.X)
                println(
                    "${sensors[i].name.padEnd(20)} | \t" +
                            "${optVal.toString().padEnd(20)} | \t" +
                            sensors[i].measuredFlow.toString().padEnd(20)
                )
                myobjval += (sensors[i].measuredFlow - optVal) * (sensors[i].measuredFlow - optVal)
            }

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

        // Dispose of model and environment
        model.dispose()
        env.dispose()
    } catch (e: GRBException) {
        println(
            ("Error code: " + e.errorCode + ". " +
                    e.message)
        )
    }
}*/