package de.uniwuerzburg.omod.calibration

import com.gurobi.gurobi.*
import com.sun.tools.javac.tree.TreeInfo.args
import de.uniwuerzburg.omod.core.models.Cell
import de.uniwuerzburg.omod.core.models.RealLocation


fun optimize(
    grid: List<Cell>,
    odMatrix: Map<Pair<Cell, Cell>, Double>,
    totalPop: Double,
    affectedAltSensors: Map<Pair<RealLocation, RealLocation>, List<List<TrafficSensor>>>,
    sensors: List<TrafficSensor>
) {

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

        val As = mutableListOf<GRBVar>()

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
                    As.addAll(a)

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

        // Dispose of model and environment
        model.dispose()
        env.dispose()
    } catch (e: GRBException) {
        println(
            ("Error code: " + e.errorCode + ". " +
                    e.message)
        )
    }

}