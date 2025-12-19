package de.uniwuerzburg.omod.calibration

import com.gurobi.gurobi.GRB
import com.gurobi.gurobi.GRBExpr
import com.gurobi.gurobi.GRBLinExpr
import com.gurobi.gurobi.GRBModel
import com.gurobi.gurobi.GRBQuadExpr
import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.differentiablemodel.LinearTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.QuadraticTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.Term

fun sseObjective(nVars: Int, sensors: List<TrafficSensor>, simCount: Map<TrafficSensor, List<Term>>) : LinearTerm {
    val obj = LinearTerm(nVars)
    for (sensor in sensors) {
        for (t in 0 until T) {
            // (m - s)^2 = m^2 - 2ms + s^2
            val s = simCount[sensor]!![t]
            val m = sensor.measuredFlow[t]
            obj.addConstant(m * m)
            obj.addTerm(s, -2 * m)
            val qTerm = QuadraticTerm(nVars, s, s,1.0)
            obj.addTerm(qTerm, 1.0)
        }
    }
    return obj
}

fun grbSseObjective(
    model: GRBModel, sensors: List<TrafficSensor>, simCount: Map<TrafficSensor, List<GRBLinExpr>>
) : GRBExpr {
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
    return obj
}