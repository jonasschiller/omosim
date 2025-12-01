package de.uniwuerzburg.omod.calibration.surrogate

import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.TrafficSensor
import de.uniwuerzburg.omod.calibration.differentiablemodel.LinearTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.QuadraticTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.Term

fun mseObjective(nVars: Int, sensors: List<TrafficSensor>, simCount: Map<TrafficSensor, List<Term>>) : LinearTerm {
    val obj = LinearTerm(nVars)
    for (sensor in sensors) {
        for (t in 0 until T) {
            // (Sm - Ss)^2 = Sm^2 - 2SmSs + Ss^2
            val simCountTerm = simCount[sensor]!![t]
            obj.addConstant(sensor.measuredFlow[t] * sensor.measuredFlow[t])
            obj.addTerm(simCountTerm, -2 * sensor.measuredFlow[t])
            val qTerm = QuadraticTerm(
                nVars,
                simCountTerm,
                simCountTerm,
                1.0
            )
            obj.addTerm(qTerm, 1.0)
        }
    }
    return obj
}
