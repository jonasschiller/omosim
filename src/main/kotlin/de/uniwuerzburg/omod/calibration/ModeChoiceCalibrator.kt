package de.uniwuerzburg.omod.calibration

import de.uniwuerzburg.omod.core.ModeChoiceFast
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.ActivityType
import de.uniwuerzburg.omod.core.models.Cell
import de.uniwuerzburg.omod.core.models.MobiAgent
import de.uniwuerzburg.omod.core.models.Mode
import de.uniwuerzburg.omod.core.models.RealLocation
import org.jetbrains.kotlinx.multik.ndarray.data.get
import smile.math.BFGS
import java.util.Random
import kotlin.math.abs
import kotlin.math.exp
import kotlin.time.measureTime
import kotlin.time.measureTimedValue

fun calibrate(
    agents: List<MobiAgent>, rng: Random, omod: Omod,
    sensors: List<TrafficSensor>,
    affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
): List<Double> {
    val mc  = ModeChoiceFast(omod.routingCache)
    println("Mode Choice Cal")

    println("Building diff model...")
    val (model, simCount) = buildModel(agents, mc, rng, omod, sensors, affectedLinks)

    var parameters = DoubleArray(1) { 0.0 }
    //parameters = lbfgs(model, parameters)
    //parameters = gradDescent(model, parameters)
    //println(parameters.toList())

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

    return parameters.toList()
}

fun lbfgs(model: DifferentiableModel, vals: DoubleArray) : DoubleArray {
    println("LBFGS-B")
    println("Start: ${model.evaluate(vals)}")
    val l = DoubleArray(model.nVars){-30.0}
    val u = DoubleArray(model.nVars){30.0}
    val solution = BFGS.minimize(model, 5, vals, l, u, 1e-5, 30)
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
    affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
) : Pair<DifferentiableModel, Map<TrafficSensor, LinearTerm>> {
    val totalPop = omod.buildings.sumOf { it.population }

    val model = DifferentiableModel(1)

    val simcount = mutableMapOf<TrafficSensor, LinearTerm>()
    for (sensor in sensors) {
        simcount[sensor] = LinearTerm(model.nVars)
    }

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

            var o = tour.first().fromActivity.location.getAggLoc()!! as Cell
            for (trip in tour) {
                val d = trip.toActivity.location.getAggLoc()!! as Cell
                val tripOD = Pair(o, d)
                if (tripOD in affectedLinks) {
                    val affectedSensors = affectedLinks[tripOD]!!
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
    return model to simcount


}