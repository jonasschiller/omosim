package de.uniwuerzburg.omod.calibration

import com.github.ajalt.mordant.table.row
import de.uniwuerzburg.omod.core.DestinationFinderDefault
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.ActivityType
import de.uniwuerzburg.omod.core.models.Cell
import de.uniwuerzburg.omod.core.models.DummyLocation
import de.uniwuerzburg.omod.core.models.MobiAgent
import de.uniwuerzburg.omod.core.models.RealLocation
import de.uniwuerzburg.omod.core.models.Weekday
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.asDNArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.expandDims
import org.jetbrains.kotlinx.multik.ndarray.operations.plusAssign
import org.locationtech.jts.geom.Coordinate
import smile.math.BFGS
import kotlin.collections.iterator


/*
    1. Trip Generation:
        - Ti = Trip generated in each region
    2. Trip Distribution:
        - Tij = Ti * Dj * Aj / sum(Dk * Ak for all k)
 */

fun calibrateK1(
    seedAgents: List<MobiAgent>,
    omod: Omod,
    sensors: List<TrafficSensor>,
    affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
) : List<Double> {
    println("Fourstep")
    val (model, simCount) = buildModel(seedAgents, omod, sensors, affectedLinks)

    var parameters = DoubleArray(omod.grid.size) { 1.0 }

    println("_".repeat(20*4 + 5*3))
    println("${"Sensor".padEnd(20)} | \t" +
            "${"Flow AltOpt".padEnd(20)} | \t" +
            "Flow Measured".padEnd(20)
    )
    println("_".repeat(20) +
            " | \t" + "_".repeat(20)  +
            " | \t" + "_".repeat(20))
    var myobjval1 = 0.0
    for ((i, sensor) in sensors.withIndex()) {
        val optVal = simCount[sensor]!!.evaluate(parameters)
        println(
            "${sensors[i].name.padEnd(20)} | \t" +
                    "${optVal.toString().padEnd(20)} | \t" +
                    sensors[i].measuredFlow.toString().padEnd(20)
        )
        myobjval1 += (sensors[i].measuredFlow - optVal) * (sensors[i].measuredFlow - optVal)
    }
    println("MSE: ${myobjval1 /sensors.size}")

    parameters = tst_lbfgs(model, parameters)

    println(parameters.toList())

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

fun tst_lbfgs(model: DifferentiableModel, vals: DoubleArray) : DoubleArray {
    println("LBFGS-B")
    println("Start: ${model.evaluate(vals)}")
    val l = DoubleArray(model.nVars){0.0}
    val u = DoubleArray(model.nVars){1e3}
    val solution = BFGS.minimize(model, 5, vals, l, u, 1e-5, 30)
    println("Solution: $solution")
    println("Confirm: ${model.evaluate(vals)}")
    return vals
}

fun buildModel(
    seedAgents: List<MobiAgent>,
    omod: Omod,
    sensors: List<TrafficSensor>,
    affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
):  Pair<DifferentiableModel, Map<TrafficSensor, LinearTerm>> {
    val modeChoiceCalibration = ModeChoiceCalibration()
    val totalPop = omod.buildings.sumOf { it.population }

    val pCar = getPCar(omod, modeChoiceCalibration)
    val mTs = computeTransitionMatrices(omod)
    val model = DifferentiableModel(omod.grid.size)
    val n = omod.grid.size

    val demand = Array(n) {
        Array(n) {
            LinearTerm(model.nVars)
        }
    }


    val relevantODs = mutableSetOf<Pair<Int, Int>>()
    for ((o, origin) in omod.grid.withIndex()) {
        for ((d, destination) in omod.grid.withIndex()) {
            val od = Pair(origin, destination)
            if (od in affectedLinks) {
                if (affectedLinks[od]!!.isNotEmpty()) {
                    relevantODs.add(Pair(o, d))
                }
            }
        }
    }

    // Step 1
    val genTrips = ActivityType.entries.associateWith { omod.grid.associateWith { 0.0 }.toMutableMap() }
    for (agent in seedAgents) {
        val activities = agent.mobilityDemand.first().activities
        var origin = activities.first()
        for ((i, destination) in activities.drop(1).withIndex()) {
            val oCell = origin.location.getAggLoc()!! as Cell
            val aType = destination.type

            //if (i == 0) {
            //    genTrips[aType]!![oCell] = genTrips[aType]!![oCell]!! + totalPop / seedAgents.size.toDouble()
            //}

            if (aType == ActivityType.HOME) {
                val o = omod.grid.indexOf(oCell)
                val dCell = destination.location.getAggLoc()!! as Cell
                val d = omod.grid.indexOf(dCell)
                demand[o][d].addConstant(totalPop / seedAgents.size.toDouble() * pCar[aType]!![o, d])
            } else {
                genTrips[aType]!![oCell] = genTrips[aType]!![oCell]!! + totalPop / seedAgents.size.toDouble()
            }
            origin = destination
        }
    }

    // Step 2
    val tMatrices = computeTransitionMatrices(omod)
    for (activity in ActivityType.entries) {
        if (activity == ActivityType.HOME) { continue }

        if (activity == ActivityType.OTHER) {
            for (o in 0 until n) {
                val entry = mutableListOf<Term>()
                val weightTerms = mutableListOf<Term>()
                val rowSumTerm = LinearTerm(model.nVars)

                for (d in 0 until n) {
                    val weight = LinearBaseTerm(model.nVars)
                    weight.addTerm(d, mTs.transitionMatrix[activity]!![o, d])
                    rowSumTerm.addTerm(weight, 1.0)
                    weightTerms.add(weight)
                }

                for (d in 0 until n) {
                    val car = pCar[activity]!![o, d]
                    val od = Pair(o, d)
                    if (od in relevantODs) {
                        val gen = genTrips[activity]!![omod.grid[o]]

                        if (gen != null) {
                            val scaledWeight = DivisionTerm(model.nVars, weightTerms[d], rowSumTerm)
                            demand[o][d].addTerm(scaledWeight, gen * car)
                        }
                    }
                }
            }
        } else {
            for (o in 0 until n) {
                var rowSum = 0.0
                for (d in 0 until n ) {
                    rowSum += mTs.transitionMatrix[activity]!![o, d]
                }

                for (d in 0 until n) {
                    val weight = mTs.transitionMatrix[activity]!![o, d]
                    val car = pCar[activity]!![o, d]
                    val od = Pair(o, d)
                    if (od in relevantODs) {
                        val gen = genTrips[activity]!![omod.grid[o]]

                        if (gen != null) {
                            demand[o][d].addConstant(gen * car * weight / rowSum)
                        }
                    }
                }
            }
        }
    }


    val simcount = mutableMapOf<TrafficSensor, LinearTerm>()
    for (sensor in sensors) {
        simcount[sensor] = LinearTerm(model.nVars)
    }

    for ((o, origin) in omod.grid.withIndex()) {
        for ((d, destination) in omod.grid.withIndex()) {
            val od = Pair(origin, destination)
            if (od in affectedLinks) {
                val affected = affectedLinks[od]!!

                for (sensor in affected) {
                    simcount[sensor]!!.addTerm(demand[o][d], 1.0)
                }
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

private fun computeTransitionMatrices(omod: Omod) : FinderMatrices {
    val finder = omod.destinationFinder as DestinationFinderDefault

    // HOME
    val homeWeights = finder.getWeightsNoOrigin(omod.grid, activityType = ActivityType.HOME) // TODO what about buffer area
    val homeProbs   = mk.ndarray( homeWeights.map { it / homeWeights.sum() } )
        .expandDims(0).asDNArray().asD2Array()

    // Transitions
    val transitionProbs = mutableMapOf<ActivityType,  D2Array<Double>>()
    for (activityType in ActivityType.entries) {
        if (activityType == ActivityType.HOME) { continue }

        val activityProbs = mk.zeros<Double>(omod.grid.size, omod.grid.size)
        for (o in omod.grid.indices) {
            val activityWeights = finder.getWeights(omod.grid[o], omod.grid, activityType = activityType)
            activityProbs[o] = mk.ndarray( activityWeights.map { it / activityWeights.sum() } )
        }

        transitionProbs[activityType] = activityProbs
    }
    return FinderMatrices(homeProbs, transitionProbs)
}

private fun getPCar(omod: Omod, modeChoiceCalibration: ModeChoiceCalibration) : Map<ActivityType, D2Array<Double>> {
    val finder = omod.destinationFinder as DestinationFinderDefault

    // Car probability
    val dummyCoord = Coordinate(0.0,0.0)
    val dummyLocation = DummyLocation(dummyCoord, dummyCoord, null, setOf())
    val carProbs = mutableMapOf<ActivityType, D2Array<Double>>()
    for (activity in ActivityType.entries) {
        carProbs[activity] = mk.zeros<Double>(omod.grid.size, omod.grid.size)
    }
    for (stratum in omod.popStrata) {
        if (stratum.stratumShare == 0.0) { continue }

        for ((socioFeatureSet, pSFSet) in stratum.iterateOptions()) {
            if (pSFSet == 0.0) { continue }

            val dummyAgent = MobiAgent(
                -1,  socioFeatureSet.hom, socioFeatureSet.mob, socioFeatureSet.age,
                dummyLocation, dummyLocation, dummyLocation, socioFeatureSet.sex
            )
            dummyAgent.carAccess = true // Car ownership probability is considered separately

            val carOwnershipP = omod.carOwnership.probability(dummyAgent, stratum)
            if (carOwnershipP == 0.0) { continue }

            for (activity in ActivityType.entries) {
                val carProbsActivity = mk.zeros<Double>(omod.grid.size, omod.grid.size)
                for (o in omod.grid.indices) {
                    val distances = finder.routingCache.getDistances(omod.grid[o], omod.grid)
                    for (d in omod.grid.indices) {
                        val weights = modeChoiceCalibration.utilitiesForCalibration(
                            distances[d].toDouble() / 1000.0, dummyAgent, activity, Weekday.UNDEFINED
                        )
                        val pTrip = weights[0] / weights.sum()
                        carProbsActivity[o, d] = stratum.stratumShare * pSFSet * carOwnershipP * pTrip
                    }
                }
                carProbs[activity]?.plusAssign(carProbsActivity)
            }
        }
    }
    return carProbs
}

private data class FinderMatrices (
    val homeP: D2Array<Double>,
    val transitionMatrix: Map<ActivityType,  D2Array<Double>>,
) {
    // Home <-> Work, Work
    val homeWorkProbs = homeP.diagonal().dot( transitionMatrix[ActivityType.WORK]!! )

    // Home <-> School, School
    val homeSchoolProbs = homeP.diagonal().dot( transitionMatrix[ActivityType.SCHOOL]!! )
}


private inline fun <reified T : Any> D2Array<T>.diagonal() : D2Array<T> {
require(this.shape[0] == 1)

val diagonal  = mk.zeros<T>(this.size, this.size)
for ( i in 0 until this.size) {
    diagonal[i, i] = this[0,i]
}
return diagonal
}