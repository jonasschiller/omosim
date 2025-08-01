package de.uniwuerzburg.omod.calibration

import com.gurobi.gurobi.*
import de.uniwuerzburg.omod.core.ActivityGeneratorDefault
import de.uniwuerzburg.omod.core.CarOwnership
import de.uniwuerzburg.omod.core.DestinationFinderDefault
import de.uniwuerzburg.omod.core.models.*
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
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.jetbrains.kotlinx.multik.ndarray.operations.toDoubleArray
import org.locationtech.jts.geom.Coordinate

fun optimize(
    grid: List<Cell>, activityGenerator: ActivityGeneratorDefault,
    modeChoiceCalibration: ModeChoiceCalibration,
    customCellFactors: Map<ActivityType, Map<Cell, Double>>,
    popStrata: List<PopStratum>,
    carOwnership: CarOwnership,
    destinationFinder: DestinationFinderDefault,
    totalPop: Double,
    affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
    sensors: List<TrafficSensor>,
) : List<ODPossibility>{
    val odProbs = determinePairProbabilities(
        grid, activityGenerator, modeChoiceCalibration, customCellFactors, popStrata, carOwnership, destinationFinder
    )
    println("nVars: ${odProbs.size}")

    val relevantODs = affectedSensors.keys.toSet()
    val odProbsByOD = mutableMapOf<Pair<RealLocation, RealLocation>, MutableList<ODPossibility>>()
    var affected = 0.0
    var notAffected = 0.0
    for (prob in odProbs) {
        val od = Pair(grid[prob.o], grid[prob.d])
        if ( od !in relevantODs ) { continue }
        if(prob.affected == AffectedByWork.NO ) {
            notAffected += 1.0
        } else {
            affected += 1.0
        }
        if ( od in odProbsByOD) {
            odProbsByOD[od]!!.add(prob)
        } else {
            odProbsByOD[od] = mutableListOf(prob)
        }
    }
    println("rel Vars: ${odProbsByOD.values.sumOf { it.size }}")
    println("Affected: ${affected}")
    println("Not Affected: ${notAffected}")


    val workTransitionMatrix = mk.zeros<Double>(grid.size, grid.size)
    for (o in grid.indices) {
        val workWeights = destinationFinder.getWeights(grid[o], grid, activityType = ActivityType.WORK, customCellFactors)
        workTransitionMatrix[o] = mk.ndarray( workWeights.map { it / workWeights.sum() } )
    }

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

        /* Complete free
        var wx = mutableListOf<Array<GRBVar>>()
        for (o in grid.indices) {
            wx.add(
                model.addVars(
                    DoubleArray(grid.size) {0.01},
                    DoubleArray(grid.size) {0.2},
                    DoubleArray(grid.size) {0.0},
                    CharArray(grid.size) {GRB.CONTINUOUS},
                    Array(grid.size) {""}
                )
            )
        }

        // Make sure work scaling doesn't change weight sums
        // TODO all flexible. For this I must remember where the x where located.
        // TODO is it just at the normal location?
        for (o in grid.indices) {
            val wSum = GRBLinExpr()
            wSum.addTerms(DoubleArray(grid.size) {1.0}, wx[o])
            model.addConstr(wSum, GRB.EQUAL, 1.0, "weightConsistency")
        }
        */
        // Mult model, TODO doesn't do what I think it do
        val workScaling =  model.addVars(
            DoubleArray(grid.size) {0.0},
            DoubleArray(grid.size) {5.0},
            DoubleArray(grid.size) {0.0},
            CharArray(grid.size) {GRB.CONTINUOUS},
            Array(grid.size) {""}
        )
        val workScalingNorm = model.addVars(
            DoubleArray(grid.size) {0.0},
            DoubleArray(grid.size) {1.0},
            DoubleArray(grid.size) {0.0},
            CharArray(grid.size) {GRB.CONTINUOUS},
            Array(grid.size) {""}
        )
       for (o in grid.indices) {
           val oWeights = workTransitionMatrix[o].toDoubleArray()
           for (idxScale in grid.indices) {
               val weightSum = GRBQuadExpr()
               for (d in grid.indices) {
                   weightSum.addTerm(oWeights[d], workScaling[d], workScalingNorm[idxScale])
               }
               model.addQConstr(weightSum, GRB.EQUAL, workScaling[idxScale], "weightConsistency")
           }

            //val weightSum = GRBLinExpr()
            //weightSum.addTerms(oWeights, workScaling)
            //model.addConstr(weightSum, GRB.EQUAL, oWeights.sum(), "weightConsistency")
        }


        for (origin in grid) {
            for (destination in grid) {
                val od = Pair(origin, destination)
                if (od in affectedSensors) {
                    val x = GRBLinExpr()
                    for (prob in odProbsByOD[od]!!) {
                        if (prob.affected == AffectedByWork.NO) {
                            x.addConstant(prob.pCar *  prob.pChain * prob.pTransition * totalPop)
                        } else {
                            x.addTerm(
                                prob.pCar *  prob.pChain * prob.pTransition * totalPop,
                                workScalingNorm[prob.d]
                            )
                        }
                    }

                    val affected = affectedSensors[od]!!
                    for (sensor in affected) {
                        val sensorSum = sensorCountExpr[sensor]!!
                        sensorSum.add(x)
                    }
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

        // TODO warmstart if it remains a MIP
        model.optimize()

        var optimstatus = model[GRB.IntAttr.Status]

        if (optimstatus == GRB.Status.INF_OR_UNBD) {
            model[GRB.IntParam.Presolve] = 0
            model.optimize()
            optimstatus = model[GRB.IntAttr.Status]
        }

        if (optimstatus == GRB.Status.OPTIMAL) {
            // Print results
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

    return odProbs
}

private fun determinePairProbabilities(
    grid: List<Cell>, activityGenerator: ActivityGeneratorDefault,
    modeChoiceCalibration: ModeChoiceCalibration,
    customCellFactors: Map<ActivityType, Map<Cell, Double>>,
    popStrata: List<PopStratum>,
    carOwnership: CarOwnership,
    destinationFinder: DestinationFinderDefault,
) : List<ODPossibility>{
    // Result: Expected trip count on each od-paid of one agent on one day
    val odPossibilities = mutableListOf<ODPossibility>()

    // Precompute important probabilities
    // HOME
    val homeWeights = destinationFinder.getWeightsNoOrigin(grid, activityType = ActivityType.HOME) // TODO what about buffer area
    val homeProbs   = mk.ndarray( homeWeights.map { it / homeWeights.sum() } )
        .expandDims(0).asDNArray().asD2Array()

    // Transitions
    val transitionProbs = mutableMapOf<ActivityType,  D2Array<Double>>()
    for (activityType in ActivityType.entries) {
        if (activityType == ActivityType.HOME) { continue }

        val activityProbs = mk.zeros<Double>(grid.size, grid.size)
        for (o in grid.indices) {
            val activityWeights = destinationFinder.getWeights(grid[o], grid, activityType = activityType, customCellFactors)
            activityProbs[o] = mk.ndarray( activityWeights.map { it / activityWeights.sum() } )
        }

        transitionProbs[activityType] = activityProbs
    }

    // Home <-> Work, Work
    val workTransitionP = transitionProbs[ActivityType.WORK]!!
    val homeWorkProbs = homeProbs.diagonal().dot(workTransitionP)
    val workProbs = homeProbs.dot(workTransitionP)

    // Home <-> School, School
    val schoolTransitionP = transitionProbs[ActivityType.SCHOOL]!!
    val homeSchoolProbs = homeProbs.diagonal().dot(schoolTransitionP)
    val schoolProbs = homeProbs.dot(schoolTransitionP)

    // Car probability
    val dummyCoord = Coordinate(0.0,0.0)
    val dummyLocation = DummyLocation(dummyCoord, dummyCoord, null, setOf())
    val carProbs = mutableMapOf<ActivityType, D2Array<Double>>()
    for (activity in ActivityType.entries) {
        carProbs[activity] = mk.zeros<Double>(grid.size, grid.size)
    }
    for (stratum in popStrata) {
        if (stratum.stratumShare == 0.0) { continue }

        for ((socioFeatureSet, pSFSet) in stratum.iterateOptions()) {
            if (pSFSet == 0.0) { continue }

            val dummyAgent = MobiAgent(
                -1,  socioFeatureSet.hom, socioFeatureSet.mob, socioFeatureSet.age,
                dummyLocation, dummyLocation, dummyLocation, socioFeatureSet.sex
            )
            dummyAgent.carAccess = true // Car ownership probability is considered separately
            val carOwnershipP = carOwnership.probability(dummyAgent, stratum)
            if (carOwnershipP == 0.0) { continue }

            for (activity in ActivityType.entries) {
                val carProbsActivity = mk.zeros<Double>(grid.size, grid.size)
                for (o in grid.indices) {
                    val distances = destinationFinder.routingCache.getDistances(grid[o], grid)
                    for (d in grid.indices) {
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

    // Get activity chains
    // TODO Multiple days
    val allChains = mutableMapOf<List<ActivityType>, Double>()
    for (stratum in popStrata) {
        if (stratum.stratumShare == 0.0) { continue }

        for ((socioFeatureSet, pSFSet) in stratum.iterateOptions()) {
            if (pSFSet == 0.0) { continue }
            val ageGrp = AgeGrp.fromInt(socioFeatureSet.age)
            val chains = activityGenerator.getChain(
                Weekday.UNDEFINED, socioFeatureSet.hom, socioFeatureSet.mob, ageGrp, ActivityType.HOME
            )
            val chainProbs = chains.weights.map { it / chains.weights.sum() }.toTypedArray()

            for ((chain, chainP) in chains.chains.zip(chainProbs)) {
                val p = stratum.stratumShare * pSFSet * chainP
                if (chain in allChains) {
                    allChains[chain] = allChains[chain]!! + p
                } else {
                    allChains[chain] = p
                }
            }
        }
    }

    // Determine unique Fixed -> Fixed chain segments
    // TODO: Later when time is important remember time windows on tour generation
    val fixedActivities = setOf(ActivityType.HOME, ActivityType.WORK, ActivityType.SCHOOL)
    val fixedSegments = mutableListOf<de.uniwuerzburg.omod.calibration.FixedSegment>()
    val segment = mutableListOf<ActivityType>()
    for ((chain, chainP) in allChains) {
        segment.clear()

        for ((i, activity) in chain.withIndex()) {
            segment.add(activity)
            if (segment.size > 1) {
                if ((activity in fixedActivities) or (i == chain.size - 1)) {
                    fixedSegments.add(
                        FixedSegment(
                            segment.toList(),
                            chainP
                        )
                    )
                    // New segment starts with last activity
                    segment.clear()
                    segment.add(activity)
                }
            }
        }
    }
    val uniqueSegments = fixedSegments.groupBy { it.chain }.map { (chain, segments) ->
        FixedSegment(
            chain,
            segments.sumOf { it.probability }
        )
    }

    // Determine od pair probabilities
    // TODO time
    for ((chain, chainP) in uniqueSegments) {
        if (chain.size <= 1) { continue }

        val startActivity = chain.first()
        val endActivity = chain.last()

        // Short chains
        if (chain.size == 2) {
            when(Pair(startActivity, endActivity)) {
                Pair(ActivityType.HOME, ActivityType.WORK) -> {
                    val carP = carProbs[endActivity]!!
                    odPossibilities.addAll( getOdPs(grid.size, chainP, homeWorkProbs, carP, affected = AffectedByWork.PURE) )
                    continue
                }
                Pair(ActivityType.HOME, ActivityType.SCHOOL) -> {
                    val carP = carProbs[endActivity]!!
                    odPossibilities.addAll( getOdPs(grid.size, chainP, homeSchoolProbs, carP) )
                    continue
                }
                Pair(ActivityType.WORK, ActivityType.HOME) -> { // TODO Tranpose homeWorkProbs?
                    val carP = carProbs[endActivity]!!
                    odPossibilities.addAll( getOdPs(grid.size, chainP, homeWorkProbs, carP, affected = AffectedByWork.PURE) )
                    continue
                }
                Pair(ActivityType.SCHOOL, ActivityType.HOME) -> {
                    val carP = carProbs[endActivity]!!
                    odPossibilities.addAll( getOdPs(grid.size, chainP, homeSchoolProbs, carP) )
                    continue
                }
                else -> {}
            }
        }

        // Setup
        var probPositionGivenLastFixed: D2Array<Double>?
        var previousProbs: D2Array<Double>
        when(startActivity) {
            ActivityType.HOME -> {
                previousProbs = homeProbs
                probPositionGivenLastFixed = when(endActivity) {
                    ActivityType.HOME -> homeProbs.diagonal()
                    ActivityType.WORK -> homeWorkProbs
                    ActivityType.SCHOOL -> homeSchoolProbs
                    else -> null
                }
            }
            ActivityType.WORK -> {
                previousProbs = workProbs
                probPositionGivenLastFixed = when(endActivity) {
                    ActivityType.HOME -> homeWorkProbs
                    ActivityType.WORK -> workProbs.diagonal()
                    else -> null
                }
            }
            ActivityType.SCHOOL -> {
                previousProbs = schoolProbs
                probPositionGivenLastFixed = when(endActivity) {
                    ActivityType.HOME -> homeSchoolProbs
                    ActivityType.SCHOOL -> schoolProbs.diagonal()
                    else -> null
                }
            }
            else -> {
                throw IllegalStateException(
                    "Last fixed activity can not be of type $startActivity !"
                )
            }
        }

        // Flexible chain components
        for (activity in chain.drop(1).dropLast(1)) {
            val affected = if (startActivity == ActivityType.WORK) {
                AffectedByWork.PURE
            } else {
                AffectedByWork.NO
            }
            val transitionActivity = if (activity == ActivityType.SHOPPING) {
                activity
            } else {
                ActivityType.OTHER
            }
            val transitionP = transitionProbs[transitionActivity]!!

            val carP = carProbs[transitionActivity]!!
            odPossibilities.addAll( getOdPs(grid.size, chainP, previousProbs.diagonal().dot(transitionP), carP, affected) )
            previousProbs = previousProbs.dot(transitionP)

            probPositionGivenLastFixed = probPositionGivenLastFixed?.dot(transitionP)
        }

        // Last chain component. Fixed most of the time.
        if (
            (endActivity == ActivityType.HOME) ||
            ((endActivity == ActivityType.WORK) && (startActivity != ActivityType.SCHOOL)) ||
            ((endActivity == ActivityType.SCHOOL) && (startActivity != ActivityType.WORK))
        ) {
            val affected = if ((startActivity == ActivityType.WORK) || (endActivity == ActivityType.WORK)) {
                if (endActivity == ActivityType.WORK) {
                    AffectedByWork.PURE
                } else {
                    AffectedByWork.MIXED
                }
            } else {
                AffectedByWork.NO
            }
            val carP = carProbs[endActivity]!!

            if (affected != AffectedByWork.MIXED) {
                odPossibilities.addAll( getOdPs(grid.size, chainP, probPositionGivenLastFixed!!.transpose(), carP, affected) )
            } else {
                // Use shortcut to go PURE mode
                odPossibilities.addAll(
                    getOdPs(grid.size, chainP, homeWorkProbs, carP,  AffectedByWork.PURE)
                )
            }

        } else {
            val transitionActivity = if (endActivity == ActivityType.SHOPPING) {
                endActivity
            } else {
                ActivityType.OTHER
            }
            val transitionP = transitionProbs[transitionActivity]!!

            val carP = carProbs[transitionActivity]!!
            odPossibilities.addAll( getOdPs(grid.size, chainP,  previousProbs.diagonal().dot(transitionP), carP) )
        }
    }

    return odPossibilities
}

private fun getOdPs(
    n: Int, pChain: Double, pTransition: D2Array<Double>, pCar:  D2Array<Double>,
    affected: AffectedByWork = AffectedByWork.NO
) : List<ODPossibility> {
    val odPossibilities = mutableListOf<ODPossibility>()
    if (pChain == 0.0) { return odPossibilities }
    for (o in 0 until n) {
        for (d in 0 until n) {
            if (pTransition[o, d] == 0.0) { continue }
            if (pCar[o, d] == 0.0) { continue }

            odPossibilities.add(
                ODPossibility(
                    o,
                    d,
                    pChain,
                    pTransition[o, d],
                    pCar[o, d],
                    affected
                )
            )
        }
    }
    return odPossibilities
}


enum class AffectedByWork {
    NO, PURE, MIXED
}

// NO:
// Contribution: pChain * pCar * pTransition
// PURE:
// Contribution: pChain * pCar * pTransition * WorkScaler[d]
// MIXED:
// Contribution:
data class ODPossibility(
    val o: Int,
    val d: Int,
    val pChain: Double,
    val pTransition: Double,
    val pCar: Double,
    val affected: AffectedByWork = AffectedByWork.NO
)

data class ODPossibilityTest(
    val pChain: Double,
    val pTransition: Double,
    val pCar: Double,
    val affects: WTerms = WTerms()
)

data class WTerms(
    val w: MutableList<Double> = mutableListOf(),
    val o: MutableList<Int> = mutableListOf(),
    val d: MutableList<Int> = mutableListOf(),
)


private data class FixedSegment(
    val chain: List<ActivityType>,
    val probability: Double
)

private inline fun <reified T : Any> D2Array<T>.diagonal() : D2Array<T> {
    require(this.shape[0] == 1)
    val diagonal  = mk.zeros<T>(this.size, this.size)
    for ( i in 0 until this.size) {
        diagonal[i, i] = this[0,i]
    }
    return  diagonal
}