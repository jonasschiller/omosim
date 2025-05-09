package de.uniwuerzburg.omod.calibration

import com.gurobi.gurobi.*
import de.uniwuerzburg.omod.core.ActivityGeneratorDefault
import de.uniwuerzburg.omod.core.CarOwnership
import de.uniwuerzburg.omod.core.DestinationFinderDefault
import de.uniwuerzburg.omod.core.models.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.asDNArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import org.locationtech.jts.geom.Coordinate
import kotlin.math.pow

object WACalibrator {
    fun determinePairProbabilities(
        grid: List<Cell>, activityGenerator: ActivityGeneratorDefault,
        modeChoiceCalibration: ModeChoiceCalibration,
        customCellFactors: Map<Cell, Double>,
        popStrata: List<PopStratum>,
        carOwnership: CarOwnership,
        destinationFinder: DestinationFinderDefault,
        totalPop: Double,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        sensors: List<TrafficSensor>
    ) : Map<Pair<Cell, Cell>, Double> {

        // Result: Expected trip count on each od-paid of one agent on one day
        val expectedCountPerAgent = mk.zeros<Double>(grid.size, grid.size)

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
        val fixedSegments = mutableListOf<FixedSegment>()
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

        val workMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(grid.size, grid.size) }
        val fixedMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(grid.size, grid.size) }
        val hFixedMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(grid.size, grid.size) }
        val nthFixedMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(grid.size, grid.size) }
        val testFixedMatrix = mk.zeros<Double>(grid.size, grid.size)
        val testMatrix = homeProbs.times(0.0)
        val expectedOther = mk.zeros<Double>(grid.size, grid.size)
        val expectedHome = mk.zeros<Double>(grid.size, grid.size)
        val expectedWork = mk.zeros<Double>(grid.size, grid.size)
        val expectedHomeWork = mk.zeros<Double>(grid.size, grid.size)
        val expectedSchool = mk.zeros<Double>(grid.size, grid.size)
        val expectedShopping = mk.zeros<Double>(grid.size, grid.size)

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
                        //val carP = carProbs[endActivity]!!
                        val p = homeWorkProbs * chainP
                        expectedCountPerAgent.plusAssign(p)
                        expectedWork.plusAssign(p)
                        expectedHomeWork.plusAssign(p)
                        fixedMatrix[ActivityType.WORK]!!.plusAssign(homeProbs.diagonal() * chainP)
                        //hFixedMatrix[ActivityType.WORK]!!.plusAssign(homeWorkProbs.transpose() * chainP)
                        //hFixedMatrix[ActivityType.WORK]!!.plusAssign( mk.identity<Double>(grid.size) * chainP)
                        nthFixedMatrix[ActivityType.WORK]!!.plusAssign( mk.identity<Double>(grid.size) * chainP)
                        testMatrix.plusAssign(homeProbs * chainP)
                        testFixedMatrix.plusAssign(homeProbs.diagonal() * chainP)
                        continue
                    }
                    Pair(ActivityType.HOME, ActivityType.SCHOOL) -> {
                        //val carP = carProbs[endActivity]!!
                        val p = homeSchoolProbs* chainP
                        expectedCountPerAgent.plusAssign(p)
                        expectedSchool.plusAssign(p)
                        fixedMatrix[ActivityType.SCHOOL]!!.plusAssign(homeProbs.diagonal() * chainP)
                        hFixedMatrix[ActivityType.SCHOOL]!!.plusAssign(homeProbs.diagonal() * chainP)
                        testMatrix.plusAssign(homeProbs * chainP)
                        testFixedMatrix.plusAssign(homeProbs.diagonal() * chainP)
                        continue
                    }
                    Pair(ActivityType.WORK, ActivityType.HOME) -> {
                        //val carP = carProbs[endActivity]!!
                        val p = homeWorkProbs.transpose() * chainP // TODO Transpose?
                        expectedCountPerAgent.plusAssign(p)
                        expectedHome.plusAssign(p)
                        workMatrix[ActivityType.HOME]!!.plusAssign(mk.identity<Double>(grid.size) * chainP)
                        testMatrix.plusAssign(workProbs * chainP)
                        testFixedMatrix.plusAssign(homeWorkProbs * chainP)
                        continue
                    }
                    Pair(ActivityType.SCHOOL, ActivityType.HOME) -> {
                        //val carP = carProbs[endActivity]!!
                        val p = homeSchoolProbs.transpose() * chainP
                        expectedCountPerAgent.plusAssign(p)
                        expectedHome.plusAssign(p)
                        fixedMatrix[ActivityType.HOME]!!.plusAssign(homeSchoolProbs * chainP)
                        testMatrix.plusAssign(schoolProbs * chainP)
                        testFixedMatrix.plusAssign(homeSchoolProbs * chainP)
                        continue
                    }
                    else -> {}
                }
            }

            var nextActivity = chain[1]

            // Setup
            var probPositionGivenLastFixed: D2Array<Double>?
            var pLastTest: D2Array<Double>? = null
            var previousProbs: D2Array<Double>
            var cumMatrix = mk.identity<Double>(grid.size)
            when(startActivity) {
                ActivityType.HOME -> {
                    testFixedMatrix.plusAssign(homeProbs.diagonal() * chainP)
                    fixedMatrix[nextActivity]!!.plusAssign(homeProbs.diagonal() * chainP)
                    pLastTest = mk.identity<Double>(grid.size)
                    hFixedMatrix[nextActivity]!!.plusAssign(homeWorkProbs * chainP) // TODO Tmp
                    previousProbs = homeProbs
                    probPositionGivenLastFixed = when(endActivity) {
                        ActivityType.HOME -> homeProbs.diagonal()
                        ActivityType.WORK -> homeWorkProbs
                        ActivityType.SCHOOL -> homeSchoolProbs
                        else -> null
                    }
                }
                ActivityType.WORK -> {
                    workMatrix[nextActivity]!!.plusAssign(cumMatrix * chainP)
                    testFixedMatrix.plusAssign(homeWorkProbs * chainP)

                    //testMatrix.plusAssign(workProbs * chainP)
                    previousProbs = workProbs
                    probPositionGivenLastFixed = when(endActivity) {
                        ActivityType.HOME -> homeWorkProbs
                        ActivityType.WORK -> workProbs.diagonal()
                        else -> null
                    }
                }
                ActivityType.SCHOOL -> {
                    testFixedMatrix.plusAssign(homeSchoolProbs * chainP)
                    fixedMatrix[nextActivity]!!.plusAssign(homeSchoolProbs * chainP)
                    //testMatrix.plusAssign(schoolProbs * chainP)
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
            var next = 2
            for (activity in chain.drop(1).dropLast(1)) {
                nextActivity = chain[next]
                next += 1

                val transitionActivity = if (activity == ActivityType.SHOPPING) {
                    activity
                } else {
                    ActivityType.OTHER
                }
                val transitionP = transitionProbs[transitionActivity]!!

                //val carP = carProbs[transitionActivity]!!
                val p = previousProbs.diagonal().dot(transitionP) * chainP
                expectedCountPerAgent.plusAssign(p)

                if (activity == ActivityType.OTHER) {
                    expectedOther.plusAssign(p)
                } else if (activity == ActivityType.HOME) {
                    expectedHome.plusAssign(p)
                } else if (activity == ActivityType.WORK) {
                    expectedWork.plusAssign(p)
                } else if (activity == ActivityType.SCHOOL) {
                    expectedSchool.plusAssign(p)
                } else if (activity == ActivityType.SHOPPING) {
                    expectedShopping.plusAssign(p)
                }

                if (startActivity == ActivityType.WORK) {
                    cumMatrix = cumMatrix.dot(transitionP)
                    workMatrix[nextActivity]!!.plusAssign(cumMatrix * chainP)
                } else {
                    if (startActivity == ActivityType.HOME) {
                        if (probPositionGivenLastFixed != null) {
                            //hFixedMatrix[nextActivity]!!.plusAssign(probPositionGivenLastFixed.dot(transitionP) * chainP)
                            hFixedMatrix[nextActivity]!!.plusAssign(pLastTest!!.dot(transitionP) * chainP)
                        }
                    }
                    fixedMatrix[nextActivity]!!.plusAssign(previousProbs.diagonal().dot(transitionP) * chainP)
                }

                testFixedMatrix.plusAssign(previousProbs.diagonal().dot(transitionP) * chainP)
                testMatrix.plusAssign(previousProbs * chainP)
                previousProbs = previousProbs.dot(transitionP)

                probPositionGivenLastFixed = probPositionGivenLastFixed?.dot(transitionP)
                pLastTest = pLastTest?.dot(transitionP)
            }


            // Last chain component. Fixed most of the time.
            if (
                (endActivity == ActivityType.HOME) ||
                ((endActivity == ActivityType.WORK) && (startActivity != ActivityType.SCHOOL)) ||
                ((endActivity == ActivityType.SCHOOL) && (startActivity != ActivityType.WORK))
            ) {
                // val carP = carProbs[endActivity]!!
                val p = probPositionGivenLastFixed!!.transpose()  * chainP
                expectedCountPerAgent.plusAssign(p)
                if (endActivity == ActivityType.HOME) {
                    expectedHome.plusAssign(p)
                }else if (endActivity == ActivityType.WORK) {
                    expectedWork.plusAssign(p)
                    if (startActivity == ActivityType.HOME) {
                        expectedHomeWork.plusAssign(p)
                    }
                }else if (endActivity == ActivityType.SCHOOL) {
                    expectedSchool.plusAssign(p)
                }
                testMatrix.plusAssign(previousProbs * chainP)
            } else {
                val transitionActivity = if (endActivity == ActivityType.SHOPPING) {
                    endActivity
                } else {
                    ActivityType.OTHER
                }
                val transitionP = transitionProbs[transitionActivity]!!

                //val carP = carProbs[transitionActivity]!!
                val p = previousProbs.diagonal().dot(transitionP) * chainP

                testMatrix.plusAssign(previousProbs * chainP)

                expectedCountPerAgent.plusAssign(p)

                if (endActivity == ActivityType.OTHER) {
                    expectedOther.plusAssign(p)
                }else if (endActivity == ActivityType.WORK) {
                    expectedWork.plusAssign(p)
                }else if (endActivity == ActivityType.SCHOOL) {
                    expectedSchool.plusAssign(p)
                }else if (endActivity == ActivityType.SHOPPING) {
                    expectedShopping.plusAssign(p)
                }

                /*if (startActivity == ActivityType.WORK) {
                    cumMatrix = cumMatrix.dot(transitionP)
                    workCumMatrixBack.plusAssign(cumMatrix * chainP)
                } else {
                    fixedMatrix.plusAssign(previousProbs.diagonal().dot(transitionP) * chainP)
                }*/
            }
        }

       val activityP = mutableMapOf<ActivityType, Double>()
       for ((chain, chainP) in uniqueSegments) {
           if (chain.size <= 1) {
               continue
           }
           for (activity in chain.drop(1)) {
               activityP[activity] = (activityP[activity] ?: 0.0) + chainP /// (chain.size - 1)
           }
       }
       val psum = activityP.values.sum()

       val result = mk.zeros<Double>(grid.size).expandDims(0).asDNArray().asD2Array()
       for (activity in ActivityType.entries) {
           val test1 = homeProbs.diagonal().dot(workTransitionP).dot(workMatrix[activity]!!)
           val fixPWork = fixedMatrix[activity]!!.plus(test1)
           val test2 = mk.ones<Double>(grid.size).expandDims(0).asDNArray().asD2Array().dot(fixPWork)
           result.plusAssign(test2)
       }

       // TODO Works
       val test1other = homeProbs.diagonal().dot(workTransitionP).dot(workMatrix[ActivityType.OTHER]!!)
       val fixPWorkOther = fixedMatrix[ActivityType.OTHER]!!.plus(test1other)
       val test2Other = mk.ones<Double>(grid.size).expandDims(0).asDNArray()
           .asD2Array().dot(fixPWorkOther)
           .diagonal().dot(transitionProbs[ActivityType.OTHER]!!)

       // TODO Works
       val test1shop= homeProbs.diagonal().dot(workTransitionP).dot(workMatrix[ActivityType.SHOPPING]!!)
       val fixPWorkshop = fixedMatrix[ActivityType.SHOPPING]!!.plus(test1shop)
       val test2shop = mk.ones<Double>(grid.size).expandDims(0).asDNArray()
           .asD2Array().dot(fixPWorkshop)
           .diagonal().dot(transitionProbs[ActivityType.SHOPPING]!!)

       // TODO Small error
       val test1home = homeProbs.diagonal().dot(workTransitionP).dot(workMatrix[ActivityType.HOME]!!)
       val fixPWorkHome = fixedMatrix[ActivityType.HOME]!!.plus(test1home)
       val test2Home = fixPWorkHome.transpose()

       // TODO How
       val test1work = homeProbs.diagonal().dot(workTransitionP).dot(workMatrix[ActivityType.WORK]!!)
       val fixPWorkWork = fixedMatrix[ActivityType.WORK]!!.plus(test1work)
       val test2Work =  mk.ones<Double>(grid.size).expandDims(0).asDNArray()
           .asD2Array().dot(fixPWorkWork.transpose())
           .diagonal().dot(transitionProbs[ActivityType.WORK]!!)

       // Test fix work
       //val testFixWork = hFixedMatrix[ActivityType.WORK]!!.transpose()
       val testFixWork = homeProbs.diagonal().dot(workTransitionP).dot(hFixedMatrix[ActivityType.WORK]!!).transpose()
           .plus(homeProbs.diagonal().dot(workTransitionP).dot(nthFixedMatrix[ActivityType.WORK]!!)) // TODO Works

       //val porigin = mk.ones<Double>(grid.size).expandDims(0).asDNArray()
       //    .asD2Array().dot(fixedMatrix[ActivityType.WORK]!!).diagonal()
       //val diag = mk.ones<Double>(grid.size, grid.size).dot(fixedMatrix[ActivityType.WORK]!!.transpose())
       //val testfixwork = porigin.dot(diag).dot(workTransitionP)

       // TODO How
       val test1school = homeProbs.diagonal().dot(workTransitionP).dot(workMatrix[ActivityType.SCHOOL]!!)
       val fixPWorkSchool = fixedMatrix[ActivityType.SCHOOL]!!.plus(test1school)
       val test2School =  mk.ones<Double>(grid.size).expandDims(0).asDNArray()
           .asD2Array().dot(fixPWorkSchool.transpose())
           .diagonal().dot(transitionProbs[ActivityType.SCHOOL]!!)

       val expecsum = test2Work.plus(test2Other).plus(test2School).plus(test2Home).plus(test2shop)

       val testExpected = mk.zeros<Double>(grid.size, grid.size)
       for (activity in ActivityType.entries) {
           val test1 = homeProbs.diagonal().dot(workTransitionP).dot(workMatrix[activity]!!)
           val fixPWork = fixedMatrix[activity]!!.plus(test1)
           val flatOnes = mk.ones<Double>(grid.size).expandDims(0).asDNArray().asD2Array()

           val setws = setOf(ActivityType.WORK, ActivityType.SCHOOL)
           val test3 = if (activity == ActivityType.HOME) {
               fixPWork.transpose()
           } else if (activity in setws) {
               flatOnes.dot(fixPWork.transpose()).diagonal().dot(transitionProbs[activity]!!)
           } else {
               flatOnes.dot(fixPWork).diagonal().dot(transitionProbs[activity]!!)
           }
           testExpected.plusAssign(test3)
       }

        // Gurobi test
        val oi = OptimizationInput(homeProbs, workMatrix, fixedMatrix, transitionProbs, carProbs)

        val demand = mutableMapOf<Pair<Int, Int>, Double>()
        val w = mutableMapOf<Pair<Int, Int>, Double>()

        for ((o, origin) in grid.withIndex()) {
            for ((d, destination) in grid.withIndex()) {
                val od = Pair(o, d)
                w[od] = workTransitionP[o, d]
                demand[od] = 0.0
            }
        }

        for (activity in listOf(ActivityType.OTHER)) {
            val fixed = oi.fixedMatrix[activity]!!
            val wtm = oi.workMatrix[activity]!!
            val pHome = oi.homeProbs.flatten()
            val trans = oi.transitionMatrix[activity]!!

            // Case OTHER
            for ((o, origin) in grid.withIndex()) {
                var s = 0.0

                for (j in grid.indices) {
                    s += fixed[j, o]
                    for (i in grid.indices) {
                        s += pHome[i] * wtm[j, o] * w[Pair(i, j)]!!
                    }
                }

                for ((d, destination) in grid.withIndex()) {
                    val t = trans[o, d]
                    val od = Pair(o, d)

                    demand[od] = demand[od]!! + t * s
                }
            }
        }

        for (d in 0 until 10) {
            println("(0, $d): Matrixresult: ${test2Other[0, d]} | GurobiTest: ${demand[Pair(0, d)]}")
        }

        // Gurobi test end

       // Format output
       //val out = mutableMapOf<Pair<Cell, Cell>, Double>()
       //for ((o, origin) in grid.withIndex()) {
       //    for ((d, destination) in grid.withIndex()) {
       //        out[Pair(origin, destination)] = expectedCountPerAgent[o][d]
       //    }
       //}

       val testFixed = mk.ones<Double>(grid.size).expandDims(0).asDNArray().asD2Array().dot(testFixedMatrix)

       val transitionTest = mk.zeros<Double>(grid.size, grid.size)
       println("Do it")
       /*for ((activity, p) in activityP) {
           if (activity == ActivityType.HOME) {
               transitionTest.plusAssign(fixPWork.transpose() * p / psum)
           } else {
               transitionTest.plusAssign(test2.diagonal().dot(transitionProbs[activity]!!) * p /psum )
           }
       }*/

       // TEST OTHER
        val staticCount = sensors.associateWith { 0.0 }.toMutableMap()
        val unexplained = sensors.associateWith { 0.0 }.toMutableMap()
        val othercar = test2Other.times(carProbs[ActivityType.OTHER]!!)
        val shopcar  = test2shop.times(carProbs[ActivityType.SHOPPING]!!)
        val homecar  = test2Home.times(carProbs[ActivityType.HOME]!!)
        val hwcar = testFixWork.times(carProbs[ActivityType.WORK]!!)

        val expeccarsum = mk.zeros<Double>(grid.size, grid.size).plus(
            expectedOther.times(carProbs[ActivityType.OTHER]!!)
        ).plus(
            expectedHome.times(carProbs[ActivityType.HOME]!!)
        ).plus(
            expectedWork.times(carProbs[ActivityType.WORK]!!)
        ).plus(
            expectedSchool.times(carProbs[ActivityType.SCHOOL]!!)
        ).plus(
            expectedShopping.times(carProbs[ActivityType.SHOPPING]!!)
        )
        for ((o, origin) in grid.withIndex()) {
            for ((d, destination) in grid.withIndex()) {
                val odPair = Pair(origin, destination)
                if (odPair in affectedLinks) {
                    val affected = affectedLinks[odPair]!!
                    for (sensor in affected) {
                        val expecexplained = othercar[o,d] * totalPop + shopcar[o,d] * totalPop + homecar[o,d] * totalPop + hwcar[o,d] * totalPop
                        staticCount[sensor] = staticCount[sensor]!! + expecexplained
                        unexplained[sensor] = unexplained[sensor]!! + (expeccarsum[o,d] * totalPop - expecexplained)
                    }
                }
            }
        }
        var mse = 0.0
        val allFlows = mutableMapOf<TrafficSensor, Double>()
        for (sensor in sensors) {
            val simFlow = staticCount[sensor]!!
            mse += (sensor.measuredFlow - simFlow).pow(2)

            allFlows[sensor] = simFlow
        }
        mse /= sensors.size

        println("MSE: $mse")
        println("------------------------------")
        println("Sensor | \t Flow OMOD | \t Flow Measured | \t Flow Unxeplained")
        var tstmse = 0.0
        var n = 0.0
        for ((i, flow) in staticCount.values.withIndex()) {
            println("${sensors[i].name} | \t $flow | \t ${sensors[i].measuredFlow}  | \t ${unexplained[sensors[i]]}")
            tstmse += (sensors[i].measuredFlow - flow).pow(2)
            n += 1
        }
        tstmse /= n
        println("Testmse: $tstmse")

        // END TEST OTHER

       optimize(grid, totalPop, affectedLinks, sensors, oi)

       // Format output
       val out = mutableMapOf<Pair<Cell, Cell>, Double>()
       for ((o, origin) in grid.withIndex()) {
           for ((d, destination) in grid.withIndex()) {
               out[Pair(origin, destination)] = expectedCountPerAgent[o][d]
           }
       }

       println(transitionTest.toList().take(10))
       println(expectedCountPerAgent.toList().take(10))


       return out
    }


    private inline fun <reified T : Any> D2Array<T>.diagonal() : D2Array<T> {
        require(this.shape[0] == 1)
        val diagonal  = mk.zeros<T>(this.size, this.size)
        for ( i in 0 until this.size) {
            diagonal[i, i] = this[0,i]
        }
        return  diagonal
    }

    private data class FixedSegment(
        val chain: List<ActivityType>,
        val probability: Double
    )

    data class OptimizationInput (
        val homeProbs: D2Array<Double>,
        val workMatrix: Map<ActivityType,  D2Array<Double>>,
        val fixedMatrix: Map<ActivityType,  D2Array<Double>>,
        val transitionMatrix: Map<ActivityType,  D2Array<Double>>,
        val carProbs: Map<ActivityType,  D2Array<Double>>,
    )


    fun optimize(
        grid: List<Cell>,
        totalPop: Double,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        sensors: List<TrafficSensor>,
        oi: OptimizationInput
    ) :  Map<Pair<RealLocation, RealLocation>, List<Double>> {
        // TODO Add H-W Tours
        // TODO Try dimensionality reduction with PCA
        // TODO Evaluate result and try to find parameters that recreate the optimal matrix
        // TODO Test parameters in real simulation

        println(grid.size)

        try {
            val env = GRBEnv()
            val model = GRBModel(env)

            val sensorCountExpr = mutableMapOf<TrafficSensor, GRBLinExpr>()
            for (sensor in sensors) {
                sensorCountExpr[sensor] = GRBLinExpr()
            }

            // Demand
            val demand = mutableMapOf<Pair<Int, Int>, GRBLinExpr>()
            val w = mutableMapOf<Pair<Int, Int>, GRBVar>()

            for ((o, origin) in grid.withIndex()) {
                val rowsum = GRBLinExpr()
                for ((d, destination) in grid.withIndex()) {
                    val od = Pair(o, d)
                    w[od] = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, "W")
                    demand[od] = GRBLinExpr()
                    rowsum.addTerm(1.0, w[od]!!)
                    //model.addConstr(w[od], GRB.EQUAL, oi.transitionMatrix[ActivityType.WORK]!![o,d], "Test")
                }
                model.addConstr(rowsum, GRB.EQUAL, 1.0, "ProbCondition")
            }

            for (activity in listOf(ActivityType.OTHER, ActivityType.SHOPPING)) { // TODO all
                val fixed = oi.fixedMatrix[activity]!!
                val wtm = oi.workMatrix[activity]!!
                val pHome = oi.homeProbs.flatten()
                val trans = oi.transitionMatrix[activity]!!
                val carP = oi.carProbs[activity]!!

                // Case OTHER
                for ((o, origin) in grid.withIndex()) {
                    val s = GRBLinExpr()

                    for (j in grid.indices) {
                        s.addConstant(fixed[j, o])
                        for (i in grid.indices) {
                            s.addTerm(pHome[i] * wtm[j, o], w[ Pair(i, j) ] )
                        }
                    }

                    for ((d, destination) in grid.withIndex()) {
                        val t = trans[o, d] * carP[o,d]
                        val od = Pair(o, d)

                        demand[od]!!.multAdd(t, s)
                    }
                }
            }

            for (activity in listOf(ActivityType.HOME)) { // TODO all
                val fixed = oi.fixedMatrix[activity]!!
                val wtm = oi.workMatrix[activity]!!
                val pHome = oi.homeProbs.flatten()
                val carP = oi.carProbs[activity]!!

                // Case OTHER
                for ((o, origin) in grid.withIndex()) {
                    for ((d, destination) in grid.withIndex()) {
                        val s = GRBLinExpr()
                        s.addConstant(fixed[o, d])

                        for (j in grid.indices) {
                            s.addTerm(pHome[o] * wtm[j, d], w[ Pair(o, j) ] )
                        }

                        val od = Pair(d, o) // Transpose

                        demand[od]!!.multAdd(carP[o,d], s)
                    }
                }
            }

            for ((o, origin) in grid.withIndex()) {
                for ((d, destination) in grid.withIndex()) {
                    val od = Pair(origin, destination)
                    if (od in affectedSensors) {
                        val affected = affectedSensors[od]!!

                        for (sensor in affected) {
                            val sensorSum = sensorCountExpr[sensor]!!
                            val dem = demand[Pair(o,d)]!!
                            sensorSum.multAdd(totalPop, dem)
                        }
                    }
                }
            }

            val sensorSimCount = model.addVars(
                DoubleArray(sensors.size) {0.0},
                null,
                DoubleArray(sensors.size) {0.0},
                CharArray(sensors.size) {GRB.CONTINUOUS},
                Array(sensors.size) {""}
            )
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
                println("MY MSE: ${myobjval /sensors.size}")

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

            println("Opt Matrix:")
            for (i in 0 until 5) {
                println(w[Pair(0, i)]!!.get(GRB.DoubleAttr.X))
            }
            println("Old Matrix:")
            for (i in 0 until 5) {
                println(oi.transitionMatrix[ActivityType.WORK]!![0,i])
            }


            // Dispose of model and environment
            model.dispose()
            env.dispose()


            return mapOf()
        } catch (e: GRBException) {
            println(
                ("Error code: " + e.errorCode + ". " +
                        e.message)
            )
        }
        return mapOf()
    }

}