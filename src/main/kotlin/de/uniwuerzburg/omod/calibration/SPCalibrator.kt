package de.uniwuerzburg.omod.calibration

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
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.plusAssign
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.locationtech.jts.geom.Coordinate

object SPCalibrator {
    fun determinePairProbabilities(
        grid: List<Cell>, activityGenerator: ActivityGeneratorDefault,
        modeChoiceCalibration: ModeChoiceCalibration,
        customCellFactors: Map<ActivityType, Map<Cell, Double>>,
        popStrata: List<PopStratum>,
        carOwnership: CarOwnership,
        destinationFinder: DestinationFinderDefault,
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
        val transitionWeights = mutableMapOf<ActivityType,  D2Array<Double>>()
        for (activityType in ActivityType.entries) {
            if (activityType == ActivityType.HOME) { continue }

            val activityProbs = mk.zeros<Double>(grid.size, grid.size)
            val activityWs = mk.zeros<Double>(grid.size, grid.size)
            for (o in grid.indices) {
                val activityWeights = destinationFinder.getWeights(grid[o], grid, activityType = activityType, customCellFactors)
                activityProbs[o] = mk.ndarray( activityWeights.map { it / activityWeights.sum() } )
                activityWs[o] = mk.ndarray( activityWeights)
            }

            transitionProbs[activityType] = activityProbs
            transitionWeights[activityType] = activityWs
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

        val wcnst  = mk.zeros<Double>(grid.size, grid.size)
        val wcoeff = mk.zeros<Double>(grid.size, grid.size)
        val wvalue = mk.zeros<Double>(grid.size, grid.size)

        for ( (oi, o) in grid.withIndex()) {
            val distances = destinationFinder.routingCache.getDistances(o, grid)
            for ((di, d) in grid.withIndex()) {
                val weightFunction = destinationFinder.locChoiceWeightFuns[ActivityType.WORK]!!

                val (cnst, coeff, value) = weightFunction.testFixedAndCoefficient(d, distances[di].toDouble())
                wcnst[oi, di] = cnst
                wcoeff[oi, di] = coeff
                wvalue[oi, di] = value
            }
        }

        // Determine od pair probabilities
        // TODO time
        val fixedMatrix = mk.zeros<Double>(grid.size, grid.size)
        val cnstMatrix = mk.zeros<Double>(grid.size, grid.size)
        val coeffsMatrix = mk.zeros<Double>(grid.size, grid.size)

        for ((chain, chainP) in uniqueSegments) {
            if (chain.size <= 1) { continue }

            val startActivity = chain.first()
            val endActivity = chain.last()

            // Short chains
            if (chain.size == 2) {
                when(Pair(startActivity, endActivity)) {
                    Pair(ActivityType.HOME, ActivityType.WORK) -> {
                        val carP = carProbs[endActivity]!!
                        val p = homeWorkProbs.times(carP) * chainP
                        expectedCountPerAgent.plusAssign(p)

                        cnstMatrix.plusAssign(homeProbs.diagonal().dot(wcnst).times(carP) * chainP)
                        coeffsMatrix.plusAssign(homeProbs.diagonal().dot(wcoeff).times(carP) * chainP)

                        continue
                    }
                    Pair(ActivityType.HOME, ActivityType.SCHOOL) -> {
                        val carP = carProbs[endActivity]!!
                        val p = homeSchoolProbs.times(carP) * chainP
                        expectedCountPerAgent.plusAssign(p)
                        fixedMatrix.plusAssign(p)
                        continue
                    }
                    Pair(ActivityType.WORK, ActivityType.HOME) -> {
                        val carP = carProbs[endActivity]!!
                        val p = homeWorkProbs.times(carP) * chainP // TODO Transpose? I think so
                        expectedCountPerAgent.plusAssign(p)

                        cnstMatrix.plusAssign(homeProbs.diagonal().dot(wcnst).times(carP) * chainP)
                        coeffsMatrix.plusAssign(homeProbs.diagonal().dot(wcoeff).times(carP) * chainP)

                        continue
                    }
                    Pair(ActivityType.SCHOOL, ActivityType.HOME) -> {
                        val carP = carProbs[endActivity]!!
                        val p = homeSchoolProbs.times(carP) * chainP
                        expectedCountPerAgent.plusAssign(p)
                        fixedMatrix.plusAssign(p)
                        continue
                    }
                    else -> {}
                }
            }

            // Setup
            var probPositionGivenLastFixed: D2Array<Double>?
            var previousProbs: D2Array<Double>
            var previousCnst: D2Array<Double> = homeProbs
            var previousCoeff: D2Array<Double> = homeProbs
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
                    previousCnst = previousCnst.dot(wcnst)
                    previousCoeff = previousCoeff.dot(wcoeff)

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
                val transitionActivity = if (activity == ActivityType.SHOPPING) {
                    activity
                } else {
                    ActivityType.OTHER
                }
                val transitionP = transitionProbs[transitionActivity]!!

                val carP = carProbs[transitionActivity]!!
                val p = previousProbs.diagonal().dot(transitionP).times(carP) * chainP
                expectedCountPerAgent.plusAssign(p)

                if (startActivity == ActivityType.WORK) {
                    cnstMatrix.plusAssign(previousCnst.diagonal().dot(transitionP).times(carP) * chainP)
                    coeffsMatrix.plusAssign(previousCoeff.diagonal().dot(transitionP).times(carP) * chainP)

                    previousCnst = previousCnst.dot(transitionP)
                    previousCoeff = previousCoeff.dot(transitionP)

                } else {
                    fixedMatrix.plusAssign(p)
                }

                previousProbs = previousProbs.dot(transitionP)
                probPositionGivenLastFixed = probPositionGivenLastFixed?.dot(transitionP)
            }

            // Last chain component. Fixed most of the time.
            if (
                (endActivity == ActivityType.HOME) ||
                ((endActivity == ActivityType.WORK) && (startActivity != ActivityType.SCHOOL)) ||
                ((endActivity == ActivityType.SCHOOL) && (startActivity != ActivityType.WORK))
            ) {
                val carP = carProbs[endActivity]!!
                val p = probPositionGivenLastFixed!!.transpose().times(carP) * chainP
                expectedCountPerAgent.plusAssign(p)
                fixedMatrix.plusAssign(p)
            } else {
                val transitionActivity = if (endActivity == ActivityType.SHOPPING) {
                    endActivity
                } else {
                    ActivityType.OTHER
                }
                val transitionP = transitionProbs[transitionActivity]!!

                val carP = carProbs[transitionActivity]!!
                val p = previousProbs.diagonal().dot(transitionP).times(carP) * chainP
                expectedCountPerAgent.plusAssign(p)
                fixedMatrix.plusAssign(p)
            }
        }

        val workWeights = transitionWeights[ActivityType.WORK]

        val workTransitionTest = mk.zeros<Double>(grid.size, grid.size) // This is the same as workTransitionP
        val workWeightTest = mk.zeros<Double>(grid.size, grid.size)
        val workWeightTest2 = mk.zeros<Double>(grid.size, grid.size)
        val recreatedWMatrix = mk.zeros<Double>(grid.size, grid.size)

        for ( (oi, o) in grid.withIndex()) {
            var sum = 0.0
            for ((di, d) in grid.withIndex()) {
                sum += wcnst[oi, di] + wvalue[oi, di]
            }

            for ((di, d) in grid.withIndex()) {
                var featsum = 0.0
                for (building in d.buildings) {
                    featsum += building.osmProperties.number_shops
                }
                coeffsMatrix[oi, di] = coeffsMatrix[oi, di] * featsum / sum
                cnstMatrix[oi, di] = cnstMatrix[oi, di] / sum
                workTransitionTest[oi, di] = (wcnst[oi, di] + wvalue[oi,di]) / sum
                workWeightTest[oi, di] = (wcnst[oi, di] + wvalue[oi,di])
                workWeightTest2[oi, di] = (wcnst[oi, di] + wcoeff[oi,di] * featsum ) / sum
            }
        }

        val testMatrix = fixedMatrix.plus(coeffsMatrix).plus(cnstMatrix)

        // Format output
        val out = mutableMapOf<Pair<Cell, Cell>, Double>()
        for ((o, origin) in grid.withIndex()) {
            for ((d, destination) in grid.withIndex()) {
                out[Pair(origin, destination)] = expectedCountPerAgent[o][d]
            }
        }
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
}