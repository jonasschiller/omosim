package de.uniwuerzburg.omod.core

import com.github.ajalt.mordant.table.grid
import de.uniwuerzburg.omod.calibration.ModeChoiceCalibration
import de.uniwuerzburg.omod.core.models.*
import de.uniwuerzburg.omod.routing.RoutingCache
import de.uniwuerzburg.omod.utils.createCumDist
import de.uniwuerzburg.omod.utils.sampleCumDist
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
import org.locationtech.jts.geom.Coordinate
import java.util.*

/**
 * Gravity model based destination finder.
 */
class DestinationFinderDefault(
    val routingCache: RoutingCache, // TODO private?
    var locChoiceWeightFuns: MutableMap<ActivityType, LocationChoiceDCWeightFun>,
) : DestinationFinder {
    private var calibrated = false
    private var firstOrderCFactors: Map<ActivityType, Map<ODZone, Double>> = mapOf()
    private var secondOrderCFactors: Map<Pair<ActivityType, ActivityType>, Map<Pair<ODZone, ODZone>, Double>> = mapOf()

    private val cellCFactors = mutableMapOf<Cell, Double>()

    var forceWMatrix: Map<Cell, DoubleArray>? = null
    /**
     * Determine the probabilistic weight that a location is a destination given an origin and activity type
     * for all possible destinations.
     *
     * @param destinations Possible destinations
     * @param activityType Activity type conducted at the destination.
     * @return Probabilistic weights
     */
    override fun getWeights(
        origin: LocationOption, destinations: List<LocationOption>, activityType: ActivityType,
        customCellFactors: Map<Cell, Double>?
    ): List<Double> {
        require(activityType != ActivityType.HOME) { "For HOME activities call getWeightsNoOrigin()!" }

        // Don't leave dummy location except OD-Matrix defines it
        if (origin is DummyLocation) {
            if (activityType !in origin.transferActivities) {
                return destinations.map { if (origin == it) 1.0 else 0.0 }
            }
        }

        val distances = routingCache.getDistances(origin, destinations)
        val weightFunction = locChoiceWeightFuns[activityType]!!

        val weights = destinations.mapIndexed { i, destination ->
            when(destination) {
                is DummyLocation -> {
                    if (activityType !in destination.transferActivities) {
                        0.0
                    } else {
                        1.0
                    }
                }
                else -> {
                    destination as RealLocation
                    val distance = distances[i].toDouble()
                    weightFunction.calcFor(destination, distance)
                }
            }
        }

        // Test
        val thisCellFactors = if (activityType == ActivityType.WORK) { // TODO Temporary
           customCellFactors ?: cellCFactors
        } else {
            mapOf()
        }
        return destinations.mapIndexed { i, destination ->
            thisCellFactors.getOrDefault(destination, 1.0) * weights[i]
        }

        /*
        if (!calibrated) { return weights }

        return if (activityType == ActivityType.WORK) {
            destinations.mapIndexed { i, destination ->
                val foFactor = firstOrderCFactors[activityType]?.get(destination.odZone) ?: 1.0
                val soFactor = secondOrderCFactors[Pair(ActivityType.HOME, activityType)]
                    ?.get(Pair(origin.odZone, destination.odZone)) ?: 1.0
                foFactor * soFactor * weights[i]
            }
        } else {
            weights
        }*/
    }

    /**
     * Determine the probabilistic weight that a location is a destination given an activity type but no origin
     * for all possible destinations.
     *
     * @param destinations Possible destinations
     * @param activityType Activity type conducted at the destination.
     * @return Probabilistic weights
     */
    override fun getWeightsNoOrigin(destinations: List<LocationOption>, activityType: ActivityType) : List<Double> {
        val weightFunction = locChoiceWeightFuns[activityType]!!

        val weights = destinations.map { destination ->
            when (destination) {
                is DummyLocation -> {
                    if (activityType !in destination.transferActivities) {
                        0.0
                    } else {
                        1.0
                    }
                }
                else -> {
                    destination as RealLocation
                    weightFunction.calcForNoOrigin(destination)
                }
            }
        }

        if (!calibrated) { return weights }

        return when(activityType) {
            ActivityType.HOME -> {
                destinations.mapIndexed { i, destination ->
                    val foFactor = firstOrderCFactors[activityType]?.get(destination.odZone) ?: 1.0
                    foFactor * weights[i]
                }
            }
            ActivityType.WORK -> destinations.mapIndexed { i, destination ->
                val foFactor = firstOrderCFactors[activityType]?.get(destination.odZone) ?: 1.0
                foFactor * weights[i]
            }
            else -> weights
        }
    }

    /**
    * Determine the probability that a location is a destination given an origin and activity type
    * for all possible destinations.
    *
    * @param origin Origin of the trip
    * @param destinations Possible destinations
    * @param activityType Activity type conducted at the destination.
    * @return Cumulative distribution of the destination probabilities
    */
    private fun getDistr(origin: LocationOption, destinations: List<LocationOption>, activityType: ActivityType
    ) : DoubleArray {
        val weights = getWeights(origin, destinations, activityType)
        return createCumDist(weights.toDoubleArray())
    }

    /**
     * Determine the probability that a location is a destination given an activity type but no origin
     * for all possible destinations.
     *
     * @param destinations Possible destinations
     * @param activityType Activity type conducted at the destination.
     * @return Cumulative distribution of the destination probabilities
     */
    private fun getDistrNoOrigin(destinations: List<LocationOption>, activityType: ActivityType) : DoubleArray {
        val weights = getWeightsNoOrigin(destinations, activityType)
        return createCumDist(weights.toDoubleArray())
    }

    /**
     * Determine activity location.
     * To speed things up only the aggregated location is computed in with the distance factored in.
     * The building level location is determined based only on the attraction values of the buildings inside the
     * aggregated location.
     * @param origin Routing cell of the trip origin.
     * @param destinations Possible destinations
     * @param activityType Activity for which a location is searched
     * @param rng Random number generator
     * @return destination
     */
    override fun getLocation(
        origin: AggLocation, destinations: List<AggLocation>,
        activityType: ActivityType, rng: Random
    ) : LocationOption {
        // TODO Test
        val aggZone = if ((forceWMatrix != null) && (destinations.size == forceWMatrix?.size) && activityType == ActivityType.WORK) {
            val distr = createCumDist(forceWMatrix!![origin]!!)
            destinations[sampleCumDist(distr, rng)]
        } else {
            // Get agg zone (might be cell or dummy is node)
            val aggCumDist = getDistr(origin, destinations, activityType)
            destinations[sampleCumDist(aggCumDist, rng)]
        }

        // Get fine-grained location
        val destination = if (aggZone is Cell) {
            val workBuildingsCumDist = getDistrNoOrigin(aggZone.buildings, activityType) // No origin for speed up
            aggZone.buildings[sampleCumDist(workBuildingsCumDist, rng)]
        } else {
            aggZone
        }
        return destination
    }

    /**
     * Calibrate the destination finder with a OD-Matrix.
     * @param zones Possible destinations (Should be all that this destination finder applies to).
     * @param odZones OD-Matrix
     */
    override fun calibrate(zones: List<AggLocation>, odZones: List<ODZone>) {
        val (kfo, vfo) = calcFirstOrderScaling(zones, odZones)
        firstOrderCFactors = mapOf(kfo to vfo)
        val (kso, vso) =  calcSecondOrderScaling(zones, odZones)
        secondOrderCFactors = mapOf(kso to vso)
        calibrated = true
    }

    fun determinePairProbabilities(
        grid: List<Cell>, activityGenerator: ActivityGeneratorDefault,
        modeChoiceCalibration: ModeChoiceCalibration,
        customCellFactors: Map<Cell, Double>,
        popStrata: List<PopStratum>,
        carOwnership: CarOwnership
    ) : Map<Pair<Cell, Cell>, Double> {
        // Result: Expected trip count on each od-paid of one agent on one day
        val expectedCountPerAgent = mk.zeros<Double>(grid.size, grid.size)

        // Precompute important probabilities
        // HOME
        val homeWeights = getWeightsNoOrigin(grid, activityType = ActivityType.HOME) // TODO what about buffer area
        val homeProbs   = mk.ndarray( homeWeights.map { it / homeWeights.sum() } )
            .expandDims(0).asDNArray().asD2Array()

        // Transitions
        val transitionProbs = mutableMapOf<ActivityType,  D2Array<Double>>()
        for (activityType in ActivityType.entries) {
            if (activityType == ActivityType.HOME) { continue }

            val activityProbs = mk.zeros<Double>(grid.size, grid.size)
            for (o in grid.indices) {
                val activityWeights = getWeights(grid[o], grid, activityType = activityType, customCellFactors)
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
                        val distances = routingCache.getDistances(grid[o], grid)
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
                        val p = homeWorkProbs.times(carP) * chainP
                        expectedCountPerAgent.plusAssign(p)
                        continue
                    }
                    Pair(ActivityType.HOME, ActivityType.SCHOOL) -> {
                        val carP = carProbs[endActivity]!!
                        val p = homeSchoolProbs.times(carP) * chainP
                        expectedCountPerAgent.plusAssign(p)
                        continue
                    }
                    Pair(ActivityType.WORK, ActivityType.HOME) -> {
                        val carP = carProbs[endActivity]!!
                        val p = homeWorkProbs.times(carP) * chainP // TODO Transpose? I think so
                        expectedCountPerAgent.plusAssign(p)
                        continue
                    }
                    Pair(ActivityType.SCHOOL, ActivityType.HOME) -> {
                        val carP = carProbs[endActivity]!!
                        val p = homeSchoolProbs.times(carP) * chainP
                        expectedCountPerAgent.plusAssign(p)
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
                val transitionActivity = if (activity == ActivityType.SHOPPING) {
                    activity
                } else {
                    ActivityType.OTHER
                }
                val transitionP = transitionProbs[transitionActivity]!!

                val carP = carProbs[transitionActivity]!!
                val p = previousProbs.diagonal().dot(transitionP).times(carP) * chainP
                expectedCountPerAgent.plusAssign(p)
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
            }
        }

        // Format output
        val out = mutableMapOf<Pair<Cell, Cell>, Double>()
        for ((o, origin) in grid.withIndex()) {
            for ((d, destination) in grid.withIndex()) {
                out[Pair(origin, destination)] = expectedCountPerAgent[o][d]
            }
        }
        return out
    }

    /**
     * Calculate calibration factors from the od-matrix.
     * First-order factor changes the probability of a destination without taking the origin into account.
     *
     * P_calibrated(destination | activity) = P_base(destination | activity) * k_firstOrder(activity, destination)
     */
    private fun calcFirstOrderScaling(zones: List<AggLocation>, odZones: List<ODZone>) : Pair<ActivityType, Map<ODZone, Double>> {
        val activity = odZones.first().originActivity
        require(activity in listOf(ActivityType.HOME, ActivityType.WORK, ActivityType.SCHOOL))
        {"Scaling origins is only implemented for fixed locations"}

        val factors = mutableMapOf<ODZone, Double>()
        val omodProbs = calcOMODProbsAsMap(zones, activity)
        val omodWeights = mutableMapOf<ODZone, Double>()
        val odWeights = mutableMapOf<ODZone, Double>()

        for (odZone in odZones) {
            // Calculate omod origin probability. For speed only on zone level.
            omodWeights[odZone] = odZone.aggLocs.sumOf { omodProbs[it]!! }
            // Calculate OD-Matrix origin probability.
            odWeights[odZone] = if (odZone.inFocusArea) {
                odZone.destinations.sumOf { it.second }
            } else {
                odZone.destinations.filter { it.first.inFocusArea }.sumOf { it.second }
            }
        }

        val weightSumOMOD = omodWeights.values.sum()
        val weightSumOD = odWeights.values.sum()

        // Calibration failed!
        if ((weightSumOMOD <= 0) || (weightSumOD <= 0)){
            throw Exception("Calculation of first order calibration factors failed! " +
                    "Possible causes: OD-Matrix has negative values, " +
                    "OD-Matrix does not intersect focus area, ... \n" +
                    "If code was changed make sure that calcFirstOrderScaling is called after " +
                    "addODZoneInfo().")
        }

        for (odZone in odZones) {
            // Normalize
            val omodProb = omodWeights[odZone]!! / weightSumOMOD
            val odProb = odWeights[odZone]!! / weightSumOD
            factors[odZone] = if (omodProb <= 0) { // Can't calibrate with k-factor if OMOD prob is 0 %
                0.0
            } else {
                odProb / omodProb
            }
        }
        return Pair(activity, factors)
    }

    /**
     * Calculate calibration factors from the od-matrix.
     * Second-order factor changes the probability of a destination with taking the origin into account.
     *
     * P_calibrated(destination | activity, origin) =
     * P_base(destination | activity, origin) * k_firstOrder(activity, destination) * k_secondOrder(activity, destination, origin)
     */
    private fun calcSecondOrderScaling(zones: List<AggLocation>, odZones: List<ODZone>)
            : Pair<Pair<ActivityType, ActivityType> ,Map<Pair<ODZone, ODZone> , Double>> {
        val activities = Pair(odZones.first().originActivity, odZones.first().destinationActivity)

        // Check if OD has valid activities. Currently allowed: HOME->WORK
        require(activities == Pair(ActivityType.HOME, ActivityType.WORK)) {
            "Only OD-Matrices with Activities HOME->WORK are currently supported"
        }

        val factors = mutableMapOf<Pair<ODZone, ODZone>, Double>()
        val priorProbs = calcOMODProbsAsMap(zones, activities.first)

        for (originOdZone in odZones) {
            val omodWeights = odZones.associateWith { 0.0 } as MutableMap<ODZone, Double>
            val odWeights = odZones.associateWith { 0.0 } as MutableMap<ODZone, Double>

            // Calculate omod transition probability. For speed only on zone level.
            for (origin in originOdZone.aggLocs) {
                val pPriorLoc = priorProbs[origin]!!
                if (pPriorLoc == 0.0) { continue }
                val wDependentZone = getWeights(origin, zones, activities.second).sum()

                for (destOdZone in odZones) {
                    val wDependentLoc = getWeights(origin, destOdZone.aggLocs, activities.second).sum()
                    omodWeights[destOdZone] = omodWeights[destOdZone]!! + (pPriorLoc * wDependentLoc / wDependentZone)
                }
            }

            // Calculate OD-Matrix origin probability.
            for ((destOdZone, transitions) in originOdZone.destinations) {
                if (destOdZone.inFocusArea || originOdZone.inFocusArea) {
                    odWeights[destOdZone] = transitions
                }
            }

            val weightSumOMOD = omodWeights.values.sum()
            val weightSumOD = odWeights.values.sum()

            // Transitions from origin are impossible. Leave unadjusted. Factor should never be used.
            if ((weightSumOMOD <= 0) || (weightSumOD <= 0)){
                for (destOdZone in originOdZone.destinations.map { it.first }) {
                    factors[Pair(originOdZone, destOdZone)] = 1.0
                }
            } else {
                for (destOdZone in odZones) {
                    // Normalize
                    val omodProb = omodWeights[destOdZone]!! / weightSumOMOD
                    val odProb = odWeights[destOdZone]!! / weightSumOD

                    // Can't calibrate with k-factor if OMOD prob is 0 %
                    factors[Pair(originOdZone, destOdZone)] = if (omodProb <= 0) {
                        0.0
                    } else {
                        odProb / omodProb
                    }
                }
            }
        }
        return Pair(activities, factors)
    }

    /**
     * Calculate probability that an activity of type x happens at certain location for all locations.
     * Used to compare OMODs od probabilities with that of the od-file.
     * Possible activity types are: HOME and WORK
     *
     * P(Location | HOME) = Distribution used for Home location assignment
     * P(Location | WORK) = sum( P(Location | WORK, Origin=x) * P(x | HOME) ) over all locations x
     *
     * @param activityType The activity type x
     * @return Probability that an activity of type x happens at certain location for all locations
     */
    private fun calcOMODProbs(zones: List<AggLocation>, activityType: ActivityType) : DoubleArray {
        require(activityType in listOf(ActivityType.HOME, ActivityType.WORK))
        {"Flexible locations are not  yet supported for k-Factor calibration!"}
        // Home distribution
        val homeWeights = getWeightsNoOrigin(zones, ActivityType.HOME)
        val totalHomeWeight = homeWeights.sum()
        val homeProbs = homeWeights.map { it / totalHomeWeight }.toDoubleArray()
        if (activityType == ActivityType.HOME) {
            return homeProbs
        }

        // Work distribution
        val workProbs = DoubleArray(zones.size) { 0.0 }
        for ((i, zone) in zones.withIndex()) {
            val workWeights = getWeights(zone, zones, ActivityType.WORK)
            val totalWorkWeight = workWeights.sum()
            for (j in zones.indices) {
                workProbs[j] += homeProbs[i] * workWeights[j] / totalWorkWeight
            }
        }
        return workProbs
    }

    /**
     * Wrapper for calcOMODProbs that returns a map instead of an array.
     */
    private fun calcOMODProbsAsMap(zones: List<AggLocation>, activityType: ActivityType) : Map<LocationOption, Double> {
        val probs = calcOMODProbs(zones, activityType)
        val map = mutableMapOf<LocationOption, Double>()
        for (i in zones.indices) {
            map[zones[i]] = probs[i]
        }
        return map
    }

    fun getCalibrationPosition() : Array<Double> {
        val x = mutableListOf<Double>()
        for (activityType in ActivityType.entries) {
            if (activityType in locChoiceWeightFuns) {
                val lcwFun = locChoiceWeightFuns[activityType]!!
                if (lcwFun is ByPopulation) { continue }
                x.addAll(lcwFun.getCalibrationPosition())
            }
        }
        return x.toTypedArray()
    }

    fun dCoeffImpact(newval: Double) {
        val oldFun = locChoiceWeightFuns[ActivityType.WORK]!!
        (oldFun as CombinedDCUtil).changeCoeff0(newval)
    }

    fun nShopsImpact(newval: Double) {
        val oldFun = locChoiceWeightFuns[ActivityType.WORK]!!
        val x = oldFun.getCalibrationPosition().copyOf()
        x[9] = newval
        locChoiceWeightFuns[ActivityType.WORK] = oldFun.createCopyFromCalibrationPosition(x)
    }

    fun updateCalibrationPosition(position: Array<Double>, grid: List<Cell>) {
        var offset = 0
        for (activityType in ActivityType.entries) {
            if (activityType in locChoiceWeightFuns) {
                val oldFun = locChoiceWeightFuns[activityType]!!
                if (oldFun is ByPopulation) { continue }
                val nParams = oldFun.getCalibrationPosition().size

                val x = position.slice(offset until offset + nParams).toTypedArray()
                locChoiceWeightFuns[activityType] = oldFun.createCopyFromCalibrationPosition(x)
                offset += nParams
            }
        }
        grid.forEach{ it.recalculateAttractions(locChoiceWeightFuns) }
    }

    fun updateCellCValues(position: Array<Double>, grid: List<Cell>) {
       cellCFactors.clear()
       for((i, cell) in grid.withIndex()) {
           cellCFactors[cell] = position[i]
       }
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

