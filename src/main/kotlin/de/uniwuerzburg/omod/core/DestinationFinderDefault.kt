package de.uniwuerzburg.omod.core

import de.uniwuerzburg.omod.core.models.*
import de.uniwuerzburg.omod.routing.RoutingCache
import de.uniwuerzburg.omod.utils.createCumDist
import de.uniwuerzburg.omod.utils.sampleCumDist
import java.util.Random

/**
 * Gravity model based destination finder.
 */
class DestinationFinderDefault(
    private val routingCache: RoutingCache,
    private var locChoiceWeightFuns: MutableMap<ActivityType, LocationChoiceDCWeightFun>,
) : DestinationFinder {
    private var calibrated = false
    private var firstOrderCFactors: Map<ActivityType, Map<ODZone, Double>> = mapOf()
    private var secondOrderCFactors: Map<Pair<ActivityType, ActivityType>, Map<Pair<ODZone, ODZone>, Double>> = mapOf()

    private val cellCFactors = mutableMapOf<Cell, Double>()

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
        val thisCellFactors = customCellFactors ?: cellCFactors
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
        // Get agg zone (might be cell or dummy is node)
        val aggCumDist = getDistr(origin, destinations, activityType)
        val aggZone = destinations[sampleCumDist(aggCumDist, rng)]

        // Get fine-grained location
        val destination = if (aggZone is Cell) {
            val workBuildingsCumDist = getDistrNoOrigin(aggZone.buildings, activityType) // No origin for speed up
            aggZone.buildings[sampleCumDist(workBuildingsCumDist, rng)]
        } else {
            aggZone
        }
        return destination
    }

    private fun updateExpectedCount(
        priorExpectedCount: Array<Array<Double>>, transitionP: Array<Array<Double>>, chainP: Double, nDim: Int
    ) {
        for(o in 0 until nDim) {
            for(d in 0 until nDim) {
                priorExpectedCount[o][d] += transitionP[o][d] * chainP
            }
        }
    }

    private fun updateExpectedCount(
        priorExpectedCount: Array<Array<Double>>, transitionP: Array<Array<Double>>,
        originP: Array<Double>,chainP: Double, nDim: Int
    ) {
        for(o in 0 until nDim) {
            for(d in 0 until nDim) {
                val pairP = originP[o] * transitionP[o][d]
                priorExpectedCount[o][d] += pairP * chainP
            }
        }
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
        grid: List<Cell>, activityGenerator: ActivityGeneratorDefault, customCellFactors: Map<Cell, Double>
    ) : Map<Pair<Cell, Cell>, Double> {
        // TODO Test for least divergence:
        // Possible reasons
        //      - Last trip Home, work, school ist approximated with OTHER

        // Result: Expected trip count on each od-paid of one agent on one day
        val expectedCountPerAgent = Array(grid.size) { Array(grid.size) { 0.0 } }

        // Precompute important probabilities
        // HOME
        val homeWeights = getWeightsNoOrigin(grid, activityType = ActivityType.HOME) // TODO what about buffer area
        val homeProbs   = homeWeights.map { it / homeWeights.sum() }.toTypedArray()

        // Transitions
        val transitionProbs =  mutableMapOf<ActivityType, Array<Array<Double>>>()
        for (activityType in ActivityType.entries) {
            if (activityType == ActivityType.HOME) { continue }

            val activityProbs = Array(grid.size) { Array(grid.size) { 0.0 } }
            for (o in grid.indices) {
                val activityWeights = getWeights(grid[o], grid, activityType = activityType, customCellFactors)
                val activityProb    = activityWeights.map { it / activityWeights.sum() }.toTypedArray()

                for (d in grid.indices) {
                    activityProbs[o][d] = activityProb[d]
                }
            }
            transitionProbs[activityType] = activityProbs
        }

        // Home - Work, Work
        val homeWorkProbs = Array(grid.size) { Array(grid.size) { 0.0 } }
        val workProbs = Array(grid.size) { 0.0 }
        for (o in grid.indices) {
            for (d in grid.indices) {
                val transitionP = transitionProbs[ActivityType.WORK]!![o][d]
                homeWorkProbs[o][d] = homeProbs[o] * transitionP
                workProbs[d] += homeProbs[o] * transitionP
            }
        }

        // Home - School, School
        val homeSchoolProbs = Array(grid.size) { Array(grid.size) { 0.0 } }
        val schoolProbs = Array(grid.size) { 0.0 }
        for (o in grid.indices) {
            for (d in grid.indices) {
                val transitionP = transitionProbs[ActivityType.SCHOOL]!![o][d]
                homeSchoolProbs[o][d] = homeProbs[o] * transitionP
                schoolProbs[d] += homeProbs[o] * transitionP
            }
        }

        // Get activity chains
        // TODO account for socio-demographic features
        // TODO test kotlin matrix lib for speed up
        // TODO cache chain pieces for speed up Fixed -> Fixed tours
        // 1. Find all unique fixed fixed tours
        // 2. Determine their probability
        // 3. Run normal code
        // Later when time is important remember time windows on tour generation
        val chains = activityGenerator.getChain(
            Weekday.UNDEFINED, HomogeneousGrp.UNDEFINED, MobilityGrp.UNDEFINED, AgeGrp.UNDEFINED, ActivityType.HOME
        )
        val chainProbs = chains.weights.map { it / chains.weights.sum() }.toTypedArray()

        // Determine od pair probabilities
        // TODO mode choice and time
        for ((chain, chainP) in chains.chains.zip(chainProbs)) {
            if (chain.size <= 1) { continue }
            var lastActivityFixed = chain.first()
            var lastActivity = chain.first()
            var previousProbs = homeProbs
            var probPositionGivenLastFixed = Array(grid.size) { Array(grid.size) { 0.0 } }
            for (start in grid.indices) {
                probPositionGivenLastFixed[start][start] = 1.0
            }
            for (activity in chain.drop(1)) {
                when(activity){
                    ActivityType.HOME -> {
                        if (lastActivity == ActivityType.WORK){
                            updateExpectedCount(expectedCountPerAgent, homeWorkProbs, chainP, grid.size)
                        } else if (lastActivity == ActivityType.SCHOOL){
                            updateExpectedCount(expectedCountPerAgent, homeSchoolProbs, chainP, grid.size)
                        } else {
                            when (lastActivityFixed) {
                                ActivityType.HOME -> {
                                    for (h in grid.indices) {
                                        for (flex in grid.indices) {
                                            expectedCountPerAgent[flex][h] += homeProbs[h] * probPositionGivenLastFixed[h][flex] * chainP
                                        }
                                    }
                                }
                                ActivityType.WORK -> {
                                    for (h in grid.indices) {
                                        for (flex in grid.indices) {
                                            for (w in grid.indices) {
                                                expectedCountPerAgent[flex][h] += homeWorkProbs[h][w] * probPositionGivenLastFixed[w][flex] * chainP
                                            }
                                        }
                                    }
                                }
                                ActivityType.SCHOOL -> {
                                    for (h in grid.indices) {
                                        for (flex in grid.indices) {
                                            for (s in grid.indices) {
                                                expectedCountPerAgent[flex][h] += homeSchoolProbs[h][s] * probPositionGivenLastFixed[s][flex] * chainP
                                            }
                                        }
                                    }
                                }
                                else -> {
                                    require(false)
                                }
                            }
                        }
                        lastActivityFixed = ActivityType.HOME
                        previousProbs = homeProbs
                        probPositionGivenLastFixed = Array(grid.size) { Array(grid.size) { 0.0 } }
                        for (start in grid.indices) {
                            probPositionGivenLastFixed[start][start] = 1.0
                        }
                    }
                    ActivityType.WORK -> {
                        if ((lastActivity == ActivityType.HOME)){
                            updateExpectedCount(expectedCountPerAgent, homeWorkProbs, chainP, grid.size)
                        } else {
                            when (lastActivityFixed) {
                                ActivityType.HOME -> {
                                    for (h in grid.indices) {
                                        for (flex in grid.indices) {
                                            for (w in grid.indices) {
                                                expectedCountPerAgent[flex][w] += homeWorkProbs[h][w] * probPositionGivenLastFixed[h][flex] * chainP
                                            }
                                        }
                                    }
                                }
                                ActivityType.WORK -> {
                                    for (w in grid.indices) {
                                        for (flex in grid.indices) {
                                            expectedCountPerAgent[flex][w] += workProbs[w] * probPositionGivenLastFixed[w][flex] * chainP
                                        }
                                    }
                                }
                                else -> {
                                    val transitionP = transitionProbs[ActivityType.OTHER]!!
                                    updateExpectedCount(expectedCountPerAgent, transitionP, previousProbs, chainP, grid.size)
                                }
                            }
                        }
                        lastActivityFixed = ActivityType.WORK
                        previousProbs = workProbs
                        probPositionGivenLastFixed = Array(grid.size) { Array(grid.size) { 0.0 } }
                        for (start in grid.indices) {
                            probPositionGivenLastFixed[start][start] = 1.0
                        }
                    }
                    ActivityType.SCHOOL -> {
                        if ((lastActivity == ActivityType.HOME)){
                            updateExpectedCount(expectedCountPerAgent, homeSchoolProbs, chainP, grid.size)
                        } else {
                            when (lastActivityFixed) {
                                ActivityType.HOME -> {
                                    for (h in grid.indices) {
                                        for (flex in grid.indices) {
                                            for (s in grid.indices) {
                                                expectedCountPerAgent[flex][s] += homeSchoolProbs[h][s] * probPositionGivenLastFixed[h][flex] * chainP
                                            }
                                        }
                                    }
                                }
                                ActivityType.SCHOOL -> {
                                    for (s in grid.indices) {
                                        for (flex in grid.indices) {
                                            expectedCountPerAgent[flex][s] += schoolProbs[s] * probPositionGivenLastFixed[s][flex] * chainP
                                        }
                                    }
                                }
                                else -> {
                                    val transitionP = transitionProbs[ActivityType.OTHER]!!
                                    updateExpectedCount(expectedCountPerAgent, transitionP, previousProbs, chainP, grid.size)
                                }
                            }
                        }
                        lastActivityFixed = ActivityType.SCHOOL
                        previousProbs = schoolProbs
                        probPositionGivenLastFixed = Array(grid.size) { Array(grid.size) { 0.0 } }
                        for (start in grid.indices) {
                            probPositionGivenLastFixed[start][start] = 1.0
                        }
                    }
                    else -> {
                        val transitionActivity = if (activity == ActivityType.SHOPPING) {
                            activity
                        } else {
                            ActivityType.OTHER
                        }
                        val transitionP = transitionProbs[transitionActivity]!!
                        val newProbs = Array(grid.size) { 0.0 }
                        for (o in grid.indices) {
                            for (d in grid.indices) {
                                val pairP = previousProbs[o] * transitionP[o][d]
                                expectedCountPerAgent[o][d] += pairP * chainP
                                newProbs[d] += pairP
                            }
                        }
                        for (start in grid.indices) {
                            val newProbsGiven = Array(grid.size) { 0.0 }
                            for (o in grid.indices) {
                                for (d in grid.indices) {
                                    newProbsGiven[d] += probPositionGivenLastFixed[start][o] * transitionP[o][d]
                                }
                            }
                            probPositionGivenLastFixed[start] = newProbsGiven
                        }
                        previousProbs = newProbs
                    }
                }
                lastActivity = activity
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
}