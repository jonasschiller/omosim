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
    override fun getWeights(origin: LocationOption, destinations: List<LocationOption>, activityType: ActivityType
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
        return destinations.mapIndexed { i, destination ->
            cellCFactors.getOrDefault(destination, 1.0) * weights[i]
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
        grid: List<Cell>, activityGenerator: ActivityGeneratorDefault
    ) : Map<Pair<Cell, Cell>, Double> {
        // TODO Clean up, Speed up, Test for last divergence:
        // Possible reasons
        //      - Last trip Home, work, school ist approximated with OTHER
        //      - No difference between origin -> destination and the other way
        //      - Home probs because of Focus area

        val expectedCountPerAgent = mutableMapOf<Pair<Cell, Cell>, Double>()
        for (origin in grid) {
            for (destination in grid) {
                expectedCountPerAgent[Pair(origin, destination)] = 0.0
            }
        }

        val homeWeights = getWeightsNoOrigin(grid, activityType = ActivityType.HOME) // TODO what about buffer area
        val homeProbs   = homeWeights.map { it / homeWeights.sum() }

        val transitionProbs =  mutableMapOf<ActivityType, Map<Pair<Cell, Cell>, Double>>()
        for (activityType in ActivityType.entries) {
            if (activityType == ActivityType.HOME) { continue }

            val activityProbs = mutableMapOf<Pair<Cell, Cell>, Double>()
            for (origin in grid) {
                val activityWeights = getWeights(origin, grid, activityType = activityType)
                val activityProb    = activityWeights.map { it / activityWeights.sum() }

                for ((j, destination) in grid.withIndex()) {
                    activityProbs[Pair(origin, destination)] = activityProb[j]
                }
            }
            transitionProbs[activityType] = activityProbs
        }

        val homeWorkProbs = mutableMapOf<Pair<Cell, Cell>, Double>()
        val workProbs = DoubleArray(grid.size) { 0.0 }
        for ((i,origin) in grid.withIndex()) {
            for ((j, destination) in grid.withIndex()) {
                val transitionP = transitionProbs[ActivityType.WORK]!![Pair(origin, destination)]!!
                homeWorkProbs[Pair(origin, destination)] = homeProbs[i] * transitionP
                workProbs[j] += homeProbs[i] * transitionP
            }
        }

        val homeSchoolProbs = mutableMapOf<Pair<Cell, Cell>, Double>()
        val schoolProbs = DoubleArray(grid.size) { 0.0 }
        for ((i,origin) in grid.withIndex()) {
            for ((j, destination) in grid.withIndex()) {
                val transitionP = transitionProbs[ActivityType.SCHOOL]!![Pair(origin, destination)]!!
                homeSchoolProbs[Pair(origin, destination)] = homeProbs[i] * transitionP
                schoolProbs[j] += homeProbs[i] * transitionP
            }
        }


        val chains = activityGenerator.getChain(
            Weekday.UNDEFINED, HomogeneousGrp.UNDEFINED, MobilityGrp.UNDEFINED, AgeGrp.UNDEFINED, ActivityType.HOME
        )
        val chainProbs = chains.weights.map { it / chains.weights.sum() }
        var coveredActivities = 0.0
        for ((chain, chainP) in chains.chains.zip(chainProbs)) {
            if (chain.size <= 1) {
                coveredActivities +=  chain.size * chainP
                continue
            } else if (chain.all { (it != ActivityType.WORK ) and ( it != ActivityType.SCHOOL ) }) {
                coveredActivities += chain.size * chainP
                // No Work or School -> Home only important
                var lastActivity = chain.first()
                var previousProbs = homeProbs
                for ((n, activity) in chain.withIndex().drop(1)) {
                    var scaler = 1
                    if ((lastActivity == ActivityType.HOME) and (chain.drop(n).contains(ActivityType.HOME))) {
                        scaler = 2
                    }
                    if (activity == ActivityType.HOME) {
                        previousProbs = homeProbs
                    } else {
                        val transitionP = transitionProbs[activity]!!
                        val newProbs = DoubleArray(grid.size) { 0.0 }
                        for ((i, origin) in grid.withIndex()) {
                            for ((j, destination) in grid.withIndex()) {
                                val pairP = previousProbs[i] * transitionP[Pair(origin, destination)]!!
                                expectedCountPerAgent[Pair(origin, destination)] =
                                    expectedCountPerAgent[Pair(origin, destination)]!! + scaler * pairP * chainP
                                newProbs[j] += pairP
                            }
                        }
                        previousProbs = newProbs.toList()
                    }
                    lastActivity = activity
                }
            } else if (chain.all { (it == ActivityType.HOME ) or ( it == ActivityType.WORK ) }) {
                coveredActivities +=  chain.size * chainP
                // Only Home and Work -> Directly
                var lastActivity = chain.first()
                for ((n, activity) in chain.withIndex().drop(1)) {
                    if (lastActivity != activity) {
                        for ((i, origin) in grid.withIndex()) {
                            for ((j, destination) in grid.withIndex()) {
                                val pair = Pair(origin, destination)
                                expectedCountPerAgent[pair] =
                                    expectedCountPerAgent[pair]!! + chainP * homeWorkProbs[pair]!!
                            }
                        }
                    }
                    lastActivity = activity
                }
            } else if (chain.all { (it == ActivityType.HOME ) or ( it == ActivityType.SCHOOL ) }) {
                coveredActivities +=  chain.size * chainP
                // Only Home and School -> Directly
                var lastActivity = chain.first()
                for ((n, activity) in chain.withIndex().drop(1)) {
                    if (lastActivity != activity) {
                        for ((i, origin) in grid.withIndex()) {
                            for ((j, destination) in grid.withIndex()) {
                                val pair = Pair(origin, destination)
                                expectedCountPerAgent[pair] =
                                    expectedCountPerAgent[pair]!! + chainP * homeSchoolProbs[pair]!!
                            }
                        }
                    }
                    lastActivity = activity
                }
            } else {
                coveredActivities +=  chain.size * chainP
                // All -> Run through all
                var lastActivity = chain.first()
                var lastActivityFixed = chain.first()
                var previousProbs = homeProbs
                var lastHomeAwayP: DoubleArray? = null
                var lastWorkAwayP: DoubleArray? = null
                var lastSchoolAwayP: DoubleArray? = null
                for ((n, activity) in chain.withIndex().drop(1)) {
                    if (activity == ActivityType.HOME) {
                        if (lastActivityFixed == ActivityType.WORK) {
                            for ((i, origin) in grid.withIndex()) {
                                for ((j, destination) in grid.withIndex()) {
                                    val pair = Pair(origin, destination)
                                    expectedCountPerAgent[pair] =
                                        expectedCountPerAgent[pair]!! + chainP * homeWorkProbs[pair]!!
                                }
                            }
                        } else if (lastActivityFixed == ActivityType.SCHOOL){
                            for ((i, origin) in grid.withIndex()) {
                                for ((j, destination) in grid.withIndex()) {
                                    val pair = Pair(origin, destination)
                                    expectedCountPerAgent[pair] =
                                        expectedCountPerAgent[pair]!! + chainP * homeSchoolProbs[pair]!!
                                }
                            }
                        } else {
                            val transitionP = transitionProbs[ActivityType.OTHER]!!
                            for ((i, origin) in grid.withIndex()) {
                                for ((j, destination) in grid.withIndex()) {
                                    val pairP = previousProbs[i] * transitionP[Pair(origin, destination)]!!
                                    expectedCountPerAgent[Pair(origin, destination)] =
                                        expectedCountPerAgent[Pair(origin, destination)]!! + pairP * chainP
                                }
                            }
                        }
                        previousProbs = homeProbs
                        lastActivityFixed = ActivityType.HOME
                    } else if (activity == ActivityType.WORK) {
                        if (lastActivityFixed == ActivityType.HOME) {
                            for ((i, origin) in grid.withIndex()) {
                                for ((j, destination) in grid.withIndex()) {
                                    val pair = Pair(origin, destination)
                                    expectedCountPerAgent[pair] =
                                        expectedCountPerAgent[pair]!! + chainP * homeWorkProbs[pair]!!
                                }
                            }
                        } else {
                            val transitionP = transitionProbs[ActivityType.OTHER]!!
                            for ((i, origin) in grid.withIndex()) {
                                for ((j, destination) in grid.withIndex()) {
                                    val pairP = previousProbs[i] * transitionP[Pair(origin, destination)]!!
                                    expectedCountPerAgent[Pair(origin, destination)] =
                                        expectedCountPerAgent[Pair(origin, destination)]!! + pairP * chainP
                                }
                            }
                        }
                        previousProbs = workProbs.toList()
                        lastActivityFixed = ActivityType.WORK
                    } else if (activity == ActivityType.SCHOOL) {
                        if (lastActivityFixed == ActivityType.HOME) {
                            for ((i, origin) in grid.withIndex()) {
                                for ((j, destination) in grid.withIndex()) {
                                    val pair = Pair(origin, destination)
                                    expectedCountPerAgent[pair] =
                                        expectedCountPerAgent[pair]!! + chainP * homeSchoolProbs[pair]!!
                                }
                            }
                        } else {
                            val transitionP = transitionProbs[ActivityType.OTHER]!!
                            for ((i, origin) in grid.withIndex()) {
                                for ((j, destination) in grid.withIndex()) {
                                    val pairP = previousProbs[i] * transitionP[Pair(origin, destination)]!!
                                    expectedCountPerAgent[Pair(origin, destination)] =
                                        expectedCountPerAgent[Pair(origin, destination)]!! + pairP * chainP
                                }
                            }
                        }
                        previousProbs = schoolProbs.toList()
                        lastActivityFixed = ActivityType.SCHOOL
                    } else {
                        val transitionP = transitionProbs[activity]!!
                        val newProbs = DoubleArray(grid.size) { 0.0 }
                        for ((i, origin) in grid.withIndex()) {
                            for ((j, destination) in grid.withIndex()) {
                                val pairP = previousProbs[i] * transitionP[Pair(origin, destination)]!!
                                expectedCountPerAgent[Pair(origin, destination)] =
                                    expectedCountPerAgent[Pair(origin, destination)]!! + pairP * chainP
                                newProbs[j] += pairP
                            }
                        }
                        previousProbs = newProbs.toList()
                    }
                    lastActivity = activity
                }

            }
        }
        println("Covered Activities: $coveredActivities")
        val totalActivities = chainProbs.zip(chains.chains).sumOf{(p, c) -> p * c.size}
        println("Total Activities: $totalActivities")
        return expectedCountPerAgent
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
       for((i, cell) in grid.withIndex()) {
           cellCFactors[cell] = position[i]
       }
    }
}