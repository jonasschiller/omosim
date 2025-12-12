package de.uniwuerzburg.omod.calibration.surrogate

import de.uniwuerzburg.omod.calibration.*
import de.uniwuerzburg.omod.calibration.CalibrationConstants.MC_SAMPLES
import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.differentiablemodel.*
import de.uniwuerzburg.omod.core.ActivityGeneratorDefault
import de.uniwuerzburg.omod.core.DestinationFinderDefault
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.*
import de.uniwuerzburg.omod.utils.diagonal
import de.uniwuerzburg.omod.utils.normalize
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.asDNArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.expandDims
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.plusAssign
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.locationtech.jts.geom.Coordinate
import kotlin.math.pow

/**
 * Surrogate model builder for the gravity model surrogate.
 *
 * @param omod The simulator for which to build a surrogate
 */
class SGGravity(
    val omod: Omod
) {
    private val modeChoiceCalibration = ModeChoiceCalibration() // TODO adapt
    private val fixActivitiesNotHome = setOf(ActivityType.WORK, ActivityType.SCHOOL)
    private val fixActivities = setOf(ActivityType.HOME) + fixActivitiesNotHome
    private val flexActivities = setOf(ActivityType.OTHER, ActivityType.SHOPPING, ActivityType.BUSINESS)

    init {
        if (omod.destinationFinder !is DestinationFinderDefault) {
            throw NotImplementedError(
                "Surrogate is not valid for the destination finder " +
                        omod.destinationFinder.javaClass.simpleName
            )
        }
        if (omod.activityGenerator !is ActivityGeneratorDefault) {
            throw NotImplementedError(
                "Surrogate is not valid for the activity generator " +
                        omod.activityGenerator.javaClass.simpleName
            )
        }
    }

    /**
     * Build surrogate for a gravity model with Sum-of-Squared-Errors objective.
     * Variables = Gravity Model attraction scalers for one activity type.
     *
     * @param activityType ActivityType for which the gravity model will be variable
     * @param sensors Sensors with measurements
     * @param affectedSensors Gives all sensors that are affected by a certain origin-destination pair
     * @return surrogate model
     */
    // TODO get population size as input
    fun buildModelSSE(
        activityType: ActivityType,
        sensors: List<TrafficSensor>,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    ): DifferentiableModel {
        val (model, tstSimCounts) = buildDiffModel(activityType, sensors, affectedSensors)

        // DEBUG CODE // TODO DELETE
        println(activityType)
        println("Evaluate Matrix Rep")
        debugEval(generateMarkovChainRep(activityType), sensors, affectedSensors)
        println("Evaluate Graph")
        for (sensor in sensors) {
            val sum = tstSimCounts[sensor]!!.map { it.evaluate(DoubleArray(omod.grid.size - 1) { 1.0 }) }.sum()
            val measured = sensor.measuredFlow.sum()
            println("${sensor.name} \t | $measured\t|$sum ")
        }
        // DEBUG ENDS
        return model
    }

    /**
     * Build surrogate for a gravity model. Returns the simulated traffic counts at each sensor.
     * Used in W-SPSA.
     * Variables = Gravity Model attraction scalers for one activity type.
     *
     * @param activityType ActivityType for which the gravity model will be variable
     * @param sensors Sensors with measurements
     * @param affectedSensors Gives all sensors that are affected by a certain origin-destination pair
     * @return surrogate model
     */
    fun buildModelSimCounts(
        activityType: ActivityType,
        sensors: List<TrafficSensor>,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    ): DifferentiableModelMultiOut {
        val (_, simCounts) = buildDiffModel(activityType, sensors, affectedSensors)

        // Create DifferentiableModelMultiOut from simulated counts
        val countsFlat = mutableListOf<Term>()
        for (sensor in sensors) {
            for (t in 0 until T) {
                countsFlat.add( simCounts[sensor]!![t] )
            }
        }
        val model = DifferentiableModelMultiOut(countsFlat.first().nVars)
        model.setRootTerms(countsFlat)
        return model
    }

    /**
     * Generate compact markov chain representation of original model.
     *
     * Compact representation explanation:
     * - The expected trip matrix can be computed with the probability matrix mPrior (K in the paper)
     * that gives the probability of an agent being in location col given his home location is row.
     * - K depends on the sequence of prior activities. For example: HSSSO
     * - All these sequences that have the same last activity can be grouped:
     *      - O' = HSSSO + HO + HWO + ... etc.
     * - The last activity O will not be included in the groups, since it is always the same and because it might
     * be dependent on the variables. Therefore: O' = HSSS + H + HW + ...
     * - In this grouping we make a distinction between the sequences that contain the variable activity vActivity
     * and those that do not:
     *      - mPriorCnst -> Does not contain vActivity
     *      - mPriorVar  -> Contains vActivity
     *
     * @param vActivity activity type for the variable gravity model
     * @return compact markov chain representation.
     */
    fun generateMarkovChainRep(vActivity: ActivityType) : SGCompactMatrixRep {
        if(vActivity == ActivityType.HOME) {
            throw NotImplementedError("Surrogate model dependent on home coefficients is not implemented!")
        }

        // Transition matrices
        val tMatrices = computeTransitionMatrices()
        val h = tMatrices[ActivityType.HOME]!!
        val mHW = h.diagonal().dot(tMatrices[ActivityType.WORK]!!)    // Work | Home
        val mHS = h.diagonal().dot(tMatrices[ActivityType.SCHOOL]!!)  // School | Home
        val hDiag = h.diagonal()

        // Compact representation:
        val mPriorVar  = ActivityType.entries.associateWith { mk.zeros<Double>(omod.grid.size, omod.grid.size) }
        val mPriorCnst = ActivityType.entries.associateWith { mk.zeros<Double>(omod.grid.size, omod.grid.size) }

        // Go through each unique fixed-fixed segment
        for ((chain, chainP) in getUniqueChainSegments()) {
            if (chain.size <= 1) { continue }

            val startActivity = chain.first()
            var nextActivity = chain[1]

            // Setup
            var mPostFixed = mk.identity<Double>(omod.grid.size) // Matrix product of matrices after vActivity that are not vActivity
            var mK: D2Array<Double>      // Location probability distributions given the home location
            var mKExVar: D2Array<Double> // The same but ignoring the var activity
            when(startActivity) {
                ActivityType.HOME -> {
                    mK = hDiag
                    mKExVar = hDiag
                    mPriorCnst[nextActivity]!!.plusAssign(hDiag * chainP)
                }
                ActivityType.WORK -> {
                    mK = mHW
                    mKExVar = mHW
                    if (vActivity == ActivityType.WORK) {
                        mPriorVar[nextActivity]!!.plusAssign(mPostFixed * chainP)
                    } else {
                        mPriorCnst[nextActivity]!!.plusAssign(mHW * chainP)
                    }
                }
                ActivityType.SCHOOL -> {
                    mK = mHS
                    mKExVar = mHS
                    if (vActivity == ActivityType.SCHOOL) {
                        mPriorVar[nextActivity]!!.plusAssign(mPostFixed * chainP)
                    } else {
                        mPriorCnst[nextActivity]!!.plusAssign(mHS * chainP)
                    }
                }
                else -> {
                    throw IllegalStateException("Last fixed activity can not be of type $startActivity !")
                }
            }

            // Run through chain cumulate mPriorVar, mPriorCnst, and mPostFixed
            var next = 2
            var occurred = (startActivity == vActivity) || (nextActivity == vActivity)
            for (activity in chain.drop(1).dropLast(1)) {
                nextActivity = chain[next]
                next += 1

                // Get transition matrix
                val mT = if (activity == ActivityType.BUSINESS) {
                    tMatrices[ActivityType.OTHER]!! // Edge case: Use other type transition matrix for business activity
                } else {
                    tMatrices[activity]!!
                }

                // Variable activity is in the past
                if ((vActivity in fixActivitiesNotHome) and (occurred)) {
                    // Cumulate all matrices after a fixed location activity has occurred
                    mPostFixed = mPostFixed.dot(mT)
                    mPriorVar[nextActivity]!!.plusAssign(mPostFixed * chainP)
                } else if ((vActivity !in fixActivitiesNotHome) and (occurred) and (nextActivity != vActivity)){
                    // Cumulate probability distributions after a flex location activity
                    // Ignores all subsequent vActivities
                    mPriorVar[nextActivity]!!.plusAssign(mKExVar * chainP)
                }
                // vActivity has not yet occured or is ignored
                else {
                    // Normal case cumulate matrices before next activity
                    mPriorCnst[nextActivity]!!.plusAssign(mK.dot(mT) * chainP)
                }

                // Update location probability distributions
                mK = mK.dot(mT)
                if ((vActivity !in fixActivitiesNotHome) and (activity != vActivity)) {
                    mKExVar = mKExVar.dot(mT)
                }

                // Check if vActivity occurrs
                if (nextActivity == vActivity) {
                    occurred = true
                }
            }
        }

        return SGCompactMatrixRep(h, mPriorVar, mPriorCnst, tMatrices, getPCar(), vActivity)
    }

    /**
     * Compact markov chain representation of original model.
     *
     * @param h home probability for each location
     * @param mPriorVar @see de.uniwuerzburg.omod.calibration.surrogate.SGGravity.generateMarkovChainRep
     * @param mPriorCnst @see de.uniwuerzburg.omod.calibration.surrogate.SGGravity.generateMarkovChainRep
     * @param tMatrices Current transition matrices for each activity
     * @param pCar mode share of car for each transition
     * @param vActivity activity type for the variable gravity model
     */
    data class SGCompactMatrixRep (
        val h: D2Array<Double>,
        val mPriorVar: Map<ActivityType, D2Array<Double>>,
        val mPriorCnst: Map<ActivityType,  D2Array<Double>>,
        val tMatrices: Map<ActivityType,  D2Array<Double>>,
        val pCar: Map<ActivityType,  D2Array<Double>>,
        val vActivity: ActivityType
    )

    /**
     * Compute the current transition matrices based on the gravity models.
     *
     * @return Transition matrices. HOME: 1xn vector, Rest: nxn matrix
     */
    private fun computeTransitionMatrices() : Map<ActivityType,  D2Array<Double>> {
        val tMatrices = mutableMapOf<ActivityType,  D2Array<Double>>()
        val finder = omod.destinationFinder as DestinationFinderDefault

        // HOME. A vector.
        val homeWeights = finder.getWeightsNoOrigin(omod.grid, activityType=ActivityType.HOME).toMutableList()
        if (!omod.populateBufferArea) { // Handle buffer area
            for ((i, cell) in omod.grid.withIndex()) {
                if (!cell.inFocusArea) {
                    homeWeights[i] = 0.0
                }
            }
        }
        val h = mk.ndarray( homeWeights.normalize()!! )
            .expandDims(0).asDNArray().asD2Array()
        tMatrices[ActivityType.HOME] = h

        // Transition matrices
        for (activityType in ActivityType.entries) {
            if (activityType == ActivityType.HOME) { continue }

            val mT = mk.zeros<Double>(omod.grid.size, omod.grid.size)
            if (finder.forcedTransitionMatrix.containsKey(activityType)) { // Handle forced matrix
                val mWeights = finder.forcedTransitionMatrix[activityType]!!
                for ((o, cell) in omod.grid.withIndex()) {
                    val weights = mWeights[cell]!!
                    mT[o] = mk.ndarray( weights.normalize()!! )
                }
            } else { // Normal case
                for (o in omod.grid.indices) {
                    val weights = finder.getWeights(omod.grid[o], omod.grid, activityType=activityType)
                    mT[o] = mk.ndarray( weights.normalize()!! )
                }
            }
            tMatrices[activityType] = mT
        }
        return tMatrices
    }

    /**
     * Determine all unique fixed-fixed chain segments and their probabilities of occurrence.
     *
     * @return unique chain segments with probabilities
     */
    private fun getUniqueChainSegments() : List<ChainSegment> {
        val activityGenerator = omod.activityGenerator as ActivityGeneratorDefault

        // Get all activity chains
        val allChains = mutableMapOf<List<ActivityType>, Double>()
        for (stratum in omod.popStrata) {
            if (stratum.stratumShare == 0.0) { continue }

            for ((socioFeatureSet, pSFSet) in stratum.iterateOptions()) {
                if (pSFSet == 0.0) { continue }

                // Get chains for that stratum
                val ageGrp = AgeGrp.fromInt(socioFeatureSet.age)
                val chains = activityGenerator.getChain(
                    Weekday.UNDEFINED, socioFeatureSet.hom, socioFeatureSet.mob, ageGrp, ActivityType.HOME
                )
                val chainProbs = chains.weights.normalize()!!.toTypedArray()

                // Cumulate probabilities
                for ((chain, chainP) in chains.chains.zip(chainProbs)) {
                    val p = stratum.stratumShare * pSFSet * chainP
                    allChains[chain] = allChains.getOrDefault(chain, 0.0) + p
                }
            }
        }

        // Determine fixed-fixed chain segments
        val segments = mutableListOf<ChainSegment>()
        for ((chain, chainP) in allChains) {
            val currentSegment = mutableListOf<ActivityType>()

            for ((i, activity) in chain.withIndex()) {
                currentSegment.add(activity)
                if (currentSegment.size > 1) {
                    // Reached fixed or last activity -> Segment finished
                    if ((activity in fixActivities) or (i == chain.size - 1)) {
                        segments.add(
                            ChainSegment(
                                currentSegment.toList(),
                                chainP
                            )
                        )
                        currentSegment.clear()
                        currentSegment.add(activity)  // New segment starts with end activity of this segment
                    }
                }
            }
        }

        // Get unique segments and cumulate probabilities
        val uniqueSegments = segments.groupBy { it.chain }.map { (chain, segments) ->
            ChainSegment(
                chain,
                segments.sumOf { it.probability }
            )
        }
        return uniqueSegments
    }

    /**
     * Fixed-fixed activity chain segment with probability of occurrence
     */
    private data class ChainSegment(
        val chain: List<ActivityType>,
        val probability: Double
    )

    /**
     * Get probability matrix for the car mode. Each entry gives the probability that the origin-destination trip
     * given by o=row and d=col is by car.
     *
     * @param weekday Weekday
     * @return Probability matrix for each activity type.
     */
    private fun getPCar(weekday: Weekday = Weekday.UNDEFINED) : Map<ActivityType, D2Array<Double>> {
        val finder = omod.destinationFinder as DestinationFinderDefault

        val pCar = mutableMapOf<ActivityType, D2Array<Double>>()
        for (activity in ActivityType.entries) {
            pCar[activity] = mk.zeros<Double>(omod.grid.size, omod.grid.size)
        }

        // Dummy location for home, work, and school of dummy agent. Irrelevant for mode choice.
        val dummyCoord = Coordinate(0.0,0.0)
        val dummyLocation = DummyLocation(dummyCoord, dummyCoord, null, setOf())

        for (stratum in omod.popStrata) {
            if (stratum.stratumShare == 0.0) { continue }

            for ((socioFeatureSet, pSFSet) in stratum.iterateOptions()) {
                if (pSFSet == 0.0) { continue }

                // Agent representing the population stratum
                val stratumAgent = MobiAgent(
                    -1,  socioFeatureSet.hom, socioFeatureSet.mob, socioFeatureSet.age,
                    dummyLocation, dummyLocation, dummyLocation, socioFeatureSet.sex
                )
                stratumAgent.carAccess = true // Car ownership probability is considered later

                // Car ownership probability
                val carOwnershipP = omod.carOwnership.probability(stratumAgent, stratum)
                if (carOwnershipP == 0.0) { continue }

                for (activity in ActivityType.entries) {
                    val pCarActivity = mk.zeros<Double>(omod.grid.size, omod.grid.size)
                    for (o in omod.grid.indices) {
                        val distances = finder.routingCache.getDistances(omod.grid[o], omod.grid)
                        for (d in omod.grid.indices) {
                            val weights = modeChoiceCalibration.utilitiesForCalibration(
                                distances[d].toDouble() / 1000.0, stratumAgent, activity, weekday
                            )
                            val pTrip = weights[0] / weights.sum()
                            pCarActivity[o, d] = stratum.stratumShare * pSFSet * carOwnershipP * pTrip
                        }
                    }
                    pCar[activity]?.plusAssign(pCarActivity)
                }
            }
        }
        return pCar
    }

    /**
     * Sample simulation to determine the distribution of trips across the time of day separated into T time slices.
     *
     * @see de.uniwuerzburg.omod.calibration.CalibrationConstants.T
     *
     * @n Sample size
     * @param weekday Weekday
     * @return Distributions where key is of size T
     */
    @Suppress("SameParameterValue")
    fun monteCarloTripStartDistribution(n: Int, weekday: Weekday = Weekday.UNDEFINED) : Map<ActivityType, DoubleArray> {
        val distr = ActivityType.entries.associateWith {
            DoubleArray(T) { 0.0 }
        }.toMutableMap()

        // Ensure results are deterministic
        omod.mainRng.setSeed(0)

        // Run Simulation
        val agents = omod.run(n, verbose = false, start_wd = weekday)
        omod.doModeChoice(agents, ModeChoiceOption.FAST, false, false)

        // Determine counts at sensors
        val visitor: TripVisitor = { _, _, destinationActivity, departureTime, _, _ ->
            val arr = distr[destinationActivity.type]!!
            val i = departureTime.determineTimeSlice()
            arr[i] = arr[i] + 1
        }
        for (agent in agents) {
            agent.mobilityDemand[0].visitTrips(visitor)
        }

        // Normalize
        for ((key, arr) in distr.entries) {
            distr[key] = arr.normalize() ?: DoubleArray(arr.size) { 0.0 }
        }

        return distr
    }

    /**
     * Determine od-Pairs that contribute to sensor measurements.
     *
     * @param affectedSensors Gives all sensors that are affected by a certain origin-destination pair
     * @return Set of relevant od-Pairs
     */
    fun getRelevantODs(
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
    ): Set<Pair<Int, Int>> {
        val relevantODs = mutableSetOf<Pair<Int, Int>>()
        for ((o, origin) in omod.grid.withIndex()) {
            for ((d, destination) in omod.grid.withIndex()) {
                val od = Pair(origin, destination)
                if (od in affectedSensors) {
                    if (affectedSensors[od]!!.isNotEmpty()) {
                        relevantODs.add(Pair(o, d))
                    }
                }
            }
        }
        return relevantODs
    }

    /**
     * Create computational graph of surrogate model.
     *
     * @param vActivity ActivityType for which the gravity model will be variable
     * @param sensors Sensors with measurements
     * @param affectedSensors Gives all sensors that are affected by a certain origin-destination pair
     * @param iThresh Performance parameter.
     * All terms with coefficients below this value will be ignored and not added to the result.
     * Higher values -> Computes faster but is a rougher approximation of the markov chain representation.
     */
    private fun buildDiffModel(
        vActivity: ActivityType,
        sensors: List<TrafficSensor>,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        iThresh: Double = (1/omod.grid.size.toDouble()).pow(1.5)
    ) : Pair<DifferentiableModel, Map<TrafficSensor, List<LinearTerm>>> {
        logger.info("Building surrogate for activity $vActivity with ${omod.grid.size - 1} variables")

        val n = omod.grid.size
        val population = omod.buildings.sumOf { it.population }
        val mrep = generateMarkovChainRep(vActivity) // Compact matrix representation
        val relevantODs = getRelevantODs(affectedSensors) // Relevant origin-destination pairs for measurements

        // Init diff model
        val model = DifferentiableModel(omod.grid.size - 1)

        // Create graph of the expected trips matrix: E(o, d | Car)
        val expectedTrips = ActivityType.entries.associateWith {
            List(n) {
                List(n) {
                    LinearTerm(model.nVars)
                }
            }
        }

        // Transition matrix containing variable terms
        val vMatrix = mutableListOf<List<Term>>()
        for (o in 0 until n) {
            // Get weight terms
            val weights = mutableListOf<Term>()
            val sum = LinearTerm(model.nVars)
            for (d in 0 until n) {
                val weight = LinearBaseTerm(model.nVars)
                if ( d != (n-1) ) {
                    weight.addTerm(d, mrep.tMatrices[mrep.vActivity]!![o, d])
                } else {
                    // Last destination is chosen as the pivot element
                    weight.addConstant(mrep.tMatrices[mrep.vActivity]!![o, d])
                }
                sum.addTerm(weight, 1.0)
                weights.add(weight)
            }

            // Normalize
            val t = mutableListOf<Term>()
            for (d in 0 until n) {
                t.add( DivisionTerm(model.nVars, weights[d], sum) )
            }

            vMatrix.add(t)
        }

        // Temporal trip distribution
        val tripStartDistr = monteCarloTripStartDistribution( MC_SAMPLES )

        // Add expected trips for each destination activity
        for (activity in ActivityType.entries) {
            addE(
                LinearTermBuilder,
                model.nVars,
                mrep,
                expectedTrips[activity]!!,
                vMatrix,
                relevantODs,
                iThresh,
                activity
            )
        }

        // Simulated traffic counts
        val simCount = mutableMapOf<TrafficSensor, List<LinearTerm>>()
        for (sensor in sensors) {
            simCount[sensor] = List(T) { LinearTerm(model.nVars) }
        }
        for ((o, origin) in omod.grid.withIndex()) {
            for ((d, destination) in omod.grid.withIndex()) {
                val od = Pair(origin, destination)
                if (od in affectedSensors) {
                    val affected = affectedSensors[od]!!
                    for (sensor in affected) {
                        for (t in 0 until T) {
                            for (activity in ActivityType.entries) {
                                simCount[sensor]!![t].addTerm(
                                    expectedTrips[activity]!![o][d], population * tripStartDistr[activity]!![t]
                                )
                            }
                        }
                    }
                }
            }
        }

        // Objective
        val obj = mseObjective(model.nVars, sensors, simCount)

        model.setRootTerm(obj)

        // Logging
        var terms = 0
        model.visit { terms += 1 }
        logger.info("Building surrogate complete. Number of terms: $terms")

        return model to simCount
    }

    /**
     * Determine expected trips matrix with the given activity at the destination.
     *
     * @param builder Term builder of the desired model
     * @param nVars Number of variables in the problem
     * @param mrep Compact markov chain representation of original model.
     * @param expectedTrips Expected trips matrix model to which the new terms are added
     * @param vMatrix Transition matrix containing variable terms
     * @param relevantODs ActivityType for which the gravity model will be variable
     * @param iThresh Performance parameter. @see de.uniwuerzburg.omod.calibration.surrogate.SGGravity.buildDiffModel
     * @param activity Activity at the destination of the trip matrix
     */
    fun <T, K> addE(
        builder: TermBuilder<T, K>,
        nVars: Int,
        mrep: SGCompactMatrixRep,
        expectedTrips: List<List<T>>,
        vMatrix : List<List<K>>,
        relevantODs: Set<Pair<Int, Int>>,
        iThresh: Double,
        activity: ActivityType
    ) {
        val tActivity = if (activity == ActivityType.BUSINESS) {
            ActivityType.OTHER // Edge case: Use other type transition matrix for business activity
        } else {
            activity
        }

        val n = omod.grid.size
        val pCar = mrep.pCar[tActivity]!!
        val mPriorCnst  = mrep.mPriorCnst[activity]!!
        val mPriorVar   = mrep.mPriorVar[activity]!!
        val mPriorVarT  = mPriorVar.transpose()

        // In the case that the segment starts at vActivity
        val vStart = List(n) { builder.new(nVars) }

        // Transition matrix
        val tMatrix = if (activity == ActivityType.HOME) {
            mk.identity<Double>(n)
        } else {
            mrep.tMatrices[tActivity]!!
        }

        // Expected trip contribution unaffected by vActivity
        val mFix = when(activity) {
            in fixActivities -> mPriorCnst.transpose().dot(tMatrix) // F = (K^T)A
            mrep.vActivity -> mk.zeros<Double>(n, n) // Not used
            else -> {
                // CASE: vActivity is flexible but not the destination
                // F = diag(iK) A
                val ones = mk.ones<Double>(1, n)
                val left = ones.dot(mPriorCnst).diagonal()
                left.dot(tMatrix)
            }
        }

        // Expected trip contribution affected by vActivity
        val mVar = when(activity) {
            in fixActivities -> {
                when (mrep.vActivity) {
                    activity -> {
                        // For segments that started at vActivity: V = (vK)^T
                        // Here we only build v (vStart)
                        val h = mrep.h.flatten()
                        for (col in 0 until n) {
                            for (row in 0 until n) {
                                builder.addVar(vStart[col], vMatrix[row][col], h[row])
                            }
                        }
                        // For other segments: V = (K^T)X
                        val left  = mPriorCnst.transpose()
                        val right = mk.identity<Double>(n)
                        builder.fromMatrixMult(
                            nVars, vMatrix, left, right, transpose=false, relevantRCs=relevantODs, cTol=iThresh
                        )
                    }
                    in fixActivitiesNotHome -> {
                        // V = ( ( diag(h)XK )^T ) A
                        val left  = mPriorVarT
                        val right = mrep.h.diagonal().transpose().dot(tMatrix)
                        builder.fromMatrixMult(nVars, vMatrix, left, right, relevantRCs=relevantODs, cTol=iThresh)
                    }
                    ActivityType.HOME -> {
                        // Destination: HOME
                        // V = ( ( KX )^T ) A
                        val left  = mk.identity<Double>(n)
                        val right = mPriorVarT.dot(tMatrix)
                        builder.fromMatrixMult(nVars, vMatrix, left, right, relevantRCs=relevantODs, cTol=iThresh)
                    }
                    else -> { throw IllegalStateException("$mrep.vActivity neither HOME nor in fixActivitiesNotHome") }
                }
            }
            in flexActivities -> {
                when (mrep.vActivity) {
                    activity -> {
                        // V = diag(iK)X
                        val ones = mk.ones<Double>(1, n)
                        val left = ones.dot(mPriorCnst).diagonal()
                        val right = mk.identity<Double>(n)
                        builder.fromMatrixMult(
                            nVars, vMatrix, left, right, transpose=false, relevantRCs=relevantODs, cTol=iThresh
                        )
                    }
                    in fixActivitiesNotHome -> {
                        // V = diag(hXK)A
                        val left = mrep.h
                        val right = mPriorVar
                        builder.fromMatrixMult(
                            nVars, vMatrix, left, right, transpose=false, relevantRCs=relevantODs, cTol=iThresh
                        )
                    }
                    else -> {
                        // V = diag(KX)A
                        val left = mk.ones<Double>(1, n).dot(mPriorVar)
                        val right = mk.identity<Double>(n)
                        builder.fromMatrixMult(
                            nVars, vMatrix, left, right, transpose=false, relevantRCs=relevantODs, cTol=iThresh
                        )
                    }
                }
            }
            else -> { throw IllegalStateException("$activity neither fixed nor flexible") }
        }

        // E += ( F + V ) odot pCar
        val fix = mFix.times(pCar)
        val tMatrixCar = tMatrix.times(pCar)
        val mPriorVarTCar = mPriorVarT.times(pCar)
        for (o in 0 until n) {
            for (d in 0 until n) {
                if (Pair(o, d) !in relevantODs) { continue }
                // F
                if (mrep.vActivity != activity) {
                    builder.addConstant(expectedTrips[o][d], fix[o, d])
                }

                // V
                if (mVar.size == 1) {
                    builder.addTerm(expectedTrips[o][d], mVar[0][o], tMatrixCar[o,d])
                } else {
                    builder.addTerm(expectedTrips[o][d], mVar[o][d], pCar[o, d])
                }
                if ((mrep.vActivity in fixActivities) and (mrep.vActivity == activity)){
                    // For segments that started at vActivity: V = (vK)^T
                    builder.addTerm(expectedTrips[o][d], vStart[d], mPriorVarTCar[o, d]) // TODO should be vStart[o] i think
                }
            }
        }
    }

    // ========== DEBUG CODE ================= // TODO Delete
    private fun getExpectedCountPerAgent(mrep: SGCompactMatrixRep) : D2Array<Double> {
        // Result: Expected trip count on each od-paid of one agent on one day
        val expectedCountPerAgent = mk.zeros<Double>(omod.grid.size, omod.grid.size)

        val varTMatrix = mrep.tMatrices[mrep.vActivity]!!

        // Flex
        for (flexActivity in listOf(ActivityType.OTHER, ActivityType.SHOPPING, ActivityType.BUSINESS)) {
            val transitionActivity = if (flexActivity == ActivityType.BUSINESS) {
                ActivityType.OTHER
            } else {
                flexActivity
            }
            val carP = mrep.pCar[transitionActivity]!!
            val fix = mrep.mPriorCnst[flexActivity]!!

            val dep = if (mrep.vActivity in fixActivitiesNotHome) {
                mrep.h.diagonal().dot(varTMatrix).dot(mrep.mPriorVar[flexActivity]!!)
            } else if (flexActivity == mrep.vActivity) {
                mk.zeros<Double>(omod.grid.size, omod.grid.size)
            } else {
                mrep.mPriorVar[flexActivity]!!.dot(varTMatrix)
            }
            val total = mk.ones<Double>(omod.grid.size).expandDims(0).asDNArray()
                .asD2Array().dot(dep.plus(fix))
                .diagonal().dot(mrep.tMatrices[transitionActivity]!!)
            expectedCountPerAgent.plusAssign(total.times(carP))
        }

        // Home
        for (fixActivity in listOf(ActivityType.HOME)) {
            val carP = mrep.pCar[fixActivity]!!

            val dep = if (mrep.vActivity in fixActivitiesNotHome) {
                mrep.h.diagonal().dot(varTMatrix).dot(mrep.mPriorVar[fixActivity]!!)
            } else {
                mrep.mPriorVar[fixActivity]!!.dot(varTMatrix)
            }
            val fix = mrep.mPriorCnst[fixActivity]!!
            val total = dep.plus(fix).transpose()

            expectedCountPerAgent.plusAssign(total.times(carP))
        }

        // School
        for (fixActivity in listOf(ActivityType.SCHOOL)) {
            val carP = mrep.pCar[fixActivity]!!
            val fix = mrep.mPriorCnst[fixActivity]!!

            if (mrep.vActivity == ActivityType.SCHOOL) {
                val schoolProbs = mrep.h.dot( mrep.tMatrices[ActivityType.SCHOOL]!! )
                val dep = schoolProbs.diagonal().dot( mrep.mPriorVar[fixActivity]!! ).transpose()
                val total = dep.plus( fix.transpose().dot( mrep.tMatrices[fixActivity]!!) )
                expectedCountPerAgent.plusAssign(total.times(carP))
            } else {
                val dep = if (mrep.vActivity in fixActivitiesNotHome) {
                    mrep.h.diagonal().dot(varTMatrix).dot(mrep.mPriorVar[fixActivity]!!)
                } else {
                    mrep.mPriorVar[fixActivity]!!.dot(varTMatrix)
                }
                val total = dep.plus(fix).transpose().dot(mrep.tMatrices[fixActivity]!!)
                expectedCountPerAgent.plusAssign(total.times(carP))
            }
        }

        // Work
        for (fixActivity in listOf(ActivityType.WORK)) {
            val carP = mrep.pCar[fixActivity]!!
            val fix = mrep.mPriorCnst[fixActivity]!!

            if (mrep.vActivity == ActivityType.WORK) {
                val workProbs = mrep.h.dot( mrep.tMatrices[ActivityType.WORK]!! )
                val dep = workProbs.diagonal().dot( mrep.mPriorVar[fixActivity]!! ).transpose()
                val total = dep.plus( fix.transpose().dot( mrep.tMatrices[fixActivity]!!) )
                expectedCountPerAgent.plusAssign(total.times(carP))
            } else {
                val dep = if (mrep.vActivity in fixActivitiesNotHome) {
                    mrep.h.diagonal().dot(varTMatrix).dot(mrep.mPriorVar[fixActivity]!!)
                } else {
                    mrep.mPriorVar[fixActivity]!!.dot(varTMatrix)
                }
                val total = dep.plus(fix).transpose().dot(mrep.tMatrices[fixActivity]!!)
                expectedCountPerAgent.plusAssign(total.times(carP))
            }
        }

        return expectedCountPerAgent
    }

    private fun debugEval(
        mrep: SGCompactMatrixRep,
        sensors: List<TrafficSensor>,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    ) {
        val expected = getExpectedCountPerAgent(mrep)
        val totalPop = omod.buildings.sumOf { it.population }

        val simCount = sensors.associateWith { 0.0 }.toMutableMap()
        for ((o, origin) in omod.grid.withIndex()) {
            for ((d, destination) in omod.grid.withIndex()) {
                val odPair = Pair(origin, destination)
                if (odPair in affectedSensors) {
                    val affected = affectedSensors[odPair]!!
                    for (sensor in affected) {

                        simCount[sensor] = simCount[sensor]!! + expected[o,d] * totalPop
                    }
                }
            }
        }
        var mse = 0.0
        val allFlows = mutableMapOf<TrafficSensor, Double>()
        for (sensor in sensors) {
            val simFlow = simCount[sensor]!!
            mse += (sensor.measuredFlow.sum() - simFlow).pow(2)

            allFlows[sensor] = simFlow
        }
        mse /= sensors.size

        println("MSE: $mse")
        println("------------------------------")
        println("Sensor | \t Flow OMOD | \t Flow Measured")
        for ((i, flow) in simCount.values.withIndex()) {
            println("${sensors[i].name} | \t $flow | \t ${sensors[i].measuredFlow.sum()}")
        }
    }
    // ========== DEBUG CODE ENDS =================
}


