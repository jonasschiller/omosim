package de.uniwuerzburg.omod.calibration.surrogate

import de.uniwuerzburg.omod.calibration.CalibrationConstants.MC_SAMPLES
import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.ModeChoiceCalibration
import de.uniwuerzburg.omod.calibration.TrafficSensor
import de.uniwuerzburg.omod.calibration.differentiablemodel.*
import de.uniwuerzburg.omod.calibration.logger
import de.uniwuerzburg.omod.core.ActivityGeneratorDefault
import de.uniwuerzburg.omod.core.DestinationFinderDefault
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.*
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
import kotlin.math.abs
import kotlin.math.floor
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
    private val fixActivities = setOf(ActivityType.WORK, ActivityType.SCHOOL)

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
        val (model, tstSimCounts) = buildDiffModel(activityType, affectedSensors, sensors)

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
        val (_, simCounts) = buildDiffModel(activityType, affectedSensors, sensors)

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
     * TODO: Explain compact representation
     *
     * Generate markov chain representation of original model.
     *
     * Compact representation:
     * - The expected trip matrix can be computed based on the probability matrix K that gives the probability of any
     * agent being in location column given his home location row.
     * - K depends on the sequence of prior activities. For example: HSSSO
     * - All these sequences that have the same last activity can be grouped:
     *      - O' = HSSSO + HO + HWO + ... etc.
     * - O will not be included in the groups. Therefore actually: O' = HSSS + H + HW + ...
     * - In this grouping we make a distinction between the sequences that contain the variable activity and those that
     * do not:
     *      - mPriorCnst -> Does not contain the activity
     *      - mPriorVar  -> Contains the activity
     *
     * @param vActivity activity type for the variable gravity model
     * @return compact markov chain representation.
     */
    fun generateMarkovChainRep(vActivity: ActivityType) : SGCompactMatrixRep {
        if(vActivity == ActivityType.HOME) {
            throw NotImplementedError("Surrogate model dependent on home coefficients is not implemented!")
        }

        val n = omod.grid.size

        // Transition matrices
        val tMatrices = computeTransitionMatrices()
        val h = tMatrices[ActivityType.HOME]!!
        val mHW = h.diagonal().dot(tMatrices[ActivityType.WORK]!!)  // Home <-> Work, Work
        val mHS = h.diagonal().dot(tMatrices[ActivityType.SCHOOL]!!)  // Home <-> School, School
        val hDiag = h.diagonal()

        // Compact representation:
        val mPriorVar = ActivityType.entries.associateWith { mk.zeros<Double>(n, n) }
        val mPriorCnst = ActivityType.entries.associateWith { mk.zeros<Double>(n, n) }

        // Go through each unique fixed-fixed segment
        for ((chain, chainP) in getUniqueChainSegments()) {
            if (chain.size <= 1) { continue }

            val startActivity = chain.first()
            var nextActivity = chain[1]

            // Setup
            var mPostFixed = mk.identity<Double>(n)
            var mK: D2Array<Double> // Location probability distribution given the home location
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

            // Run through chain cumulate mPriorVar and mPriorCnst
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
                if ((vActivity in fixActivities) and (occurred)) {
                    // Cumulate all matrices after a fixed location activity has occurred
                    mPostFixed = mPostFixed.dot(mT)
                    mPriorVar[nextActivity]!!.plusAssign(mPostFixed * chainP)
                } else if ((vActivity !in fixActivities) and (occurred) and (nextActivity != vActivity)){
                    // Cumulate location probabilities after a flex location activity
                    // Ignores all subsequent flex location activities of the same time
                    mPriorVar[nextActivity]!!.plusAssign(mKExVar * chainP)
                }
                // Has not (yet) occured or is ignored
                else {
                    // Normal case cumulate matrices before next activity
                    mPriorCnst[nextActivity]!!.plusAssign(mK.dot(mT) * chainP)
                }

                // Update location probability distributions
                mK = mK.dot(mT)
                if ((vActivity !in fixActivities) and (activity != vActivity)) {
                    mKExVar = mKExVar.dot(mT)
                }

                if (nextActivity == vActivity) {
                    occurred = true
                }
            }
        }

        return SGCompactMatrixRep(h, mPriorVar, mPriorCnst, tMatrices, getPCar(), vActivity)
    }

    /**
     * Matrices that describe the probability of an agent to be somewhere before an activity:
     *  - homeP -> Home probability for each cell.
     *  - dependentMatrix -> Probability part dependent on the currently calibrated activity.
     *  - fixedMatrix -> Independent part
     *
     * Matrices that describe transition probabilities:
     *  - transitionMatrix
     *  - carP mode share of car mode for each transition
     *
     *  varActivityType: Activity those transition matrix is going to be calibrated.
     */
    data class SGCompactMatrixRep (
        val h: D2Array<Double>,
        val mPriorVar: Map<ActivityType, D2Array<Double>>,
        val mPriorCnst: Map<ActivityType,  D2Array<Double>>,
        val tMatrices: Map<ActivityType,  D2Array<Double>>,
        val pCar: Map<ActivityType,  D2Array<Double>>,
        val vActivity: ActivityType
    )

    private fun computeTransitionMatrices() : Map<ActivityType,  D2Array<Double>> {
        val transitionProbs = mutableMapOf<ActivityType,  D2Array<Double>>()
        val finder = omod.destinationFinder as DestinationFinderDefault

        // HOME
        val homeWeights = finder.getWeightsNoOrigin(omod.grid, activityType=ActivityType.HOME).toMutableList()
        if (!omod.populateBufferArea) { // Handle buffer area
            for ((i, cell) in omod.grid.withIndex()) {
                if (!cell.inFocusArea) {
                    homeWeights[i] = 0.0
                }
            }
        }
        val homeProbs   = mk.ndarray( homeWeights.map { it / homeWeights.sum() } )
            .expandDims(0).asDNArray().asD2Array()
        transitionProbs[ActivityType.HOME] = homeProbs

        // Transitions
        for (activityType in ActivityType.entries) {
            if (activityType == ActivityType.HOME) { continue }

            val activityProbs = mk.zeros<Double>(omod.grid.size, omod.grid.size)
            if (finder.forcedTransitionMatrix.containsKey(activityType)) { // If matrix is forced
                val fMatrix = finder.forcedTransitionMatrix[activityType]!!
                for ((o, cell) in omod.grid.withIndex()) {
                    val activityWeights = fMatrix[cell]!!
                    activityProbs[o] = mk.ndarray( activityWeights.map { it / activityWeights.sum() } )
                }
            } else { // Normal case
                for (o in omod.grid.indices) {
                    val activityWeights = finder.getWeights(omod.grid[o], omod.grid, activityType = activityType)
                    activityProbs[o] = mk.ndarray( activityWeights.map { it / activityWeights.sum() } )
                }
            }
            transitionProbs[activityType] = activityProbs
        }
        return transitionProbs
    }

    private fun getUniqueChainSegments() : List<ChainSegment> {
        val activityGenerator = omod.activityGenerator as ActivityGeneratorDefault

        // Get activity chains
        val allChains = mutableMapOf<List<ActivityType>, Double>()
        for (stratum in omod.popStrata) {
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
        val fixedActivities = setOf(ActivityType.HOME, ActivityType.WORK, ActivityType.SCHOOL)
        val fixedSegments = mutableListOf<ChainSegment>()
        val segment = mutableListOf<ActivityType>()
        for ((chain, chainP) in allChains) {
            segment.clear()

            for ((i, activity) in chain.withIndex()) {
                segment.add(activity)
                if (segment.size > 1) {
                    if ((activity in fixedActivities) or (i == chain.size - 1)) {
                        fixedSegments.add(
                            ChainSegment(
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
            ChainSegment(
                chain,
                segments.sumOf { it.probability }
            )
        }
        return uniqueSegments
    }

    private data class ChainSegment(
        val chain: List<ActivityType>,
        val probability: Double
    )

    private fun getPCar() : Map<ActivityType, D2Array<Double>> {
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

    @Suppress("SameParameterValue")
    fun monteCarloTripStartDistribution(n: Int) : Map<ActivityType, DoubleArray> {
        val distr = ActivityType.entries.associateWith {
            DoubleArray(T) { 0.0 }
        }.toMutableMap()

        // Ensure results are deterministic
        omod.mainRng.setSeed(0)

        // Run Simulation
        val agents = omod.run(n, verbose = false)
        omod.doModeChoice(agents, ModeChoiceOption.FAST, false, false)

        val visitor: TripVisitor = { _, _, destinationActivity, departureTime, _, _ ->
            val arr = distr[destinationActivity.type]!!
            val mod = departureTime.minute + departureTime.hour * 60
            val i = floor((mod % 1440.0) / 1440.0 * T).toInt()
            arr[i] = arr[i] + 1
        }

        // Determine counts at sensors
        for (agent in agents) {
            agent.mobilityDemand[0].visitTrips(visitor)
        }

        for ((key, arr) in distr.entries) {
            val sum = arr.sum()
            distr[key] = if (sum == 0.0) {
                DoubleArray(arr.size) { 0.0 }
            } else {
                arr.map { it / sum }.toDoubleArray()
            }
        }

        return distr
    }

    private fun buildDiffModel(
        activityType: ActivityType,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        sensors: List<TrafficSensor>,
        irrelevancyThreshold: Double = (1/omod.grid.size.toDouble()).pow(1.5)
    ) : Pair<DifferentiableModel, Map<TrafficSensor, List<LinearTerm>>> {
        logger.info("Building surrogate for activity $activityType with ${omod.grid.size -1} variables")
        val mrep = generateMarkovChainRep(activityType)

        val n = omod.grid.size
        val totalPop = omod.buildings.sumOf { it.population }

        val relevantODs = determineRelevantODs(affectedSensors)

        // Init diff model
        val diffModel = DifferentiableModel(omod.grid.size - 1)

        // Create expected trips matrix dependent on the variable transition matrix: E(o, d | M)
        val expectedTrips = ActivityType.entries.associateWith {
            List(n) {
                List(n) {
                    LinearTerm(diffModel.nVars)
                }
            }
        }
        val varTransitionMatrix = mutableListOf<List<Term>>()
        for (o in 0 until n) {
            val entry = mutableListOf<Term>()
            val weightTerms = mutableListOf<Term>()
            val rowSumTerm = LinearTerm(diffModel.nVars)
            for (d in 0 until n) {
                val weight = LinearBaseTerm(diffModel.nVars)
                if ( d != (n-1)) {
                    weight.addTerm(d, mrep.tMatrices[mrep.vActivity]!![o, d])
                } else {
                    weight.addConstant(mrep.tMatrices[mrep.vActivity]!![o, d])
                }
                rowSumTerm.addTerm(weight, 1.0)
                weightTerms.add(weight)
            }
            for (d in 0 until n) {
                val scaledWeight = DivisionTerm(diffModel.nVars, weightTerms[d], rowSumTerm)
                entry.add(scaledWeight)
            }
            varTransitionMatrix.add(entry)
        }

        // Flexible destination
        for (activity in listOf( ActivityType.OTHER, ActivityType.SHOPPING, ActivityType.BUSINESS)) {
            addFlexE(
                LinearTermBuilder,
                diffModel.nVars,
                mrep,
                expectedTrips[activity]!!,
                varTransitionMatrix,
                relevantODs,
                irrelevancyThreshold,
                activity
            )
        }

        // Destination home
        addHomeE(
            LinearTermBuilder,
            diffModel.nVars,
            mrep,
            expectedTrips[ActivityType.HOME]!!,
            varTransitionMatrix,
            relevantODs,
            irrelevancyThreshold
        )

        // Destination School and Work
        for (activity in listOf( ActivityType.SCHOOL, ActivityType.WORK)) {
            addFixE(
                LinearTermBuilder,
                diffModel.nVars,
                mrep,
                expectedTrips[activity]!!,
                varTransitionMatrix,
                relevantODs,
                irrelevancyThreshold,
                activity
            )
        }

        // Temporal trip distribution
        val tripStartDistr = monteCarloTripStartDistribution( MC_SAMPLES )

        // Simulated Traffic Counts
        val simCount = mutableMapOf<TrafficSensor, List<LinearTerm>>()
        for (sensor in sensors) {
            simCount[sensor] = List(T) { LinearTerm(diffModel.nVars) }
        }
        for ((o, origin) in omod.grid.withIndex()) {
            for ((d, destination) in omod.grid.withIndex()) {
                val od = Pair(origin, destination)
                if (od in affectedSensors) {
                    val affected = affectedSensors[od]!!

                    for (sensor in affected) {
                        for (t in 0 until T) {
                            for (activity in ActivityType.entries) {
                                /*if(demand[activity]!![o][d].terms.size == 0) { continue }
                                if(activity == ActivityType.BUSINESS) {continue}*/
                                simCount[sensor]!![t].addTerm(
                                    expectedTrips[activity]!![o][d], totalPop * tripStartDistr[activity]!![t]
                                )
                            }
                        }
                    }
                }
            }
        }

        // Objective
        val obj = LinearTerm(diffModel.nVars)
        for (sensor in sensors) {
            for (t in 0 until T) {
                // (Sm - Ss)^2 = Sm^2 - 2SmSs + Ss^2
                val simCountTerm = simCount[sensor]!![t]
                obj.addConstant(sensor.measuredFlow[t] * sensor.measuredFlow[t])
                obj.addTerm(simCountTerm, -2 * sensor.measuredFlow[t])
                val qTerm = QuadraticTerm(
                    diffModel.nVars,
                    simCountTerm,
                    simCountTerm,
                    1.0
                )
                obj.addTerm(qTerm, 1.0)
            }
        }

        diffModel.setRootTerm(obj)

        var terms = 0
        diffModel.visit { terms += 1 }
        logger.info("Building surrogate complete. Number of terms: $terms")
        return diffModel to simCount
    }



    fun <T, K> addFlexE(
        demandBuilder: TermBuilder<T, K>,
        nVars: Int,
        mrep: SGCompactMatrixRep,
        expectedTrips: List<List<T>>,
        varTransitionMatrix : List<List<K>>,
        relevantODs: Set<Pair<Int, Int>>,
        irrelevancyThreshold: Double,
        flexActivity: ActivityType
    ) {
        val n = omod.grid.size

        val transitionActivity = if (flexActivity == ActivityType.BUSINESS) {
            ActivityType.OTHER
        } else {
            flexActivity
        }
        val mCarP = mrep.pCar[transitionActivity]!!
        val mFixed = mrep.mPriorCnst[flexActivity]!!

        if (flexActivity == mrep.vActivity) {
            for (o in 0 until n) {
                var cnst = 0.0

                for (a in 0 until n) {
                    cnst += mFixed[a, o]
                }

                for (d in 0 until n) {
                    if (Pair(o, d) !in relevantODs) { continue }
                    demandBuilder.addVar(expectedTrips[o][d], varTransitionMatrix[o][d], cnst * mCarP[o, d])
                }
            }
        } else {
            val mTransition = mrep.tMatrices[transitionActivity]!!
            val mDep = mrep.mPriorVar[flexActivity]!!
            val pHome = mrep.h.flatten()

            for (o in 0 until n) {
                val s = demandBuilder.new(nVars)
                var cnst = 0.0

                for (a in 0 until n) {
                    cnst += mFixed[a, o]
                    for (b in 0 until n) {
                        val coeff = if (mrep.vActivity in fixActivities) {
                            pHome[a] * mDep[b, o]
                        } else {
                            mDep[a, b]
                        }

                        // Ignore very small terms
                        if (abs(coeff) <= irrelevancyThreshold) {
                            continue
                        }

                        if (mrep.vActivity in fixActivities) {
                            demandBuilder.addVar(s, varTransitionMatrix[a][b], coeff)
                        } else {
                            demandBuilder.addVar(s, varTransitionMatrix[b][o], coeff)
                        }
                    }
                }
                demandBuilder.addConstant(s, cnst)

                for (d in 0 until n) {
                    if (Pair(o, d) !in relevantODs) { continue }
                    val t = mTransition[o, d] * mCarP[o, d]
                    demandBuilder.addTerm(expectedTrips[o][d], s, t)
                }
            }
        }

    }

    fun <T, K> addHomeE(
        builder: TermBuilder<T, K>,
        nVars: Int,
        mrep: SGCompactMatrixRep,
        expectedTrips: List<List<T>>,
        varTransitionMatrix : List<List<K>>,
        relevantODs: Set<Pair<Int, Int>>,
        irrelevancyThreshold: Double
    ) {
        val n = omod.grid.size

        val carP = mrep.pCar[ActivityType.HOME]!!
        val fix = mrep.mPriorCnst[ActivityType.HOME]!!.transpose().times(carP)

        val depExpr = if(mrep.vActivity in fixActivities) {
            val left = mrep.mPriorVar[ActivityType.HOME]!!.transpose()
            val right = mrep.h.diagonal().transpose()
            builder.fromMatrixMult(
                nVars,
                varTransitionMatrix, left, right,
                relevantRCs = relevantODs, irrelevancyThreshold=irrelevancyThreshold
            )
        } else {
            val left = mk.identity<Double>(n)
            val right = mrep.mPriorVar[ActivityType.HOME]!!.transpose()
            builder.fromMatrixMult(
                nVars,
                varTransitionMatrix, left, right,
                relevantRCs = relevantODs,
                irrelevancyThreshold=irrelevancyThreshold
            )
        }

        for (o in 0 until n) {
            for (d in 0 until n) {
                if (Pair(o, d) !in relevantODs) { continue }
                builder.addTerm(expectedTrips[o][d], depExpr[o][d], carP[o, d])
                builder.addConstant(expectedTrips[o][d], fix[o, d])
            }
        }

    }

    fun <T, K> addFixE(
        builder: TermBuilder<T, K>,
        nVars: Int,
        mrep: SGCompactMatrixRep,
        expectedTrips: List<List<T>>,
        varTransitionMatrix : List<List<K>>,
        relevantODs: Set<Pair<Int, Int>>,
        irrelevancyThreshold: Double,
        fixActivity: ActivityType
    ) {
        val n = omod.grid.size

        val carP = mrep.pCar[fixActivity]!!
        val fix = mrep.mPriorCnst[fixActivity]!!.transpose()
            .dot(mrep.tMatrices[fixActivity]!!)
            .times(carP)

        // Tour starting at var activity
        val pHome = mrep.h.flatten()
        val dMatrixT = mrep.mPriorVar[fixActivity]!!.transpose()
        val varStartProbs = List(n) { builder.new(nVars) }
        for (col in 0 until n) {
            for (row in 0 until n) {
                builder.addVar(varStartProbs[col], varTransitionMatrix[row][col], pHome[row])
            }
        }

        // Tour not starting at var activity
        val depExpr = if (mrep.vActivity == fixActivity) {
            val left = mrep.mPriorCnst[fixActivity]!!.transpose()
            val right = mk.identity<Double>(n)
            builder.fromMatrixMult(
                nVars,
                varTransitionMatrix, left, right,
                transpose = false,
                relevantRCs = relevantODs,
                irrelevancyThreshold=irrelevancyThreshold
            )
        } else if (mrep.vActivity in fixActivities) {
            val left = mrep.mPriorVar[fixActivity]!!.transpose()
            val right = mrep.h
                .diagonal()
                .transpose()
                .dot(mrep.tMatrices[fixActivity]!!)
            builder.fromMatrixMult(
                nVars,
                varTransitionMatrix, left, right,
                relevantRCs = relevantODs,
                irrelevancyThreshold=irrelevancyThreshold
            )
        } else {
            val left = mk.identity<Double>(n)
            val right = mrep.mPriorVar[fixActivity]!!
                .transpose()
                .dot(mrep.tMatrices[fixActivity]!!)
            builder.fromMatrixMult(
                nVars,
                varTransitionMatrix, left, right,
                relevantRCs = relevantODs,
                irrelevancyThreshold=irrelevancyThreshold
            )
        }

        for (o in 0 until n) {
            for (d in 0 until n) {
                if (Pair(o, d) !in relevantODs) { continue }
                builder.addTerm(expectedTrips[o][d], depExpr[o][d], carP[o, d])

                if (mrep.vActivity == fixActivity) {
                    builder.addTerm(expectedTrips[o][d], varStartProbs[d], dMatrixT[o][d] * carP[o, d])
                } else {
                    builder.addConstant(expectedTrips[o][d], fix[o,d])
                }
            }
        }
    }

    fun determineRelevantODs(
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

            val dep = if (mrep.vActivity in fixActivities) {
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

            val dep = if (mrep.vActivity in fixActivities) {
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
                val dep = if (mrep.vActivity in fixActivities) {
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
                val dep = if (mrep.vActivity in fixActivities) {
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


