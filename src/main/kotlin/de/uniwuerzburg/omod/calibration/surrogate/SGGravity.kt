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
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import org.locationtech.jts.geom.Coordinate
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.pow

class SGGravity(
    val omod: Omod
) {
    private val modeChoiceCalibration = ModeChoiceCalibration()
    private val fixActivities = setOf(ActivityType.WORK, ActivityType.SCHOOL)

    init {
        if (omod.destinationFinder !is DestinationFinderDefault) {
            throw NotImplementedError(
                "MetaModel is not valid for the destination finder: " +
                        omod.destinationFinder.javaClass.simpleName
            )
        }
        if (omod.activityGenerator !is ActivityGeneratorDefault) {
            throw NotImplementedError(
                "MetaModel is not valid for the activity generator finder: " +
                        omod.activityGenerator.javaClass.simpleName
            )
        }
    }

    fun buildModelMSE(
        activityType: ActivityType,
        sensors: List<TrafficSensor>,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    ): DifferentiableModel {
        // - Clean up rest and document
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

    fun generateMarkovChainRep(varActivityType: ActivityType) : SGCompactMatrixRep {
        if(varActivityType == ActivityType.HOME) {
            throw NotImplementedError("Surrogate model dependent on home coefficients is not implemented!")
        }

        val dependentMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(omod.grid.size, omod.grid.size) }
        val fixedMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(omod.grid.size, omod.grid.size) }
        val finderMatrices = computeTransitionMatrices()

        // Determine od pair probabilities
        for ((chain, chainP) in getUniqueChainSegments()) {
            if (chain.size <= 1) { continue }

            val startActivity = chain.first()
            val endActivity = chain.last()

            // Short chains
            if (chain.size == 2) {
                when(Pair(startActivity, endActivity)) {
                    Pair(ActivityType.HOME, ActivityType.WORK) -> {
                        fixedMatrix[ActivityType.WORK]!!.plusAssign(finderMatrices.homeP.diagonal() * chainP)
                        continue
                    }
                    Pair(ActivityType.HOME, ActivityType.SCHOOL) -> {
                        fixedMatrix[ActivityType.SCHOOL]!!.plusAssign(finderMatrices.homeP.diagonal() * chainP)
                        continue
                    }
                    Pair(ActivityType.WORK, ActivityType.HOME) -> {
                        if (varActivityType == ActivityType.WORK) {
                            dependentMatrix[ActivityType.HOME]!!.plusAssign(mk.identity<Double>(omod.grid.size) * chainP)
                        } else {
                            fixedMatrix[ActivityType.HOME]!!.plusAssign(finderMatrices.homeWorkProbs * chainP)
                        }
                        continue
                    }
                    Pair(ActivityType.SCHOOL, ActivityType.HOME) -> {
                        if (varActivityType == ActivityType.SCHOOL) {
                            dependentMatrix[ActivityType.HOME]!!.plusAssign(mk.identity<Double>(omod.grid.size) * chainP)
                        } else {
                            fixedMatrix[ActivityType.HOME]!!.plusAssign(finderMatrices.homeSchoolProbs * chainP)
                        }
                        continue
                    }
                    else -> {}
                }
            }

            var nextActivity = chain[1]

            // Setup
            var cumMatrix = mk.identity<Double>(omod.grid.size)
            var previousProbsDepHome: D2Array<Double>
            var previousProbsDepHomeExVarActivity: D2Array<Double>
            when(startActivity) {
                ActivityType.HOME -> {
                    previousProbsDepHome = finderMatrices.homeP.diagonal()
                    previousProbsDepHomeExVarActivity = finderMatrices.homeP.diagonal()
                    fixedMatrix[nextActivity]!!.plusAssign(finderMatrices.homeP.diagonal() * chainP)
                }
                ActivityType.WORK -> {
                    previousProbsDepHome = finderMatrices.homeWorkProbs
                    previousProbsDepHomeExVarActivity = finderMatrices.homeWorkProbs
                    if (varActivityType == ActivityType.WORK) {
                        dependentMatrix[nextActivity]!!.plusAssign(cumMatrix * chainP)
                    } else {

                        fixedMatrix[nextActivity]!!.plusAssign(finderMatrices.homeWorkProbs * chainP)
                    }
                }
                ActivityType.SCHOOL -> {
                    previousProbsDepHome = finderMatrices.homeSchoolProbs
                    previousProbsDepHomeExVarActivity = finderMatrices.homeSchoolProbs
                    if (varActivityType == ActivityType.SCHOOL) {
                        dependentMatrix[nextActivity]!!.plusAssign(cumMatrix * chainP)
                    } else {
                        fixedMatrix[nextActivity]!!.plusAssign(finderMatrices.homeSchoolProbs * chainP)
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
                val transitionP = finderMatrices.transitionMatrix[transitionActivity]!!

                // Number of varActivityType activities that already occurred
                val nBefore = chain.take(next-1).count { it == varActivityType }

                if ((varActivityType in fixActivities) and (startActivity == varActivityType)) {
                    cumMatrix = cumMatrix.dot(transitionP)
                    dependentMatrix[nextActivity]!!.plusAssign(cumMatrix * chainP)
                } else if ((varActivityType !in fixActivities) and (nBefore >= 1) and (nextActivity != varActivityType)){
                    dependentMatrix[nextActivity]!!.plusAssign(previousProbsDepHomeExVarActivity * chainP)
                } else {
                    fixedMatrix[nextActivity]!!.plusAssign(previousProbsDepHome.dot(transitionP) * chainP)
                }

                previousProbsDepHome = previousProbsDepHome.dot(transitionP)
                if ((varActivityType !in fixActivities) and (activity != varActivityType)) {
                    previousProbsDepHomeExVarActivity = previousProbsDepHomeExVarActivity.dot(transitionP)
                }
            }
        }

        val m3rep = SGCompactMatrixRep(
            finderMatrices.homeP,
            dependentMatrix,
            fixedMatrix,
            finderMatrices.transitionMatrix,
            getPCar(),
            varActivityType
        )
        return m3rep
    }

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
        val m3rep = generateMarkovChainRep(activityType)

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
                    weight.addTerm(d, m3rep.transitionMatrix[m3rep.varActivityType]!![o, d])
                } else {
                    weight.addConstant(m3rep.transitionMatrix[m3rep.varActivityType]!![o, d])
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
                m3rep,
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
            m3rep,
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
                m3rep,
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

    private fun computeTransitionMatrices() : TransitionMatrixStore {
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

        // Transitions
        val transitionProbs = mutableMapOf<ActivityType,  D2Array<Double>>()
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
        return TransitionMatrixStore(homeProbs, transitionProbs)
    }

    fun <T, K> addFlexE(
        demandBuilder: TermBuilder<T, K>,
        nVars: Int,
        m3rep: SGCompactMatrixRep,
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
        val mCarP = m3rep.carP[transitionActivity]!!
        val mFixed = m3rep.fixedMatrix[flexActivity]!!

        if (flexActivity == m3rep.varActivityType) {
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
            val mTransition = m3rep.transitionMatrix[transitionActivity]!!
            val mDep = m3rep.dependentMatrix[flexActivity]!!
            val pHome = m3rep.homeP.flatten()

            for (o in 0 until n) {
                val s = demandBuilder.new(nVars)
                var cnst = 0.0

                for (a in 0 until n) {
                    cnst += mFixed[a, o]
                    for (b in 0 until n) {
                        val coeff = if (m3rep.varActivityType in fixActivities) {
                            pHome[a] * mDep[b, o]
                        } else {
                            mDep[a, b]
                        }

                        // Ignore very small terms
                        if (abs(coeff) <= irrelevancyThreshold) {
                            continue
                        }

                        if (m3rep.varActivityType in fixActivities) {
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
        m3rep: SGCompactMatrixRep,
        expectedTrips: List<List<T>>,
        varTransitionMatrix : List<List<K>>,
        relevantODs: Set<Pair<Int, Int>>,
        irrelevancyThreshold: Double
    ) {
        val n = omod.grid.size

        val carP = m3rep.carP[ActivityType.HOME]!!
        val fix = m3rep.fixedMatrix[ActivityType.HOME]!!.transpose().times(carP)

        val depExpr = if(m3rep.varActivityType in fixActivities) {
            val left = m3rep.dependentMatrix[ActivityType.HOME]!!.transpose()
            val right = m3rep.homeP.diagonal().transpose()
            builder.fromMatrixMult(
                nVars,
                varTransitionMatrix, left, right,
                relevantRCs = relevantODs, irrelevancyThreshold=irrelevancyThreshold
            )
        } else {
            val left = mk.identity<Double>(n)
            val right = m3rep.dependentMatrix[ActivityType.HOME]!!.transpose()
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
        m3rep: SGCompactMatrixRep,
        expectedTrips: List<List<T>>,
        varTransitionMatrix : List<List<K>>,
        relevantODs: Set<Pair<Int, Int>>,
        irrelevancyThreshold: Double,
        fixActivity: ActivityType
    ) {
        val n = omod.grid.size

        val carP = m3rep.carP[fixActivity]!!
        val fix = m3rep.fixedMatrix[fixActivity]!!.transpose()
            .dot(m3rep.transitionMatrix[fixActivity]!!)
            .times(carP)

        // Tour starting at var activity
        val pHome = m3rep.homeP.flatten()
        val dMatrixT = m3rep.dependentMatrix[fixActivity]!!.transpose()
        val varStartProbs = List(n) { builder.new(nVars) }
        for (col in 0 until n) {
            for (row in 0 until n) {
                builder.addVar(varStartProbs[col], varTransitionMatrix[row][col], pHome[row])
            }
        }

        // Tour not starting at var activity
        val depExpr = if (m3rep.varActivityType == fixActivity) {
            val left = m3rep.fixedMatrix[fixActivity]!!.transpose()
            val right = mk.identity<Double>(n)
            builder.fromMatrixMult(
                nVars,
                varTransitionMatrix, left, right,
                transpose = false,
                relevantRCs = relevantODs,
                irrelevancyThreshold=irrelevancyThreshold
            )
        } else if (m3rep.varActivityType in fixActivities) {
            val left = m3rep.dependentMatrix[fixActivity]!!.transpose()
            val right = m3rep.homeP
                .diagonal()
                .transpose()
                .dot(m3rep.transitionMatrix[fixActivity]!!)
            builder.fromMatrixMult(
                nVars,
                varTransitionMatrix, left, right,
                relevantRCs = relevantODs,
                irrelevancyThreshold=irrelevancyThreshold
            )
        } else {
            val left = mk.identity<Double>(n)
            val right = m3rep.dependentMatrix[fixActivity]!!
                .transpose()
                .dot(m3rep.transitionMatrix[fixActivity]!!)
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

                if (m3rep.varActivityType == fixActivity) {
                    builder.addTerm(expectedTrips[o][d], varStartProbs[d], dMatrixT[o][d] * carP[o, d])
                } else {
                    builder.addConstant(expectedTrips[o][d], fix[o,d])
                }
            }
        }
    }

    private data class ChainSegment(
        val chain: List<ActivityType>,
        val probability: Double
    )

    private data class TransitionMatrixStore (
        val homeP: D2Array<Double>,
        val transitionMatrix: Map<ActivityType,  D2Array<Double>>,
    ) {
        // Home <-> Work, Work
        val homeWorkProbs = homeP.diagonal().dot( transitionMatrix[ActivityType.WORK]!! )

        // Home <-> School, School
        val homeSchoolProbs = homeP.diagonal().dot( transitionMatrix[ActivityType.SCHOOL]!! )
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
        val homeP: D2Array<Double>,
        val dependentMatrix: Map<ActivityType, D2Array<Double>>,
        val fixedMatrix: Map<ActivityType,  D2Array<Double>>,
        val transitionMatrix: Map<ActivityType,  D2Array<Double>>,
        val carP: Map<ActivityType,  D2Array<Double>>,
        val varActivityType: ActivityType
    )

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
    private fun getExpectedCountPerAgent(m3rep: SGCompactMatrixRep) : D2Array<Double> {
        // Result: Expected trip count on each od-paid of one agent on one day
        val expectedCountPerAgent = mk.zeros<Double>(omod.grid.size, omod.grid.size)

        val varTMatrix = m3rep.transitionMatrix[m3rep.varActivityType]!!

        // Flex
        for (flexActivity in listOf(ActivityType.OTHER, ActivityType.SHOPPING, ActivityType.BUSINESS)) {
            val transitionActivity = if (flexActivity == ActivityType.BUSINESS) {
                ActivityType.OTHER
            } else {
                flexActivity
            }
            val carP = m3rep.carP[transitionActivity]!!
            val fix = m3rep.fixedMatrix[flexActivity]!!

            val dep = if (m3rep.varActivityType in fixActivities) {
                m3rep.homeP.diagonal().dot(varTMatrix).dot(m3rep.dependentMatrix[flexActivity]!!)
            } else if (flexActivity == m3rep.varActivityType) {
                mk.zeros<Double>(omod.grid.size, omod.grid.size)
            } else {
                m3rep.dependentMatrix[flexActivity]!!.dot(varTMatrix)
            }
            val total = mk.ones<Double>(omod.grid.size).expandDims(0).asDNArray()
                .asD2Array().dot(dep.plus(fix))
                .diagonal().dot(m3rep.transitionMatrix[transitionActivity]!!)
            expectedCountPerAgent.plusAssign(total.times(carP))
        }

        // Home
        for (fixActivity in listOf(ActivityType.HOME)) {
            val carP = m3rep.carP[fixActivity]!!

            val dep = if (m3rep.varActivityType in fixActivities) {
                m3rep.homeP.diagonal().dot(varTMatrix).dot(m3rep.dependentMatrix[fixActivity]!!)
            } else {
                m3rep.dependentMatrix[fixActivity]!!.dot(varTMatrix)
            }
            val fix = m3rep.fixedMatrix[fixActivity]!!
            val total = dep.plus(fix).transpose()

            expectedCountPerAgent.plusAssign(total.times(carP))
        }

        // School
        for (fixActivity in listOf(ActivityType.SCHOOL)) {
            val carP = m3rep.carP[fixActivity]!!
            val fix = m3rep.fixedMatrix[fixActivity]!!

            if (m3rep.varActivityType == ActivityType.SCHOOL) {
                val schoolProbs = m3rep.homeP.dot( m3rep.transitionMatrix[ActivityType.SCHOOL]!! )
                val dep = schoolProbs.diagonal().dot( m3rep.dependentMatrix[fixActivity]!! ).transpose()
                val total = dep.plus( fix.transpose().dot( m3rep.transitionMatrix[fixActivity]!!) )
                expectedCountPerAgent.plusAssign(total.times(carP))
            } else {
                val dep = if (m3rep.varActivityType in fixActivities) {
                    m3rep.homeP.diagonal().dot(varTMatrix).dot(m3rep.dependentMatrix[fixActivity]!!)
                } else {
                    m3rep.dependentMatrix[fixActivity]!!.dot(varTMatrix)
                }
                val total = dep.plus(fix).transpose().dot(m3rep.transitionMatrix[fixActivity]!!)
                expectedCountPerAgent.plusAssign(total.times(carP))
            }
        }

        // Work
        for (fixActivity in listOf(ActivityType.WORK)) {
            val carP = m3rep.carP[fixActivity]!!
            val fix = m3rep.fixedMatrix[fixActivity]!!

            if (m3rep.varActivityType == ActivityType.WORK) {
                val workProbs = m3rep.homeP.dot( m3rep.transitionMatrix[ActivityType.WORK]!! )
                val dep = workProbs.diagonal().dot( m3rep.dependentMatrix[fixActivity]!! ).transpose()
                val total = dep.plus( fix.transpose().dot( m3rep.transitionMatrix[fixActivity]!!) )
                expectedCountPerAgent.plusAssign(total.times(carP))
            } else {
                val dep = if (m3rep.varActivityType in fixActivities) {
                    m3rep.homeP.diagonal().dot(varTMatrix).dot(m3rep.dependentMatrix[fixActivity]!!)
                } else {
                    m3rep.dependentMatrix[fixActivity]!!.dot(varTMatrix)
                }
                val total = dep.plus(fix).transpose().dot(m3rep.transitionMatrix[fixActivity]!!)
                expectedCountPerAgent.plusAssign(total.times(carP))
            }
        }

        return expectedCountPerAgent
    }

    private fun debugEval(
        m3rep: SGCompactMatrixRep,
        sensors: List<TrafficSensor>,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    ) {
        val expected = getExpectedCountPerAgent(m3rep)
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


