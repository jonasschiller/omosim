package de.uniwuerzburg.omod.calibration

import com.gurobi.gurobi.*
import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.CalibrationConstants.mcSamples
import de.uniwuerzburg.omod.calibration.algorithms.BFGS
import de.uniwuerzburg.omod.calibration.differentiablemodel.*
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
import org.jetbrains.kotlinx.multik.ndarray.operations.plusAssign
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.locationtech.jts.geom.Coordinate
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.pow

class MetaModel private constructor(
    val omod: Omod
) {
    private val modeChoiceCalibration = ModeChoiceCalibration()
    private val fixActivities = setOf(ActivityType.WORK, ActivityType.SCHOOL)

    companion object {
        fun build(omod: Omod): MetaModel? {
            if (omod.destinationFinder !is DestinationFinderDefault) {
                logger.warn(
                    "MetaModel is not valid for the destination finder: " +
                            omod.destinationFinder.javaClass.simpleName
                )
                return null
            }
            if (omod.activityGenerator !is ActivityGeneratorDefault) {
                logger.warn(
                    "MetaModel is not valid for the activity generator finder: " +
                            omod.activityGenerator.javaClass.simpleName
                )
                return null
            }
            return MetaModel(omod)
        }
    }

    fun getDiffModel(
        activityType: ActivityType,
        sensors: List<TrafficSensor>,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    ): DifferentiableModel {
        val m3rep = generateMatrixRep(activityType)

        println("Building diff model...")
        val (model, _) = buildDiffModel(m3rep, affectedSensors, sensors)
        return model
    }

    fun getDiffModelSimCounts(
        activityType: ActivityType,
        sensors: List<TrafficSensor>,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    ): DifferentiableModelMultiOut {
        val m3rep = generateMatrixRep(activityType)

        println("Building diff model multi out...")
        val (_, simCounts) = buildDiffModel(m3rep, affectedSensors, sensors)

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

    fun calibrateK1(
        activityType: ActivityType,
        sensors: List<TrafficSensor>,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    ): List<Double> {
        val m3rep = generateMatrixRep(activityType)

        println("Building diff model...")
        val (model, simCount) = buildDiffModel(m3rep, affectedSensors, sensors)

        var parameters = DoubleArray(omod.grid.size - 1) { 1.0 }
        parameters = (BFGS.run(model, parameters).toList() + listOf(1.0)).toDoubleArray()

        println("_".repeat(20*4 + 5*3))
        println("${"Sensor".padEnd(20)} | \t" +
                "${"T".padEnd(20)} | \t" +
                "${"Flow AltOpt".padEnd(20)} | \t" +
                "Flow Measured".padEnd(20)
        )
        println("_".repeat(20) +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20)  +
                " | \t" + "_".repeat(20))
        var oval = 0.0
        for ((i, sensor) in sensors.withIndex()) {
            for (seg in listOf(Pair(0, T))) {
                var optVal = 0.0
                var mVal = 0.0
                for (t in seg.first until seg.second) {
                    optVal += simCount[sensor]!![t].evaluate(parameters)
                    mVal += sensors[i].measuredFlow[t]
                }
                println(
                    "${sensors[i].name.padEnd(20)} | \t" +
                    "${(seg.first.toString() + "-" + seg.second).padEnd(20)} | \t" +
                    "${optVal.toString().padEnd(20)} | \t" +
                    mVal.toString().padEnd(20)
                )
            }
            for ( t in 0 until T) {
                val optVal = simCount[sensor]!![t].evaluate(parameters)
                oval += (sensors[i].measuredFlow[t] - optVal) * (sensors[i].measuredFlow[t] - optVal)
            }
        }
        println("MSE: ${oval /sensors.size}")

        return parameters.toList()
    }

    fun calibrateMatrix(
            activityType: ActivityType,
            sensors: List<TrafficSensor>,
            affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    ) : D2Array<Double>? {
        println("GENERAL META MODEL OPTIMIZE")
        val m3rep = generateMatrixRep(activityType)

        return optimizeTMatrix(m3rep, affectedSensors, sensors)
    }

    private fun generateMatrixRep(activityType: ActivityType) : MetaModelMatrixRep {
        require(activityType != ActivityType.HOME) { "Meta model dependent on home coefficients not implemented!" }

        return if (activityType in fixActivities) {
            generateMatrixRepForFixed(activityType)
        } else {
            generateMatrixRepForFlex(activityType)
        }
    }

    private fun generateMatrixRepForFixed(varActivityType: ActivityType) : MetaModelMatrixRep {
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
            when(startActivity) {
                ActivityType.HOME -> {
                    previousProbsDepHome = finderMatrices.homeP.diagonal()
                    fixedMatrix[nextActivity]!!.plusAssign(finderMatrices.homeP.diagonal() * chainP)
                }
                ActivityType.WORK -> {
                    previousProbsDepHome = finderMatrices.homeWorkProbs
                    if (varActivityType == ActivityType.WORK) {
                        dependentMatrix[nextActivity]!!.plusAssign(cumMatrix * chainP)
                    } else {
                        fixedMatrix[nextActivity]!!.plusAssign(finderMatrices.homeWorkProbs * chainP)
                    }
                }
                ActivityType.SCHOOL -> {
                    previousProbsDepHome = finderMatrices.homeSchoolProbs
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

                if (startActivity == varActivityType) {
                    cumMatrix = cumMatrix.dot(transitionP)
                    dependentMatrix[nextActivity]!!.plusAssign(cumMatrix * chainP)
                } else {
                    fixedMatrix[nextActivity]!!.plusAssign(previousProbsDepHome.dot(transitionP) * chainP)
                }

                previousProbsDepHome = previousProbsDepHome.dot(transitionP)
            }
        }

        val m3rep = MetaModelMatrixRep(
            finderMatrices.homeP,
            dependentMatrix,
            fixedMatrix,
            finderMatrices.transitionMatrix,
            getPCar(),
            varActivityType
        )
        return m3rep
    }

    private fun generateMatrixRepForFlex(varActivityType: ActivityType) : MetaModelMatrixRep {
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
                        fixedMatrix[ActivityType.HOME]!!.plusAssign(finderMatrices.homeWorkProbs * chainP)
                        continue
                    }
                    Pair(ActivityType.SCHOOL, ActivityType.HOME) -> {
                        fixedMatrix[ActivityType.HOME]!!.plusAssign(finderMatrices.homeSchoolProbs * chainP)
                        continue
                    }
                    else -> {}
                }
            }

            var nextActivity = chain[1]

            // Setup
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
                    fixedMatrix[nextActivity]!!.plusAssign(finderMatrices.homeWorkProbs * chainP)
                }
                ActivityType.SCHOOL -> {
                    previousProbsDepHome = finderMatrices.homeSchoolProbs
                    previousProbsDepHomeExVarActivity = finderMatrices.homeSchoolProbs
                    fixedMatrix[nextActivity]!!.plusAssign(finderMatrices.homeSchoolProbs * chainP)
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

                val nBefore = chain.take(next-1).count { it == varActivityType }
                if ((nBefore >= 1) and (nextActivity != varActivityType)) {
                    dependentMatrix[nextActivity]!!.plusAssign(previousProbsDepHomeExVarActivity * chainP)
                } else {
                    fixedMatrix[nextActivity]!!.plusAssign(previousProbsDepHome.dot(transitionP) * chainP)
                }

                previousProbsDepHome = previousProbsDepHome.dot(transitionP)
                if (activity != varActivityType) {
                    previousProbsDepHomeExVarActivity = previousProbsDepHomeExVarActivity.dot(transitionP)
                }
            }
        }

        val m3rep = MetaModelMatrixRep(
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

    private fun getUniqueChainSegments() : List<FixedSegment> {
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
        return uniqueSegments
    }

    @Suppress("SameParameterValue")
    private fun monteCarloTripStartDistribution(n: Int) : Map<ActivityType, Array<Double>> {
        val distr = ActivityType.entries.associateWith {
            Array(T) { 0.0 }
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
            distr[key] = arr.map {
                val sum = arr.sum()
                if (sum == 0.0) {
                    0.0
                } else {
                    it / sum
                }
            }.toTypedArray()
        }

        return distr
    }

    private fun buildDiffModel(
        m3rep: MetaModelMatrixRep,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        sensors: List<TrafficSensor>,
        irrelevantFactorThreshold: Double = (1/omod.grid.size.toDouble()).pow(1.5)
    ) : Pair<DifferentiableModel, Map<TrafficSensor, List<LinearTerm>>> {
        val n = omod.grid.size
        val totalPop = omod.buildings.sumOf { it.population }
        println("Building diff model with grid size: $n")

        val relevantODs = determineRelevantODs(affectedSensors)

        // Init diff model
        val diffModel = DifferentiableModel(omod.grid.size - 1)

        // Create demand matrix dependent on the variable transition matrix: demand(o, d | M)
        val demand = ActivityType.entries.associateWith {
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

        // Demand with flexible destination
        for (activity in listOf( ActivityType.OTHER, ActivityType.SHOPPING, ActivityType.BUSINESS)) {
            addFlexDemand(
                DMDemandBuilder,
                diffModel.nVars,
                m3rep,
                demand[activity]!!,
                varTransitionMatrix,
                relevantODs,
                irrelevantFactorThreshold,
                activity
            )
        }

        // Demand for home
        addHomeDemand(DMDemandBuilder, diffModel.nVars, m3rep, demand[ActivityType.HOME]!!, varTransitionMatrix, relevantODs, irrelevantFactorThreshold)

        // School and Work
        for (activity in listOf( ActivityType.SCHOOL, ActivityType.WORK)) {
            addFixDemand(
                DMDemandBuilder,
                diffModel.nVars,
                m3rep,
                demand[activity]!!,
                varTransitionMatrix,
                relevantODs,
                irrelevantFactorThreshold,
                activity
            )
        }

        // Temporal trip distribution
        val tripStartDistr = monteCarloTripStartDistribution( mcSamples )

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
                                    demand[activity]!![o][d], totalPop * tripStartDistr[activity]!![t]
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
        return diffModel to simCount
    }

    private fun optimizeTMatrix(
        m3rep: MetaModelMatrixRep,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        sensors: List<TrafficSensor>,
        irrelevantFactorThreshold: Double = (1/omod.grid.size.toDouble()).pow(1.5)
    ) : D2Array<Double>? {
        val n = omod.grid.size
        val totalPop = omod.buildings.sumOf { it.population }
        println("Optimization 1 start with grid size: ${n}")

        val relevantODs = determineRelevantODs(affectedSensors)

        try {
            // Setup
            val env = GRBEnv()
            val model = GRBModel(env)

            // Create demand matrix dependent on the variable transition matrix: demand(o, d | M)
            val demand = ActivityType.entries.associateWith {
                List(n) {
                    List(n) {
                        GRBLinExpr()
                    }
                }
            }

            // Transition matrix
            val varTransitionMatrix = List<List<GRBVar>>(n) { o ->
                model.addVars(
                    DoubleArray(n) { 0.0 },
                    DoubleArray(n) { 1.0 },
                    DoubleArray(n) { 0.0 },
                    CharArray(n) { GRB.CONTINUOUS },
                    Array(n) { d -> "W_${o}_$d" }
                ).toList()
            }

            // Ensure that each row of W is a proper probability distribution
            for (o in 0 until n) {
                val rowSum = GRBLinExpr()
                for (d in 0 until n) {
                    rowSum.addTerm(1.0, varTransitionMatrix[o][d])
                }
                model.addConstr(rowSum, GRB.EQUAL, 1.0, "ProbCondition")
            }

            // Demand with flexible destination
            for (activity in listOf(ActivityType.OTHER, ActivityType.SHOPPING, ActivityType.BUSINESS)) {
                addFlexDemand(
                    GRBDemandBuilder,
                    n,
                    m3rep,
                    demand[activity]!!,
                    varTransitionMatrix,
                    relevantODs,
                    irrelevantFactorThreshold,
                    activity
                )
            }

            // Demand for home
            addHomeDemand(
                GRBDemandBuilder,
                n,
                m3rep,
                demand[ActivityType.HOME]!!,
                varTransitionMatrix,
                relevantODs,
                irrelevantFactorThreshold
            )

            // School and Work
            for (activity in listOf(ActivityType.SCHOOL, ActivityType.WORK)) {
                addFixDemand(
                    GRBDemandBuilder,
                    n,
                    m3rep,
                    demand[activity]!!,
                    varTransitionMatrix,
                    relevantODs,
                    irrelevantFactorThreshold,
                    activity
                )
            }

            // Temporal trip distribution
            val tripStartDistr = monteCarloTripStartDistribution(mcSamples)

            // Simulated Traffic Counts
            val sensorCountExpr = mutableMapOf<TrafficSensor, List<GRBLinExpr>>()
            for (sensor in sensors) {
                sensorCountExpr[sensor] = List(T) { GRBLinExpr() }
            }
            for ((o, origin) in omod.grid.withIndex()) {
                for ((d, destination) in omod.grid.withIndex()) {
                    // Temporary variables to avoid GRBLinExpr().multAdd(). multAdd() leads to explosion in memory usage.
                    val demandVars = mutableMapOf<ActivityType, GRBVar> ()
                    for (activity in ActivityType.entries) {
                        val v = model.addVar( 0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "demand")
                        model.addConstr( demand[activity]!![o][d], GRB.EQUAL, v,"demandEq")
                        demandVars[activity] = v
                    }

                    val od = Pair(origin, destination)
                    if (od in affectedSensors) {
                        val affected = affectedSensors[od]!!
                        for (sensor in affected) {
                            for (t in 0 until T) {
                                for (activity in ActivityType.entries) {
                                    sensorCountExpr[sensor]!![t]
                                        .addTerm(totalPop * tripStartDistr[activity]!![t], demandVars[activity]!!)
                                }
                            }
                        }
                    }
                }
            }
            val sensorSimCount = List(T) {
                model.addVars(
                    DoubleArray(sensors.size) { 0.0 },
                    null,
                    DoubleArray(sensors.size) { 0.0 },
                    CharArray(sensors.size) { GRB.CONTINUOUS },
                    Array(sensors.size) { "" }
                )
            }

            for ((i, sensor) in sensors.withIndex()) {
                for (t in 0 until T) {
                    model.addConstr(
                        sensorCountExpr[sensor]!![t],
                        GRB.EQUAL,
                        sensorSimCount[t][i]!!,
                        "cnteq"
                    )
                }
            }

            // Objective
            val obj = GRBQuadExpr()
            for ((i, sensor) in sensors.withIndex()) {
                for (t in 0 until T) {
                    // (Sm - Ss)^2 = Sm^2 - 2SmSs + Ss^2
                    obj.addConstant(sensor.measuredFlow[t] * sensor.measuredFlow[t])
                    obj.addTerm(-2 * sensor.measuredFlow[t], sensorSimCount[t][i])
                    obj.addTerm(1.0, sensorSimCount[t][i], sensorSimCount[t][i])
                }
            }
            model.setObjective(obj, GRB.MINIMIZE)

            model.optimize()

            var status = model[GRB.IntAttr.Status]

            if (status == GRB.Status.INF_OR_UNBD) {
                model[GRB.IntParam.Presolve] = 0
                model.optimize()
                status = model[GRB.IntAttr.Status]
            }

            if (status == GRB.Status.OPTIMAL) {
                // Print result
                val oval = model[GRB.DoubleAttr.ObjVal]
                println("Optimal objective: $oval")
            } else if (status == GRB.Status.INFEASIBLE) {
                println("Model is infeasible")
            } else if (status == GRB.Status.UNBOUNDED) {
                println("Model is unbounded")
            } else {
                println(
                    "Optimization was stopped with status = $status"
                )
            }

            // Ideal transition matrix
            val result = mk.ones<Double>(omod.grid.size, omod.grid.size)
            for(o in omod.grid.indices) {
                for (d in omod.grid.indices) {
                    result[o, d] = varTransitionMatrix[o][d].get(GRB.DoubleAttr.X)
                }
            }

            // Dispose of model and environment
            model.dispose()
            env.dispose()
            return result
        } catch (e: GRBException) {
            println(
                ("Error code: " + e.errorCode + ". " + e.message)
            )
        }
        return null
    }

    private fun <T, K> fromMatrixSandwichT(
        demandBuilder: DemandBuilder<T, K>,
        nVars: Int,
        mVar: List<List<K>>,
        left: D2Array<Double>,
        right: D2Array<Double>,
        transpose: Boolean = true,
        relevantRCs: Set<Pair<Int, Int>>? = null,
        irrelevantFactorThreshold: Double
    ) : List<List<T>> {
        val n = left.shape[0] // Assume all matrices are square and same size

        val result = List(n) {
            List(n) {
                demandBuilder.createSum(nVars)
            }
        }

        for (row in 0 until n) {
            for (col in 0 until  n) {
                if (relevantRCs != null) {
                    if (Pair(row, col) !in relevantRCs) {
                        continue
                    }
                }

                val activeEntry = result[row][col]

                for (i in 0 until n) {
                    if (right[i, col] == 0.0) { continue }
                    for (j in 0 until n) {
                        if (left[row, j] == 0.0) { continue }
                        val coeff = left[row, j] * right[i, col]

                        if (abs(coeff) <= irrelevantFactorThreshold) {
                            continue
                        }

                        if (transpose) {
                            demandBuilder.addTermToSum(activeEntry, mVar[i][j], coeff)
                        } else {
                            demandBuilder.addTermToSum(activeEntry, mVar[j][i], coeff)
                        }
                    }
                }
            }
        }
        return result
    }

    private fun computeTransitionMatrices() : FinderMatrices {
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
        return FinderMatrices(homeProbs, transitionProbs)
    }

    private fun <T, K> addFlexDemand(
        demandBuilder: DemandBuilder<T, K>,
        nVars: Int,
        m3rep: MetaModelMatrixRep,
        demand: List<List<T>>,
        varTransitionMatrix : List<List<K>>,
        relevantODs: Set<Pair<Int, Int>>,
        irrelevantFactorThreshold: Double,
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
                    demandBuilder.addTerm(demand[o][d], varTransitionMatrix[o][d], cnst * mCarP[o, d])
                }
            }
        } else {
            val mTransition = m3rep.transitionMatrix[transitionActivity]!!
            val mDep = m3rep.dependentMatrix[flexActivity]!!
            val pHome = m3rep.homeP.flatten()

            for (o in 0 until n) {
                val s = demandBuilder.createSum(nVars)
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
                        if (abs(coeff) <= irrelevantFactorThreshold) {
                            continue
                        }

                        if (m3rep.varActivityType in fixActivities) {
                            demandBuilder.addTermToSum(s, varTransitionMatrix[a][b], coeff)
                        } else {
                            demandBuilder.addTermToSum(s, varTransitionMatrix[b][o], coeff)
                        }
                    }
                }
                demandBuilder.addConstToSum(s, cnst)

                for (d in 0 until n) {
                    if (Pair(o, d) !in relevantODs) { continue }
                    val t = mTransition[o, d] * mCarP[o, d]
                    demandBuilder.addSum(demand[o][d], s, t)
                }
            }
        }

    }

    private fun <T, K> addHomeDemand(
        demandBuilder: DemandBuilder<T, K>,
        nVars: Int,
        m3rep: MetaModelMatrixRep,
        demand: List<List<T>>,
        varTransitionMatrix : List<List<K>>,
        relevantODs: Set<Pair<Int, Int>>,
        irrelevantFactorThreshold: Double
    ) {
        val n = omod.grid.size

        val carP = m3rep.carP[ActivityType.HOME]!!
        val fix = m3rep.fixedMatrix[ActivityType.HOME]!!.transpose().times(carP)

        val depExpr = if(m3rep.varActivityType in fixActivities) {
            val left = m3rep.dependentMatrix[ActivityType.HOME]!!.transpose()
            val right = m3rep.homeP.diagonal().transpose()
            fromMatrixSandwichT(
                demandBuilder,
                nVars,
                varTransitionMatrix, left, right,
                relevantRCs = relevantODs, irrelevantFactorThreshold=irrelevantFactorThreshold
            )
        } else {
            val left = mk.identity<Double>(n)
            val right = m3rep.dependentMatrix[ActivityType.HOME]!!.transpose()
            fromMatrixSandwichT(
                demandBuilder,
                nVars,
                varTransitionMatrix, left, right,
                relevantRCs = relevantODs,
                irrelevantFactorThreshold=irrelevantFactorThreshold
            )
        }

        for (o in 0 until n) {
            for (d in 0 until n) {
                if (Pair(o, d) !in relevantODs) { continue }
                demandBuilder.addSum(demand[o][d], depExpr[o][d], carP[o, d])
                demandBuilder.addConstant(demand[o][d], fix[o, d])
            }
        }

    }

    private fun <T, K> addFixDemand(
        demandBuilder: DemandBuilder<T, K>,
        nVars: Int,
        m3rep: MetaModelMatrixRep,
        demand: List<List<T>>,
        varTransitionMatrix : List<List<K>>,
        relevantODs: Set<Pair<Int, Int>>,
        irrelevantFactorThreshold: Double,
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
        val varStartProbs = List(n) { demandBuilder.createSum(nVars) }
        for (col in 0 until n) {
            for (row in 0 until n) {
                demandBuilder.addTermToSum(varStartProbs[col], varTransitionMatrix[row][col], pHome[row])
            }
        }

        // Tour not starting at var activity
        val depExpr = if (m3rep.varActivityType == fixActivity) {
            val left = m3rep.fixedMatrix[fixActivity]!!.transpose()
            val right = mk.identity<Double>(n)
            fromMatrixSandwichT(
                demandBuilder,
                nVars,
                varTransitionMatrix, left, right,
                transpose = false,
                relevantRCs = relevantODs,
                irrelevantFactorThreshold=irrelevantFactorThreshold
            )
        } else if (m3rep.varActivityType in fixActivities) {
            val left = m3rep.dependentMatrix[fixActivity]!!.transpose()
            val right = m3rep.homeP
                .diagonal()
                .transpose()
                .dot(m3rep.transitionMatrix[fixActivity]!!)
            fromMatrixSandwichT(
                demandBuilder,
                nVars,
                varTransitionMatrix, left, right,
                relevantRCs = relevantODs,
                irrelevantFactorThreshold=irrelevantFactorThreshold
            )
        } else {
            val left = mk.identity<Double>(n)
            val right = m3rep.dependentMatrix[fixActivity]!!
                .transpose()
                .dot(m3rep.transitionMatrix[fixActivity]!!)
            fromMatrixSandwichT(
                demandBuilder,
                nVars,
                varTransitionMatrix, left, right,
                relevantRCs = relevantODs,
                irrelevantFactorThreshold=irrelevantFactorThreshold
            )
        }

        for (o in 0 until n) {
            for (d in 0 until n) {
                if (Pair(o, d) !in relevantODs) { continue }
                demandBuilder.addSum(demand[o][d], depExpr[o][d], carP[o, d])

                if (m3rep.varActivityType == fixActivity) {
                    demandBuilder.addSum(demand[o][d], varStartProbs[d], dMatrixT[o][d] * carP[o, d])
                } else {
                    demandBuilder.addConstant(demand[o][d], fix[o,d])
                }
            }
        }
    }

    private data class FixedSegment(
        val chain: List<ActivityType>,
        val probability: Double
    )

    private data class FinderMatrices (
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
    private data class MetaModelMatrixRep (
        val homeP: D2Array<Double>,
        val dependentMatrix: Map<ActivityType, D2Array<Double>>,
        val fixedMatrix: Map<ActivityType,  D2Array<Double>>,
        val transitionMatrix: Map<ActivityType,  D2Array<Double>>,
        val carP: Map<ActivityType,  D2Array<Double>>,
        val varActivityType: ActivityType
    )

    private fun determineRelevantODs(
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
     * Wrapper Object that allows creating demand matrix for use in both DifferentialModel and Gurobi
     */
    private interface DemandBuilder<T,K>  {
        fun addTerm(demand: T, v: K, coefficient: Double) {}
        fun addConstant(demand: T, constant: Double) {}
        fun addSum(demand: T, sum: T, coefficient: Double) {}
        fun createSum(nVars: Int) : T
        fun addTermToSum(s: T, v: K, coefficient: Double)
        fun addConstToSum(s: T, const: Double)
    }

    private object DMDemandBuilder: DemandBuilder<LinearTerm, Term> {
        override fun addTerm(demand: LinearTerm, v: Term, coefficient: Double) {
            demand.addTerm(v, coefficient)
        }

        override fun addConstant(demand: LinearTerm, constant: Double) {
            demand.addConstant(constant)
        }

        override fun addSum(demand: LinearTerm, sum: LinearTerm, coefficient: Double) {
            demand.addTerm(sum, coefficient)
        }

        override fun createSum(nVars: Int): LinearTerm {
            return LinearTerm(nVars)
        }

        override fun addTermToSum(s: LinearTerm, v: Term, coefficient: Double) {
            s.addTerm(v, coefficient)
        }

        override fun addConstToSum(s: LinearTerm, const: Double) {
            s.addConstant(const)
        }
    }


    private object GRBDemandBuilder: DemandBuilder<GRBLinExpr, GRBVar> {
        override fun addTerm(demand: GRBLinExpr, v: GRBVar, coefficient: Double) {
            demand.addTerm(coefficient, v)
        }

        override fun addConstant(demand: GRBLinExpr, constant: Double) {
            demand.addConstant(constant)
        }

        override fun addSum(demand: GRBLinExpr, sum: GRBLinExpr, coefficient: Double) {
            demand.multAdd(coefficient, sum)
        }

        override fun createSum(nVars: Int): GRBLinExpr {
            return GRBLinExpr()
        }

        override fun addTermToSum(s: GRBLinExpr, v: GRBVar, coefficient: Double) {
            s.addTerm(coefficient, v)
        }

        override fun addConstToSum(s: GRBLinExpr, const: Double) {
            s.addConstant(const)
        }
    }
}

private inline fun <reified T : Any> D2Array<T>.diagonal() : D2Array<T> {
    require(this.shape[0] == 1)

    val diagonal  = mk.zeros<T>(this.size, this.size)
    for ( i in 0 until this.size) {
        diagonal[i, i] = this[0,i]
    }
    return diagonal
}


