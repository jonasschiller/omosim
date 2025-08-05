package de.uniwuerzburg.omod.calibration

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
import smile.math.BFGS
import kotlin.math.abs
import kotlin.math.pow
import kotlin.time.measureTime
import kotlin.time.measureTimedValue

class DefaultMetaModel(
    val omod: Omod
) : MetaModel {
    private val modeChoiceCalibration = ModeChoiceCalibration()
    private val fixActivities = setOf(ActivityType.WORK, ActivityType.SCHOOL)

    override fun build(omod: Omod): DefaultMetaModel? {
        if (omod.destinationFinder !is DestinationFinderDefault) {
            logger.warn(
                "DefaultMetaModel is not valid for the destination finder: " +
                "${omod.destinationFinder.javaClass.simpleName}"
            )
            return null
        }
        if (omod.activityGenerator !is ActivityGeneratorDefault) {
            logger.warn(
                "DefaultMetaModel is not valid for the activity generator finder: " +
                "${omod.activityGenerator.javaClass.simpleName}"
            )
            return null
        }
        return DefaultMetaModel(omod)
    }

    override fun calibrateK1(
        activityType: ActivityType,
        sensors: List<TrafficSensor>,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    ): List<Double> {
        val m3rep = generateMatrixRep(activityType)
        val exp = getExpectedCountPerAgent(m3rep)

        println("GENERAL META MODEL")
        eval(exp, sensors, affectedLinks)
        println("Building diff model...")
        val (model, simCount) = buildDiffModel(m3rep, affectedLinks, sensors)

        var parameters = DoubleArray(omod.grid.size - 1) { 1.0 }
        parameters = (lbfgs(model, parameters).toList() + listOf(1.0)).toDoubleArray()
        //parameters = gradDescent(model, parameters)

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
            val optVal = simCount[sensor]!!.evaluate(parameters)
            println(
                "${sensors[i].name.padEnd(20)} | \t" +
                        "${optVal.toString().padEnd(20)} | \t" +
                        sensors[i].measuredFlow.toString().padEnd(20)
            )
            myobjval += (sensors[i].measuredFlow - optVal) * (sensors[i].measuredFlow - optVal)
        }
        println("MSE: ${myobjval /sensors.size}")

        return parameters.toList()
    }

    fun lbfgs(model: DifferentiableModel, vals: DoubleArray) : DoubleArray {
        println("LBFGS-B")
        println("Start: ${model.evaluate(vals)}")
        val l = DoubleArray(model.nVars){0.0}
        val u = DoubleArray(model.nVars){1e3}
        val solution = BFGS.minimize(model, 5, vals, l, u, 1e-5, 10)
        println("Solution: $solution")
        println("Confirm: ${model.evaluate(vals)}")
        return vals
    }

    fun gradDescent(model: DifferentiableModel, vals: DoubleArray, iterations: Int = 1000, lr: Double = 1.0e-9) :
            DoubleArray
    {
        val currentValues = vals.copyOf()
        val gradients = DoubleArray(vals.size) { 0.0 }

        val evaltime = measureTime {
            val loss = model.evaluate(currentValues)
            println("Start loss: $loss")
        }
        println("Eval time: $evaltime")

        for (i in 0 until iterations) {
            val (loss, time) = measureTimedValue {
                // Determine gradients
                for ((i, value) in currentValues.withIndex()) {
                    gradients[i] = model.gradient(i, currentValues)
                }

                for ((i, value) in currentValues.withIndex()) {
                    currentValues[i] -= lr * gradients[i]
                }
                val loss = model.evaluate(currentValues)
                loss
            }

            print("Iteration $i, vals ${currentValues.toList().toString()}, loss: $loss \t took $time \r")
        }
        val loss = model.evaluate(currentValues)
        println("Solution loss: $loss")
        return currentValues
    }

    fun eval(
        expected: D2Array<Double>,
        sensors: List<TrafficSensor>,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    ) {
        val totalPop = omod.buildings.sumOf { it.population }

        val simCount = sensors.associateWith { 0.0 }.toMutableMap()
        for ((o, origin) in omod.grid.withIndex()) {
            for ((d, destination) in omod.grid.withIndex()) {
                val odPair = Pair(origin, destination)
                if (odPair in affectedLinks) {
                    val affected = affectedLinks[odPair]!!
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
            mse += (sensor.measuredFlow - simFlow).pow(2)

            allFlows[sensor] = simFlow
        }
        mse /= sensors.size

        println("MSE: $mse")
        println("------------------------------")
        println("Sensor | \t Flow OMOD | \t Flow Measured")
        for ((i, flow) in simCount.values.withIndex()) {
            println("${sensors[i].name} | \t $flow | \t ${sensors[i].measuredFlow}")
        }

        // TODO Turn this test into a unit test
    }

    private fun getExpectedCountPerAgent(m3rep: MetaModelMatrixRep) : D2Array<Double> {
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

    private fun generateMatrixRep(activityType: ActivityType) : MetaModelMatrixRep {
        require(activityType != ActivityType.HOME) { "Meta model dependent on home coefficients not implemented!" }

        return if (activityType in fixActivities) {
            generateMatrixRepForFixed(activityType)
        } else {
            generateMatrixRepForFlex(activityType)
        }
    }

    private fun generateMatrixRepForFixed(varActivityType: ActivityType) : MetaModelMatrixRep {
        // TODO School and Work can be combined see tag: EndOfExplorativeTesting1, class FixGradDescent
        val dependentMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(omod.grid.size, omod.grid.size) }
        val fixedMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(omod.grid.size, omod.grid.size) }

        val finderMatrices = computeTransitionMatrices()

        // Determine od pair probabilities
        // TODO time
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
        // TODO if this version has staying power. Integrate into flex version.
        val dependentMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(omod.grid.size, omod.grid.size) }
        val fixedMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(omod.grid.size, omod.grid.size) }

        val finderMatrices = computeTransitionMatrices()

        // Determine od pair probabilities
        // TODO time
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
        // TODO Multiple days
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
        return uniqueSegments
    }

    private fun buildDiffModel(
        m3rep: MetaModelMatrixRep,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        sensors: List<TrafficSensor>,
        irrelevantFactorThreshold: Double = (1/omod.grid.size.toDouble()).pow(1.5)
    ) : Pair<DifferentiableModel, Map<TrafficSensor, LinearTerm>> {
        val n = omod.grid.size
        val totalPop = omod.buildings.sumOf { it.population }
        println("Building diff model with grid size: $n")

        val relevantODs = mutableSetOf<Pair<Int, Int>>()
        for ((o, origin) in omod.grid.withIndex()) {
            for ((d, destination) in omod.grid.withIndex()) {
                val od = Pair(origin, destination)
                if (od in affectedLinks) {
                    if (affectedLinks[od]!!.isNotEmpty()) {
                        relevantODs.add(Pair(o, d))
                    }
                }
            }
        }

        // Init diff model
        val diffModel = DifferentiableModel(omod.grid.size - 1)

        // Create demand matrix dependent on the variable transition matrix: demand(o, d | M)
        val demand = Array(n) {
            Array(n) {
                LinearTerm(diffModel.nVars)
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
        for (flexActivity in listOf(ActivityType.OTHER, ActivityType.SHOPPING, ActivityType.BUSINESS)) {
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
                        demand[o][d].addTerm(varTransitionMatrix[o][d], cnst * mCarP[o, d])
                    }
                }
            } else {
                val mTransition = m3rep.transitionMatrix[transitionActivity]!!
                val mDep = m3rep.dependentMatrix[flexActivity]!!
                val pHome = m3rep.homeP.flatten()

                for (o in 0 until n) {
                    val s = LinearTerm(diffModel.nVars)
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
                                s.addTerm(varTransitionMatrix[a][b], coeff)
                            } else {
                                s.addTerm(varTransitionMatrix[b][o], coeff)
                            }
                        }
                    }
                    s.addConstant(cnst)

                    for (d in 0 until n) {
                        if (Pair(o, d) !in relevantODs) { continue }
                        val t = mTransition[o, d] * mCarP[o, d]
                        demand[o][d].addTerm(s, t)
                    }
                }
            }
        }

        // Demand for home
        for (fixActivity in listOf(ActivityType.HOME)) {
            val carP = m3rep.carP[fixActivity]!!
            val fix = m3rep.fixedMatrix[fixActivity]!!.transpose().times(carP)

            val depExpr = if(m3rep.varActivityType in fixActivities) {
                val left = m3rep.dependentMatrix[fixActivity]!!.transpose()
                val right = m3rep.homeP.diagonal().transpose()
                diffModelFromMatrixSandwichT(
                    diffModel.nVars,
                    varTransitionMatrix, left, right,
                    relevantRCs = relevantODs, irrelevantFactorThreshold=irrelevantFactorThreshold
                )
            } else {
                val left = mk.identity<Double>(n)
                val right = m3rep.dependentMatrix[fixActivity]!!.transpose()
                diffModelFromMatrixSandwichT(
                    diffModel.nVars,
                    varTransitionMatrix, left, right,
                    relevantRCs = relevantODs,
                    irrelevantFactorThreshold=irrelevantFactorThreshold
                )
            }

            for (o in 0 until n) {
                for (d in 0 until n) {
                    if (Pair(o, d) !in relevantODs) { continue }
                    demand[o][d].addTerm(depExpr[o][d], carP[o, d])
                    demand[o][d].addConstant(fix[o, d])
                }
            }
        }

        // School and Work
        for (fixActivity in listOf(ActivityType.SCHOOL, ActivityType.WORK)) {
            val carP = m3rep.carP[fixActivity]!!
            val fix = m3rep.fixedMatrix[fixActivity]!!.transpose()
                .dot(m3rep.transitionMatrix[fixActivity]!!)
                .times(carP)

            // Tour starting at var activity
            val pHome = m3rep.homeP.flatten()
            val dMatrixT = m3rep.dependentMatrix[fixActivity]!!.transpose()
            val varStartProbs = Array(n) {LinearTerm(diffModel.nVars)}
            for (col in 0 until n) {
                for (row in 0 until n) {
                    varStartProbs[col].addTerm(varTransitionMatrix[row][col], pHome[row])
                }
            }

            // Tour not starting at var activity
            val depExpr = if (m3rep.varActivityType == fixActivity) {
                val left = m3rep.fixedMatrix[fixActivity]!!.transpose()
                val right = mk.identity<Double>(n)
                diffModelFromMatrixSandwichT(
                    diffModel.nVars,
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
                diffModelFromMatrixSandwichT(
                    diffModel.nVars,
                    varTransitionMatrix, left, right,
                    relevantRCs = relevantODs,
                    irrelevantFactorThreshold=irrelevantFactorThreshold
                )
            } else {
                val left = mk.identity<Double>(n)
                val right = m3rep.dependentMatrix[fixActivity]!!
                    .transpose()
                    .dot(m3rep.transitionMatrix[fixActivity]!!)
                diffModelFromMatrixSandwichT(
                    diffModel.nVars,
                    varTransitionMatrix, left, right,
                    relevantRCs = relevantODs,
                    irrelevantFactorThreshold=irrelevantFactorThreshold
                )
            }

            for (o in 0 until n) {
                for (d in 0 until n) {
                    if (Pair(o, d) !in relevantODs) { continue }
                    demand[o][d].addTerm(depExpr[o][d], carP[o, d])

                    if (m3rep.varActivityType == fixActivity) {
                        demand[o][d].addTerm(varStartProbs[d], dMatrixT[o][d] * carP[o, d])
                    } else {
                        demand[o][d].addConstant(fix[o, d])
                    }
                }
            }
        }

        val simcount = mutableMapOf<TrafficSensor, LinearTerm>()
        for (sensor in sensors) {
            simcount[sensor] = LinearTerm(diffModel.nVars)
        }

        for ((o, origin) in omod.grid.withIndex()) {
            for ((d, destination) in omod.grid.withIndex()) {
                val od = Pair(origin, destination)
                if (od in affectedLinks) {
                    val affected = affectedLinks[od]!!

                    for (sensor in affected) {
                        val sensorSumTest = simcount[sensor]!!
                        val demTest = demand[o][d]
                        sensorSumTest.addTerm(demTest, totalPop)
                    }
                }
            }
        }

        // Objective
        val obj = LinearTerm(diffModel.nVars)
        for ((i, sensor) in sensors.withIndex()) {
            // (Sm - Ss)^2 = Sm^2 - 2SmSs + Ss^2
            val simCountTerm = simcount[sensor]!!
            obj.addConstant(sensor.measuredFlow * sensor.measuredFlow)
            obj.addTerm(simCountTerm, -2 * sensor.measuredFlow)
            val qTerm = QuadraticTerm(
                diffModel.nVars,
                simCountTerm,
                simCountTerm,
                1.0
            )
            obj.addTerm(qTerm, 1.0)
        }

        diffModel.setRootTerm(obj)
        return diffModel to simcount
    }

    fun diffModelFromMatrixSandwichT(
        nVars: Int,
        mVar: List<List<Term>>,
        left: D2Array<Double>,
        right: D2Array<Double>,
        transpose: Boolean = true,
        relevantRCs: Set<Pair<Int, Int>>? = null,
        irrelevantFactorThreshold: Double
    ) : Array<Array<LinearTerm>> {
        val n = left.shape[0] // Assume all matrices are square and same size

        val result = Array(n) {
            Array(n) {
                LinearTerm(nVars)
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
                            activeEntry.addTerm(mVar[i][j], coeff)
                        } else {
                            activeEntry.addTerm(mVar[j][i], coeff)
                        }
                    }
                }
            }
        }
        return result
    }

    private fun computeTransitionMatrices() : FinderMatrices{
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
            for (o in omod.grid.indices) {
                val activityWeights = finder.getWeights(omod.grid[o], omod.grid, activityType = activityType)
                activityProbs[o] = mk.ndarray( activityWeights.map { it / activityWeights.sum() } )
            }

            transitionProbs[activityType] = activityProbs
        }
        return FinderMatrices(homeProbs, transitionProbs)
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
}

private inline fun <reified T : Any> D2Array<T>.diagonal() : D2Array<T> {
    require(this.shape[0] == 1)

    val diagonal  = mk.zeros<T>(this.size, this.size)
    for ( i in 0 until this.size) {
        diagonal[i, i] = this[0,i]
    }
    return diagonal
}