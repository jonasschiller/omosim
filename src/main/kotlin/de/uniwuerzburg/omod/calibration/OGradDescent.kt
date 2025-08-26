package de.uniwuerzburg.omod.calibration

import de.uniwuerzburg.omod.calibration.differentiablemodel.DifferentiableModel
import de.uniwuerzburg.omod.calibration.differentiablemodel.DivisionTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.LinearBaseTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.LinearTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.QuadraticTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.Term
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
import org.jetbrains.kotlinx.multik.ndarray.operations.expandDims
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.plusAssign
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.locationtech.jts.geom.Coordinate
import smile.math.BFGS
import java.io.FileOutputStream
import kotlin.math.*
import kotlin.time.measureTime
import kotlin.time.measureTimedValue

object OGradDescent {
    fun run(
        grid: List<Cell>,
        activityGenerator: ActivityGeneratorDefault,
        modeChoiceCalibration: ModeChoiceCalibration,
        customCellFactors: Map<ActivityType, Map<Cell, Double>>,
        popStrata: List<PopStratum>,
        carOwnership: CarOwnership,
        destinationFinder: DestinationFinderDefault,
        totalPop: Double,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        sensors: List<TrafficSensor>
    )  : Pair<D2Array<Double>, DoubleArray> {
        val oi = prepareOptInputSchoolDep(
            grid, activityGenerator, modeChoiceCalibration, customCellFactors, popStrata, carOwnership,
            destinationFinder
        )
        println("OGradDescent")
        val expected = getExpectedCountPerAgent(oi, grid)
        eval(expected, sensors, totalPop, grid, affectedLinks)

        val (pair, time) = measureTimedValue {
            buildDiffModel(grid, totalPop, affectedLinks, sensors, oi, destinationFinder)
        }
        val (diffModel, simCount) = pair

        println("-----")
        println(time)
        println("-----")

        val seedVals = DoubleArray(grid.size) { 1.0 }
            //doubleArrayOf(-0.01, -0.01, 0.01, 0.01, 0.01, 0.01, 0.01)//DoubleArray(5) {0.01} //(280.688, 727.141, 611.394, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        //val parameters = gradDescent(diffModel, seedVals, iterations = 100, lr = 1.0e-8)
        //gradDescent(diffModel, seedVals, iterations = 100, lr = 1.0e-8)
        val (parameters, lbfgstime) = measureTimedValue { lbfgs(diffModel, seedVals) }
        println("LBFGS time: $lbfgstime")

        println("Parameters:")
        println(parameters.toList())


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


        val shopMax = grid.maxOf { it.buildings.sumOf { it.osmProperties.number_shops } }
        val officeMax = grid.maxOf { it.buildings.sumOf { it.osmProperties.number_offices } }
        val schoolMax = grid.maxOf { it.buildings.sumOf { it.osmProperties.number_schools } }
        val uniMax = grid.maxOf { it.buildings.sumOf { it.osmProperties.number_universities } }
        val powMax = grid.maxOf { it.buildings.sumOf { it.osmProperties.number_place_of_worship } }
        val allDistance = mutableListOf<Float>()
        for (o in 0 until grid.size) {
            val distances = destinationFinder.routingCache.getDistances(grid[o], grid)
            allDistance.addAll(distances.toList())
        }
        val distanceMax = allDistance.max()

        val matrix = mk.zeros<Double>(grid.size, grid.size)

        for (o in 0 until grid.size) {
            val distances = destinationFinder.routingCache.getDistances(grid[o], grid)
            val weights = mutableListOf<Double>()
            for (d in 0 until grid.size) {
                /*val nShops = grid[d].buildings.sumOf { it.osmProperties.number_shops / shopMax}
                val nOffice = grid[d].buildings.sumOf { it.osmProperties.number_offices / officeMax}
                val nSchool = grid[d].buildings.sumOf { it.osmProperties.number_schools  /schoolMax}
                val nUni = grid[d].buildings.sumOf { it.osmProperties.number_universities / uniMax}
                val nPOW = grid[d].buildings.sumOf { it.osmProperties.number_place_of_worship / powMax}
                /*val retailArea = grid[d].buildings.filter { it.osmProperties.landuse == Landuse.RETAIL }.sumOf { it.osmProperties.area }
                val indArea = grid[d].buildings.filter { it.osmProperties.landuse == Landuse.INDUSTRIAL }.sumOf { it.osmProperties.area }
                val commArea = grid[d].buildings.filter { it.osmProperties.landuse == Landuse.COMMERCIAL }.sumOf { it.osmProperties.area }
                val resArea = grid[d].buildings.filter { it.osmProperties.landuse == Landuse.RESIDENTIAL }.sumOf { it.osmProperties.area }*/
                var attraction = 0.0
                attraction += parameters[2] * nShops
                attraction += parameters[3] * nOffice
                attraction += parameters[4] * nSchool
                attraction += parameters[5] * nUni
                attraction += parameters[6] * nPOW
                /*weight += parameters[5] * retailArea
                weight += parameters[6] * indArea
                weight += parameters[7] * commArea
                weight += parameters[8] * resArea*/

                val distance = max(Double.MIN_VALUE, distances[d].toDouble() / distanceMax)
                val deterrence = exp(parameters[0] * distance + parameters[1] * ln(distance))

                val weight = attraction * deterrence*/

                val weight = parameters[d] * oi.transitionMatrix[ActivityType.OTHER]!![o, d]

                weights.add(weight)
            }

            val wsum = weights.sum()
            for (d in 0 until grid.size) {
                matrix[o, d] = weights[d] / wsum
            }
        }


        FileOutputStream("LBFGSTargetMatrix.csv").apply {
            val writer = bufferedWriter()
            for(o in grid.indices) {
                for (d in grid.indices) {
                    writer.write("${matrix!![o, d]};")
                }
                writer.newLine()
            }
            writer.flush()
        }

        return Pair(matrix, parameters)
    }

    fun lbfgs(model: DifferentiableModel, vals: DoubleArray) : DoubleArray {
        println("LBFGS-B")
        println("Start: ${model.evaluate(vals)}")
        val l = DoubleArray(model.nVars){0.0}
        val u = DoubleArray(model.nVars){1e3}
        /*l[0] = -1e3
        l[1] = -1e3
        u[0] = 0.0
        u[1] = 0.0*/
        val solution = BFGS.minimize(model, 5, vals, l, u, 1e-5, 10)
        println("Solution: $solution")
        println("Confirm: ${model.evaluate(vals)}")
        return vals
    }

    fun gradDescent(model: DifferentiableModel, vals: DoubleArray, iterations: Int = 10, lr: Double = 1.0e-9) :
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

    fun prepareOptInputSchoolDep(
        grid: List<Cell>, activityGenerator: ActivityGeneratorDefault,
        modeChoiceCalibration: ModeChoiceCalibration,
        customCellFactors: Map<ActivityType, Map<Cell, Double>>,
        popStrata: List<PopStratum>,
        carOwnership: CarOwnership,
        destinationFinder: DestinationFinderDefault,
    ) : OptimizationInput {
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

        // Home <-> School, School
        val schoolTransitionP = transitionProbs[ActivityType.SCHOOL]!!
        val homeSchoolProbs = homeProbs.diagonal().dot(schoolTransitionP)

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

        val fixedMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(grid.size, grid.size) }
        val otherMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(grid.size, grid.size) }

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
                        fixedMatrix[ActivityType.WORK]!!.plusAssign(homeProbs.diagonal() * chainP)
                        continue
                    }
                    Pair(ActivityType.HOME, ActivityType.SCHOOL) -> {
                        fixedMatrix[ActivityType.SCHOOL]!!.plusAssign(homeProbs.diagonal() * chainP)
                        continue
                    }
                    Pair(ActivityType.WORK, ActivityType.HOME) -> {
                        fixedMatrix[ActivityType.HOME]!!.plusAssign(homeWorkProbs * chainP)
                        continue
                    }
                    Pair(ActivityType.SCHOOL, ActivityType.HOME) -> {
                        fixedMatrix[ActivityType.HOME]!!.plusAssign(homeSchoolProbs * chainP)
                        continue
                    }
                    else -> {}
                }
            }

            var nextActivity = chain[1]

            // Setup
            var cumMatrix = mk.identity<Double>(grid.size)
            var previousProbsH: D2Array<Double>
            var previousProbsHnotO: D2Array<Double>
            when(startActivity) {
                ActivityType.HOME -> {
                    previousProbsH = homeProbs.diagonal()
                    previousProbsHnotO = homeProbs.diagonal()
                    fixedMatrix[nextActivity]!!.plusAssign(homeProbs.diagonal() * chainP)
                }
                ActivityType.WORK -> {
                    previousProbsH = homeWorkProbs
                    previousProbsHnotO = homeWorkProbs
                    fixedMatrix[nextActivity]!!.plusAssign(homeWorkProbs * chainP)
                }
                ActivityType.SCHOOL -> {
                    previousProbsH = homeSchoolProbs
                    previousProbsHnotO = homeSchoolProbs
                    fixedMatrix[nextActivity]!!.plusAssign(homeSchoolProbs * chainP)
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

                val oBefore = chain.take(next-1).count { it == ActivityType.OTHER }
                if ((oBefore >= 1) and (nextActivity != ActivityType.OTHER)) {
                    otherMatrix[nextActivity]!!.plusAssign(previousProbsHnotO * chainP)
                } else {
                    fixedMatrix[nextActivity]!!.plusAssign(previousProbsH.dot(transitionP) * chainP)
                }

                previousProbsH = previousProbsH.dot(transitionP)
                if (activity != ActivityType.OTHER) {
                    previousProbsHnotO = previousProbsHnotO.dot(transitionP)
                }
            }
        }

        val oi = OptimizationInput(
            homeProbs,
            otherMatrix,
            fixedMatrix,
            transitionProbs,
            carProbs
        )

        return oi
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
        val otherMatrix: Map<ActivityType, D2Array<Double>>,
        val fixedMatrix: Map<ActivityType,  D2Array<Double>>,
        val transitionMatrix: Map<ActivityType,  D2Array<Double>>,
        val carProbs: Map<ActivityType,  D2Array<Double>>,
    )

    private fun getExpectedCountPerAgent(oi: OptimizationInput, grid: List<Cell>) : D2Array<Double> {
        // Result: Expected trip count on each od-paid of one agent on one day
        val expectedCountPerAgent = mk.zeros<Double>(grid.size, grid.size)

        val oTMatrix = oi.transitionMatrix[ActivityType.OTHER]!!

        // Flex
        for (flexActivity in listOf(ActivityType.OTHER, ActivityType.SHOPPING, ActivityType.BUSINESS)) {
            val transitionActivity = if (flexActivity == ActivityType.BUSINESS) {
                ActivityType.OTHER
            } else {
                flexActivity
            }
            val carP = oi.carProbs[transitionActivity]!!

            if (flexActivity == ActivityType.OTHER) {
                val fix = oi.fixedMatrix[flexActivity]!!
                val total = mk.ones<Double>(grid.size).expandDims(0).asDNArray()
                    .asD2Array().dot(fix)
                    .diagonal().dot(oi.transitionMatrix[transitionActivity]!!)
                expectedCountPerAgent.plusAssign(total.times(carP))
            } else {
                val oDep = oi.otherMatrix[flexActivity]!!.dot(oTMatrix)
                val fix = oi.fixedMatrix[flexActivity]!!
                val total = mk.ones<Double>(grid.size).expandDims(0).asDNArray()
                    .asD2Array().dot(oDep.plus(fix))
                    .diagonal().dot(oi.transitionMatrix[transitionActivity]!!)
                expectedCountPerAgent.plusAssign(total.times(carP))
            }
        }

        // Home
        for (fixActivity in listOf(ActivityType.HOME)) {
            val carP = oi.carProbs[fixActivity]!!

            val oDep = oi.otherMatrix[fixActivity]!!.dot(oTMatrix)
            val fix = oi.fixedMatrix[fixActivity]!!
            val total = oDep.plus(fix).transpose()

            expectedCountPerAgent.plusAssign(total.times(carP))
        }

        // School
        for (fixActivity in listOf(ActivityType.SCHOOL)) {
            val carP = oi.carProbs[fixActivity]!!

            val oDep = oi.otherMatrix[fixActivity]!!.dot(oTMatrix)
            val fix = oi.fixedMatrix[fixActivity]!!
            val total = oDep.plus(fix).transpose().dot(oi.transitionMatrix[fixActivity]!!)

            expectedCountPerAgent.plusAssign(total.times(carP))
        }

        // Work
        for (fixActivity in listOf(ActivityType.WORK)) {
            val carP = oi.carProbs[fixActivity]!!

            val oDep = oi.otherMatrix[fixActivity]!!.dot(oTMatrix)
            val fix = oi.fixedMatrix[fixActivity]!!
            val total = oDep.plus(fix).transpose().dot(oi.transitionMatrix[fixActivity]!!)

            expectedCountPerAgent.plusAssign(total.times(carP))
        }

        return expectedCountPerAgent
    }

    fun buildDiffModel(
        grid: List<Cell>,
        totalPop: Double,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        sensors: List<TrafficSensor>,
        oi: OptimizationInput,
        destinationFinder: DestinationFinderDefault
    ) : Pair<DifferentiableModel, Map<TrafficSensor, LinearTerm>> {
        val n = grid.size
        println("Building diff model with grid size: $n")

        val relevantODs = mutableSetOf<Pair<Int, Int>>()
        for ((o, origin) in grid.withIndex()) {
            for ((d, destination) in grid.withIndex()) {
                val od = Pair(origin, destination)
                if (od in affectedSensors) {
                    if (affectedSensors[od]!!.isNotEmpty()) {
                        relevantODs.add(Pair(o, d))
                    }
                }
            }
        }

        val diffModel = DifferentiableModel(grid.size)

        // Create demand matrix dependent on the work matrix: demand(o, d | W)
        val demand = Array(n) {
            Array(n) {
                LinearTerm(diffModel.nVars)
            }
        }

        val shopMax = grid.maxOf { it.buildings.sumOf { it.osmProperties.number_shops } }
        val officeMax = grid.maxOf { it.buildings.sumOf { it.osmProperties.number_offices } }
        val schoolMax = grid.maxOf { it.buildings.sumOf { it.osmProperties.number_schools } }
        val uniMax = grid.maxOf { it.buildings.sumOf { it.osmProperties.number_universities } }
        val powMax = grid.maxOf { it.buildings.sumOf { it.osmProperties.number_place_of_worship } }
        val allDistance = mutableListOf<Float>()
        for (o in 0 until n) {
            val distances = destinationFinder.routingCache.getDistances(grid[o], grid)
            allDistance.addAll(distances.toList())
        }
        val distanceMax = allDistance.max()

        val O = mutableListOf<List<Term>>()
        for (o in 0 until n) {
            val oO = mutableListOf<Term>()
            val distances = destinationFinder.routingCache.getDistances(grid[o], grid)

            val weightTerms = mutableListOf<Term>()
            val rowSumTerm = LinearTerm(diffModel.nVars)
            for (d in 0 until n) {
               val nShops = grid[d].buildings.sumOf { it.osmProperties.number_shops / shopMax}
               val nOffice = grid[d].buildings.sumOf { it.osmProperties.number_offices / officeMax}
               val nSchool = grid[d].buildings.sumOf { it.osmProperties.number_schools / schoolMax}
               val nUni = grid[d].buildings.sumOf { it.osmProperties.number_universities / uniMax}
               val nPOW = grid[d].buildings.sumOf { it.osmProperties.number_place_of_worship / powMax}
               /*val retailArea = grid[d].buildings.filter { it.osmProperties.landuse == Landuse.RETAIL }.sumOf { it.osmProperties.area }
               val indArea = grid[d].buildings.filter { it.osmProperties.landuse == Landuse.INDUSTRIAL }.sumOf { it.osmProperties.area }
               val commArea = grid[d].buildings.filter { it.osmProperties.landuse == Landuse.COMMERCIAL }.sumOf { it.osmProperties.area }
               val resArea = grid[d].buildings.filter { it.osmProperties.landuse == Landuse.RESIDENTIAL }.sumOf { it.osmProperties.area }*/
               /*val attraction = LinearBaseTerm(diffModel.nVars)
                attraction.addTerm(0, nShops)
                attraction.addTerm(1, nOffice)
                attraction.addTerm(2, nSchool)
                attraction.addTerm(3, nUni)
                attraction.addTerm(4, nPOW)*/
               /*weight.addTerm(5, retailArea)
               weight.addTerm(6, indArea)
               weight.addTerm(7, commArea)
               weight.addTerm(8, resArea)*/

               /*val distance = max(Double.MIN_VALUE, distances[d].toDouble() / distanceMax)
               val detUtil = LinearBaseTerm(diffModel.nVars)
               detUtil.addTerm(0, distance)
               detUtil.addTerm(1, ln(distance))
               val deterrence = ExponentialTerm(diffModel.nVars, detUtil)


               val weight = QuadraticTerm(diffModel.nVars, attraction, deterrence, 1.0)*/
               val weight = LinearBaseTerm(diffModel.nVars)
               //weight.addTerm(d, 1.0)
               weight.addTerm(d, oi.transitionMatrix[ActivityType.OTHER]!![o, d])
               //weight.addTerm(d, attraction)

               rowSumTerm.addTerm(weight, 1.0)
               weightTerms.add(weight)
            }
            for (d in 0 until n) {
                val scaledWeight = DivisionTerm(diffModel.nVars, weightTerms[d], rowSumTerm)
                oO.add(scaledWeight)
            }
            O.add(oO)
        }

        // Demand with flexible destination
        for (flexActivity in listOf(ActivityType.OTHER, ActivityType.SHOPPING, ActivityType.BUSINESS)) {
            val transitionActivity = if (flexActivity == ActivityType.BUSINESS) {
                ActivityType.OTHER
            } else {
                flexActivity
            }
            val mCarP = oi.carProbs[transitionActivity]!!

            if (flexActivity == ActivityType.OTHER) {
                val mFixed = oi.fixedMatrix[flexActivity]!!
                for (o in 0 until n) {
                    var cnst = 0.0

                    for (a in 0 until n) {
                        cnst += mFixed[a, o]
                    }

                    for (d in 0 until n) {
                        if (Pair(o, d) !in relevantODs) { continue }
                        demand[o][d].addTerm(O[o][d], cnst * mCarP[o, d])
                    }
                }
            } else {
                val mFixed = oi.fixedMatrix[flexActivity]!!
                val mTransition = oi.transitionMatrix[transitionActivity]!!
                val mODep = oi.otherMatrix[flexActivity]!!

                for (o in 0 until n) {
                    val s = LinearTerm(diffModel.nVars)
                    var cnst = 0.0

                    for (a in 0 until n) {
                        cnst += mFixed[a, o]
                        for (b in 0 until n) {
                            val coeff = mODep[a, b]

                            if (abs(coeff) <= (1/n.toDouble()).pow(1.5)) {
                                continue
                            }

                            s.addTerm(O[b][o], coeff) // TODO Store coeffs in matrix
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
            val carP = oi.carProbs[fixActivity]!!

            val left = mk.identity<Double>(n)
            val right = oi.otherMatrix[fixActivity]!!.transpose()
            val oExpr = diffModelFromMatrixSandwichT(O, left, right, relevantRCs = relevantODs, transpose = true)

            val fix = oi.fixedMatrix[fixActivity]!!.transpose().times(carP)

            for (o in 0 until n) {
                for (d in 0 until n) {
                    if (Pair(o, d) !in relevantODs) { continue }
                    demand[o][d].addTerm(oExpr[o][d], carP[o, d])
                    demand[o][d].addConstant(fix[o, d])
                }
            }
        }

        // School
        for (fixActivity in listOf(ActivityType.SCHOOL)) {
            val carP = oi.carProbs[fixActivity]!!

            val left = mk.identity<Double>(n)
            val right = oi.otherMatrix[fixActivity]!!.transpose().dot(oi.transitionMatrix[fixActivity]!!)
            val oExpr = diffModelFromMatrixSandwichT(O, left, right, relevantRCs = relevantODs)

            val fix = oi.fixedMatrix[fixActivity]!!.transpose()
                .dot(oi.transitionMatrix[fixActivity]!!)
                .times(carP)

            for (o in 0 until n) {
                for (d in 0 until n) {
                    if (Pair(o, d) !in relevantODs) { continue }
                    demand[o][d].addTerm(oExpr[o][d], carP[o, d])
                    demand[o][d].addConstant(fix[o, d])
                }
            }
        }

        for (fixActivity in listOf(ActivityType.WORK)) {
            val carP = oi.carProbs[fixActivity]!!

            val left = mk.identity<Double>(n)
            val right = oi.otherMatrix[fixActivity]!!.transpose().dot(oi.transitionMatrix[fixActivity]!!)
            val oExpr = diffModelFromMatrixSandwichT(O, left, right, relevantRCs = relevantODs)

            val fix = oi.fixedMatrix[fixActivity]!!.transpose()
                .dot(oi.transitionMatrix[fixActivity]!!)
                .times(carP)

            for (o in 0 until n) {
                for (d in 0 until n) {
                    if (Pair(o, d) !in relevantODs) { continue }
                    demand[o][d].addTerm(oExpr[o][d], carP[o, d])
                    demand[o][d].addConstant(fix[o, d])
                }
            }
        }

        val simcount = mutableMapOf<TrafficSensor, LinearTerm>()
        for (sensor in sensors) {
            simcount[sensor] = LinearTerm(diffModel.nVars)
        }

        for ((o, origin) in grid.withIndex()) {
            for ((d, destination) in grid.withIndex()) {
                val od = Pair(origin, destination)
                if (od in affectedSensors) {
                    val affected = affectedSensors[od]!!

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
        mVar: List<List<Term>>,
        left: D2Array<Double>,
        right: D2Array<Double>,
        transpose: Boolean = true,
        relevantRCs: Set<Pair<Int, Int>>? = null
    ) : Array<Array<LinearTerm>> {
        val n = left.shape[0] // Assume all matrices are square and same size

        val result = Array(n) {
            Array(n) {
                LinearTerm(n * n)
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

                        if (abs(coeff) <= (1/n.toDouble()).pow(1.5)) {
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

    fun eval(
        expected: D2Array<Double>,
        sensors: List<TrafficSensor>,
        totalPop: Double,
        grid: List<Cell>,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
    ) {
        val simCount = sensors.associateWith { 0.0 }.toMutableMap()
        for ((o, origin) in grid.withIndex()) {
            for ((d, destination) in grid.withIndex()) {
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
        println("OBJ-VAL: $mse")
        mse /= sensors.size

        println("MSE: $mse")
        println("------------------------------")
        println("Sensor | \t Flow OMOD | \t Flow Measured")
        for ((i, flow) in simCount.values.withIndex()) {
            println("${sensors[i].name} | \t $flow | \t ${sensors[i].measuredFlow}")
        }
    }
}