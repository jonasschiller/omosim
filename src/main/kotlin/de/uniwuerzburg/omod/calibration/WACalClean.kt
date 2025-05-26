package de.uniwuerzburg.omod.calibration

import com.gurobi.gurobi.*
import de.uniwuerzburg.omod.calibration.WACalClean.diagonal
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
import java.io.FileOutputStream
import kotlin.math.abs
import kotlin.math.pow
import kotlin.time.measureTime
import kotlin.time.measureTimedValue

object WACalClean {
    fun run(
        grid: List<Cell>,
        activityGenerator: ActivityGeneratorDefault,
        modeChoiceCalibration: ModeChoiceCalibration,
        customCellFactors: Map<Cell, Double>,
        popStrata: List<PopStratum>,
        carOwnership: CarOwnership,
        destinationFinder: DestinationFinderDefault,
        totalPop: Double,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        sensors: List<TrafficSensor>
    ) : Triple<List<Double>, Map<Pair<Cell, Cell>, Double>, D2Array<Double>> {
        val oi = prepareOptInputWorkDep(
            grid, activityGenerator, modeChoiceCalibration, customCellFactors, popStrata, carOwnership,
            destinationFinder
        )
        val expected = getExpectedCountPerAgent(oi, grid)
        eval(expected, sensors, totalPop, grid, affectedLinks)

        val (wMatrix, time) = measureTimedValue {
            optimize(grid, totalPop, affectedLinks, sensors, oi)
        }

        println("-----")
        println(time)
        println("-----")

        val calvals = optimizeStep2Smpl(grid, wMatrix!!, oi.transitionMatrix[ActivityType.WORK]!!)

        // Optimized
        val newCellFactors = grid.zip(calvals).toMap()
        val oiopt = prepareOptInputWorkDep(
            grid, activityGenerator, modeChoiceCalibration, newCellFactors, popStrata, carOwnership,
            destinationFinder
        )
        val expectedOpt = getExpectedCountPerAgent(oiopt, grid)
        eval(expectedOpt, sensors, totalPop, grid, affectedLinks)

        // Format output
        val out = mutableMapOf<Pair<Cell, Cell>, Double>()
        for ((o, origin) in grid.withIndex()) {
            for ((d, destination) in grid.withIndex()) {
                out[Pair(origin, destination)] = expectedOpt[o][d]
            }
        }
        return Triple(calvals, out, wMatrix)
    }

    fun prepareOptInputWorkDep(
        grid: List<Cell>, activityGenerator: ActivityGeneratorDefault,
        modeChoiceCalibration: ModeChoiceCalibration,
        customCellFactors: Map<Cell, Double>,
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

        val workMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(grid.size, grid.size) }
        val fixedMatrix = ActivityType.entries.associateWith { mk.zeros<Double>(grid.size, grid.size) }

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
                        workMatrix[ActivityType.HOME]!!.plusAssign(mk.identity<Double>(grid.size) * chainP)
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
            when(startActivity) {
                ActivityType.HOME -> {
                    previousProbsH = homeProbs.diagonal()
                    fixedMatrix[nextActivity]!!.plusAssign(homeProbs.diagonal() * chainP)
                }
                ActivityType.WORK -> {
                    previousProbsH = homeWorkProbs
                    workMatrix[nextActivity]!!.plusAssign(cumMatrix * chainP)
                }
                ActivityType.SCHOOL -> {
                    previousProbsH = homeSchoolProbs
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

                if (startActivity == ActivityType.WORK) {
                    cumMatrix = cumMatrix.dot(transitionP)
                    workMatrix[nextActivity]!!.plusAssign(cumMatrix * chainP)
                } else {
                    fixedMatrix[nextActivity]!!.plusAssign(previousProbsH.dot(transitionP) * chainP)
                }

                previousProbsH = previousProbsH.dot(transitionP)
            }
        }

        val oi = OptimizationInput(
            homeProbs,
            workMatrix,
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
        val workMatrix: Map<ActivityType,  D2Array<Double>>,
        val fixedMatrix: Map<ActivityType,  D2Array<Double>>,
        val transitionMatrix: Map<ActivityType,  D2Array<Double>>,
        val carProbs: Map<ActivityType,  D2Array<Double>>,
    )

    private fun getExpectedCountPerAgent(oi: OptimizationInput, grid: List<Cell>) : D2Array<Double> {
        // Result: Expected trip count on each od-paid of one agent on one day
        val expectedCountPerAgent = mk.zeros<Double>(grid.size, grid.size)

        val wTMatrix = oi.transitionMatrix[ActivityType.WORK]!!
        val workProbs = oi.homeProbs.dot(wTMatrix)

        // Flex
        for (flexActivity in listOf(ActivityType.OTHER, ActivityType.SHOPPING, ActivityType.BUSINESS)) {
            val transitionActivity = if (flexActivity == ActivityType.BUSINESS) {
                ActivityType.OTHER
            } else {
                flexActivity
            }
            val carP = oi.carProbs[transitionActivity]!!

            val wDep = oi.homeProbs.diagonal().dot(wTMatrix).dot(oi.workMatrix[flexActivity]!!)
            val fix = oi.fixedMatrix[flexActivity]!!
            val total = mk.ones<Double>(grid.size).expandDims(0).asDNArray()
                .asD2Array().dot(wDep.plus(fix))
                .diagonal().dot(oi.transitionMatrix[transitionActivity]!!)

            expectedCountPerAgent.plusAssign(total.times(carP))
        }

        // Home
        for (fixActivity in listOf(ActivityType.HOME)) {
            val carP = oi.carProbs[fixActivity]!!

            val wDep = oi.homeProbs.diagonal().dot(wTMatrix).dot(oi.workMatrix[fixActivity]!!)
            val fix = oi.fixedMatrix[fixActivity]!!
            val total = wDep.plus(fix).transpose()

            expectedCountPerAgent.plusAssign(total.times(carP))
        }

        // School
        for (fixActivity in listOf(ActivityType.SCHOOL)) {
           val carP = oi.carProbs[fixActivity]!!

           val wDep = oi.homeProbs.diagonal().dot(wTMatrix).dot(oi.workMatrix[fixActivity]!!)
           val fix = oi.fixedMatrix[fixActivity]!!
           val total = wDep.plus(fix).transpose().dot(oi.transitionMatrix[fixActivity]!!)

           expectedCountPerAgent.plusAssign(total.times(carP))
        }

        // Work
        for (fixActivity in listOf(ActivityType.WORK)) {
            val carP = oi.carProbs[fixActivity]!!

            val wDep = workProbs.diagonal().dot(oi.workMatrix[fixActivity]!!).transpose()
            val fix = oi.fixedMatrix[fixActivity]!!.transpose().dot(oi.transitionMatrix[fixActivity]!!)
            val total = wDep.plus(fix)

            expectedCountPerAgent.plusAssign(total.times(carP))
        }

        return expectedCountPerAgent
    }

    fun optimize(
        grid: List<Cell>,
        totalPop: Double,
        affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
        sensors: List<TrafficSensor>,
        oi: OptimizationInput
    ) : D2Array<Double>? {
        val n = grid.size
        println("Optimization 1 start with grid size: ${n}")

        try {
            // Setup
            val env = GRBEnv()
            val model = GRBModel(env)

            // Create demand matrix dependent on the work matrix: demand(o, d | W)
            val demand = Array(n) {
                Array(n) {
                    GRBLinExpr()
                }
            }

            val w = Array<Array<GRBVar>> (n) { o ->
                model.addVars(
                    DoubleArray(n) { 0.0 },
                    DoubleArray(n) { 1.0 },
                    DoubleArray(n) { 0.0},
                    CharArray(n) {GRB.CONTINUOUS},
                    Array(n) { d -> "W_${o}_$d"}
                )
            }

            // Ensure that each row of W is a proper probability distribution
            for (o in 0 until n) {
                val rowSum = GRBLinExpr()
                for (d in 0 until n) {
                    rowSum.addTerm(1.0, w[o][d])
                    //model.addConstr(w[o][d], GRB.EQUAL, oi.transitionMatrix[ActivityType.WORK]!![o,d], "Test")
                }
                model.addConstr(rowSum, GRB.EQUAL, 1.0, "ProbCondition")
            }

            // Demand with flexible destination
            for (flexActivity in listOf(ActivityType.OTHER, ActivityType.SHOPPING, ActivityType.BUSINESS)) {
                val transitionActivity = if (flexActivity == ActivityType.BUSINESS) {
                    ActivityType.OTHER
                } else {
                    flexActivity
                }
                val mFixed = oi.fixedMatrix[flexActivity]!!
                val mWDep = oi.workMatrix[flexActivity]!!
                val pHome = oi.homeProbs.flatten()
                val mTransition = oi.transitionMatrix[transitionActivity]!!
                val mCarP = oi.carProbs[transitionActivity]!! // TODO transition or flex?

                for (o in 0 until n) {
                    val s = GRBLinExpr()
                    var cnst = 0.0

                    for (a in 0 until n) {
                        cnst += mFixed[a, o]
                        for (b in 0 until n) {
                            val coeff = pHome[a] * mWDep[b, o]
                            if (abs(coeff) <= (1/n.toDouble()).pow(2)) {
                                continue
                            }

                            s.addTerm(coeff, w[a][b]) // TODO Store coeffs in matrix
                        }
                    }
                    s.addConstant(cnst)

                    for (d in 0 until n) {
                        val t = mTransition[o, d] * mCarP[o, d]
                        demand[o][d].multAdd(t, s)
                    }
                }
            }

            // Demand for home
            for (fixActivity in listOf(ActivityType.HOME)) {
                val carP = oi.carProbs[fixActivity]!!

                val left = oi.workMatrix[fixActivity]!!.transpose()
                val right = oi.homeProbs.diagonal().transpose()
                val wExpr = exprFromMatrixSandwichT(w, left, right)

                val fix = oi.fixedMatrix[fixActivity]!!.transpose().times(carP)

                for (o in 0 until n) {
                    for (d in 0 until n) {
                        demand[o][d].multAdd(carP[o, d], wExpr[o][d])
                        demand[o][d].addConstant(fix[o, d])
                    }
                }
            }

            // School
            for (fixActivity in listOf(ActivityType.SCHOOL)) {
                val carP = oi.carProbs[fixActivity]!!

                val left = oi.workMatrix[fixActivity]!!.transpose()
                val right = oi.homeProbs
                    .diagonal()
                    .transpose()
                    .dot(oi.transitionMatrix[fixActivity]!!)
                val wExpr = exprFromMatrixSandwichT(w, left, right)

                val fix = oi.fixedMatrix[fixActivity]!!.transpose()
                    .dot(oi.transitionMatrix[fixActivity]!!)
                    .times(carP)

                for (o in 0 until n) {
                    for (d in 0 until n) {
                        demand[o][d].multAdd(carP[o, d], wExpr[o][d])
                        demand[o][d].addConstant(fix[o, d])
                    }
                }
            }

            for (fixActivity in listOf(ActivityType.WORK)) {
                val carP = oi.carProbs[fixActivity]!!
                val pHome = oi.homeProbs.flatten()

                // Start at work
                val mWT = oi.workMatrix[fixActivity]!!.transpose()
                val wProbs = Array(n) { GRBLinExpr() }
                for (col in 0 until n) {
                    val activeExpr = wProbs[col]
                    for (row in 0 until n) {
                        activeExpr.addTerm(pHome[row], w[row][col])
                    }
                }

                // Not start at work
                val left = oi.fixedMatrix[fixActivity]!!.transpose()
                val right = mk.identity<Double>(n)
                val wExprNSW = exprFromMatrixSandwichT(w, left, right, transpose = false)

                for (o in 0 until n) {
                    for (d in 0 until n) {
                        demand[o][d].multAdd(carP[o, d], wExprNSW[o][d])
                        demand[o][d].multAdd(mWT[o][d] * carP[o, d], wProbs[d])
                    }
                }
            }

            val sensorCountExpr = mutableMapOf<TrafficSensor, GRBLinExpr>()
            for (sensor in sensors) {
                sensorCountExpr[sensor] = GRBLinExpr()
            }

            for ((o, origin) in grid.withIndex()) {
                for ((d, destination) in grid.withIndex()) {
                    val od = Pair(origin, destination)
                    if (od in affectedSensors) {
                        val affected = affectedSensors[od]!!

                        for (sensor in affected) {
                            val sensorSum = sensorCountExpr[sensor]!!
                            val dem = demand[o][d]
                            sensorSum.multAdd(totalPop, dem)
                        }
                    }
                }
            }

            val sensorSimCount = model.addVars(
                DoubleArray(sensors.size) {0.0},
                null,
                DoubleArray(sensors.size) {0.0},
                CharArray(sensors.size) { GRB.CONTINUOUS},
                Array(sensors.size) {""}
            )
            for ((i, sensor) in sensors.withIndex()) {
                val sensorSum = sensorCountExpr[sensor]!!
                model.addConstr(sensorSum, GRB.EQUAL, sensorSimCount[i], "cnteq")
            }

            // Objective
            val obj = GRBQuadExpr()
            for ((i, sensor) in sensors.withIndex()) {
                // (Sm - Ss)^2 = Sm^2 - 2SmSs + Ss^2
                obj.addConstant(sensor.measuredFlow * sensor.measuredFlow)
                obj.addTerm(-2 * sensor.measuredFlow, sensorSimCount[i])
                obj.addTerm(1.0, sensorSimCount[i], sensorSimCount[i])
            }
            model.setObjective(obj, GRB.MINIMIZE)

            model.optimize()

            var optimstatus = model[GRB.IntAttr.Status]

            if (optimstatus == GRB.Status.INF_OR_UNBD) {
                model[GRB.IntParam.Presolve] = 0
                model.optimize()
                optimstatus = model[GRB.IntAttr.Status]
            }

            if (optimstatus == GRB.Status.OPTIMAL) {
                val objval = model[GRB.DoubleAttr.ObjVal]
                println("Optimal objective: $objval")


                println("_".repeat(20*4 + 5*3))
                println("MSE: ${objval /sensors.size}")
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
                    val optVal = sensorSimCount[i].get(GRB.DoubleAttr.X)
                    println(
                        "${sensors[i].name.padEnd(20)} | \t" +
                                "${optVal.toString().padEnd(20)} | \t" +
                                sensors[i].measuredFlow.toString().padEnd(20)
                    )
                    myobjval += (sensors[i].measuredFlow - optVal) * (sensors[i].measuredFlow - optVal)
                }

            } else if (optimstatus == GRB.Status.INFEASIBLE) {
                println("Model is infeasible")
            } else if (optimstatus == GRB.Status.UNBOUNDED) {
                println("Model is unbounded")
            } else {
                println(
                    "Optimization was stopped with status = "
                            + optimstatus
                )
            }

            val result = mk.ones<Double>(grid.size, grid.size)
            for(o in grid.indices) {
                for (d in grid.indices) {
                    result[o, d] = w[o][d].get(GRB.DoubleAttr.X)
                }
            }

            // Dispose of model and environment
            model.dispose()
            env.dispose()
            return result
        } catch (e: GRBException) {
            println(
                ("Error code: " + e.errorCode + ". " +
                        e.message)
            )
        }
        return null
    }

    fun exprFromMatrixSandwichTTester(
        mVarTest: D2Array<Double>,
        left: D2Array<Double>,
        right: D2Array<Double>,
        transpose: Boolean = true,
    ) : D2Array<Double> {
        val n = left.shape[0] // Assume all matrices are square and same size

        val result = mk.zeros<Double>(n,n)

        for (row in 0 until n) {
            for (col in 0 until  n) {
                // TODO store coeff in matrix and get rid of too small ones
                for (i in 0 until n) {
                    if (right[i, col] == 0.0) { continue }
                    for (j in 0 until n) {
                        if (left[row, j] == 0.0) { continue }
                        if (transpose) {
                            result[row, col] += left[row, j] * right[i, col] * mVarTest[i, j]
                        } else {
                            result[row, col] += left[row, j] * right[i, col] * mVarTest[j, i]
                        }


                    }
                }
            }
        }
        return result
    }

    fun exprFromMatrixSandwichT(
        mVar: Array<Array<GRBVar>>,
        left: D2Array<Double>,
        right: D2Array<Double>,
        transpose: Boolean = true,
        ) : Array<Array<GRBLinExpr>> {
        val n = left.shape[0] // Assume all matrices are square and same size

        val result = Array(n) {
            Array(n) {
                GRBLinExpr()
            }
        }

        for (row in 0 until n) {
            for (col in 0 until  n) {
                val activeEntry = result[row][col]

                // TODO store coeff in matrix and get rid of too small ones
                for (i in 0 until n) {
                    if (right[i, col] == 0.0) { continue }
                    for (j in 0 until n) {
                        if (left[row, j] == 0.0) { continue }
                        val coeff = left[row, j] * right[i, col]

                        if (abs(coeff) <= (1/n.toDouble()).pow(2)) {
                            continue
                        }

                        if (transpose) {
                            activeEntry.addTerm(coeff, mVar[i][j])
                        } else {
                            activeEntry.addTerm(coeff, mVar[j][i])
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
        mse /= sensors.size

        println("MSE: $mse")
        println("------------------------------")
        println("Sensor | \t Flow OMOD | \t Flow Measured")
        for ((i, flow) in simCount.values.withIndex()) {
            println("${sensors[i].name} | \t $flow | \t ${sensors[i].measuredFlow}")
        }
    }

    fun optimizeStep2Smpl(
        grid: List<Cell>,
        bestT: D2Array<Double>,
        seedMatrix: D2Array<Double>,
    ) : List<Double> {
        try {
            val env = GRBEnv()
            val model = GRBModel(env)

            val dScaler = model.addVars(
                DoubleArray(grid.size - 1) {0.0}, // TODO doesn't work if unbounded at the upper limit
                null,//DoubleArray(grid.size - 1) {15.0},
                DoubleArray(grid.size - 1) {0.0},
                CharArray(grid.size - 1) { GRB.CONTINUOUS},
                Array(grid.size - 1) {"DestScaler"}
            )


            // Objective
            val obj = GRBQuadExpr()
            for(o in grid.indices) {
                for(d in grid.indices.drop(1)) {
                    val targetValue = bestT[o,d]
                    val seedvalue = seedMatrix[o,d]

                    val targetProp = targetValue / bestT[o, 0]
                    val seedProp = seedvalue/ seedMatrix[o,0]

                    obj.addConstant(targetProp * targetProp)
                    obj.addTerm(-2 * targetProp * seedProp, dScaler[d-1])
                    obj.addTerm(seedProp * seedProp, dScaler[d-1], dScaler[d-1])
                }
            }
            model.setObjective(obj, GRB.MINIMIZE)


            model.optimize()

            var optimstatus = model[GRB.IntAttr.Status]

            if (optimstatus == GRB.Status.INF_OR_UNBD) {
                model[GRB.IntParam.Presolve] = 0
                model.optimize()
                optimstatus = model[GRB.IntAttr.Status]
            }

            if (optimstatus == GRB.Status.OPTIMAL) {
                val objval = model[GRB.DoubleAttr.ObjVal]
                println("Optimal objective: $objval")
            } else if (optimstatus == GRB.Status.INFEASIBLE) {
                println("Model is infeasible")
            } else if (optimstatus == GRB.Status.UNBOUNDED) {
                println("Model is unbounded")
            } else {
                println(
                    "Optimization was stopped with status = "
                            + optimstatus
                )
            }

            val scalerList = listOf(1.0) + dScaler.map { it.get(GRB.DoubleAttr.X) }

            val newMatrix = mk.ones<Double>(grid.size, grid.size)
            for(o in grid.indices) {
                var isrowsum = 0.0
                for (d in grid.indices) {
                    isrowsum += seedMatrix[o, d] * scalerList[d]
                }

                for (d in grid.indices) {
                    newMatrix[o, d] = seedMatrix[o, d] *  scalerList[d] / isrowsum
                }
            }



            FileOutputStream("TargetMatrix.csv").apply {
                val writer = bufferedWriter()
                for(o in grid.indices) {
                    for (d in grid.indices) {
                        writer.write("${bestT[o, d]};")
                    }
                    writer.newLine()
                }
                writer.flush()
            }

            FileOutputStream("FindMatrix.csv").apply {
                val writer = bufferedWriter()
                for(o in grid.indices) {
                    for (d in grid.indices) {
                        writer.write("${newMatrix[o, d]};")
                    }
                    writer.newLine()
                }
                writer.flush()
            }

            FileOutputStream("SeedMatrix.csv").apply {
                val writer = bufferedWriter()
                for(o in grid.indices) {
                    for (d in grid.indices) {
                        writer.write("${seedMatrix[o, d]};")
                    }
                    writer.newLine()
                }
                writer.flush()
            }


            // Dispose of model and environment
            model.dispose()
            env.dispose()

            return  scalerList
        } catch (e: GRBException) {
            println(
                ("Error code: " + e.errorCode + ". " +
                        e.message)
            )
        }
        return listOf()
    }
}