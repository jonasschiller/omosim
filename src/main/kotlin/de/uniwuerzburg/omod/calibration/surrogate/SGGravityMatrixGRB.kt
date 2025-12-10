package de.uniwuerzburg.omod.calibration.surrogate

import com.gurobi.gurobi.*
import de.uniwuerzburg.omod.calibration.CalibrationConstants.MC_SAMPLES
import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.TrafficSensor
import de.uniwuerzburg.omod.calibration.differentiablemodel.TermBuilder
import de.uniwuerzburg.omod.calibration.logger
import de.uniwuerzburg.omod.calibration.surrogate.SGGravity.MetaModelMatrixRep
import de.uniwuerzburg.omod.core.models.ActivityType
import de.uniwuerzburg.omod.core.models.RealLocation
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.set
import kotlin.math.pow

fun SGGravity.calibrateTransitionMatrix(
    activityType: ActivityType,
    sensors: List<TrafficSensor>,
    affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
) : D2Array<Double>? {
    logger.info(
        "[Experimental] Calibrating transition matrix for activity $activityType directly." +
            "Number of variables: ${omod.grid.size}"
    )
    val m3rep = generateMarkovChainRep(activityType)
    val matrix = optimizeTMatrix(m3rep, affectedSensors, sensors)
    logger.info("Transition matrix for activity $activityType optimized.")
    return matrix
}

fun SGGravity.optimizeTMatrix(
    m3rep: MetaModelMatrixRep,
    affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
    sensors: List<TrafficSensor>,
    irrelevantFactorThreshold: Double = (1/omod.grid.size.toDouble()).pow(1.5)
) : D2Array<Double>? {
    val n = omod.grid.size
    val totalPop = omod.buildings.sumOf { it.population }
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
                GRBLinExprBuilder,
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
            GRBLinExprBuilder,
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
                GRBLinExprBuilder,
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
        val tripStartDistr = monteCarloTripStartDistribution(MC_SAMPLES)

        // Simulated Traffic Counts
        val simCount = mutableMapOf<TrafficSensor, List<GRBLinExpr>>()
        for (sensor in sensors) {
            simCount[sensor] = List(T) { GRBLinExpr() }
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
                                simCount[sensor]!![t]
                                    .addTerm(totalPop * tripStartDistr[activity]!![t], demandVars[activity]!!)
                            }
                        }
                    }
                }
            }
        }

        // Objective
        val obj = grbMseObjective(model, sensors, simCount)
        model.setObjective(obj, GRB.MINIMIZE)

        model.optimize()

        val success = handleGrbStatus(model)
        if (success) {
            val oval = model[GRB.DoubleAttr.ObjVal]
            logger.info("Gurobi optimization finished with optimal objective: $oval")

            // Ideal transition matrix
            val result = mk.ones<Double>(omod.grid.size, omod.grid.size)
            for(o in omod.grid.indices) {
                for (d in omod.grid.indices) {
                    result[o, d] = varTransitionMatrix[o][d].get(GRB.DoubleAttr.X)
                }
            }
            model.dispose()
            env.dispose()
            return result
        }

        model.dispose()
        env.dispose()
    }  catch (e: GRBException) {
        logger.error("Gurobi Error! Error code: ${e.errorCode}. ${e.message}")
    }
    return null
}

object GRBLinExprBuilder: TermBuilder<GRBLinExpr, GRBVar> {
    override fun addTerm(term: GRBLinExpr, v: GRBVar, coefficient: Double) {
        term.addTerm(coefficient, v)
    }

    override fun addConstant(term: GRBLinExpr, constant: Double) {
        term.addConstant(constant)
    }

    override fun addSum(term: GRBLinExpr, sum: GRBLinExpr, coefficient: Double) {
        term.multAdd(coefficient, sum)
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