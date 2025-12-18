package de.uniwuerzburg.omod.calibration.surrogate

import com.gurobi.gurobi.*
import de.uniwuerzburg.omod.calibration.CalibrationConstants.MC_SAMPLES
import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import de.uniwuerzburg.omod.calibration.TrafficSensor
import de.uniwuerzburg.omod.calibration.differentiablemodel.TermBuilder
import de.uniwuerzburg.omod.calibration.grbMseObjective
import de.uniwuerzburg.omod.calibration.handleGrbStatus
import de.uniwuerzburg.omod.calibration.logger
import de.uniwuerzburg.omod.core.models.ActivityType
import de.uniwuerzburg.omod.core.models.RealLocation
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.set
import kotlin.math.pow

/**
 * Optimize the transition matrix of one activity directly with the gurobi solver
 * while ignoring the structure of the gravity model.
 *
 * Requirement: Gurobi must be installed and Gurobi_HOME must be added to the path.
 *
 * @param activityType Activity for which the transition matrix is optimized.
 * @param sensors Sensors with measurements
 * @param affectedSensors Gives all sensors that are affected by a certain origin-destination pair
 * @param iThresh Performance parameter.
 * All terms with coefficients below this value will be ignored and not added to the result.
 * Higher values -> Computes faster but is a rougher approximation of the markov chain representation.
 * @return Optimal transition matrix
 */
fun SGGravity.optimizeTMatrix(
    activityType: ActivityType,
    sensors: List<TrafficSensor>,
    affectedSensors: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>,
    iThresh: Double = (1/omod.grid.size.toDouble()).pow(1.5)
) : D2Array<Double>? {
    logger.info(
        "[Experimental] Calibrating transition matrix for activity $activityType directly." +
                "Number of variables: ${omod.grid.size}"
    )
    val m3rep = generateMarkovChainRep(activityType)

    val n = omod.grid.size
    val totalPop = omod.buildings.sumOf { it.population }
    val relevantODs = getRelevantODs(affectedSensors)

    try {
        // Setup
        val env = GRBEnv()
        val model = GRBModel(env)

        // Create gurobi expression of the expected trips matrix: E(o, d | Car)
        val expectedTrips = ActivityType.entries.associateWith {
            List(n) {
                List(n) {
                    GRBLinExpr()
                }
            }
        }

        // Transition matrix
        val vMatrix = List<List<GRBVar>>(n) { o ->
            model.addVars(
                DoubleArray(n) { 0.0 },
                DoubleArray(n) { 1.0 },
                DoubleArray(n) { 0.0 },
                CharArray(n) { GRB.CONTINUOUS },
                Array(n) { d -> "W_${o}_$d" }
            ).toList()
        }

        // Ensure that each row of vMatrix is a proper probability distribution
        for (o in 0 until n) {
            val rowSum = GRBLinExpr()
            for (d in 0 until n) {
                rowSum.addTerm(1.0, vMatrix[o][d])
            }
            model.addConstr(rowSum, GRB.EQUAL, 1.0, "PCondition")
        }

        // Temporal trip distribution
        val tripStartDistr = monteCarloTripStartDistribution(MC_SAMPLES)

        // Add expected trips for each destination activity
        for (activity in ActivityType.entries) {
            addE(
                GRBLinExprBuilder,
                n,
                m3rep,
                expectedTrips[activity]!!,
                vMatrix,
                relevantODs,
                iThresh,
                activity
            )
        }

        // Simulated Traffic Counts
        val simCount = mutableMapOf<TrafficSensor, List<GRBLinExpr>>()
        for (sensor in sensors) {
            simCount[sensor] = List(T) { GRBLinExpr() }
        }
        for ((o, origin) in omod.grid.withIndex()) {
            for ((d, destination) in omod.grid.withIndex()) {
                // Temporary variables to avoid GRBLinExpr().multAdd(). multAdd() leads to explosion in memory usage.
                val eODA = mutableMapOf<ActivityType, GRBVar> ()
                for (activity in ActivityType.entries) {
                    val v = model.addVar( 0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "demand")
                    model.addConstr( expectedTrips[activity]!![o][d], GRB.EQUAL, v,"demandEq")
                    eODA[activity] = v
                }

                // Add expected trips to simulated traffic counts
                val od = Pair(origin, destination)
                if (od in affectedSensors) {
                    val affected = affectedSensors[od]!!
                    for (sensor in affected) {
                        for (t in 0 until T) {
                            for (activity in ActivityType.entries) {
                                simCount[sensor]!![t]
                                    .addTerm(totalPop * tripStartDistr[activity]!![t], eODA[activity]!!)
                            }
                        }
                    }
                }
            }
        }

        // Set Objective
        val obj = grbMseObjective(model, sensors, simCount)
        model.setObjective(obj, GRB.MINIMIZE)

        // Solve
        model.optimize()

        val success = handleGrbStatus(model)
        if (success) {
            val oval = model[GRB.DoubleAttr.ObjVal]
            logger.info("Optimization (gurobi) finished with optimal objective: $oval")

            // Ideal transition matrix
            val result = mk.ones<Double>(omod.grid.size, omod.grid.size)
            for(o in omod.grid.indices) {
                for (d in omod.grid.indices) {
                    result[o, d] = vMatrix[o][d].get(GRB.DoubleAttr.X)
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

/**
 * Term builder for Gurobi.
 * Used to generate Gurobi Terms from matrix multiplications of the form: AXB,
 * where X is a matrix filled with variable terms.
 */
object GRBLinExprBuilder: TermBuilder<GRBLinExpr, GRBVar> {
    override fun addVar(term: GRBLinExpr, v: GRBVar, coefficient: Double) {
        term.addTerm(coefficient, v)
    }

    override fun addConstant(term: GRBLinExpr, constant: Double) {
        term.addConstant(constant)
    }

    override fun addTerm(term: GRBLinExpr, other: GRBLinExpr, coefficient: Double) {
        term.multAdd(coefficient, other)
    }

    override fun new(nVars: Int): GRBLinExpr {
        return GRBLinExpr()
    }
}