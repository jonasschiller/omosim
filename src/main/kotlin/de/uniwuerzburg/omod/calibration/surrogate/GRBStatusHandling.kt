package de.uniwuerzburg.omod.calibration.surrogate

import com.gurobi.gurobi.GRB
import com.gurobi.gurobi.GRBModel
import de.uniwuerzburg.omod.calibration.logger

/**
 * Call after optimize()
 */
internal fun handleGrbStatus(model: GRBModel) : Boolean {
    var status = model[GRB.IntAttr.Status]
    // INFEASIBLE or UNBOUNDED: Retry without Presolve
    if (status == GRB.Status.INF_OR_UNBD) {
        model[GRB.IntParam.Presolve] = 0
        model.optimize()
        status = model[GRB.IntAttr.Status]
    }
    when(status) {
        GRB.Status.OPTIMAL    -> return true
        GRB.Status.INFEASIBLE -> logger.error("Gurobi error: Model is infeasible")
        GRB.Status.UNBOUNDED  ->  logger.error("Gurobi error: Model is unbounded")
        else -> logger.error("Gurobi error: Optimization was stopped with status = $status")
    }
    return false
}