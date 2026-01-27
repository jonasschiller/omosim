package de.uniwuerzburg.omosim.core

import de.uniwuerzburg.omosim.core.models.*
import java.util.*
import kotlin.math.exp

/**
 * Determines with an MNL model.
 * The MNL model is only used for agent's above the minimum driving age,
 * younger agent's never have car ownership.
 *
 * @param carOwnershipUtility Coefficients of the MNL
 * @param minDrivingAge Minimum driving age
 */
class CarOwnershipMNL(
    private val carOwnershipUtility: CarOwnershipUtility,
    private val minDrivingAge: Int
) : CarOwnership {
    override fun determine(agent: MobiAgent, stratum: PopStratum, rng: Random) : Boolean {
        return rng.nextDouble() < probability(agent, stratum)
    }

    override fun probability(agent: MobiAgent, stratum: PopStratum): Double {
        return if ((agent.age != null) && (agent.age!! < minDrivingAge)) {
            0.0
        } else {
            pOwnership( agent.homogenousGroup, agent.mobilityGroup, agent.ageGrp )
        }
    }

    private fun pOwnership(
        homogenousGroup: HomogeneousGrp, mobilityGroup: MobilityGrp, age: AgeGrp
    ) : Double {
        val utility = carOwnershipUtility.calc( homogenousGroup, mobilityGroup, age )
        val p = 1 / (1 + exp(-utility))
        return p
    }
}