package de.uniwuerzburg.omosim.core

import de.uniwuerzburg.omosim.core.models.MobiAgent
import de.uniwuerzburg.omosim.core.models.PopStratum
import java.util.*

/**
 * Determines car ownership with a fixed probability.
 * The probability is taken from the CarOwnership property of the stratum.
 * The probability is only applied to agent's that have an age higher than the minimum driving age.
 *
 * @param minDrivingAge Minimum driving age. Agents younger than this will never have car ownership.
 */
class CarOwnershipFixedProbability(
    private val minDrivingAge: Int
): CarOwnership {
    override fun determine(agent: MobiAgent, stratum: PopStratum, rng: Random) : Boolean {
       return  rng.nextDouble() < probability(agent, stratum)
    }

    override fun probability(agent: MobiAgent, stratum: PopStratum): Double {
        return if ((agent.age != null) && (agent.age < minDrivingAge)){
            0.0
        } else {
           stratum.carOwnership
        }
    }
}