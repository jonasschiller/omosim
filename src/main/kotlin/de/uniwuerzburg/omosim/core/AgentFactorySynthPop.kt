package de.uniwuerzburg.omosim.core

import de.uniwuerzburg.omosim.core.models.ActivityType
import de.uniwuerzburg.omosim.core.models.AggLocation
import de.uniwuerzburg.omosim.core.models.LocationOption
import de.uniwuerzburg.omosim.core.models.MobiAgent
import java.util.ArrayList
import java.util.Random
import de.uniwuerzburg.omosim.core.models.*
import de.uniwuerzburg.omosim.io.synthpop.SynthPopAgent
import de.uniwuerzburg.omosim.io.synthpop.getSynthPopAgents
import org.apache.commons.lang3.ObjectUtils
import java.io.File
import kotlin.math.exp
import kotlin.math.ln


/** Load synthetic population data directly from Synthetic Population files
 * Create agents based on that data
 * Assign home location based on census data
 * Assign work and school location based on destination choice model
 * Assign sociodemographic features based on the data
 * Assign car ownership based on the data
 */
class AgentFactorySynthPop(
    override val destinationFinder: DestinationFinder,
    val synthPopFile: File,
    val wfhRate: Double? =null,
    val soRate: Double? =null
) : AgentFactoryDefault {

    /**
     * Population size is based on census data and share thereof.
     * Load the agents from a file and assign home, work, and school locations.
     * Throws an error if the population size in the file is smaller than the requested size.
     * @param homes Home locations
     * @param zones Possible home locations
     * @param rng Random number generator
     * @return Population of agents
     */
    override fun createAgentsFromHomes(
        homes: List<LocationOption>,
        zones: List<AggLocation>,
        rng: Random,
    ): List<MobiAgent> {
        var synthPopAgents =
            getSynthPopAgents(synthPopFile)
        val agents = ArrayList<MobiAgent>()

        if (wfhRate != null) {synthPopAgents= adjustWorkFromHomeRate(synthPopAgents,wfhRate)}
        if (soRate != null) {synthPopAgents= adjustSharedOfficeRate(synthPopAgents,soRate)  }
        val shuffledAgents = synthPopAgents.shuffled(rng)
        var agentIdx = 0
        for ((id, home) in homes.withIndex()) {
            // Hole Agenten aus der gesampelten Liste, ggf. wiederhole die Liste
            val baseAgent = shuffledAgents[agentIdx % shuffledAgents.size]
            agentIdx++

            // Fixed locations
            val work = destinationFinder.getLocation(home.getAggLoc()!!, zones, ActivityType.WORK, rng)
            val school = destinationFinder.getLocation(home.getAggLoc()!!, zones, ActivityType.SCHOOL, rng)
            // Shared Office Location is determined later based on the given locations
            val agent = MobiAgentSSWCBase(
                id,
                baseAgent.homogenousGroup,
                baseAgent.mobilityGroup,
                baseAgent.age,
                home,
                work,
                school,
                baseAgent.sex,
                carAccess = baseAgent.carAccess,
                sharedOffice=null,
                homeOfficeRate = baseAgent.homeOfficeRate,
                sharedOfficeRate = baseAgent.sharedOfficeRate,
                drtLikelihood = baseAgent.drtLike,
            )
            agents.add(agent)
        }
        agents.shuffle(rng)
        return agents
    }

    private fun adjustWorkFromHomeRate(agents: List<SynthPopAgent>, wfhRate: Double): List<SynthPopAgent> {
        val logits=agents.map { logit(it.homeOfficeRate) }.toDoubleArray()
        val alpha=findAlpha(logits, wfhRate)
        val adjusted = logits.map { l ->
            sigmoid(l + alpha)
        }.toDoubleArray()
        for (i in agents.indices) {
            agents[i].homeOfficeRate=adjusted[i]
        }
        return agents
    }

    private fun adjustSharedOfficeRate(agents: List<SynthPopAgent>, soRate: Double): List<SynthPopAgent> {
        val logits=agents.map { logit(it.sharedOfficeRate) }.toDoubleArray()
        val alpha=findAlpha(logits, soRate)
        val adjusted = logits.map { l ->
            sigmoid(l + alpha)
        }.toDoubleArray()
        for (i in agents.indices) {
            agents[i].sharedOfficeRate=adjusted[i]
        }
        return agents
    }


    fun sigmoid(x: Double): Double {
        return 1.0 / (1.0 + exp(-x))
    }

    fun logit(p: Double): Double {
        val eps = 1e-9
        val pClamped = p.coerceIn(eps, 1.0 - eps)
        return ln(pClamped / (1.0 - pClamped))
    }

    fun meanWfh(alpha: Double, logits: DoubleArray): Double {
        var sum = 0.0
        for (l in logits) {
            sum += sigmoid(l + alpha)
        }
        return sum / logits.size
    }

    fun f(alpha: Double, logits: DoubleArray, target: Double): Double {
        return meanWfh(alpha, logits) - target
    }

    fun findAlpha(
        logits: DoubleArray,
        target: Double,
        tol: Double = 1e-6,
        maxIter: Int = 100
    ): Double {

        var low = -10.0
        var high = 10.0

        // expand bounds if needed
        while (f(low, logits, target) > 0) {
            low *= 2
        }
        while (f(high, logits, target) < 0) {
            high *= 2
        }

        for (i in 0 until maxIter) {
            val mid = (low + high) / 2.0
            val value = f(mid, logits, target)

            if (kotlin.math.abs(value) < tol) {
                return mid
            }

            if (value < 0) {
                low = mid
            } else {
                high = mid
            }
        }

        return (low + high) / 2.0
    }


}


