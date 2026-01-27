package de.uniwuerzburg.omosim.core

import de.uniwuerzburg.omosim.core.models.ActivityType
import de.uniwuerzburg.omosim.core.models.AggLocation
import de.uniwuerzburg.omosim.core.models.LocationOption
import de.uniwuerzburg.omosim.core.models.MobiAgent
import java.util.ArrayList
import java.util.Random
import de.uniwuerzburg.omosim.core.models.*
import de.uniwuerzburg.omosim.io.synthpop.getSynthPopAgents
import java.io.File
import kotlin.collections.mapOf


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
        val synthPopAgents =
            getSynthPopAgents(synthPopFile)
        val agents = ArrayList<MobiAgent>()
        val shuffledAgents = synthPopAgents.shuffled(rng)
        var agentIdx = 0
        val sharedOfficeRateMapping = mapOf(1 to 0.05, 2 to 0.3, 3 to 0.5, 4 to 0.7, 5 to 0.95)
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
                homeOfficeDays = baseAgent.homeOfficeDays,
                sharedOfficeRate = sharedOfficeRateMapping.getOrDefault(baseAgent.sharedOfficeLike, 0.0)*baseAgent.sharedOfficeDays/5.0,
                drtLikelihood = baseAgent.drtLike,
            )
            agents.add(agent)
        }
        agents.shuffle(rng)
        return agents
    }
}


