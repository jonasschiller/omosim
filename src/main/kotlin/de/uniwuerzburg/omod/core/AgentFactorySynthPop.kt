package de.uniwuerzburg.omod.core

import de.uniwuerzburg.omod.core.models.ActivityType
import de.uniwuerzburg.omod.core.models.AggLocation
import de.uniwuerzburg.omod.core.models.LocationOption
import de.uniwuerzburg.omod.core.models.MobiAgent
import java.util.ArrayList
import java.util.Random
import de.uniwuerzburg.omod.core.models.*
import de.uniwuerzburg.omod.io.synthpop.getSynthPopAgents
import java.io.FileInputStream
import java.io.InputStream
import java.util.*
import kotlin.random.Random.Default.nextDouble


/** Load synthetic population data directly from Synthetic Population files
 * Create agents based on that data
 * Assign home location based on census data
 * Assign work and school location based on destination choice model
 * Assign sociodemographic features based on the data
 * Assign car ownership based on the data
 */
class AgentFactorySynthPop(
    override val destinationFinder: DestinationFinder,
) : AgentFactoryDefault {

    /**
     * Population size is based on census data and share thereof.
     * Load the agents from a file and assign home, work, and school locations.
     * Throws an error if the population size in the file is smaller than the requested size.
     *
     * @param share Share of the population to simulate
     * @param zones Possible home locations
     * @param populateBufferArea False: Only place agents in the focus area
     * @param rng Random number generator
     * @return Population of agents
     */
    override fun createAgentsFromHomes(
        homes: List<LocationOption>,
        zones: List<AggLocation>,
        rng: Random,
    ): List<MobiAgent> {
        val synthPopAgents =
            getSynthPopAgents("C:\\Daten\\Forschung\\Sustainable Work Culture\\Daten\\Korea\\Synthetic Population\\Sejong+Daejeon\\travel_survey_preprocessed_Sejong+Daejeon.csv")
        val agents = ArrayList<MobiAgent>(homes.size)
        val shuffledAgents = synthPopAgents.shuffled(rng)
        var agentIdx = 0

        for ((id, home) in homes.withIndex()) {
            // Hole Agenten aus der gesampelten Liste, ggf. wiederhole die Liste
            val baseAgent = shuffledAgents[agentIdx % shuffledAgents.size]
            agentIdx++

            // Fixed locations
            val work = destinationFinder.getLocation(home.getAggLoc()!!, zones, ActivityType.WORK, rng)
            val school = destinationFinder.getLocation(home.getAggLoc()!!, zones, ActivityType.SCHOOL, rng)
            // currently use the center of the cell as shared office
            // Implement function based on provided list?
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
                home.getAggLoc()!!.getCentralBuilding(),
                homeOfficeDays = baseAgent.homeOfficeDays,
                sharedOfficeRate = 0.4,
            )
            agents.add(agent)
        }
        agents.shuffle(rng)
        return agents
    }
}


