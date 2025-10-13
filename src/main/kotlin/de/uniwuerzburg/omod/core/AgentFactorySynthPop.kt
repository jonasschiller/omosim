package de.uniwuerzburg.omod.core

import de.uniwuerzburg.omod.core.models.ActivityType
import de.uniwuerzburg.omod.core.models.AggLocation
import de.uniwuerzburg.omod.core.models.LocationOption
import de.uniwuerzburg.omod.core.models.MobiAgent
import java.util.ArrayList
import java.util.Random
import de.uniwuerzburg.omod.core.models.*
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
) : AgentFactoryDefault
    {

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
        val synthPopAgents = getSynthPopAgents()
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
            val agent = MobiAgentBase(
                id, baseAgent.homogenousGroup, baseAgent.mobilityGroup, baseAgent.age, home, home.getAggLoc()!!.getCentralBuilding(),work, school, baseAgent.sex, carAccess = baseAgent.carAccess
            )
            agents.add(agent)
        }
        agents.shuffle(rng)
        return agents
    }


    fun getSynthPopAgents(): List<MobiAgentSSWC> {
        val SynthPop =
            readCsv(FileInputStream("C:\\Daten\\Forschung\\Sustainable Work Culture\\Daten\\Korea\\Synthetic Population\\Sejong+Daejeon\\travel_survey_preprocessed_Sejong+Daejeon.csv"))
        val agents: List<MobiAgentSSWC> = SynthPop.map { it ->
            MobiAgentSSWC(
                id = 0,
                // Working columns gives 11 different jobs and other
                // Assuming that other jobs are non-working people
                homogenousGroup = when (it.occupation) {
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 -> HomogeneousGrp.WORKING
                    97 -> HomogeneousGrp.NON_WORKING
                    else -> HomogeneousGrp.UNDEFINED
                },
                // Mobility Group is based on car ownership and bicycle ownership and driving regularly and driver license

                mobilityGroup = when {
                    it.driverLicense == 2 -> MobilityGrp.NOT_CAR
                    it.driveRegularly == 2 && it.carGroup >= 1 -> MobilityGrp.CAR_MIXED
                    it.driveRegularly == 2 && it.carGroup == 0 -> MobilityGrp.NOT_CAR
                    it.driveRegularly == 1 && it.carGroup >= 1 -> MobilityGrp.CAR_USER
                    it.carGroup == 0 -> MobilityGrp.CAR_MIXED
                    else -> MobilityGrp.UNDEFINED
                },
                age = it.age,
                sex = if (it.sex == 1) {
                    Sex.MALE
                } else {
                    Sex.FEMALE
                },
                carAccess = it.carGroup > 0,
                homeOfficeDays = it.homeOfficeDays,
                sharedOfficeRate= nextDouble(),
                DRTprobability = 0.0,
            )
        }
        return agents
    }

        //This Function reads in the csv file with the synthetic population data based on the ConTab GAN model
        fun readCsv(inputStream: InputStream): List<SynthPopAgent> {
            val reader = inputStream.bufferedReader()
            reader.readLine()
            return reader.lineSequence()
                .filter { it.isNotBlank() }
                .map {
                    val parts = it.split(',', ignoreCase = false, limit = 14)
                    SynthPopAgent().apply {
                        homeProvince = parts[0].trim()
                        homeAdministrative = parts[1].trim()
                        sex = parts[2].trim().toInt()
                        age = parts[3].trim().toInt()
                        houseType = parts[4].trim().toInt()
                        driverLicense = parts[5].trim().toInt()
                        driveRegularly = parts[6].trim().toInt()
                        commuteToFixedWorkplace = parts[7].trim().toInt()
                        occupation = parts[8].trim().toInt()
                        homeOfficeDays = parts[9].trim().toInt()
                        carGroup = parts[10].trim().toInt()
                        bicycleGroup = parts[11].trim().toInt()
                        modeChoice = parts[12].trim().toInt()
                    }
                }.toList()
        }
}