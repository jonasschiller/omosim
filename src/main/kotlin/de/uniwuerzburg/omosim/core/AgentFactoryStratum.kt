package de.uniwuerzburg.omosim.core

import de.uniwuerzburg.omosim.core.models.*
import de.uniwuerzburg.omosim.utils.*
import kotlinx.coroutines.*
import java.util.*

/**
 * Creates agents by determining socio-demographic features as well as work and school locations.
 */
class AgentFactoryStratum (
    override val destinationFinder: DestinationFinder,
    private val carOwnership: CarOwnership,
    private val popStrata: List<PopStratum>,
    private val dispatcher: CoroutineDispatcher
) : AgentFactoryDefault {
    //strata Distr contains the cumulative distribution of the population strata
    private val strataDistr: DoubleArray = createCumDist(popStrata.map{it.stratumShare}.toDoubleArray())


    /**
     * @param homes Home location
     * @param zones Options for work and school locations
     * @param rng Random number generator
     * @return Population of agents
     */
    override fun createAgentsFromHomes(
        homes: List<LocationOption>, zones: List<AggLocation>, rng: Random
    ) : List<MobiAgent> {
        val agents = ArrayList<MobiAgent>(homes.size)
        for (chunk in homes.withIndex().chunked(AppConstants.nAllowedCoroutines)) {
            runBlocking(dispatcher) {
                val agentsFutures = mutableListOf<Deferred<MobiAgent>>()
                for ((id, home) in chunk) {
                    val coroutineRng = Random(rng.nextLong())
                    val agent = async {
                        createAgent(id, home, home.getAggLoc()!!, zones, coroutineRng)
                    }
                    agentsFutures.add(agent)
                }
                agents.addAll(agentsFutures.awaitAll())
            }
        }
        agents.shuffle(rng)
        return agents
    }

    /**
     * Create agent with given home.
     *
     * @param home Home of the agent. Either a Building or a DummyLocation.
     * @param homeZone Routing cell of the home. Either a Cell or a DummyLocation
     * @return Agent
     */
    private fun createAgent(
        id: Int, home: LocationOption, homeZone: AggLocation, zones: List<AggLocation>, rng: Random
    ) : MobiAgent {
        // Sociodemographic features
        val stratum = popStrata[sampleCumDist(strataDistr, rng)]
        val featureSet = stratum.sampleSocDemFeatures(rng)

        // Fixed locations
        val work = destinationFinder.getLocation(homeZone, zones, ActivityType.WORK, rng)
        val school = destinationFinder.getLocation(homeZone, zones, ActivityType.SCHOOL, rng)

        val agent = MobiAgentBase(
            id, featureSet.hom, featureSet.mob, featureSet.age, home, work, school, featureSet.sex
        )
        agent.carAccess = carOwnership.determine(agent, stratum, rng)
        return agent
    }
}