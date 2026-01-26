package de.uniwuerzburg.omod.core

import de.uniwuerzburg.omod.core.models.AggLocation
import de.uniwuerzburg.omod.core.models.Building
import de.uniwuerzburg.omod.core.models.MobiAgent
import de.uniwuerzburg.omod.core.models.MobiAgentSSWC
import java.util.*

/**
 * Creates the population of agents.
 */
interface AgentFactory {
    /**
     * Initialize population based on a share of the existing population.
     * Assigns socio-demographic features, and home, work, and school locations.
     *
     * @param share Share of the population to simulate
     * @param zones Possible home locations
     * @param populateBufferArea False: Only place agents in the focus area
     * @param rng Random number generator
     * @return Population of agents
     */
    fun createAgents(
        share: Double, zones: List<AggLocation>, populateBufferArea: Boolean, rng: Random
    ) : List<MobiAgent>

    /**
     * Initialize population with fixed number of agents.
     * Assigns socio-demographic features, and home, work, and school locations.
     *
     * @param nFocus number of agents in focus areas
     * @param zones Possible home locations
     * @param populateBufferArea False: Only place agents in the focus area
     * @param rng Random number generator
     * @return Population of agents
     */
    fun createAgents(
        nFocus: Int, zones: List<AggLocation>, populateBufferArea: Boolean, rng: Random
    ) : List<MobiAgent>

    /** Initialize population with fixed number of agents.
     * Assigns socio-demographic features, and home, work, and school locations.
     * Assign shared office locations based on the provided list or if empty  compute shared office locations based on k-means clustering.
     * @param nFocus number of agents in focus areas
     * @param zones Possible home locations
     * @param populateBufferArea False: Only place agents in the focus area
     * @param rng Random number generator
     * @param sharedOfficeLocation List of shared office locations
     * @return Population of agents
     */

    fun createAgents(
        nFocus: Int,  zones: List<AggLocation>, populateBufferArea:Boolean, rng: Random, sharedOfficeLocation: List<Building>?
    ) : List<MobiAgentSSWC>

    /**
     * Initialize population based on a share of the existing population.
     * Assigns socio-demographic features, and home, work, and school locations.
     * Assign shared office locations based on the provided list or if empty  compute shared office locations based on k-means clustering.
     * @param nFocus  Number of agents to simulate
     * @param zones Possible home locations
     * @param populateBufferArea False: Only place agents in the focus area
     * @param rng Random number generator
     * @param sharedOfficeLocation List of shared office locations
     * @return Population of agents
     */

    fun createAgents(
        shareOfPop: Double,  zones: List<AggLocation>, populateBufferArea:Boolean, rng: Random, sharedOfficeLocation: List<Building>?
    ) : List<MobiAgentSSWC>

    /**
     * Initialize population based on a share of the existing population.
     * Assigns socio-demographic features, and home, work, and school locations.
     * Assign shared office locations based on the provided list or if empty  compute shared office locations based on k-means clustering.
     * @param share Share of the population to simulate
     * @param zones Possible home locations
     * @param populateBufferArea False: Only place agents in the focus area
     * @param rng Random number generator
     * @param sharedOfficeLocation List of shared office locations
     * @return Population of agents
     */

}