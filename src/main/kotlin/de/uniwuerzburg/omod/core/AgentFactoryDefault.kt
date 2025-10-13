package de.uniwuerzburg.omod.core

import de.uniwuerzburg.omod.core.models.*
import de.uniwuerzburg.omod.utils.*
import kotlinx.coroutines.*
import java.util.*

/**
 * Creates agents by determining socio-demographic features as well as work and school locations.
 */
interface AgentFactoryDefault : AgentFactory {

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
    val destinationFinder: DestinationFinder
    override fun createAgents(
        share: Double, zones: List<AggLocation>, populateBufferArea: Boolean, rng: Random
    ): List<MobiAgent> {
        print("Creating Population...\r")

        // Zones are aggregates created to reduce Routing effort
        // First routing between the zones then within zone
        val grid = zones.filterIsInstance<Cell>()
        // Select buildings in focus area or all buildings, selects building from each cell and flattens the list
        val buildings = if (populateBufferArea) {
            grid.flatMap { it.buildings }
        } else {
            grid.flatMap { it.buildings }.filter { it.inFocusArea }
        }


        // Determine populations of dummy zones
        // Attraction for ActivityType Home based on number of houses and POIs
        val weightSumCells = destinationFinder.getWeightsNoOrigin(grid, ActivityType.HOME).sum()
        //Calculate total population in cells might not be high as most buildings don't have population assigned
        val totalPopCells = grid.sumOf { it.population }
        // calculate home weights for all zones (cells and dummy locations)
        // Dummy locations only used if OD File is provided
        val homeWeightsZones = destinationFinder.getWeightsNoOrigin(zones, ActivityType.HOME)

        // Can mostly ignore this part since OD matrices not used in most cases
        val dummyZones = mutableListOf<DummyLocation>()
        val dummyZonePopulation = mutableListOf<Int>()
        if (populateBufferArea) {
            for ((i, zone) in zones.withIndex()) {
                if (zone is DummyLocation) {
                    val hWeight = homeWeightsZones[i]
                    val synthPop = totalPopCells * (hWeight / weightSumCells)
                    dummyZones.add(zone)
                    dummyZonePopulation.add(synthPop.toInt())
                }
            }
        }

        // Determine number of agents, get the total number of agents to create based on building population and selected share
        val totalAgentsBuildings = (buildings.sumOf { it.population } * share).toInt()
        // total population in dummy zones
        val totalAgentsDummy = (dummyZonePopulation.sum() * share).toInt()
        val totalNAgents = totalAgentsBuildings + totalAgentsDummy

        val homes = ArrayList<LocationOption>(totalNAgents)
        // Create a distribution based on building population to sample homes
        // Just adds buildings according to their population
        val buildingPopDistr = createCumDistWOR(buildings.map { it.population.toInt() }.toIntArray())
        // Distribute agents to buildings according to their population
        for (id in 0 until totalAgentsBuildings) {
            val i = sampleCumDistWOR(buildingPopDistr, rng)
            homes.add( buildings[i] )
        }
        // Living at dummy location
        val dummyPopDistr = createCumDistWOR(dummyZonePopulation.toIntArray())
        for (id in 0 until totalAgentsDummy) {
            val i = sampleCumDistWOR(dummyPopDistr, rng)
            homes.add( dummyZones[i] )
        }

        val agents = createAgentsFromHomes(homes, zones, rng)
        println("Creating Population...  Done!")
        return agents
    }

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
    override fun createAgents(
        nFocus: Int, zones: List<AggLocation>, populateBufferArea: Boolean, rng: Random
    ): List<MobiAgent> {
        logger.info("Creating Population... ")

        // Home distributions inside and outside of focus area
        val insideHWeights = getHomeWeightsRestricted(zones, true).toDoubleArray()
        val outsideHWeight = getHomeWeightsRestricted(zones, false).toDoubleArray()
        val insideCumDist = createCumDist(insideHWeights)
        val outsideCumDist = createCumDist(outsideHWeight)

        val totalNAgents = if (populateBufferArea) {
            // Check the rough proportions of in and out of focus area homes for emergency break
            val hWeights = destinationFinder.getWeightsNoOrigin(zones, ActivityType.HOME)
            val inShare = insideHWeights.sum() / hWeights.sum()
            (nFocus / inShare).toInt()
        } else {
            nFocus
        }

        // Create agent until n life in the focus area
        val homes = ArrayList<LocationOption>(totalNAgents)
        for (id in 0 until totalNAgents) {
            // Get home zone (might be cell or dummy is node)
            val inside = id < nFocus // Should agent live inside focus area
            val homeCumDist = if (inside) insideCumDist else outsideCumDist
            val homeZoneID = sampleCumDist(homeCumDist, rng)
            val homeZone = zones[homeZoneID]

            // Get home location
            val home = if (homeZone is Cell) { // Home is building
                val buildingsHomeDist = createCumDist(
                    getHomeWeightsRestricted(homeZone.buildings, inside).toDoubleArray()
                )
                homeZone.buildings[sampleCumDist(buildingsHomeDist, rng)]
            } else { // IS dummy location
                homeZone
            }
            homes.add(home)
        }

        val agents = createAgentsFromHomes(homes, zones, rng)
        logger.info("Creating Population... Done!")
        return agents
    }
    fun createAgentsFromHomes(
        homes: List<LocationOption>, zones: List<AggLocation>, rng: Random
    ) : List<MobiAgent>
    /**
     * Get the home weights of all destinations in the focus area and set the others to zero; or the other way around.
     *
     * @param destinations Destination options
     * @param inside True: Only the weights inside the focus area. False: Only those outside.
     * @return List of weights
     */
    fun getHomeWeightsRestricted(destinations: List<LocationOption>, inside: Boolean) : List<Double> {
        val originalWeights = destinationFinder.getWeightsNoOrigin(destinations, ActivityType.HOME)
        return if (inside) {
            originalWeights.mapIndexed { i, weight -> destinations[i].inFocusArea.toDouble() * weight }
        } else {
            originalWeights.mapIndexed { i, weight -> (!destinations[i].inFocusArea).toDouble() * weight }
        }
    }
}