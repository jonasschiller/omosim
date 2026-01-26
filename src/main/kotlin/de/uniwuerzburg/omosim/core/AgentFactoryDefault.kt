package de.uniwuerzburg.omosim.core

import de.uniwuerzburg.omosim.core.models.*
import de.uniwuerzburg.omosim.utils.*
import kotlinx.coroutines.*
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.Point
import org.locationtech.jts.geom.GeometryFactory
import java.util.*
import smile.clustering.kmeans

private val gf = GeometryFactory()

private fun Coordinate.toPoint(): Point =
    gf.createPoint(this)
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
        logger.get()?.info("Creating Population(%)... ")

        // Zones are aggregates created to reduce Routing effort
        // First routing between the zones then within zone
        val grid = zones.filterIsInstance<Cell>()
        // Select buildings in focus area or all buildings, selects building from each cell and flattens the list
        val buildings = if (populateBufferArea) {
            //All Buildings
            grid.flatMap { it.buildings }
        } else {
            // Only buildings in focus area
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
        repeat(totalAgentsBuildings) {
            val i = sampleCumDistWOR(buildingPopDistr, rng)
            homes.add(buildings[i])
        }
        // Living at dummy location
        val dummyPopDistr = createCumDistWOR(dummyZonePopulation.toIntArray())
        repeat(totalAgentsDummy) {
            val i = sampleCumDistWOR(dummyPopDistr, rng)
            homes.add(dummyZones[i])
        }

        val agents = createAgentsFromHomes(homes, zones, rng)
        logger.get()?.info("Creating Population(%)... Done!")
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
        logger.get()?.info("Creating Population(#)... ")

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
        logger.get()?.info("Creating Population(#)... Done!")
        return agents
    }

    override fun createAgents(
        nFocus: Int,
        zones: List<AggLocation>,
        populateBufferArea: Boolean,
        rng: Random,
        sharedOfficeLocation: List<Building>?
    ): List<MobiAgentSSWC> {
        val grid = zones.filterIsInstance<Cell>()

        val buildings = if (populateBufferArea) {
            grid.flatMap { it.buildings }
        } else {
            grid.flatMap { it.buildings }.filter { it.inFocusArea }
        }
        val agents = createAgents(nFocus, zones, populateBufferArea, rng)
        if (sharedOfficeLocation.isNullOrEmpty()) {
            assignSharedOfficeKMeans(agents as List<MobiAgentSSWC>,buildings)
        } else
            assignSharedOfficeLocationsList(agents as List<MobiAgentSSWC>, buildings, sharedOfficeLocation)
        return agents as List<MobiAgentSSWC>
    }

    override fun createAgents(
        shareOfPop: Double,
        zones: List<AggLocation>,
        populateBufferArea: Boolean,
        rng: Random,
        sharedOfficeLocation: List<Building>?
    ): List<MobiAgentSSWC> {
        val grid = zones.filterIsInstance<Cell>()

        val buildings = if (populateBufferArea) {
            grid.flatMap { it.buildings }
        } else {
            grid.flatMap { it.buildings }.filter { it.inFocusArea }
        }
        val agents = createAgents(shareOfPop, zones, populateBufferArea, rng)
        if (sharedOfficeLocation.isNullOrEmpty()) {
            assignSharedOfficeKMeans(agents as List<MobiAgentSSWC>,buildings)
        } else
            assignSharedOfficeLocationsList(agents as List<MobiAgentSSWC>, buildings, sharedOfficeLocation)
        return agents as List<MobiAgentSSWC>
    }

    fun createAgentsFromHomes(
        homes: List<LocationOption>, zones: List<AggLocation>, rng: Random
    ): List<MobiAgent>

    /**
     * Get the home weights of all destinations in the focus area and set the others to zero; or the other way around.
     *
     * @param destinations Destination options
     * @param inside True: Only the weights inside the focus area. False: Only those outside.
     * @return List of weights
     */
    fun getHomeWeightsRestricted(destinations: List<LocationOption>, inside: Boolean): List<Double> {
        val originalWeights = destinationFinder.getWeightsNoOrigin(destinations, ActivityType.HOME)
        return if (inside) {
            originalWeights.mapIndexed { i, weight -> destinations[i].inFocusArea.toDouble() * weight }
        } else {
            originalWeights.mapIndexed { i, weight -> (!destinations[i].inFocusArea).toDouble() * weight }
        }
    }

    /**
     * Assign shared office locations to agents based on a provided list of shared office locations or using K-Means clustering.
     *
     * @param agents List of agents
     * @param buildings List of all buildings in the area
     * @param sharedOfficeLocation List of shared office locations if provided
     * @return List of agents with assigned shared office locations
     */

    private fun assignSharedOfficeLocationsList(agents: List<MobiAgentSSWC>, buildings: List<Building>, sharedOfficeLocation: List<RealLocation>): List<MobiAgent> {
        val sharedOfficePairs = sharedOfficeLocation.map { sharedOffice ->
            Pair(sharedOffice, gf.createPoint(Coordinate(sharedOffice.latlonCoord.y, sharedOffice.latlonCoord.x)))
        }

         // Vorberechne die Points für alle Agenten einmalig (statt pro Iteration neu zu erzeugen)
         val agentPoints: List<Point> = agents.map { agent ->
             gf.createPoint(Coordinate(agent.home.latlonCoord.y, agent.home.latlonCoord.x))
         }

         // Für jeden Agenten das nächste Shared-Office anhand der vorberechneten Punkte finden
         agents.zip(agentPoints).forEach { (agent, agentPoint) ->
            val nearest = sharedOfficePairs.minByOrNull { (_, buildingPoint) ->
                buildingPoint.distance(agentPoint)
            }
             val distance= nearest?.second?.distance(agentPoint)
             if (distance != null && distance > 1500) {
                 println("Warning: Assigned shared office is more than 1.5 km away from agent home location ")

             }

            nearest?.let { agent.sharedOffice = it.first }
        }

         return agents
    }




    /**
     * Data class representing a home location with coordinates and clustering information.
     * @param coord Pair of latitude and longitude coordinates
     * @param cluster Assigned cluster ID
     * @param centerLat Latitude of the assigned cluster center
     * @param centerLon Longitude of the assigned cluster center
     * @param distanceKm Distance from the home location to the cluster center in kilometers
     */
    data class HomeLocation(
        val coord: Coordinate,
        var cluster: Int = -1,
        var centerCoord: Coordinate? = null,
        var distanceKm: Double = Double.POSITIVE_INFINITY
    )


    /**
     * Calculate distances from home locations to their assigned cluster centers.
     * @param clusterCenters Array of cluster center coordinates as Point
     * @param homes List of home locations with coordinates and assigned clusters.
     * @return Pair of maximum distance and 90th percentile distance in kilometers.
     */
    fun getDistance(
        clusterCenters: Array<Point?>,
        homes: List<HomeLocation>
    ): Pair<Double, Double> {

        val distances = homes.mapNotNull { home ->
            val center = clusterCenters.getOrNull(home.cluster) ?: run {
                home.centerCoord = null
                home.distanceKm = Double.POSITIVE_INFINITY
                return@mapNotNull null
            }

            home.centerCoord = Coordinate(center.coordinate)
            val distance = center.distance(home.coord.toPoint())
            home.distanceKm = distance
            distance
        }

        if (distances.isEmpty()) return 0.0 to 0.0

        val sorted = distances.sorted()
        val p90Index = (0.9 * (sorted.size - 1)).toInt()

        return sorted.last() to sorted[p90Index]
    }


    private fun assignSharedOfficeKMeans(
        agents: List<MobiAgentSSWC>,
        buildings: List<Building>
    ): List<MobiAgent> {

        val homes = agents.map {
            HomeLocation(Coordinate(it.home.latlonCoord.y, it.home.latlonCoord.x))
        }.toMutableList()

        val centers = calculateClusters(homes) ?: return agents

        return assignSharedOfficeLocations(agents, buildings, centers, homes)
    }


    /**
     * Assign shared office locations to agents based on clustering of home locations.
     *
     * @param agents List of agents
     * @param buildings List of possible shared office buildings
     * @return List of agents with assigned shared office locations
     */
    private fun assignSharedOfficeLocations(
        agents: List<MobiAgent>,
        buildings: List<Building>,
        clusterCenters: Array<Point?>,
        homes: List<HomeLocation>
    ): List<MobiAgent> {

        val closestBuildings = findClosestBuilding(clusterCenters, buildings)

        return agents.mapIndexed { index, agent ->
            val clusterId = homes.getOrNull(index)?.cluster ?: -1
            val office = closestBuildings.getOrNull(clusterId)

            if (agent is MobiAgentSSWCBase) {
                agent.copy(sharedOffice = office)
            } else agent
        }
    }


    /**
     * Find the closest building to each cluster center.
     *
     * @param clusterCenters Array of cluster center coordinates as Point
     * @param buildings List of buildings to search from
     * @return List of closest buildings to each cluster center
     */

    fun findClosestBuilding(
        clusterCenters: Array<Point?>,
        buildings: List<Building>
    ): List<Building> {

        if (clusterCenters.isEmpty() || buildings.isEmpty()) return emptyList()

        val buildingPoints = buildings.associateWith { it.latlonCoord.toPoint() }

        return clusterCenters.map { center ->
            center?.let {
                buildingPoints.minByOrNull { (_, point) ->
                    it.distance(point)
                }?.key
            } ?: buildings.first()
        }
    }


    /**
     * Calculate clusters for home locations using K-Means clustering.
     * Increases the number of clusters until the 90th percentile of distances to cluster centers is less than or equal to 6 km.
     * @param homes List of home locations with coordinates.
     * @return Array of cluster center coordinates as Point, or null if clustering fails.
    **/
    fun calculateClusters(homes: MutableList<HomeLocation>): Array<Point?>? {

        var percentile90 = Double.MAX_VALUE
        var nClusters = 1
        var centers: Array<Point?> = emptyArray()

        val data = homes.map { doubleArrayOf(it.coord.x, it.coord.y) }.toTypedArray()

        while (percentile90 > 6.0) {
            nClusters++

            val kmeans = kmeans(data, nClusters, 100, 10)

            homes.forEach { home ->
                home.cluster = kmeans.predict(doubleArrayOf(home.coord.x, home.coord.y))
            }

            centers = kmeans.centers()
                .map { (x, y) -> Coordinate(x, y).toPoint() }
                .toTypedArray()

            percentile90 = getDistance(centers, homes).second
        }

        return centers
    }

}