package de.uniwuerzburg.omod.core.models

import de.uniwuerzburg.omod.core.LocationChoiceDCWeightFun

/**
 * A location that is inside the area where OSM data is available, i.e. not a dummy location
 */
interface RealLocation : LocationOption {
    val population: Double
    val attractions: Map<Int, Double>

    fun recalculateAttractions(dcFunctions: Map<ActivityType, LocationChoiceDCWeightFun>)
}