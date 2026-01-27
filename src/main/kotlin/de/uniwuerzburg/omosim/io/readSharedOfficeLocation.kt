package de.uniwuerzburg.omosim.io

import de.uniwuerzburg.omosim.core.models.Building
import de.uniwuerzburg.omosim.utils.CRSTransformer
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.GeometryFactory
import java.io.File
import java.util.Random

//Reads in the shared office csv file and returns a list of SharedOffice locations with their coordinates

fun readSharedOfficeLocations(
    osmBuildings: List<Building>, transformer: CRSTransformer,
    geometryFactory: GeometryFactory, sharedOfficeLocationFile: File,
    rng: Random
) : List<Building> {
    logger.info("Start reading shared office data...")
    // Goal is to create a subset of buildings which are closest to a shared office location
    val officeLocations= mutableListOf<Pair<Double, Double>>()
    sharedOfficeLocationFile.forEachLine { line ->
        //Skip the header line
        if (line.startsWith("fid")) return@forEachLine
        val tokens = line.split(",")
        if (tokens.size < 3) return@forEachLine
        officeLocations.add(Pair(tokens[2].toDouble(), tokens[3].toDouble()))
    }
    val selectedBuildings = mutableSetOf<Building>()
    // Precompute building points
    val buildingPoints = osmBuildings.map { building ->
        Pair(building, building.point)
    }
    // For each shared office location, find nearest building
    for ((lon, lat) in officeLocations) {
        val officeCoord=geometryFactory.createPoint(Coordinate(lat, lon))
        val officePoint = transformer.toModelCRS(officeCoord)

        val nearest = buildingPoints.minByOrNull { (_, buildingPoint) ->
            buildingPoint.distance(officePoint)
        }

        nearest?.let { selectedBuildings.add(it.first) }
    }

    logger.info("Selected ${selectedBuildings.size} shared office buildings")

    return selectedBuildings.toList()
}