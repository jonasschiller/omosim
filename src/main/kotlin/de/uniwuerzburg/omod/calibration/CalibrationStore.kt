package de.uniwuerzburg.omod.calibration

import de.uniwuerzburg.omod.core.LocationChoiceDCWeightFun
import de.uniwuerzburg.omod.core.models.ActivityType
import de.uniwuerzburg.omod.core.models.Building
import de.uniwuerzburg.omod.core.models.Cell
import de.uniwuerzburg.omod.io.json.readJson
import de.uniwuerzburg.omod.io.json.writeJson
import kotlinx.serialization.Serializable
import java.io.BufferedWriter
import java.io.File

@Serializable
class CalibrationInfo (
    val osmID: Long,
    val attractionCalibration: Map<ActivityType, Double>
)

fun write(buildings: List<Building>, file: File, dcFunctions: Map<ActivityType, LocationChoiceDCWeightFun>) {
    val store = mutableListOf<CalibrationInfo>()
    for (building in buildings) {
        val attractionCalibration = mutableMapOf<ActivityType, Double>()
        for ((activityType, dcFunction) in dcFunctions.entries) {
            val scaler = building.attractionScaler[dcFunction.id]

            if (scaler != null) {
                attractionCalibration[activityType] = scaler
            }
        }

        val info = CalibrationInfo(
            building.osmID,
            attractionCalibration
        )
        store.add(info)
    }

    writeJson(store, file)
}

fun read(cells: List<Cell>, buildings: List<Building>, file: File, dcFunctions: Map<ActivityType, LocationChoiceDCWeightFun>) {
    val store = readJson<List<CalibrationInfo>>(file)
    val mapStore = store.map { it.osmID to it.attractionCalibration }.toMap()

    for (building in buildings) {
        for ((activityType, dcFunction) in dcFunctions.entries) {
            val calibrationValue =  mapStore[building.osmID]?.get(activityType)
            if (calibrationValue != null) {
                building.attractionScaler[dcFunction.id] = calibrationValue
            }
        }
    }

    for (cell in cells) {
        cell.recalculateAttractions(dcFunctions.values.toList())
    }
}