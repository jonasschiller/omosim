package de.uniwuerzburg.omod.calibration

import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.models.ActivityType
import de.uniwuerzburg.omod.core.models.RealLocation

interface MetaModel {
    fun build(omod: Omod): MetaModel?
    fun calibrateK1(
        activityType: ActivityType,
        sensors: List<TrafficSensor>,
        affectedLinks: Map<Pair<RealLocation, RealLocation>, List<TrafficSensor>>
    ): List<Double>
}