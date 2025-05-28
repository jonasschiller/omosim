package de.uniwuerzburg.omod.calibration

import org.locationtech.jts.geom.Geometry

class TrafficSensor(
    val name: String,
    val measuredFlow: Double,
    val flowDirection: Direction,
    val field: Geometry
)