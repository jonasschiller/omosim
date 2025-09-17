package de.uniwuerzburg.omod.io.json

import de.uniwuerzburg.omod.core.models.Mode
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Output format of trip
 */
@Serializable
@SerialName("AltModes")
data class AltMode (
    val mode: Mode,
    val startTime: String = "",
    val distanceKilometer: Double?,
    val timeMinute: Double?,
    val selected: Boolean?,
    val utility: Double?=null
)