package de.uniwuerzburg.omod.calibration

import de.uniwuerzburg.omod.calibration.CalibrationConstants.T
import java.time.LocalTime
import kotlin.math.floor

fun LocalTime.determineTimeSlice() : Int {
    val mod = this.minute + this.hour * 60
    return floor((mod % 1440.0) / 1440.0 * T).toInt()
}